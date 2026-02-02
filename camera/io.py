import datetime as dt
import PIL.Image as Image
import numpy as np
import loguru as lg
import geocoder
import requests
import piexif
import yaml
import os
import re

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# retrieve the longitude and latitude based on the IP address
try:
    # get coordinates based on IP
    g = geocoder.ip('me')
    LATITUDE, LONGITUDE = g.latlng if g.ok else [None, None]
    
    # get altitude based on coordinates
    query = ('https://api.open-elevation.com/api/v1/lookup'
             f'?locations={LATITUDE:.6f},{LONGITUDE:.6f}')
    r = requests.get(query, timeout=5).json()
    ALTITUDE = r['results'][0]['elevation']

    # update the config file with the last successful location retrieval
    config['last_location']['lon'] = LONGITUDE
    config['last_location']['lat'] = LATITUDE
    config['last_location']['alt'] = ALTITUDE

    # automatically update the last successful location retriaval
    with open('config.yaml', 'w') as f:
        yaml.safe_dump(config, f)
except Exception:
    LATITUDE, LONGITUDE, ALTITUDE = [
        config['last_location']['lat'],
        config['last_location']['lon'],
        config['last_location']['alt']
    ]

IMGS_DIR = os.path.abspath("images")
LOGS_DIR = os.path.abspath("logs")
if not os.path.exists('logs'):
    os.makedirs('logs')


def image_dir_join(*path):
    return os.path.join(IMGS_DIR, *path)


def log_dir_join(*path):
    return os.path.join(LOGS_DIR, *path)


def get_session_path(directory, date=None):
    if date is None:
        date = dt.datetime.now()
    date_str = date.strftime(r"%Y%m%d")
    dir_path = os.path.join(directory, f"session_{date_str}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def get_raw_image_files(directory):
    return sorted([f for f in os.listdir(directory) if re.match(r'image\d+\.tiff$', f)])


def read_raw_images(directory):
    files = get_raw_image_files(directory)
    images = []
    metas = []
    for fname in files:
        path = os.path.join(directory, fname)

        img, meta_dict = read_raw_image(path)
        if img is not None and meta_dict is not None:
            images.append(img)
            metas.append(meta_dict)

    return images, metas, files


def read_raw_image(img_path):
    # transform to real number
    pil_img = Image.open(img_path)
    assert pil_img.mode in ['I;16', 'L'], f'Image mode is {pil_img.mode}, expected single channel image.'

    # check image mode
    if pil_img.mode == 'I;16':
        dtype = np.float64
        max_val = 65535.0
    elif pil_img.mode == 'L':
        dtype = np.float32
        max_val = 255.0

    img = np.array(Image.open(img_path), dtype=dtype) / max_val

    if img is None or img.ndim != 2:
        lg.logger.error(f"Image not found or not single channel: {img_path}")
        return None, None

    return img[:, ::-1], get_meta(piexif.load(img_path))


def save_image(path, img, meta=None):
    lg.logger.debug(f'Saving image...')
    # transform uint16 to uint8
    # Pillow does not support uint16 RGB images
    # OpenCV does not support EXIF metadata
    # Tifffile does not support EXIF but supports custom metadata
    if img.dtype == np.dtypes.UInt16DType() and img.ndim > 2:
        lg.logger.warning(f"Pillow does not support 16-bit RGB images. Reducing resolution to 8 bits for image {path}")
        img_bytes = np.uint8(img / 256)
        mode = 'RGB'
    else:
        if img.ndim > 2:
            mode = 'RGB'
        elif img.dtype == np.dtypes.UInt16DType():
            mode = 'I;16'  # non-RGB images can be 16-bit long
        elif img.dtype == np.dtypes.UInt8DType():
            mode = 'L'
        else:
            mode = None
        img_bytes = img
    pil_img = Image.fromarray(img_bytes[:, ::-1], mode=mode)
    exif = None
    if meta is not None:
        exif = get_exif_bytes(meta)
    pil_img.save(path, exif=exif, compression=None)
    lg.logger.debug(f'Saved image at {path}')


def get_exif_bytes(meta):
        exif = {"0th": {}, "Exif": {}, "GPS": {}}
       
        # Camera info
        exif["0th"][piexif.ImageIFD.Make] = meta["CameraMaker"]
        exif["0th"][piexif.ImageIFD.Model] = meta['CameraModel']
        exif["0th"][piexif.ImageIFD.UniqueCameraModel] = f"{meta['CameraMaker']} {meta['CameraModel']}"
        exif["0th"][piexif.ImageIFD.DateTime] = meta['DateTime']
        exif["0th"][piexif.ImageIFD.Software] = meta.get("Software", "HDR-LUCID-CAMERA")
        exif["0th"][piexif.ImageIFD.Artist] = meta.get("Artist", "Evripidis Gkanias")
        exif["0th"][piexif.ImageIFD.Copyright] = meta.get("Copyright", "2025, Lund Vision Group")
        
        # Exposure and image info
        exif["Exif"][piexif.ExifIFD.BodySerialNumber] = meta['CameraSerialNumber']
        exif["Exif"][piexif.ExifIFD.DateTimeOriginal] = meta['DateTime']
        if "ExposureTime" in meta:
            exif["Exif"][piexif.ExifIFD.ExposureTime] = rational(meta['ExposureTime'], 1000000)
        exif["Exif"][piexif.ExifIFD.BrightnessValue] = rational(meta['CameraBrightness'], 10)
        exif["Exif"][piexif.ExifIFD.ISOSpeedRatings] = int(meta['ISOSpeedRatings'])
        exif["Exif"][piexif.ExifIFD.FNumber] = rational(meta['FNumber'], 10)
        exif["Exif"][piexif.ExifIFD.FocalLength] = rational(meta['FocalLength'], 10)
        exif["Exif"][piexif.ExifIFD.LensMake] = meta['LensMaker']
        exif["Exif"][piexif.ExifIFD.LensModel] = meta['LensModel']
        exif["Exif"][piexif.ExifIFD.Temperature] = rational(meta['CameraTemperature'], 10)

        if 'Gamma' in meta:
            exif["Exif"][piexif.ExifIFD.Gamma] = rational(meta['Gamma'], 10)

        # GPS info
        deg_lat, min_lat, sec_lat, sig_lat = dms(meta['Latitude'])
        deg_lon, min_lon, sec_lon, sig_lon = dms(meta['Longitude'])

        exif["GPS"] = {
            piexif.GPSIFD.GPSLatitudeRef: 'N' if sig_lat >= 0 else 'S',
            piexif.GPSIFD.GPSLatitude: (rational(deg_lat, 1), rational(min_lat, 1), rational(sec_lat, 100)),
            piexif.GPSIFD.GPSLongitudeRef: 'E' if sig_lon >= 0 else 'W',
            piexif.GPSIFD.GPSLongitude: (rational(deg_lon, 1), rational(min_lon, 1), rational(sec_lon, 100)),
            piexif.GPSIFD.GPSAltitudeRef: 0 if meta['Altitude'] >= 0 else 1,
            piexif.GPSIFD.GPSAltitude: rational(abs(meta['Altitude']), 1)
        }

        exif_bytes = piexif.dump(exif)
        return exif_bytes


def get_meta(exif):
    # Camera info
    meta = {
        "CameraMaker": exif["0th"][piexif.ImageIFD.Make],
        "CameraModel": exif["0th"][piexif.ImageIFD.Model],
        "DateTime": exif["Exif"][piexif.ExifIFD.DateTimeOriginal],
        "Software": exif["0th"][piexif.ImageIFD.Software],
        "Artists": exif["0th"][piexif.ImageIFD.Artist],
        "Copyright": exif["0th"][piexif.ImageIFD.Copyright],
    
        # Exposure and image info
        "CameraSerialNumber": exif["Exif"][piexif.ExifIFD.BodySerialNumber],
        "ExposureTime": unrational(*exif["Exif"][piexif.ExifIFD.ExposureTime]),
        "CameraBrightness": unrational(*exif["Exif"][piexif.ExifIFD.BrightnessValue]),
        "ISOSpeedRatings": exif["Exif"][piexif.ExifIFD.ISOSpeedRatings],
        "FNumber": unrational(*exif["Exif"][piexif.ExifIFD.FNumber]),
        "FocalLength": unrational(*exif["Exif"][piexif.ExifIFD.FocalLength]),
        "LensMaker": exif["Exif"][piexif.ExifIFD.LensMake],
        "LensModel": exif["Exif"][piexif.ExifIFD.LensModel],
        "CameraTemperature": unrational(*exif["Exif"][piexif.ExifIFD.Temperature]),

        # GPS info
        "Latitude": undms(*([unrational(*val) for val in exif["GPS"][piexif.GPSIFD.GPSLatitude]] +
                            [1.0 if exif["GPS"][piexif.GPSIFD.GPSLatitudeRef] == b'N' else -1])),
        "Longitude": undms(*([unrational(*val) for val in exif["GPS"][piexif.GPSIFD.GPSLongitude]] +
                            [1.0 if exif["GPS"][piexif.GPSIFD.GPSLongitudeRef] == b'E' else -1])),
        "Altitude": unrational(*exif["GPS"][piexif.GPSIFD.GPSAltitude]) * (1 if exif["GPS"][piexif.GPSIFD.GPSAltitudeRef] >= 0 else -1)
    }

    if piexif.ExifIFD.Gamma in exif["Exif"]:
        meta['Gamma'] = unrational(*exif["Exif"][piexif.ExifIFD.Gamma])
        
    return meta


def save_metadata(dir_path, meta):
    path = os.path.join(dir_path, 'info.txt')
    with open(path, 'w') as f:
        yaml.safe_dump(meta, f)
    lg.logger.debug(f'Saved metadata at: {path}.')


def dms(decimal):
    sign = np.sign(decimal)
    decimal_degrees = abs(decimal)

    degrees = int(decimal_degrees)
    decimal_part = decimal_degrees - degrees

    minutes = int(decimal_part * 60)
    decimal_minutes = (decimal_part * 60) - minutes

    seconds = round(decimal_minutes * 60)

    return degrees, minutes, seconds, sign


def undms(degrees, minutes, seconds, sign):
    """Convert DMS (degrees, minutes, seconds, sign) back to decimal degrees."""
    decimal = float(np.round(abs(degrees) + minutes / 60.0 + seconds / 3600.0, decimals=4))
    return sign * decimal


def rational(value, precision=100000):
    numerator = int(np.round(value * precision))
    denominator = precision
    return (numerator, denominator)


def unrational(*rational_tuple):
    """Convert a rational tuple (numerator, denominator) back to float."""
    numerator, denominator = rational_tuple
    return float(numerator) / float(denominator)
