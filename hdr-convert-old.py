import os
import sys
import cv2
import numpy as np
import yaml
import datetime as dt

import loguru as lg

LOG_FILE = os.path.join('logs', f'{dt.datetime.now().strftime(r'%Y%m%d-%H%M%S')}-hdr-convert.log')
if not os.path.exists('logs'):
    os.makedirs('logs')

def read_images(directory, base_name):
    angles = ['000', '045', '090', '135']
    filenames = [f"{base_name}_{angle}.tiff" for angle in angles]
    images = []
    for fname in filenames:
        path = os.path.join(directory, fname)
        img = cv2.imread(path, cv2.IMREAD_COLOR_BGR)
        if img is None:
            lg.logger.error(f"Image not found: {path}")
            raise FileNotFoundError(f"Image not found: {path}")
        images.append(img.astype(np.float32))
    return images

def compute_stokes(I0, I45, I90, I135):
    S0 = I0 + I90
    S1 = I0 - I90
    S2 = I45 - I135
    return S0, S1, S2

def save_images(directory, stokes, base_name):
    names = [f"{base_name}_S0.tiff", f"{base_name}_S1.tiff", f"{base_name}_S2.tiff"]
    for s, name in zip(stokes, names):
        s_norm = cv2.normalize(s, None, 0, 65535, cv2.NORM_MINMAX)
        s_uint16 = s_norm.astype(np.uint16)
        out_path = os.path.join(directory, name)
        cv2.imwrite(out_path, s_uint16)
        lg.logger.debug(f"Saved {out_path}")

def find_base_names(directory):
    files = os.listdir(directory)
    base_names = set()
    for fname in files:
        if fname.endswith('.tiff'):
            parts = fname.split('_')
            if len(parts) == 2 and parts[1][:3] in {'000', '045', '090', '135'}:
                base_names.add(parts[0])
    return sorted(base_names)

# Function to add color map disk legend at bottom right
def add_colormap_disk(img, radius=48, margin=10):
    disk_size = radius * 2
    legend = np.zeros((disk_size, disk_size, 3), dtype=np.uint8)
    cy, cx = radius, radius
    y, x = np.ogrid[:disk_size, :disk_size]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    angles = (np.arctan2(-(x - cx), (y - cy)) * 180 / np.pi) % 180  # 0-180 deg
    hue = ((angles / 180.0) * 179).astype(np.uint8)
    value = np.ones_like(hue, dtype=np.uint8) * 255
    hsv = np.zeros_like(legend)
    hsv[..., 0] = hue
    hsv[..., 1] = 255
    hsv[..., 2] = value
    legend_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    legend[mask] = legend_bgr[mask]
    # Transparent background outside disk
    legend[~mask] = img[0,0] if img.ndim == 3 else 0

    # Place legend at bottom right
    h, w = img.shape[:2]
    img_out = img.copy()
    y1 = h - disk_size - margin
    y2 = h - margin
    x1 = w - disk_size - margin
    x2 = w - margin
    # Blend disk with background if needed
    mask3 = np.stack([mask]*3, axis=-1)
    roi = img_out[y1:y2, x1:x2]
    roi[mask3] = legend[mask3]
    img_out[y1:y2, x1:x2] = roi
    return img_out

# Function to map angle and DoLP to HSV and then to BGR
def angle_dolp_to_bgr(angle_channel, dolp_channel):
    hue = ((angle_channel % 180) / 180.0 * 179).astype(np.uint8)
    value = (np.clip(dolp_channel, 0, 1) * 255).astype(np.uint8)
    hsv = np.zeros((*hue.shape, 3), dtype=np.uint8)
    hsv[..., 0] = hue
    hsv[..., 1] = 255
    hsv[..., 2] = value
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

def dop_to_bgr(dop):
    return (np.clip(dop, 0, 1) * 255).astype(np.uint8)

def main():
    if len(sys.argv) < 2:
        directory = input("Enter the directory containing the images (empty to process all): ").strip()
    else:
        directory = sys.argv[1]

    if directory == "" or directory == "last":
        directories = []
        hdr_dir = os.path.join('images', 'HDR')
        daemon_dir = os.path.join('images', 'Daemon')
        if os.path.exists(daemon_dir):
            daemon_sess = os.listdir(os.path.join('images', 'Daemon'))
            modes = ['HDR'] + [os.path.join('Daemon', dirr) for dirr in daemon_sess]
        else:
            modes = ['HDR']
        for mode in modes:
            # lg.logger.info(f"{mode=}")
            for dir_ in os.listdir(os.path.join('images', mode)):
                directories.append(os.path.join(mode, dir_))
    
        if directory == "":
            directory = directories
        elif directory == "last":
            directory = [sorted(directories, reverse=True)[0]]
    else:
        directory = [directory]

    print(f"{directory=}")
        
    for dir_ in directory:

        dir_ = os.path.join('images', dir_)
        base_names = find_base_names(dir_)
        if not base_names:
            lg.logger.error("No valid base names found in the directory.")
            return
        lg.logger.info(f"Found base names: {base_names}")
        
        overwrite = True
        if any(['HDR' in f for f in os.listdir(dir_)]):
            overwrite = input(f"HDR files exist in '{dir_}'. Would you like to overwite? [y/N] ").lower() in ['y', 'yes', 'ja']
        if not overwrite:
            continue

        # Read exposure times from info.txt
        info_path = os.path.join(dir_, "info.txt")
        exposure_times = {}
        with open(info_path, "r") as f:
            info = yaml.safe_load(f)
            exposure_times = info['exposures (ms)']

        # Collect all base names and their exposure times
        angle_labels = ['000', '045', '090', '135']
        hdr_images = {angle: [] for angle in angle_labels}
        
        for bname in base_names:
            imgs = read_images(dir_, bname)
            for i, angle in enumerate(angle_labels):
                hdr_images[angle].append(imgs[i])

        hdr_times = np.array(exposure_times, dtype=np.float32) * 1e-3  # transform to seconds

        # Create HDR image for each angle
        hdr_results = {}
        for angle in angle_labels:
            imgs = hdr_images[angle]
            if len(imgs) < 2:
                lg.logger.error(f"Not enough images for HDR merge at angle {angle}")
                continue
            for k in range(len(imgs)):
                imgs[k] = np.uint8(imgs[k])
            
            merge_debevec = cv2.createMergeDebevec()
            hdr = merge_debevec.process(imgs, times=hdr_times.copy())
            hdr_results[angle] = hdr

            # Save HDR as 16-bit tiff (tone-mapped for visualization)
            tonemap = cv2.createTonemap(gamma=2.2)
            ldr = tonemap.process(hdr)
            ldr_16 = np.clip(ldr * 65535, 0, 65535).astype(np.uint16)
            out_path = os.path.join(dir_, f"polar_HDR_{angle}.tiff")
            cv2.imwrite(out_path, ldr_16)
            lg.logger.debug(f"Saved HDR image {out_path}")
        lg.logger.info("Saved HDR images from indivicual filters.")

        # Compute Stokes parameters, angle and degree of polarisation from HDR images
        if all(angle in hdr_results for angle in angle_labels):
            I0 = hdr_results['000']
            I45 = hdr_results['045']
            I90 = hdr_results['090']
            I135 = hdr_results['135']
            S0, S1, S2 = compute_stokes(I0, I45, I90, I135)
            save_images(dir_, (S0, S1, S2), "polar_HDR")
            lg.logger.info("Saved Stokes parameters.")

            # Calculate intensity, angle, and degree of polarization
            angle = 0.5 * np.arctan2(S2, S1)
            # Calculate angle of polarization in degrees for each channel
            angle_deg = np.degrees(angle)

            # Split into R, G, B channels
            angle_deg_r = angle_deg[..., 2]
            angle_deg_g = angle_deg[..., 1]
            angle_deg_b = angle_deg[..., 0]

            # Calculate DoLP for value channel (normalized to 0-1)
            DoLP = np.sqrt(S1**2 + S2**2) / (S0 + 1e-8)
            DoLP_clip = np.clip(DoLP, 0, 1)

            # Create RGB images for each channel using DoLP as value
            rgb_r = angle_dolp_to_bgr(angle_deg_r, DoLP_clip[..., 2])
            rgb_g = angle_dolp_to_bgr(angle_deg_g, DoLP_clip[..., 1])
            rgb_b = angle_dolp_to_bgr(angle_deg_b, DoLP_clip[..., 0])

            # Add disk legend to each image
            rgb_r = add_colormap_disk(rgb_r)
            rgb_g = add_colormap_disk(rgb_g)
            rgb_b = add_colormap_disk(rgb_b)

            # Save images
            out_names = [
                "polar_HDR_AOP_red.tiff",
                "polar_HDR_AOP_green.tiff",
                "polar_HDR_AOP_blue.tiff",
                "polar_HDR_DOP.tiff"
            ]
            out_images = [
                rgb_r, rgb_g, rgb_b,
                dop_to_bgr(DoLP)
            ]
            for img, name in zip(out_images, out_names):
                out_path = os.path.join(dir_, name)
                cv2.imwrite(out_path, img)
                lg.logger.debug(f"Saved {out_path}")

            lg.logger.success(f"HDR Stokes and polarization images saved in {dir_}")
        else:
            lg.logger.error("HDR images for all angles not available, skipping HDR Stokes computation.")

if __name__ == "__main__":

    print_level = 'INFO'
    
    lg.logger.remove()
    lg.logger.add(sys.stderr, level=print_level, colorize=True, backtrace=True, diagnose=True)
    lg.logger.add(LOG_FILE, level=print_level, colorize=False, backtrace=True, diagnose=True)
    lg.logger.add(sys.stderr, level='INFO')

    main()