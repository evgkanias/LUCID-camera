import numpy as np
import cv2


def demosaic_bayer(img, blur_kernel=3):
    # read the first 4x4 block
    r00 = img[0:-2:2, 0:-2:2]
    g01 = img[0:-2:2, 1:-2:2]
    r02 = img[0:-2:2, 2::2]

    g10 = img[1:-2:2, 0:-2:2]
    b11 = img[1:-2:2, 1:-2:2]
    g12 = img[1:-2:2, 2::2]
    b13 = img[1:-2:2, 3::2]

    r20 = img[2::2, 0:-2:2]
    g21 = img[2::2, 1:-2:2]
    r22 = img[2::2, 2::2]
    g23 = img[2::2, 3::2]

    b31 = img[3::2, 1:-2:2]
    g32 = img[3::2, 2::2]
    b33 = img[3::2, 3::2]

    # calculate the missing channels
    g11 = (0.25 * g01 + 0.25 * g10 + 0.25 * g12 + 0.25 * g21).astype(img.dtype)
    r11 = (0.25 * r00 + 0.25 * r02 + 0.25 * r20 + 0.25 * r22).astype(img.dtype)

    b12 = (0.5 * b11 + 0.5 * b13).astype(img.dtype)
    r12 = (0.5 * r02 + 0.5 * r22).astype(img.dtype)

    b21 = (0.5 * b11 + 0.5 * b31).astype(img.dtype)
    r21 = (0.5 * r20 + 0.5 * r22).astype(img.dtype)

    b22 = (0.25 * b11 + 0.25 * b13 + 0.25 * b31 + 0.25 * b33).astype(img.dtype)
    g22 = (0.25 * g12 + 0.25 * g21 + 0.25 * g23 + 0.25 * g32).astype(img.dtype)

    rgb = np.zeros((img.shape[0] - 2, img.shape[1] - 2, 3), dtype=img.dtype)
    rgb[0::2, 0::2, 0] = r11
    rgb[0::2, 0::2, 1] = g11
    rgb[0::2, 0::2, 2] = b11
    rgb[0::2, 1::2, 0] = r12
    rgb[0::2, 1::2, 1] = g12
    rgb[0::2, 1::2, 2] = b12
    rgb[1::2, 0::2, 0] = r21
    rgb[1::2, 0::2, 1] = g21
    rgb[1::2, 0::2, 2] = b21
    rgb[1::2, 1::2, 0] = r22
    rgb[1::2, 1::2, 1] = g22
    rgb[1::2, 1::2, 2] = b22

    if blur_kernel > 0:
        rgb = cv2.blur(rgb, (blur_kernel,) * 2)

    return rgb


def demosaic_bayer_simple(img):
    r00 = img[0::2, 0::2]
    g01 = img[0::2, 1::2]
    g10 = img[1::2, 0::2]
    b11 = img[1::2, 1::2]
    g0 = 0.5 * g01 + 0.5 * g10
    rgb0 = np.transpose([r00, g0, b11], axes=(1, 2, 0))

    r20 = img[2::2, 0::2]
    g21 = img[2::2, 1::2]
    g1 = 0.5 * g21 + 0.5 * g10[:-1, :]
    rgb1 = np.transpose([r20, g1, b11[1:]], axes=(1, 2, 0))

    r02 = img[0::2, 2::2]
    g12 = img[1::2, 2::2]
    g2 = 0.5 * g01[:, :-1] + 0.5 * g12
    rgb2 = np.transpose([r02, g2, b11[:, 1:]], axes=(1, 2, 0))

    r22 = img[2::2, 2::2]
    g3 = 0.5 * g12[:-1, :] + 0.5 * g21[:, :-1]
    rgb3 = np.transpose([r22, g3, b11[1:, 1:]], axes=(1, 2, 0))

    rgb = np.empty((rgb0.shape[0] + rgb1.shape[0], rgb0.shape[1] + rgb2.shape[1], rgb0.shape[2]), dtype=img.dtype)
    rgb[0::2, 0::2] = rgb0
    rgb[1::2, 0::2] = rgb1
    rgb[0::2, 1::2] = rgb2
    rgb[1::2, 1::2] = rgb3

    return rgb


def demosaic_bayer_malvar(img):
    r02 = img[0:-5:2, 2:-3:2]
    g03 = img[0:-5:2, 3:-2:2]

    b11 = img[1:-4:2, 1:-4:2]
    g12 = img[1:-4:2, 2:-3:2]
    b13 = img[1:-4:2, 3:-2:2]
    g14 = img[1:-4:2, 4:-1:2]

    r20 = img[2:-3:2, 0:-5:2]
    g21 = img[2:-3:2, 1:-4:2]
    r22 = img[2:-3:2, 2:-3:2]
    g23 = img[2:-3:2, 3:-2:2]
    r24 = img[2:-3:2, 4:-1:2]
    g25 = img[2:-3:2, 5::2]

    g30 = img[3:-2:2, 0:-5:2]
    b31 = img[3:-2:2, 1:-4:2]
    g32 = img[3:-2:2, 2:-3:2]
    b33 = img[3:-2:2, 3:-2:2]
    g34 = img[3:-2:2, 4:-1:2]
    b35 = img[3:-2:2, 5::2]

    g41 = img[4:-1:2, 1:-4:2]
    r42 = img[4:-1:2, 2:-3:2]
    g43 = img[4:-1:2, 3:-2:2]
    r44 = img[4:-1:2, 4:-1:2]

    g52 = img[5::2, 2:-3:2]
    b53 = img[5::2, 3:-2:2]

    R00 = 8 * r22
    G00 = 4 * r22 + 2 * (g12 + g21 + g23 + g32) - (r02 + r20 + r24 + r42)
    B00 = 6 * r22 + 2 * (b11 + b13 + b31 + b33) - 1.5 * (r20 + r24 + r02 + r42)

    R01 = 5 * g32 + 4 * (r22 + r42) - (g12 + g52 + g21 + g41 + g23 + g43) + 0.5 * (g30 + g34)
    G01 = 8 * g23
    B01 = 5 * g32 + 4 * (b31 + b33) - (g30 + g34 + g21 + g41 + g23 + g43) + 0.5 * (g12 + g52)

    R10 = 5 * g23 + 4 * (r22 + r24) - (g21 + g25 + g12 + g14 + g32 + g34) + 0.5 * (g03 + g43)
    G10 = 8 * g32
    B10 = 5 * g23 + 4 * (b13 + b33) - (g03 + g43 + g12 + g41 + g32 + g34) + 0.5 * (g21 + g25)

    R11 = 6 * b33 + 2 * (r22 + r24 + r42 + r44) - 1.5 * (b31 + b35 + b13 + b53)
    G11 = 4 * b33 + 2 * (g23 + g32 + g34 + g43) - (b13 + b31 + b35 + b53)
    B11 = 8 * b33

    rgb00 = np.transpose([R00, G00, B00], axes=(1, 2, 0)) / 8
    rgb01 = np.transpose([R01, G01, B01], axes=(1, 2, 0)) / 8
    rgb10 = np.transpose([R10, G10, B10], axes=(1, 2, 0)) / 8
    rgb11 = np.transpose([R11, G11, B11], axes=(1, 2, 0)) / 8

    rgb = np.empty((rgb00.shape[0] + rgb01.shape[0], rgb10.shape[1] + rgb11.shape[1], rgb00.shape[2]),
                   dtype=img.dtype)
    rgb[0::2, 0::2] = rgb00
    rgb[1::2, 0::2] = rgb01
    rgb[0::2, 1::2] = rgb10
    rgb[1::2, 1::2] = rgb11

    return rgb


def demosaic_bayer_cv(img):
    # Assumes BGGR Bayer pattern
    return cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)


def demosaic_polarisation(img):
    # Assumes 2x2 pattern: [000, 045; 090, 135]
    h, w = img.shape[:2]
    return {
        '000': img[0:h:2, 0:w:2],
        '135': img[0:h:2, 1:w:2],
        '045': img[1:h:2, 0:w:2],
        '090': img[1:h:2, 1:w:2]
    }


def get_stokes(pol_dict):
    I000 = np.float32(pol_dict['000']) / 255.0
    I045 = np.float32(pol_dict['045']) / 255.0
    I090 = np.float32(pol_dict['090']) / 255.0
    I135 = np.float32(pol_dict['135']) / 255.0

    I_1 = I000 + I090 + np.finfo(float).eps
    I_2 = I045 + I135 + np.finfo(float).eps
    I = (I_1 + I_2) / 2

    Q = (I000 - I090) / I_1
    U = (I045 - I135) / I_2

    return {
        'S0': I,
        'S1': Q,
        'S2': U
    }


def get_polarisation_angle(stokes):
    return (0.5 * np.arctan2(stokes['S1'], stokes['S2'])) % np.pi


def get_polarisation_degree(stokes):
    return np.clip(np.sqrt(np.square(stokes['S1']) + np.square(stokes['S2'])), 0, 1)


# Function to add color map disk legend at bottom right
def add_colormap_disk(img, radius=48, margin=10):
    disk_size = radius * 2
    legend = np.zeros((disk_size, disk_size, 3), dtype=np.uint8)
    cy, cx = radius, radius
    y, x = np.ogrid[:disk_size, :disk_size]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    angles = (np.arctan2((y - cy), -(x - cx)) * 180 / np.pi) % 180  # 0-180 deg
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
def angle_dolp_to_rgb(angle_channel, dolp_channel):
    hue = np.round((np.nan_to_num(angle_channel, nan=0) % 180) / 180.0 * 179).astype(np.uint8)
    value = (np.clip(np.nan_to_num(dolp_channel, nan=0), 0, 1) * 255).astype(np.uint8)
    hsv = np.zeros((*hue.shape, 3), dtype=np.uint8)
    hsv[..., 0] = hue
    hsv[..., 1] = 255
    hsv[..., 2] = value
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def dop_to_rgb(dop):
    return (np.clip(np.nan_to_num(dop, nan=0), 0, 1) * 65535).astype(np.uint16)
