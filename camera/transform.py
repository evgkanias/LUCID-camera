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
