import camera
import camera.io as cio
import camera.transform as trans

import argparse as ap
import loguru as lg
import datetime as dt
import numpy as np
import cv2
import sys
import os

camera.SAVE_DIR = cio.image_dir_join('Leaf')
LOG_FILE = os.path.join(cio.LOGS_DIR, f"{dt.datetime.now().strftime(r'%Y%m%d-%H%M%S')}-leaf-capture.log")

parser = ap.ArgumentParser(
    prog='LUCID Polarisation Camera leaf capture',
    description='This programme connects to a LUCID camera with a polarised light sensor and allows capturing a series '
                'of images from a leaf with a rotating polariser above it.',
    epilog='Copyright (c) Evripidis Gkanias, 2026',
    add_help=True
)

parser.add_argument('-e', '--exposure', type=float, default=None,
                    help=f'set the exposure (in milliseconds); default is None')
parser.add_argument('-pl', '--print-level', default='INFO',
                    help='set the level of information to print: INFO mode prints the basic outputs, DEBUG mode prints '
                         'more outputs for debugging, and ERROR mode prints only the errors; default is INFO')
parser.add_argument('-pf', '--pixel-format',
                    help=f'set the pixel format on the camera; default is "{camera.PIXEL_FORMAT.name}"')
parser.add_argument('-ie', '--image-extension',
                    help=f'set the extension of the image that indicates the encoding format; supported extensions are:'
                         f' jpeg, jpg, bmp, raw, ply, tiff, png; default is "{camera.IMG_EXT}"')
parser.add_argument('-sd', '--save-directory',
                    help=f'set the directory where to save the images; default is in "{camera.SAVE_DIR}"')
parser.add_argument('-tr', '--transform-images',
                    help=f'flag to transform all the images')


if __name__ == '__main__':
    args = parser.parse_args()

    lg.logger.remove()
    lg.logger.add(sys.stderr, level=args.print_level.upper(), colorize=True, backtrace=True, diagnose=True)
    lg.logger.add(LOG_FILE, level=args.print_level.upper(), colorize=False, backtrace=True, diagnose=True)
    lg.logger.info(f'Print level is set to {args.print_level.upper()}.')

    if args.pixel_format is not None:
        valid_keys = [
            key for key in camera.PixelFormat.__dict__.keys() if
            key.startswith("Polarized") or
            key.startswith("Bayer") or
            key.startswith("Mono")
        ]
        assert args.pixel_format in valid_keys, (
            f"Pixel format is not supported. Available options are {', '.join(valid_keys)}"
        )
        camera.PIXEL_FORMAT = camera.PixelFormat[args.pixel_format]
        lg.logger.info(f'Pixel format is set to "{camera.PIXEL_FORMAT.name}".')

    if args.image_extension is not None:
        assert args.image_extension in ['jpeg', 'jpg', 'bmp', 'raw', 'ply', 'tiff', 'png'], (
            "Image extension is not supported."
        )
        camera.IMG_EXT = args.image_extension
        lg.logger.info(f'Image extension is set to "{camera.IMG_EXT}".')

    if args.save_directory is not None:
        camera.SAVE_DIR = os.path.abspath(args.save_directory)
        if not os.path.exists(camera.SAVE_DIR):
            os.makedirs(camera.SAVE_DIR, exist_ok=True)
            lg.logger.debug(f'Directory has been created: "{camera.SAVE_DIR}".')
        lg.logger.info(f'Directory is set as "{camera.SAVE_DIR}".')

    if args.exposure is None:
        lg.logger.info('Using auto exposure mode.')
    exposure = args.exposure

    cam = camera.Camera()
    session_dir = os.path.join(camera.SAVE_DIR, cam.timestamp)

    pol_angs = []

    while (pol_ang := input("Enter the angle of polarisation: ")) not in ["", "exit", "quite"]:
        cam(exposure, identity=pol_ang)
        pol_angs.append(pol_ang)
    lg.logger.success(f"Captured {len(pol_angs)} images and saved in {session_dir}")

    if args.transform_images:
        lg.logger.info(f"Transforming all images in {session_dir}")
        for pol_ang in pol_angs:
            img_path = os.path.join(session_dir, f"image_{pol_ang}.{camera.IMG_EXT}")
            img, meta = cio.read_raw_image(img_path)

            img_pol = trans.demosaic_polarisation(img)
            for ang in img_pol:
                # Extact RGB from polarisation images
                img_pol[ang] = trans.demosaic_bayer_malvar(img_pol[ang])

            stokes = trans.get_stokes(img_pol)
            for st in stokes:
                s_norm = cv2.normalize(stokes[st], None, 0, 65535, cv2.NORM_MINMAX)
                st16 = np.nan_to_num(s_norm, nan=0).astype(np.uint16)
                cio.save_image(os.path.join(session_dir, f'image_{pol_ang}_{st}.tiff'), st16, meta=meta)

            # Angle and degree of polarisation per channel
            angle = trans.get_polarisation_angle(stokes)
            angle_deg = np.degrees(angle)
            DoLP = trans.get_polarisation_degree(stokes)

            colours = ['red', 'green', 'blue']
            for c, colour in enumerate(colours):
                rgb = trans.angle_dolp_to_rgb(angle_deg[..., c], DoLP[..., c])
                rgb = trans.add_colormap_disk(rgb)

                # Save image
                out_path = os.path.join(session_dir, f'image_{pol_ang}_AOP_{colour}.tiff')
                cio.save_image(out_path, rgb, meta=meta)
                # cio.save_image_cv2(out_path, rgb)

            dop_rgb = trans.dop_to_rgb(DoLP)
            out_path = os.path.join(session_dir, f'image_{pol_ang}_DOP.tiff')
            cio.save_image(out_path, dop_rgb, meta=meta)
            lg.logger.success(f"All images are processed in {session_dir}")
