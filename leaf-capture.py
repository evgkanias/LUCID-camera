import camera
import camera.io as cio
import argparse as ap
import loguru as lg
import datetime as dt
import sys
import os

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

    while (pol_ang := input("Enter the angle of polarisation: ")) not in ["", "exit", "quite"]:
        cam(exposure, identity=pol_ang)
