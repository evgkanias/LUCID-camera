import camera
import camera.io as cio

import argparse as ap
import loguru as lg
import datetime as dt
import sys
import os

DEFAULT_EXPOSURES = cio.config['default_exposures']  # in milliseconds
EXPOSURE_TYPE = cio.config['camera']['exposure_type'].lower()  # 'auto' or 'manual'

LOG_FILE = os.path.join(cio.LOGS_DIR, f'{dt.datetime.now().strftime(r'%Y%m%d-%H%M%S')}-hdr-capture.log')

parser = ap.ArgumentParser(
    prog='LUCID Polarisation Camera HDR',
    description='This programme connects to a LUCID camera with a polarisated light sensor and allows capturing a series of images with different exposures, enabling a high dynamic range (HDR) for each polarisation angle.',
    epilog='Copyright (c) Evripidis Gkanias, 2025',
    add_help=True
)

parser.add_argument('-e', '--exposures', nargs='*', type=float,
                    help=f'set a list of exposures (in milliseconds); default is {DEFAULT_EXPOSURES}')
parser.add_argument('-s', '--step-size', type=int,
                    help=f'set the step size among the exposures when using auto-exposure; default is {camera.STEP_SIZE}')
parser.add_argument('-n', '--number-exposures', type=int,
                    help=f'set the number of exposures when using auto-exposure; default is {camera.NB_EXPOSURES}')
parser.add_argument('-pl', '--print-level', default='INFO',
                    help='set the level of information to pring: INFO mode prints the basic outputs, DEBUG mode prints more outputs for debuging, and ERROR mode prints only the errors; default is INFO')
parser.add_argument('-pf', '--pixel-format',
                    help=f'set the pixel format on the camera; default is "{camera.PIXEL_FORMAT.name}"')
parser.add_argument('-ie', '--image-extension',
                    help=f'set the extension of the image that indicates the encoding format; supportyed extensions are: jpeg, jpg, bmp, raw, ply, tiff, png; default is "{camera.IMG_EXT}"')
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

    if EXPOSURE_TYPE == 'manual':
        lg.logger.info('Using manual exposure mode.')
        if args.exposures is not None:
            exposures = args.exposures
        else:
            exposures = DEFAULT_EXPOSURES
        
        cam = camera.HDRCamera()
        cam(*exposures)
    elif EXPOSURE_TYPE == 'auto':
        lg.logger.info('Using auto exposure mode.')
        if args.step_size is not None:
            step_size = args.step_size
        else:
            step_size = None

        if args.number_exposures is not None:
            nb_exposures = args.number_exposures
        else:
            nb_exposures = None
        if args.exposures is None:
            exposures = []
        else:
            exposures = args.exposures

        cam = camera.HDRCameraAuto()
        cam(*exposures, step_size=step_size, nb_exposures=nb_exposures)
