import argparse as ap
import datetime as dt
import os
import sys

import loguru as lg

import camera
import camera.io as cio
import camera.night as cn

LOG_FILE = os.path.join(
    cio.LOGS_DIR,
    f"{dt.datetime.now().strftime(r'%Y%m%d-%H%M%S')}-night-capture.log",
)

parser = ap.ArgumentParser(
    prog="LUCID Night Capture",
    description=(
        "Capture a long night-sky exposure by stacking several camera-limited "
        "sub-exposures, e.g. 18 x 10 s for a 180 s effective exposure."
    ),
    epilog="Copyright (c) Evripidis Gkanias, 2026",
)

parser.add_argument(
    "-t", "--total-exposure",
    type=float,
    default=cn.DEFAULT_EXPOSURE,
    help="total effective exposure time in seconds; default: 300",
)
parser.add_argument(
    "-s", "--sub-exposure",
    type=float,
    default=None,
    help=(
        "individual exposure time in seconds; default: camera maximum, "
        "usually 10 s"
    ),
)
parser.add_argument(
    "-m", "--stack-method",
    choices=["sum", "mean", "median", "sigma"],
    default="sum",
    help="frame-combination method; sigma clipping removes satellite trails",
)
parser.add_argument(
    "--sigma",
    type=float,
    default=3.0,
    help="sigma-clipping threshold; default: 3.0",
)
parser.add_argument(
    "-sd", "--save-directory",
    default=None,
    help="directory in which to save the night session",
)
parser.add_argument(
    "-pf", "--pixel-format",
    default=None,
    help="camera pixel format, e.g. Mono16",
)
parser.add_argument(
    "-pl", "--print-level",
    default="INFO",
    help="log level: DEBUG, INFO, WARNING, or ERROR",
)

args = parser.parse_args()

lg.logger.remove()
lg.logger.add(sys.stderr, level=args.print_level.upper(), colorize=True)
lg.logger.add(LOG_FILE, level=args.print_level.upper(), colorize=False)

if args.pixel_format is not None:
    camera.PIXEL_FORMAT = camera.PixelFormat[args.pixel_format]
    lg.logger.info(f"Pixel format: {camera.PIXEL_FORMAT.name}")

if args.save_directory is not None:
    camera.SAVE_DIR = os.path.abspath(args.save_directory)
    os.makedirs(camera.SAVE_DIR, exist_ok=True)
    lg.logger.info(f"Save directory: {camera.SAVE_DIR}")

cam = cn.NightCamera()
input("Setting up the dark frame. Place the lid on the lens and press enter.")
dark_frame = cam(total_exposure=10, save=True)

input(f"Starting the long exposure of {args.total_exposure} seconds. Remove the lens's lid and press enter to start.")
cam(
    total_exposure=args.total_exposure,
    sub_exposure=args.sub_exposure,
    stack_method=args.stack_method,
    sigma=args.sigma,
    dark_frame=dark_frame
)
