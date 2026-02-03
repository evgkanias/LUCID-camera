import camera.io as cio

import argparse as ap
import loguru as lg
import datetime as dt
import yaml
import sys
import os

LOG_FILE = os.path.join(cio.LOGS_DIR, f"{dt.datetime.now().strftime(r'%Y%m%d-%H%M%S')}-overwrite-location.log")
with open("known-locations.yaml", 'r') as stream:
    KNOWN_LOCATIONS = yaml.safe_load(stream)

parser = ap.ArgumentParser(description="Overwrite the location in the images based on known locations.",
                           epilog='Copyright (c) Evripidis Gkanias, 2026',
                           add_help=True)

parser.add_argument(
    "sessions", nargs="*", type=str,
    default=[],
    help="Directories containing images to convert ('last' for the most recent one)"
)
parser.add_argument("-k", "--known-location",
                    type=str,
                    default=None,
                    help="Set the location based on a know location stored in the file.")
parser.add_argument("-c", "--coordinates", nargs=2,
                    type=float,
                    default=None,
                    help="Set the coordinates based on longitude and latitude.")
parser.add_argument(
    "-p", "--print-level",
    type=str,
    default="info",
    help="Set the logging level (e.g. debug, info, warning, error, critical)"
)
parser.add_argument(
    '-a', '--all',
    action='store_true',
    help='Answer yes to all questions'
)


def main(directory, longitude, latitude, altitude):
    if directory is None:
        directory = input("Enter the directory containing the images (empty to exit): ").strip().replace("'", "")

    if directory == "":
        return sys.exit(0)

    directory = get_absolute_directory(directory)
    dir_ = cio.image_dir_join(directory)

    if not os.path.exists(dir_):
        lg.logger.error(f"Directory not found: {dir_}")
        return

    base_names = cio.get_raw_image_files(dir_)
    if not base_names:
        lg.logger.error(f"No valid base names found in {dir_}")

    mata_path = os.path.join(dir_, 'info.txt')
    with open(mata_path, 'r') as f:
        info = yaml.safe_load(f)
    
    images, metas, files = cio.read_raw_images(dir_)
    for img, meta, file_ in zip(images, metas, files):
        if longitude is not None:
            info[file_[:-5]]['Longitude'] = longitude
            meta['Longitude'] = longitude
        if latitude is not None:
            info[file_[:-5]]['Latitude'] = latitude
            meta['Latitude'] = latitude
        if altitude is not None:
            info[file_[:-5]]['Altitude'] = altitude
            meta['Altitude'] = altitude

        path = os.path.join(dir_, file_)
        cio.save_image(path, img, meta)
    
    with open(mata_path, 'w') as f:
        yaml.safe_dump(info, f)

    lg.logger.success(f"All location metadata have been replaced in in {dir_}")


def get_absolute_directory(directory):
    if os.path.exists(directory):
        return os.path.abspath(directory)

    directories = []
    daemon_dir = cio.image_dir_join('Daemon')
    leaf_dir = cio.image_dir_join('Leaf')

    modes = ['HDR']
    if os.path.exists(daemon_dir):
        daemon_sess = os.listdir(daemon_dir)
        modes += [os.path.join('Daemon', ddir) for ddir in daemon_sess]
    if os.path.exists(leaf_dir):
        leaf_sess = os.listdir(leaf_dir)
        modes += [os.path.join('Leaf', ldir) for ldir in leaf_sess]

    for mode in modes:
        for dir_ in os.listdir(cio.image_dir_join(mode)):
            if dir_ == directory:
                return os.path.join(mode, directory)
            else:
                directories.append(os.path.join(mode, dir_))

    if directory == "last":
        return sorted(directories, reverse=True)[0]
    else:
        return None


if __name__ == "__main__":
    args = parser.parse_args()

    lg.logger.remove()
    lg.logger.add(sys.stderr, level=args.print_level.upper(), colorize=True, backtrace=True, diagnose=True)
    lg.logger.add(LOG_FILE, level=args.print_level.upper(), colorize=False, backtrace=True, diagnose=True)
    # lg.logger.disable('camera.io')

    if args.known_location is not None:
        lon = KNOWN_LOCATIONS[args.known_location]['lon']
        lat = KNOWN_LOCATIONS[args.known_location]['lat']
        alt = KNOWN_LOCATIONS[args.known_location]['alt']
    elif args.coordinates is not None:
        lon, lat = args.coordinates
        alt = None
    else:
        lg.logger.error("Please specify a location to overwrite.")
        sys.exit(1)

    if len(args.sessions) > 0:
        for session in args.sessions:
            main(session, lon, lat, alt)
    else:
        while True:
            main(None, lon, lat, alt)
