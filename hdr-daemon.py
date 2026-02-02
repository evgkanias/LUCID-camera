import os
import threading
import time
import datetime as dt
import loguru as lg
import sys

import camera
import camera.io as cio
import argparse

EXPOSURE_TYPE = cio.config['camera']['exposure_type'].lower()
CAM_ARGS = []
CAM_KWARGS = {}
if EXPOSURE_TYPE == 'manual':
    HDRCameraClass = camera.HDRCamera
    CAM_ARGS.extend(cio.config['default_exposures'])  # in milliseconds
elif EXPOSURE_TYPE == 'auto':
    HDRCameraClass = camera.HDRCameraAuto
    CAM_KWARGS['step_size'] = cio.config['default_step_size']
    CAM_KWARGS['nb_exposures'] = cio.config['default_nb_exposures']
else:
    raise ValueError(f"Invalid exposure type: {EXPOSURE_TYPE}. Must be 'manual' or 'auto'.")
CHECK_INTERVAL = cio.config['daemon']['intervals']
DATE_START = dt.datetime.now()
DATE_STR = DATE_START.strftime(r'%Y%m%d')
TIME_STR = DATE_START.strftime(r'%H%M%S')
SAVE_DIR = cio.image_dir_join("Daemon")
LOG_FILE = cio.log_dir_join(f'{DATE_STR}-{TIME_STR}-hdr-daemon.log')


def user_input_listener(stop_event):
    while not stop_event.is_set():
        user_input = input("Type 'exit' or 'quit' and then 'Enter' at any time to terminate the process\n")
        if user_input.strip().lower() in ("exit", "quit"):
            stop_event.set()


def main():
    stop_event = threading.Event()
    input_thread = threading.Thread(target=user_input_listener, args=(stop_event,), daemon=True)
    input_thread.start()

    image_count = 0

    while not stop_event.is_set():
        
        # Wait until camera is free
        while not stop_event.is_set():
            try:
                # Try to reserve camera
                cam = HDRCameraClass()
                break
            except Exception as e:
                lg.logger.warning('Camera is probably in use. Waiting for 5 sec...\n{e}')
                time.sleep(5)
        if stop_event.is_set():
            break

        camera.SAVE_DIR = cio.get_session_path(SAVE_DIR, DATE_START)
        start_time = camera.tic()

        # Capture HDR image
        cam(*CAM_ARGS, **CAM_KWARGS)
        image_count += 1

        # Release camera for other use
        del cam

        # Calculate time to next start
        elapsed = camera.toc(start_time)
        remaining = CHECK_INTERVAL - elapsed
        if remaining > 0:
            for _ in range(int(remaining)):
                if stop_event.is_set():
                    break
                time.sleep(1)

    if image_count > 0:
        lg.logger.success(f"Successfully captured {image_count} HDR images for this session.")
    else:
        lg.logger.info(f"Exiting HDR capture script.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDR Camera Daemon")
    parser.add_argument(
        "-e", "--exposures",
        type=float,
        nargs="+",
        default=CAM_ARGS,
        help="List of exposure times in milliseconds (e.g. --exposures 10 5 1 0.5 0.1 0.05 0.01)"
    )
    parser.add_argument(
        "-n", "--nb-exposures",
        type=int,
        default=CAM_KWARGS.get('nb_exposures', None),
        help="Number of exposures for auto-exposure mode (default: from config file)"
    )
    parser.add_argument(
        "-s", "--step-size",
        type=int,
        default=CAM_KWARGS.get('step_size', None),
        help="Step size (proportional to exposure time) for auto-exposure mode (default: from config file)"
    )
    parser.add_argument(
        "-i", "--interval",
        type=int,
        default=CHECK_INTERVAL,
        help="Time interval between captures in seconds (default: 1800)"
    )
    parser.add_argument(
        "-pl", "--print-level",
        type=str,
        default="INFO",
        help="Logger print level (e.g. DEBUG, INFO, WARNING, ERROR)"
    )
    args = parser.parse_args()

    lg.logger.remove()
    lg.logger.add(sys.stderr, level=args.print_level.upper(), colorize=True, backtrace=True, diagnose=True)
    lg.logger.add(LOG_FILE, level=args.print_level.upper(), colorize=False, backtrace=True, diagnose=True)

    # Override global variables with parsed arguments
    DEFAULT_EXPOSURES = args.exposures
    CHECK_INTERVAL = args.interval

    main()
