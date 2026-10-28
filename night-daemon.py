import argparse as ap
import datetime as dt
import sys
import threading
import time

import loguru as lg

import camera
import camera.io as cio
from camera.night import NightCamera

parser = ap.ArgumentParser(description="LUCID Night Capture Daemon")

parser.add_argument(
    "-t", "--total-exposure",
    type=float,
    default=180.0,
    help="effective exposure time per night image, in seconds",
)
parser.add_argument(
    "-s", "--sub-exposure",
    type=float,
    default=None,
    help="individual sub-exposure time in seconds; default: camera maximum",
)
parser.add_argument(
    "-m", "--stack-method",
    choices=["mean", "median", "sigma"],
    default="sigma",
)
parser.add_argument("--sigma", type=float, default=3.0)
parser.add_argument(
    "-i", "--interval",
    type=float,
    default=3600.0,
    help="seconds between the start of consecutive night captures",
)
parser.add_argument(
    "-pl", "--print-level",
    default="INFO",
)

args = parser.parse_args()

date_start = dt.datetime.now()
date_str = date_start.strftime(r"%Y%m%d")
time_str = date_start.strftime(r"%H%M%S")

save_dir = cio.image_dir_join("Night")
log_file = cio.log_dir_join(f"{date_str}-{time_str}-night-daemon.log")

lg.logger.remove()
lg.logger.add(sys.stderr, level=args.print_level.upper(), colorize=True)
lg.logger.add(log_file, level=args.print_level.upper(), colorize=False)


def stop_listener(stop_event):
    while not stop_event.is_set():
        text = input("Type 'exit' or 'quit' and press Enter to stop.\n")
        if text.strip().lower() in ("exit", "quit"):
            stop_event.set()


def main():
    stop_event = threading.Event()
    threading.Thread(
        target=stop_listener,
        args=(stop_event,),
        daemon=True,
    ).start()

    nb_captures = 0

    while not stop_event.is_set():
        cam = None

        while not stop_event.is_set():
            try:
                cam = NightCamera()
                break
            except Exception as exc:
                lg.logger.warning(f"Camera unavailable: {exc}. Retrying in 5 s.")
                time.sleep(5)

        if stop_event.is_set():
            break

        camera.SAVE_DIR = cio.get_session_path(save_dir, date_start)

        try:
            cam(
                total_exposure=args.total_exposure,
                sub_exposure=args.sub_exposure,
                stack_method=args.stack_method,
                sigma=args.sigma,
            )
            nb_captures += 1
        except Exception as exc:
            lg.logger.error(f"Night capture failed: {exc}")
        finally:
            del cam

        elapsed = args.total_exposure
        remaining = args.interval - elapsed

        if remaining > 0:
            lg.logger.info(f"Waiting {remaining:.1f} s until next capture.")
            for _ in range(int(remaining)):
                if stop_event.is_set():
                    break
                time.sleep(1)

    lg.logger.success(f"Captured {nb_captures} night images.")


if __name__ == "__main__":
    main()