import os
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'  # Suppress OpenCV logging

import camera.transform as trans
import camera.io as cio

import sys
import cv2
import numpy as np
import datetime as dt
import loguru as lg
import argparse

GAMMA_CORRECTION = cio.config['convert']['gamma']
LOG_FILE = cio.log_dir_join(f'{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}-hdr-convert.log')
UINT16_MAX = 65520.0
ORDERED_ANGLES = [0, 45, 90, 135]
# /Volumes/T9/SA2025/Q1 Experimental Setups/EG_Q4_A1_20251119070450


def main(directory=None, overwrite=False, yes_to_all=False):
    if directory is None:
        directory = input("Enter the directory containing the images (empty to process all): ").strip().replace("'", "")

    if directory in ["", "last"]:
        directories = []
        daemon_dir = cio.image_dir_join('Daemon')

        if os.path.exists(daemon_dir):
            daemon_sess = os.listdir(daemon_dir)
            modes = ['HDR'] + [os.path.join('Daemon', dirr) for dirr in daemon_sess]
        else:
            modes = ['HDR']
        
        for mode in modes:
            for dir_ in os.listdir(cio.image_dir_join(mode)):
                directories.append(os.path.join(mode, dir_))
        
        if directory == "":
            directory = directories
        elif directory == "last":
            directory = [sorted(directories, reverse=True)[0]]
    else:
        directory = [directory]
    
    if overwrite is None:
        overwrite = False
        overwrite_all = False
        skip_all = False
    else:
        overwrite_all = overwrite and yes_to_all
        skip_all = not overwrite and yes_to_all
    
    for dir_ in directory:
        dir_ = cio.image_dir_join(dir_)

        if not os.path.exists(dir_):
            lg.logger.error(f"Directory not found: {dir_}")
            continue
        
        base_names = cio.get_raw_image_files(dir_)
        if not base_names:
            lg.logger.error(f"No valid base names found in {dir_}")

        exists = any(['HDR' in f for f in os.listdir(dir_)])
        if exists and not overwrite_all and not skip_all and overwrite is None:
            res = input(f"HDR files exist in {dir_}'. Overwrite? [a/y/N] ").split(" ")
            overwrite = res[0].lower() in ['y', 'yes', 'ja']
            if any([r.lower() in ['a', 'all', '-a', '--all'] for r in res]):
                yes_to_all = True
        
            overwrite_all = overwrite and yes_to_all
            skip_all = not overwrite and yes_to_all

            overwrite = True if overwrite_all else False if skip_all else None

        if exists and not overwrite:
            continue
        
        # read images and exposures
        images, metas, files = cio.read_raw_images(dir_)
        if len(images) < 2:
            lg.logger.error(f"Need at least two images for HDR. Found {len(images)} in {dir_}")
            continue

        img_pol = {'000': [], '045': [], '090': [], '135': []}
        for image in images:
            img_pol_ = trans.demosaic_polarisation(image)
            for ang in img_pol_:
                # Extact RGB from polarisation images
                img_pol_rgb = trans.demosaic_bayer_malvar(img_pol_[ang])
                img_pol[ang].append(img_pol_rgb)

        exposures = np.array([meta['ExposureTime'] for meta in metas], dtype=np.float32)

        meta = metas[0]
        meta['Gamma'] = GAMMA_CORRECTION
        del meta["ExposureTime"]
        hdr = {}
        for ang in img_pol:
            # HDR merge
            imgs = [np.uint8(img * 255) for img in img_pol[ang]]
            calibrate = cv2.createCalibrateDebevec()
            response = calibrate.process(imgs, times=exposures)
            merge_debevec = cv2.createMergeDebevec()
            hdr_ = merge_debevec.process(imgs, exposures.copy(), response)
            lg.logger.debug("Generated HDR image.")
            
            # LDR and gamma correction
            tonemap = cv2.createTonemap(gamma=GAMMA_CORRECTION)
            hdr[ang] = tonemap.process(hdr_)
            ldr16 = np.clip(np.nan_to_num(hdr[ang], nan=0) * 65535, 0, 65535).astype(np.uint16)
            cio.save_image(os.path.join(dir_, f'image_HDR_{ang}.tiff'), ldr16, meta=meta)

        # Compute Stokes parameters per channel
        stokes = trans.get_stokes(hdr)
        for st in stokes:
            s_norm = cv2.normalize(stokes[st], None, 0, 65535, cv2.NORM_MINMAX)
            st16 = np.nan_to_num(s_norm, nan=0).astype(np.uint16)
            cio.save_image(os.path.join(dir_, f'image_HDR_{st}.tiff'), st16, meta=meta)

        # Angle and degree of polarisation per channel
        angle = trans.get_polarisation_angle(stokes)
        angle_deg = np.degrees(angle)
        DoLP = trans.get_polarisation_degree(stokes)
        
        colours = ['red', 'green', 'blue']
        for c, colour in enumerate(colours):
            rgb = trans.angle_dolp_to_rgb(angle_deg[..., c], DoLP[..., c])
            rgb = trans.add_colormap_disk(rgb)
            
            # Save image
            out_path = os.path.join(dir_, f'image_HDR_AOP_{colour}.tiff')
            cio.save_image(out_path, rgb, meta=meta)
            # cio.save_image_cv2(out_path, rgb)
        
        dop_rgb = trans.dop_to_rgb(DoLP)
        out_path = os.path.join(dir_, 'image_HDR_DOP.tiff')
        cio.save_image(out_path, dop_rgb, meta=meta)
        # cio.save_image_cv2(out_path, dop_rgb)
        lg.logger.success(f"All HDR images are processed in {dir_}")

    lg.logger.success("HDR polarisation processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDR Image Converter")

    parser.add_argument(
        "directory",
        type=str,
        nargs="?",
        default=None,
        help="Directory containing images to convert (default: process all directories or 'last' for the most recent one)"
    )
    parser.add_argument(
        "-p", "--print-level",
        type=str,
        default="info",
        help="Set the logging level (e.g. debug, info, warning, error, critical)"
    )
    parser.add_argument(
        "-o", "--overwrite",
        action="store_true",
        help="Overwrite existing HDR files without prompting"
    )
    parser.add_argument(
        '-s', '--skip-overwrite',
        action='store_true',
        help='Skip overwriting existing HDR files'
    )
    parser.add_argument(
        '-a', '--all',
        action='store_true',
        help='Answer yes to all questions'
    )
    args = parser.parse_args()

    lg.logger.remove()
    lg.logger.add(sys.stderr, level=args.print_level.upper(), colorize=True, backtrace=True, diagnose=True)
    lg.logger.add(LOG_FILE, level=args.print_level.upper(), colorize=False, backtrace=True, diagnose=True)
    lg.logger.disable('camera.io')

    main(args.directory,
         overwrite=True if args.overwrite else False if args.skip_overwrite else None,
         yes_to_all=args.all)
