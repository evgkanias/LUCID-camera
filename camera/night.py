"""
Long-exposure night-sky capture for LUCID cameras.

The camera is limited to a maximum single exposure (typically 10 s).
This module captures many maximum-length sub-exposures and combines them
into one high-signal-to-noise night image.

An effective exposure of e.g. 180 s is obtained from 18 x 10 s frames.
"""

import camera
import camera.io as cio

import os
import time
import datetime as dt

import numpy as np
import loguru as lg

SAVE_DIR = cio.image_dir_join('Night')
MAX_EXPOSURE = 10  # sec
DEFAULT_EXPOSURE = cio.config["night"]["exposure_s"]


class NightCamera(camera.Camera):
    """
    Camera for long night-sky exposures using stacked sub-exposures.

    Parameters
    ----------
    total_exposure : float
        Desired total integration time in seconds, e.g. 120-300.
    sub_exposure : float or None
        Duration of each individual exposure in seconds. If None, the
        camera's maximum supported exposure time is used.
    stack_method : str
        'mean', 'median', or 'sigma'.
        'sigma' rejects outliers such as satellites and aircraft.
    sigma : float
        Sigma threshold used when stack_method='sigma'.
    """
    def __call__(self, total_exposure=180.0, sub_exposure=None, stack_method="sum", sigma=3.0, dark_frame=None, save=True):
        if total_exposure <= 0:
            raise ValueError("total_exposure must be positive.")

        if sub_exposure is None:
            sub_exposure = float(MAX_EXPOSURE)
        sub_exposure = float(min(sub_exposure, float(MAX_EXPOSURE)))
        sub_exposure_us = sub_exposure * 1e+6
        # max_exposure_us = float(MAX_EXPOSURE)
        # max_exposure_s = max_exposure_us * 1e-6

        # Continuous acquisition: no software trigger per frame.
        # This allows back-to-back exposures with no inter-frame delay.

        # Disable automatic exposure and set the sub-exposure once.
        if self.nodes["ExposureAuto"].value != "Off":
            lg.logger.debug(f'ExposureAuto old value = {self.nodes["ExposureAuto"].value}')
            self.nodes["ExposureAuto"].value = "Off"

        # Disable triggering.
        if self.nodes["TriggerMode"] is not None:
            self.nodes["TriggerMode"].value = "Off"

        if self.nodes["ExposureTime"] is None or self.nodes["TriggerSoftware"] is None:
            raise RuntimeError("ExposureTime or TriggerSoftware node is unavailable.")

        # Exposure is set once, before the stream starts.
        # self.nodes["ExposureTime"].value = sub_exposure_us

        # Request the fastest sustainable frame rate for this exposure.
        # A frame period equal to the exposure time gives 100% duty cycle.
        min_fps = float(self.nodes["AcquisitionFrameRate"].min)
        max_fps = float(self.nodes["AcquisitionFrameRate"].max)
        fps = float(np.clip(1.0 / sub_exposure, min_fps, max_fps))

        self.nodes["AcquisitionFrameRate"].value = fps
        lg.logger.debug(f'  Set frame rate at  = {self.nodes["AcquisitionFrameRate"].value:.2f} Hz')

        if sub_exposure_us > self.nodes['ExposureTime'].max or sub_exposure_us < self.nodes['ExposureTime'].min:
            lg.logger.debug(
                f'  Exposure time should be in range {self.nodes["ExposureTime"].min} - {self.nodes["ExposureTime"].max} us, '
                f'but got {sub_exposure_us} us.')
            sub_exposure_us = np.clip(sub_exposure_us, self.nodes['ExposureTime'].min, self.nodes['ExposureTime'].max)
            sub_exposure = sub_exposure_us * 1e-6

        self.nodes['ExposureTime'].value = sub_exposure_us
        lg.logger.info(f'  Set exposure at    = {self.nodes["ExposureTime"].value * 1e-6:.2f} sec')

        nb_frames = max(1, int(round(total_exposure / sub_exposure)))
        effective_exposure = nb_frames * sub_exposure

        tl_stream_nodemap = self._device.tl_stream_nodemap
        tl_stream_nodemap["StreamAutoNegotiatePacketSize"].value = True
        tl_stream_nodemap["StreamPacketResendEnable"].value = True

        self.timestamp = camera.get_timestamp()
        session_dir = os.path.join(SAVE_DIR, self.timestamp)
        if save:
            os.makedirs(session_dir, exist_ok=True)

        frames = []
        t_start = camera.tic()

        lg.logger.info(
            f"Night capture: {nb_frames} x {sub_exposure:.1f} s = "
            f"{effective_exposure:.1f} s effective exposure."
        )

        self._device.start_stream()

        try:
            # The first buffer may have been exposed before the new exposure
            # setting fully took effect, so discard it.
            # self.trigger_software_once_armed()
            warmup = self._device.get_buffer()
            self._device.requeue_buffer(warmup)

            for i in range(nb_frames):
                # timestamp = dt.datetime.now()

                frame = self._acquire_raw_frame()
                frames.append(frame.astype(np.float64))
                frame_path = os.path.join(session_dir, f"raw_{i:02d}.{camera.IMG_EXT}")

                if save:
                    self._meta[f'raw_{i:02d}'] = self.get_meta(dt.datetime.now())
                    self._meta[f'raw_{i:02d}']["ExposureTime"] = float(self.nodes["ExposureTime"].value) * 1e-6
                    cio.save_image(frame_path, frame, meta=self._meta[f'raw_{i:02d}'])

                lg.logger.info(
                    f"Captured frame {i + 1}/{nb_frames} "
                    f"({sub_exposure:.3f} s)."
                )

                # Small pause reduces thermal buildup and USB/GigE congestion.
                # time.sleep(0.05)

        finally:
            self._device.stop_stream()

        stack = np.stack(frames, axis=0)

        if dark_frame is not None:
            dark = np.asarray(dark_frame, dtype=np.float64)
            if dark.shape != stack.shape[1:]:
                raise ValueError(
                    f"Dark-frame shape {dark.shape} does not match "
                    f"image shape {stack.shape[1:]}."
                )
            stack -= dark[None, :, :]
            np.clip(stack, 0, None, out=stack)

        exposure_cum = 0.0
        night_image = None
        for i in range(nb_frames):

            night_image = self._combine(stack[:i+1], method=stack_method, sigma=sigma)
            night_image = np.clip(np.rint(night_image), 0, 65535).astype(np.uint16)
            exposure_cum += self._meta[f'raw_{i:02d}']['ExposureTime']

            if save:
                night_path = os.path.join(session_dir, f"image_{i:02d}.{camera.IMG_EXT}")
                self._meta[f'image_{i:02d}'] = self.get_meta(dt.datetime.now())
                self._meta[f'image_{i:02d}']["ExposureTime"] = exposure_cum
                self._meta[f'image_{i:02d}']["NightSubExposures"] = i + 1
                self._meta[f'image_{i:02d}']["NightSubExposure"] = sub_exposure
                self._meta[f'image_{i:02d}']["NightStackMethod"] = stack_method
                self._meta[f'image_{i:02d}']["NightDarkFrame"] = dark_frame is not None
                self._meta[f'image_{i:02d}']["Software"] = "LUCID-camera NightCapture"

                cio.save_image(night_path, night_image, meta=self._meta[f'image_{i:02d}'])

                elapsed = camera.toc(t_start)
                lg.logger.success(
                    f"Saved night image with {sub_exposure * (i + 1):.1f} s "
                    f"exposure in {elapsed:.1f} s: {night_path}"
                )

        if save:
            cio.save_metadata(session_dir, self._meta)

        return night_image

    def _acquire_raw_frame(self):
        """Trigger one exposure and return its Mono16 image array."""
        buf = self._device.get_buffer()

        try:
            raw_bytes = bytes(buf.data)
            width = buf.width
            height = buf.height
            nb_pixels = buf.width * buf.height

            if camera.PIXEL_FORMAT is camera.PixelFormat.Mono16:
                frame = np.frombuffer(
                    raw_bytes,
                    dtype="<u2",
                    count=nb_pixels,
                ).reshape(height, width).copy()
            else:
                frame = np.frombuffer(
                    raw_bytes,
                    dtype=np.uint8,
                    count=nb_pixels,
                ).reshape(height, width).copy()

            return frame
        finally:
            self._device.requeue_buffer(buf)

    @staticmethod
    def _combine(stack, method="sigma", sigma=3.0):
        """Combine frames without quantising to 8-bit or uint16 prematurely."""
        if method == "sum":
            return np.sum(stack, axis=0)

        if method == "mean":
            return stack.mean(axis=0)

        if method == "median":
            return np.median(stack, axis=0)

        if method == "sigma":
            centre = np.median(stack, axis=0)
            mad = np.median(np.abs(stack - centre[None, :, :]), axis=0)
            # 1.4826 converts MAD to an approximate standard deviation.
            scale = 1.4826 * mad
            scale = np.maximum(scale, 1.0)  # avoid dividing by zero

            lower = centre - sigma * scale
            upper = centre + sigma * scale

            mask = (stack >= lower[None, :, :]) & (stack <= upper[None, :, :])
            weight = mask.astype(np.float64)
            weight_sum = weight.sum(axis=0)
            weight_sum = np.maximum(weight_sum, 1.0)

            return (stack * weight).sum(axis=0) / weight_sum

        raise ValueError("stack_method must be 'sum', 'mean', 'median', or 'sigma'.")