# LUCID-camera

Python tools for controlling LUCID Vision cameras, capturing individual images,
and acquiring high-dynamic-range (HDR) image brackets.

The package is intended for scientific imaging with polarisation-sensitive
LUCID cameras. It provides reproducible camera control, raw image acquisition,
metadata embedding, geolocation-aware logging, and HDR bracket capture.

## Features

- LUCID Arena SDK integration.
- Automatic detection of connected LUCID devices.
- Software-triggered image acquisition.
- Manual and automatic exposure control.
- Support for Mono, Bayer, and polarised pixel formats.
- TIFF, PNG, JPEG, BMP, and RAW image output.
- EXIF-style metadata, including camera, lens, exposure, gain, GPS, and
  timestamp information.
- Manual HDR bracket capture.
- Automatic HDR bracket generation around the current auto-exposure value.
- Periodic HDR acquisition using `hdr-daemon.py`.
- HDR conversion and tone mapping.

## Repository structure

```text
LUCID-camera/
├── camera/
│   ├── __init__.py        # Camera, HDRCamera, and HDRCameraAuto classes
│   ├── night.py           # NightCamera classes
│   └── io.py              # Configuration, image saving, metadata, and utilities
├── config.yaml            # Default camera, HDR, and conversion settings
├── hdr-capture.py         # Capture an HDR bracket
├── hdr-daemon.py          # Repeatedly capture HDR brackets
├── hdr-convert.py         # Convert an HDR bracket into an HDR image
├── hdr-convert-old.py     # Legacy HDR conversion script
├── leaf-capture.py        # Capture a single image
├── known-locations.yaml   # Known capture locations
└── overwrite-location.py  # Correct location metadata in existing images
```

## Requirements

- Python 3.8 or later
- LUCID Arena SDK
- A supported LUCID Vision camera
- Python dependencies:

```bash
pip install numpy pillow loguru pyyaml piexif geocoder requests
```

Install the LUCID Arena SDK from the
[LUCID Vision Labs downloads page](https://thinklucid.com/downloads-hub/).
The Python `arena_api` package must be installed and able to detect the camera.

## Installation

Clone the repository:

```bash
git clone [https://github.com/evgkanias/LUCID-camera.git](https://github.com/evgkanias/LUCID-camera.git)
cd LUCID-camera
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Default settings are read from `config.yaml`.

```yaml
camera:
  exposure_type: auto
  f_number: 16.0
  focal_length: 1.8
  pixel_format: Mono16
  target_brightness: 10

convert:
  gamma: 2.2

daemon:
  intervals: 600

night:
  exposures_s: 300  # 5 min

default_exposures:
  - 3000.0
  - 300.0
  - 30.0
```

### Main settings

| Setting | Description |
|---|---|
| `camera.exposure_type` | `auto` uses `HDRCameraAuto`; `manual` uses `HDRCamera`. |
| `camera.pixel_format` | Camera pixel format, e.g. `Mono16`. |
| `camera.target_brightness` | Target brightness used by automatic exposure. |
| `camera.f_number` | Lens f-number stored in image metadata. |
| `camera.focal_length` | Lens focal length stored in image metadata. |
| `convert.gamma` | Gamma value used during HDR conversion. |
| `daemon.intervals` | Delay between daemon captures, in seconds. |
| `default_exposures` | Default HDR exposure bracket, in milliseconds. |
| `default_step_size` | Exposure spacing used by automatic HDR capture. |
| `default_nb_exposures` | Number of exposures used by automatic HDR capture. |

## Capturing a single image

Use `leaf-capture.py` to acquire one image:

```bash
python leaf-capture.py
```

To specify an exposure time in seconds:

```bash
python leaf-capture.py -e 0.1
```

The resulting image is saved in a timestamped session directory under the
configured image directory.

## HDR capture

### Manual HDR bracket

`HDRCamera` captures a user-defined set of exposures, sorted from longest to
shortest.

```bash
python hdr-capture.py -e 3.0 0.3 0.03
```

Exposures are given in **milliseconds**. The command above captures:

```text
3.000 s
0.300 s
0.030 s
```

To use the exposures defined in `config.yaml`:

```bash
python hdr-capture.py
```

### Automatic HDR bracket

`HDRCameraAuto` measures the current auto-exposure value and creates a
symmetrical exposure bracket around it.

```bash
python hdr-capture.py
```

Customise the bracket with:

```bash
python hdr-capture.py -s 2 -n 5
```

| Option | Description |
|---|---|
| `-s`, `--step-size` | Relative exposure spacing. |
| `-n`, `--number-exposures` | Total number of exposures in the bracket. |
| `-e`, `--exposures` | Optional explicit exposure list, in milliseconds. |

For example, with a measured exposure of 0.1 s, a step size of 2, and five
exposures, the bracket contains exposures above and below the measured value,
including the measured exposure itself.

### Periodic HDR capture

```bash
python hdr-daemon.py
```

The daemon repeatedly captures HDR brackets using the configured interval. It
is useful for time-lapse HDR acquisition or long-term monitoring.

## HDR conversion

After capturing a bracket, convert it into an HDR image:

```bash
python hdr-convert.py
```

The conversion uses the gamma value defined in `config.yaml`.

## Camera classes

### `Camera`

`Camera` is the base class for all acquisition modes.

```python
import camera

cam = camera.Camera()

# Capture one image with a 100 ms exposure.
cam(exposure=0.1)
```

The class:

- Detects and connects to a LUCID device.
- Stores the initial camera settings.
- Configures software triggering.
- Disables automatic white balance.
- Acquires and saves images.
- Restores the initial settings when the object is deleted.

### `HDRCamera`

`HDRCamera` captures a bracket of manually specified exposures.

```python
import camera

cam = camera.HDRCamera()

# Exposure times are given in seconds.
cam(3.0, 0.3, 0.03)
```

The exposures are automatically sorted from longest to shortest before
acquisition.

### `HDRCameraAuto`

`HDRCameraAuto` estimates the current exposure using the camera’s automatic
exposure mode and captures a bracket around it.

```python
import camera

cam = camera.HDRCameraAuto()

# Five exposures around the current auto-exposure value.
cam(step_size=2, nb_exposures=5)
```

You can also provide a reference exposure in seconds:

```python
cam(0.1, step_size=2, nb_exposures=5)
```

## Acquisition behaviour

The camera uses software triggering. When the exposure time changes, the
current implementation triggers the camera twice:

1. The first trigger applies the new exposure setting.
2. The second trigger captures the image using the updated exposure.

This is implemented in `Camera.acquire_and_save_buffer()`:

```python
self.trigger_software_once_armed()
buf.append(self._device.get_buffer())

# The exposure setting takes effect on the next frame.
self.trigger_software_once_armed()
buf.append(self._device.get_buffer())
```

As a result, an HDR bracket with \(N\) exposures requires approximately
\(2N\) exposure periods.

### Reducing HDR acquisition time

If your camera applies a new `ExposureTime` value to the next triggered frame,
the first trigger is unnecessary. You can modify
`Camera.acquire_and_save_buffer()` to use one trigger per image:

```python
def acquire_and_save_buffer(self, exposure, identity=None):
    if identity is None:
        identity = ""
    elif isinstance(identity, int):
        identity = f"{identity}"
    else:
        identity = f"{identity}"

    if exposure is not None:
        # Clip exposure to the camera-supported range.
        exposure = np.clip(
            exposure,
            self.nodes["ExposureTime"].min,
            self.nodes["ExposureTime"].max,
        )

        # Set a frame rate compatible with the exposure time.
        min_frame_rate = self.nodes["AcquisitionFrameRate"].min
        max_frame_rate = self.nodes["AcquisitionFrameRate"].max
        frame_rate = np.clip(1e6 / exposure, min_frame_rate, max_frame_rate)
        self.nodes["AcquisitionFrameRate"].value = frame_rate

        # Set the exposure; it applies to the next triggered frame.
        self.nodes["ExposureTime"].value = exposure

    # Trigger once: capture the next frame with the current settings.
    self.trigger_software_once_armed()
    timestamp = dt.datetime.now()
    buf = self._device.get_buffer()

    raw_bytes = bytes(buf.data)
    width = buf.width
    height = buf.height
    num_pixels = width * height

    if PIXEL_FORMAT is PixelFormat.Mono16:
        img_data = np.frombuffer(
            raw_bytes, dtype="<u2", count=num_pixels
        ).reshape(height, width)
    else:
        img_data = np.frombuffer(
            raw_bytes, dtype=np.uint8, count=num_pixels
        ).reshape(height, width)

    meta = self.get_meta(timestamp)

    if identity != "":
        identity = f"_{identity}"

    img_path = os.path.join(
        SAVE_DIR,
        self.timestamp,
        f"image{identity}.{IMG_EXT}",
    )
    cio.save_image(img_path, img_data, meta=meta)

    self._device.requeue_buffer(buf)

    return meta
```

With this change, an HDR bracket of \(N\) exposures requires approximately
\(N\) exposure periods rather than \(2N\).

| HDR bracket | Current implementation | Single-trigger implementation |
|---|---:|---:|
| 3 exposures | ~6 exposure periods | ~3 exposure periods |
| 5 exposures | ~10 exposure periods | ~5 exposure periods |

If the first image after changing exposure appears to use the previous
exposure value, retain one discarded warm-up frame per session, immediately
after starting the stream. Do not place the warm-up inside the exposure loop.

## Output files

Each acquisition creates a timestamped session directory:

```text
images/HDR/
└── 20261029134200/
    ├── image_0.tiff
    ├── image_1.tiff
    ├── image_2.tiff
    └── metadata.yaml
```

| File | Description |
|---|---|
| `image_0.tiff` | First image in the HDR bracket. |
| `image_1.tiff` | Second image in the HDR bracket. |
| `metadata.yaml` | Acquisition metadata for the full session. |

The exact image extension depends on the `image.extension` setting in
`config.yaml` or the `--image-extension` command-line argument.

## Metadata

Each image includes metadata describing the acquisition conditions:

- Camera manufacturer, model, serial number, and temperature.
- Target brightness.
- Lens maker, model, f-number, and focal length.
- Exposure time in seconds.
- Gain converted to an approximate ISO rating.
- Acquisition date and time.
- Latitude, longitude, and altitude.

Location values are obtained through `camera.io` and can be corrected with
`overwrite-location.py`.

## Command-line options

### `hdr-capture.py`

| Option | Description |
|---|---|
| `-e`, `--exposures` | Explicit HDR exposure list in milliseconds. |
| `-s`, `--step-size` | Exposure spacing for automatic HDR capture. |
| `-n`, `--number-exposures` | Number of exposures for automatic HDR capture. |
| `-pf`, `--pixel-format` | Camera pixel format. |
| `-ie`, `--image-extension` | Output image format. |
| `-sd`, `--save-directory` | Directory in which to save images. |
| `-pl`, `--print-level` | Logging level: `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |

### Common examples

```bash
# Manual HDR bracket
python hdr-capture.py -e 3.0 0.3 0.03

# Automatic HDR bracket with five exposures
python hdr-capture.py -n 5

# Use Mono16 output
python hdr-capture.py -pf Mono16

# Save TIFF images in a custom directory
python hdr-capture.py -sd /path/to/output -ie tiff
```

## Troubleshooting

### No camera detected

- Connect the camera and wait for it to enumerate.
- Ensure the LUCID Arena SDK is installed.
- Check that the camera is not in use by another application.
- The package retries device detection for up to 60 seconds.

### Exposure time is rejected

The requested exposure is clipped to the range reported by the camera’s
`ExposureTime` node. The base class defines a nominal maximum of 10 s, but the
connected camera’s supported range takes precedence.

### HDR images have unexpected exposure

If exposure changes appear to lag by one frame, retain a single warm-up frame
per acquisition session. This indicates that the camera applies node changes
on the frame following the setting change.

### Images are too dark or too bright

- Check the lens aperture.
- Check camera gain.
- Verify the exposure bracket.
- For automatic capture, adjust `target_brightness` in `config.yaml`.

## License

Copyright (c) Evripidis Gkanias.

Add your preferred license text here, for example MIT, BSD-3-Clause, or
GPL-3.0.