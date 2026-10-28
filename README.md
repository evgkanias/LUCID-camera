# LUCID-camera

A Python package and command-line toolkit for controlling LUCID Vision cameras,
capturing calibrated images, generating HDR images, and acquiring long night-sky
exposures by accumulating multiple camera-limited sub-exposures.

The package is designed for scientific imaging workflows where raw sensor data,
exposure metadata, and reproducible acquisition settings are important.

## Features

- LUCID Arena-based camera control.
- Manual and automatic exposure control.
- Software-triggered image acquisition.
- Mono16 and other supported pixel formats.
- TIFF image output with EXIF-style metadata.
- HDR bracketed capture.
- HDR conversion and tone mapping.
- Long-exposure night-sky capture using linear frame accumulation.
- Scheduled image acquisition through daemon scripts.
- Geolocation-aware metadata through `camera.io`.

## Repository structure

```text
LUCID-camera/
├── camera/
│   ├── __init__.py       # Core Camera class and acquisition logic
│   ├── io.py             # Configuration, image saving, metadata, and utilities
│   ├── night.py          # Optional: long-exposure night-sky acquisition
│   └── transform.py      # Optional: long-exposure night-sky acquisition
├── config.yaml           # Default camera and HDR settings
├── hdr-capture.py        # Capture an HDR bracket sequence
├── hdr-daemon.py         # Repeatedly capture HDR brackets
├── hdr-convert.py        # Convert HDR brackets into an HDR image
├── hdr-convert-old.py    # Legacy HDR conversion script
├── leaf-capture.py       # Single-image capture utility
├── night-capture.py      # Optional: capture one long night-sky exposure
├── night-daemon.py       # Optional: repeatedly capture night-sky exposures
├── known-locations.yaml  # Known capture locations
└── overwrite-location.py # Utility for correcting location metadata
```

## Requirements

- Python 3.8 or later
- LUCID Arena SDK
- A supported LUCID Vision camera
- Required Python packages:

```bash
pip install numpy pillow loguru pyyaml piexif geocoder requests
```

Install the LUCID Arena SDK from
[LUCID Vision Labs](https://thinklucid.com/downloads-hub/).
The Python `arena_api` package must be able to detect the connected camera.

## Configuration

Camera defaults are read from `config.yaml`.

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

default_exposures:
  - 3000.0
  - 300.0
  - 30.0
```

### Important settings

| Setting | Description |
|---|---|
| `camera.exposure_type` | Default exposure mode, such as `auto` or `manual`. |
| `camera.pixel_format` | Camera pixel format, typically `Mono16` for scientific imaging. |
| `camera.target_brightness` | Target brightness used by automatic exposure routines. |
| `convert.gamma` | Gamma applied during HDR conversion. |
| `daemon.intervals` | Default delay between daemon captures, in seconds. |
| `default_exposures` | Exposure bracket used by the HDR capture scripts, in milliseconds. |

## HDR imaging

### Capture an HDR bracket

```bash
python hdr-capture.py
```

This captures a sequence of images using the exposure bracket defined in
`config.yaml`. The resulting images are saved as TIFF files with acquisition
metadata.

### Capture HDR images periodically

```bash
python hdr-daemon.py
```

The daemon repeatedly captures HDR brackets. The interval between captures is
controlled by the `daemon.intervals` setting or the script’s command-line
arguments.

### Convert captured brackets

```bash
python hdr-convert.py
```

This combines the captured exposure bracket into an HDR image using the
configured gamma value.

## Night-sky long exposure

Most machine-vision cameras limit a single exposure to a few seconds. The
night-capture feature approximates a much longer exposure by capturing several
maximum-length sub-exposures and **summing their raw pixel values**.

For example, a camera limited to 10 s per exposure can approximate a 180 s
exposure by capturing:

```text
18 frames × 10 s = 180 s effective exposure
```

The frames are accumulated linearly rather than averaged:

\[
I_{\text{night}} = \sum_{i=1}^{N} I_i
\]

For a linear sensor, this approximates the signal that would be collected in a
single exposure of duration \(Nt\), where \(t\) is the individual sub-exposure
duration.

### Output format

The final night image is saved as a **32-bit single-channel float TIFF**
(`mode='F'`). This avoids clipping bright stars or the Milky Way, because the
accumulated signal can exceed the 16-bit maximum of 65,535.

For example, 18 saturated 16-bit frames can produce values up to:

```text
18 × 65,535 = 1,179,630
```

The float TIFF preserves this full accumulated range. An optional 16-bit
preview can also be generated for quick visual inspection.

### Capture one night image

```bash
# 3-minute effective exposure using the camera maximum sub-exposure
python night-capture.py -t 180

# 5-minute effective exposure
python night-capture.py -t 300

# 2-minute effective exposure using 8 s sub-exposures
python night-capture.py -t 120 -s 8
```

### Command-line options

| Option | Description | Default |
|---|---|---|
| `-t`, `--total-exposure` | Total effective exposure time in seconds. | `180` |
| `-s`, `--sub-exposure` | Individual exposure duration in seconds. | Camera maximum |
| `-sd`, `--save-directory` | Directory in which to save the session. | Repository default |
| `-pf`, `--pixel-format` | Camera pixel format, e.g. `Mono16`. | Config value |
| `-pl`, `--print-level` | Logging level. | `INFO` |

### Repeated night capture

```bash
# One 3-minute night image every hour
python night-daemon.py -t 180 -i 3600

# One 5-minute night image every 30 minutes
python night-daemon.py -t 300 -i 1800
```

The daemon continues until `exit` or `quit` is entered in the terminal.

## Python API

### Standard camera capture

```python
import camera

cam = camera.Camera()
cam()
```

### Night-sky capture

```python
from camera.night import NightCamera

cam = NightCamera()

# Approximate a 180 s exposure using 10 s sub-exposures.
cam(
    total_exposure=180.0,
    sub_exposure=10.0,
)
```

The camera’s maximum supported exposure time is read directly from the
`ExposureTime` node. If `sub_exposure` is omitted, the module uses the
camera’s maximum allowed exposure.

### Dark-frame correction

If you have a master dark frame with the same dimensions and exposure settings,
pass it to the night capture call:

```python
import numpy as np
from PIL import Image

dark = np.asarray(Image.open("master_dark.tiff"), dtype=np.float64)

cam = NightCamera()
cam(
    total_exposure=300.0,
    sub_exposure=10.0,
    dark_frame=dark,
)
```

The dark frame is multiplied by the number of sub-exposures before subtraction,
because dark current accumulates over the full effective exposure time.

## Output files

Each acquisition creates a timestamped session directory:

```text
images/
└── 20261028-133000/
    ├── night_0000.tiff
    ├── night_0001.tiff
    ├── ...
    ├── night_0017.tiff
    ├── night_180s_float.tiff
    ├── night_180s_preview.tiff
    └── metadata.yaml
```

| File | Description |
|---|---|
| `night_XXXX.tiff` | Individual raw sub-exposure. |
| `night_XXXs_float.tiff` | Linearly accumulated long-exposure image in float32 format. |
| `night_XXXs_preview.tiff` | Optional 16-bit display preview. |
| `metadata.yaml` | Acquisition metadata, including effective exposure and stack method. |

## Recommended night workflow

1. Place the camera on a stable tripod or mount.
2. Focus manually on a bright star or distant light source.
3. Set aperture, gain, and white balance before capture.
4. Take a dark-frame sequence with the lens capped.
5. Create a master dark frame by averaging the dark frames.
6. Run `night-capture.py` with the desired effective exposure.
7. Process the float TIFF in Python, Fiji, or another HDR-aware tool.

### Example processing

```python
import numpy as np
from PIL import Image

image = np.asarray(Image.open("night_180s_float.tiff"), dtype=np.float32)

# Simple display normalisation
display = image / image.max()
display = np.clip(display ** (1 / 2.2), 0, 1)

Image.fromarray((display * 255).astype(np.uint8)).save("night_preview.png")
```

## Notes and limitations

- The night feature assumes a linear sensor response.
- Gain should remain fixed during a night sequence.
- Avoid changing focus, aperture, or exposure settings during acquisition.
- Frame accumulation does not compensate for star trailing; each sub-exposure
  still has its own motion blur.
- For long focal lengths or prominent star trailing, reduce the sub-exposure
  duration.
- Satellite and aircraft trails are preserved by linear summation, matching the
  behaviour of a true long exposure.
- Thermal noise and dark current accumulate with exposure time; dark-frame
  subtraction is strongly recommended for scientific use.

## License

Copyright (c) Evripidis Gkanias.

Add your preferred license text here, for example MIT, BSD-3-Clause, or GPL-3.0.