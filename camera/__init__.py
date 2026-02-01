import camera.io as cio

import loguru as lg
import numpy as np
import datetime as dt
import time
import os

try:
    from arena_api.system import system
    from arena_api.enums import PixelFormat
except ImportError:
    system = None
    PixelFormat = None

TAB1 = "  "
TAB2 = "    "

MAX_EXPOSURE = 10000000.0  # in microseconds
PERC_1 = 0.2
PERC_2 = 0.05

FNUMBER = cio.config['camera']['f_number']
FOCAL_LENGTH = cio.config['camera']['focal_length']

try:
    PIXEL_FORMAT = PixelFormat[cio.config["camera"]["pixel_format"]]
except TypeError:
    PIXEL_FORMAT = cio.config["camera"]["pixel_format"]
IMG_EXT = cio.config['image']['extension']

SETTINGS_KEYS = [
    'TriggerMode',
    'TriggerSource',
    'TriggerSelector',
    'TriggerSoftware',
    'TriggerArmed',
    'ExposureAuto',
    'ExposureTime',
    'PixelFormat',
    'Width',
    'Height',
    'AcquisitionFrameRateEnable',
    'AcquisitionFrameRate',
    'BalanceWhiteAuto',
    'DeviceTemperature',
    'DeviceVendorName',
    'DeviceModelName',
    'DeviceSerialNumber',
    'TargetBrightness',
    'Gain'
]
RESTORE_EXEMPT = [
    'TriggerArmed',
    'DeviceTemperature',
    'DeviceVendorName',
    'DeviceModelName',
    'DeviceSerialNumber',
]

SAVE_DIR = cio.image_dir_join('HDR')


class HDRCamera:
    def __init__(self, device=None):
        devices = create_devices_with_tries()
        if device is None:
            device = 0
        self._device = devices[0]

        self._nodemap = self._device.nodemap
        self.nodes = self._nodemap.get_node(SETTINGS_KEYS)
        
        # Save initial settings to restore later
        self._initial_settings = self._store_initial()

        # Prepare trigger mode
        lg.logger.debug(f'{TAB1}Prepare trigger mode')
        lg.logger.debug(f"{TAB2}Trigger selector = 'FrameStart'")
        self.nodes['TriggerSelector'].value = 'FrameStart'
        lg.logger.debug(f"{TAB2}Trigger mode = 'On'")
        self.nodes['TriggerMode'].value = 'On'
        lg.logger.debug(f"{TAB2}Trigger source = 'Software'")
        self.nodes['TriggerSource'].value = 'Software'
        lg.logger.debug(f"{TAB2}Acquitision Frame Rate Enable = 'True'")
        self.nodes['AcquisitionFrameRateEnable'].value = True
        lg.logger.debug(f"{TAB2}Pixel Format = {PIXEL_FORMAT.name}")
        self.nodes['PixelFormat'].value = PIXEL_FORMAT

        self.nodes['BalanceWhiteAuto'].value = 'Off'

        self.nodes['TargetBrightness'].value = int(cio.config['camera']['target_brightness'])

        self.timestamp = get_timestamp()
        
    def __call__(self, *exposures):
        lg.logger.info('Acquire HDR images example started')
    
        # set default exposures
        if exposures is None:
            return HDRCameraAuto.__call__(self, *exposures)
        else:
            exposures = list(exposures)
            for i in range(len(exposures)):
                exposures[i] *= 1e3  # convert to microseconds

        # ensure exposures are sorted from longest to shortest
        exposures = sorted(exposures, reverse=True)
        lg.logger.info(f'{TAB1}Using exposures (in sec): {", ".join([f"{exp * 1e-6:.6f}" for exp in exposures])}')

        # Disable automatic exposure
        lg.logger.debug(f'{TAB1}Disable automatic exposure')
        if self.nodes['ExposureAuto'].value != 'Off':
            lg.logger.debug(f'ExposureAuto old value = {self.nodes["ExposureAuto"].value}')
            self.nodes['ExposureAuto'].value = 'Off'

        # Get exposure time and software trigger nodes
        lg.logger.debug(f'{TAB1}Get exposure time and software trigger nodes')
        
        if self.nodes['ExposureTime'] is None or self.nodes['TriggerSoftware'] is None:
            raise Exception('ExposureTime or TriggerSoftware node is not available')
        
        if not self.nodes['ExposureTime'].is_writable or not self.nodes['TriggerSoftware'].is_writable:
            raise Exception('ExposureTime or TriggerSoftware node is not writable')

        # setup stream values
        tl_stream_nodemap = self._device.tl_stream_nodemap
        tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
        tl_stream_nodemap['StreamPacketResendEnable'].value = True

        # acquire and save images
        lg.logger.debug(f'{TAB1}Acquire and save images')

        # update timestamp and create directory for the images
        self.timestamp = get_timestamp()
        session_dir = os.path.join(SAVE_DIR, self.timestamp)
        os.makedirs(session_dir)

        t_start = tic()
        self._device.start_stream()

        meta = {}
        for i, exposure in enumerate(exposures):
            meta[f'image_{i}'] = self.acquire_and_save_buffer(exposure, count=i)
        meta['pixel_format'] = PIXEL_FORMAT.name
        cio.save_metadata(session_dir, meta)

        self._device.stop_stream()
        t_end = toc(t_start)
        lg.logger.success(f'{TAB1}Acquired and saved {len(exposures)} images in {t_end:.3f} sec')
        lg.logger.success(f'{TAB1}Location: {session_dir}')

    def acquire_and_save_buffer(self, exposure, count=0):
        # Set frame rate
        min_frame_rate = self.nodes['AcquisitionFrameRate'].min
        max_frame_rate = self.nodes['AcquisitionFrameRate'].max
        frame_rate = np.clip(1e6 / exposure, min_frame_rate, max_frame_rate)
        self.nodes['AcquisitionFrameRate'].value = frame_rate
        lg.logger.debug(f'{TAB2}Set frame rate at  = {frame_rate:.2f} Hz')
        
        if exposure > self.nodes['ExposureTime'].max or exposure < self.nodes['ExposureTime'].min:
            lg.logger.debug(f'{TAB2}Exposure time should be in range {self.nodes["ExposureTime"].min} - {self.nodes["ExposureTime"].max} us, '
                              f'but got {exposure} us.')
            exposure = np.clip(exposure, self.nodes['ExposureTime'].min, self.nodes['ExposureTime'].max)
            lg.logger.debug(f'{TAB2}Set exposure at    = {exposure/1e6:.2f} sec')
            
        # Set exposure time 
        lg.logger.info(f'Getting image {count+1} with exposure time {exposure * 1e-6:9,.6f} sec')
        self.nodes['ExposureTime'].value = exposure

        t_start = tic()
        
        # after the exposure time is set, the setting does not take place on the device until the next frame.
        # thus, two images are retrieved, and we use the second one.
        buf = [None, None]
        timestamp = None
        for i in range(2):
            self.trigger_software_once_armed()
            timestamp = dt.datetime.now()
            buf[i] = self._device.get_buffer()
        lg.logger.info(f'{TAB1}Snaped in {toc(t_start):5,.2f} sec')
        
        t_start = tic()

        # Extract raw data and metadate
        # The buffer contains Mono16 data (16 bits per pixel, split into two bytes)
        # Each pixel is 2 bytes (uint16), so total bytes should be width * height * 2

        # Get the raw bytes
        raw_bytes = bytes(buf[-1].data)

        # Calculate expected image shape
        width = buf[-1].width
        height = buf[-1].height
        num_pixels = width * height

        # Convert to numpy array of uint16 (little-endian)
        if PIXEL_FORMAT is PixelFormat.Mono16:
            img_data = np.frombuffer(raw_bytes, dtype='<u2', count=num_pixels).reshape(height, width)
        else:
            img_data = np.frombuffer(raw_bytes, dtype=np.uint8, count=num_pixels).reshape(height, width)
        meta = self.get_meta(timestamp)

        # Save RAW image as a TIF
        img_path = os.path.join(SAVE_DIR, self.timestamp, f"image{count}.{IMG_EXT}")
        cio.save_image(img_path, img_data, meta=meta)
        lg.logger.debug(f'{TAB2}in {toc(t_start):.2f} sec: {os.path.abspath(img_path)}')

        # Requeue image buffer
        for buffer in buf:
            self._device.requeue_buffer(buffer)

        return meta
    
    def trigger_software_once_armed(self):
        trigger_armed = False

        while not trigger_armed:
            trigger_armed = bool(self.nodes['TriggerArmed'].value)
        
        self.nodes['TriggerSoftware'].execute()

    def get_meta(self, timestamp):
        meta = {}

        # Save device info
        meta['CameraMaker'] = self.nodes['DeviceVendorName'].value
        meta['CameraModel'] = self.nodes['DeviceModelName'].value
        meta['CameraSerialNumber'] = str(self.nodes['DeviceSerialNumber'].value)
        meta['CameraBrightness'] = self.nodes['TargetBrightness'].value
        meta['CameraTemperature'] = self.nodes['DeviceTemperature'].value
        
        # Lens info
        meta['LensMaker'] = 'Fujifilm'
        meta['LensModel'] = 'FE185C057HA-1'
        meta['FNumber'] = FNUMBER
        meta['FocalLength'] = FOCAL_LENGTH

        # Store the exposure time (in seconds)
        meta['ExposureTime'] = self.nodes['ExposureTime'].value*1e-6
        # Store the gain as equivalent ISO rating (reducing precision to 1%)
        meta['ISOSpeedRatings'] = int(np.round(100*(10**(self.nodes['Gain'].value/10))**0.5))
        # Store the timestamp
        meta['DateTime'] = timestamp.strftime(r"%Y:%m:%d %H:%M:%S")
        
        meta['Latitude'] = cio.LATITUDE
        meta['Longitude'] = cio.LONGITUDE
        meta['Altitude'] = cio.ALTITUDE
        
        return meta

    def _store_initial(self):
        '''
        Store initial node values, return their values at the end
        '''
        initials = {}
        for key in SETTINGS_KEYS:
            if key not in self.nodes.keys():
                lg.logger.warning(f'{TAB1}{key} does not exist in NodeMap.')
            elif hasattr(self.nodes[key], 'value'):
                lg.logger.debug(f'{TAB1}Store initial setting: {key} = {self.nodes[key].value}')
                initials[key] = self.nodes[key].value
            else:
                lg.logger.debug(f'{TAB1}Store initial setting: {key} has no value')

        return initials
    
    def _restore_initial(self):
        '''
        Restore initial node values, return their values at the end
        '''
        for key in self._initial_settings.keys():
            if key in RESTORE_EXEMPT:
                lg.logger.debug(f'{TAB1}Exempt from restoring: {key}')
                continue

            try:
                if hasattr(self.nodes[key], 'value'):
                    initial_node = self.nodes[key].value
                    lg.logger.debug(f'{TAB1}Restoring initial setting: {key} from {initial_node} to {self._initial_settings[key]}')
                    self.nodes[key].value = self._initial_settings[key]
                    lg.logger.debug(f'{key} restored initial setting: {initial_node} --> {self._initial_settings[key]}')
                else:
                    lg.logger.debug(f'{key} has no value, cannot be restored!')
            except Exception:
                lg.logger.warning(f'{key} could not be restored!')
                lg.logger.warning(f'{TAB1}Attempted to change {self.nodes[key].value} to {self._initial_settings[key]}')

    def __del__(self):
        # Clean up ------------------------------------------------

        # Restore initial settings
        lg.logger.debug(f'{TAB1}Restoring initial settings')
        self._restore_initial()

        # Destroy all created devices. This call is optional and will automatically be called for any remaining devices when the system module is unloading.
        system.destroy_device()
        lg.logger.debug(f'{TAB1}Destroyed all created devices')


STEP_SIZE = cio.config['default_step_size']
NB_EXPOSURES = cio.config['default_nb_exposures']


class HDRCameraAuto(HDRCamera):

    def __call__(self, *exposures, step_size=None, nb_exposures=None):
        if len(exposures) > 0:
            sorted_exposures = sorted(exposures, reverse=True)
            current_exposure = sorted_exposures[len(exposures) // 2] * 1e3
            lg.logger.info(f'Using provided exposures to determine current exposure time: {current_exposure * 1e-6:.6f} sec')
        else:
            # Enable automatic exposure
            lg.logger.debug(f'{TAB1}Enable automatic exposure')
            if self.nodes['ExposureAuto'].value != 'Continuous':
                lg.logger.debug(f'ExposureAuto old value = {self.nodes["ExposureAuto"].value}')
                self.nodes['ExposureAuto'].value = 'Continuous'
            
            # Trigger once to update exposure
            self._device.start_stream()
            self.trigger_software_once_armed()
            self._device.stop_stream()

            # measure current exposure
            current_exposure = self.nodes['ExposureTime'].value
            lg.logger.info(f'Using Current exposure time from auto-exposure: {current_exposure * 1e-6:.6f} sec')
            
        if step_size is None:
            step_size = STEP_SIZE
        if nb_exposures is None or nb_exposures < 2:
            nb_exposures = NB_EXPOSURES
        exposures = []
        for i in range(nb_exposures // 2):
            exposures.append(current_exposure * (1 + step_size * (i + 1)))
            exposures.append(current_exposure / (1 + step_size * (i + 1)))
        if nb_exposures % 2 == 1:
            exposures.append(current_exposure)
        exposures = sorted(exposures, reverse=True)

        super().__call__(*[exposure * 1e-3 for exposure in exposures])


def create_devices_with_tries():
    '''
    This function waits for the user to connect a device before raising
        an exception
    '''

    tries = 0
    tries_max = 6
    sleep_time_secs = 10
    while tries < tries_max:  # Wait for device for 60 seconds
        devices = system.create_device()
        if not devices:
            lg.logger.debug(
                f'{TAB1}Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
                f'secs for a device to be connected!')
            for sec_count in range(sleep_time_secs):
                time.sleep(1)
                lg.logger.debug(
                    f'{TAB1}{sec_count + 1 } seconds passed ',
                    '.' * sec_count, end='\r')
            tries += 1
        else:
            lg.logger.debug(f'{TAB1}Created {len(devices)} device(s)')
            return devices
    else:
        raise Exception(f'{TAB1}No device found! Please connect a device and run '
                        f'the example again.')


def get_timestamp():
    return dt.datetime.now().strftime(r'%Y%m%d%H%M%S')


def tic():
    return time.time()


def toc(t_start):
    return time.time() - t_start
