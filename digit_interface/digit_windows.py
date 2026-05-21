# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import logging
import typing

import cv2
import numpy as np

try:
    from .digit_handler_windows import DigitHandler
except ImportError:
    from digit_handler_windows import DigitHandler

logger = logging.getLogger(__name__)


class DigitDefaults(object):
    STREAMS: typing.Dict = {
        # VGA resolution support 30 (default) and 15 fps
        "VGA": {
            "resolution": {"width": 640, "height": 480},
            "fps": {"30fps": 30, "15fps": 15},
        },
        # QVGA resolution support 60 (default) and 30 fps
        "QVGA": {
            "resolution": {"width": 320, "height": 240},
            "fps": {"60fps": 60, "30fps": 30},
        },
    }
    LIGHTING_MIN: int = 0
    LIGHTING_MAX: int = 15


class Digit(DigitDefaults):
    __LIGHTING_SCALER = 17

    def __init__(self, serial: str = None, name: str = None, device_index: int = None) -> None:
        """
        DIGIT Device class for a single DIGIT
        :param serial: DIGIT device serial
        :param name: Human friendly identifier name for the device
        :param device_index: Windows camera device index (alternative to serial)
        """
        self.serial: str = serial
        self.name: str = name
        self.device_index: int = device_index

        self.__dev: typing.Optional[cv2.VideoCapture] = None

        self.dev_name: str = ""
        self.manufacturer: str = ""
        self.model: str = ""
        self.revision: int = 200  # Default to newer revision

        self.resolution: typing.Dict = {}
        self.fps: int = 0
        self.intensity: int = 0

        if self.serial is not None:
            logger.debug(f"Digit object constructed with serial {self.serial}")
            self.populate(serial)
        elif self.device_index is not None:
            logger.debug(f"Digit object constructed with device index {self.device_index}")
            self.populate_by_index(device_index)

    def connect(self) -> None:
        """
        Connect to the DIGIT device using DirectShow backend on Windows
        """
        logger.info(f"{self.serial}:Connecting to DIGIT")
        
        # Use DirectShow backend (CAP_DSHOW) on Windows
        if self.device_index is not None:
            self.__dev = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        else:
            # Fallback: try to parse device_index from dev_name if available
            try:
                idx = int(self.dev_name.split("_")[-1])
                self.__dev = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            except (ValueError, IndexError):
                logger.error(f"Cannot determine camera index for {self.serial}")
                raise Exception(f"Error: Cannot determine camera index for {self.serial}")
        
        if not self.__dev.isOpened():
            logger.error(
                f"Cannot open video capture device {self.serial} - {self.dev_name}"
            )
            raise Exception(f"Error opening video stream: {self.dev_name}")
        
        # FIX 1: BUFFER A 1 (Evita lo slittamento dell'immagine nel tempo)
        self.__dev.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # FIX 2: CODEC MJPEG (Evita lo sfarfallio e i blocchi del frame)
        self.__dev.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Risoluzione standard
        self.set_resolution(self.STREAMS["QVGA"])
        
        # FIX 3: 30 FPS FISSI (Mantiene la porta USB stabile senza sovraccaricarla)
        self.set_fps(self.STREAMS["QVGA"]["fps"]["30fps"])
        
        # Accende i LED
        self.set_intensity(15)
        
        # FIX 4: DISABILITA AUTO-ESPOSIZIONE (Evita che l'immagine lampeggi scura/chiara)
        self.__dev.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.__dev.set(cv2.CAP_PROP_EXPOSURE, -5)

    def set_resolution(self, resolution: typing.Dict) -> None:
        """
        Sets stream resolution based on supported streams in Digit.STREAMS
        :param resolution: QVGA or VGA from Digit.STREAMS
        :return: None
        """
        self.resolution = resolution["resolution"]
        width = self.resolution["width"]
        height = self.resolution["height"]
        logger.debug(f"{self.serial}:Stream resolution set to {width}w x {height}h")
        
        # Windows: Use CAP_PROP_FRAME_WIDTH and CAP_PROP_FRAME_HEIGHT
        self.__dev.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.__dev.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def set_fps(self, fps: int) -> None:
        """
        Sets the stream fps, only valid values from Digit.STREAMS are accepted.
        This should typically be called after the resolution is set as the stream fps defaults to the
        highest fps
        :param fps: Stream FPS
        :return: None
        """
        self.fps = fps
        logger.debug(f"{self.serial}:Stream FPS set to {self.fps}")
        self.__dev.set(cv2.CAP_PROP_FPS, self.fps)

    def set_intensity(self, intensity: int) -> int:
        """
        Sets all LEDs to specific intensity, this is a global control.
        :param intensity: Value between 0 and 15 where 0 is all LEDs off and 15 all
        LEDS full intensity
        :return: Returns the set intensity
        """
        if self.revision < 200:
            # Deprecated version 1.01 (1b) is not supported
            intensity = int(intensity / self.__LIGHTING_SCALER)
            logger.warning(
                "You are using a previous version of the firmware "
                "which does not support independent RGB control, update your DIGIT firmware."
            )
        self.intensity = self.set_intensity_rgb(intensity, intensity, intensity)
        return self.intensity

    def set_intensity_rgb(
        self, intensity_r: int, intensity_g: int, intensity_b: int
    ) -> int:
        """
        Sets LEDs to specific intensity, per LED control
        Permitted values are between 0 (off/dim) and 15 (full brightness)
        :param intensity_r: Red value
        :param intensity_g: Green value
        :param intensity_b: Blue value
        :return: Returns the set intensity
        """
        if not all(
            [x in range(0, 16) for x in (intensity_r, intensity_g, intensity_b)]
        ):
            raise ValueError("RGB values must be between 0 and 15.")
        intensity = (intensity_r << 8) | (intensity_g << 4) | intensity_b
        logger.debug(
            f"{self.serial}:LED intensity set to {intensity} (R: {intensity_r} G: {intensity_g} B: {intensity_b})"
        )
        self.intensity = intensity
        self.__dev.set(cv2.CAP_PROP_ZOOM, self.intensity)
        return self.intensity

    def get_frame(self, transpose: bool = False) -> np.ndarray:
        """
        Returns a single image frame for the device
        :param transpose: Show direct output from the image sensor, WxH instead of HxW
        :return: Image frame array
        """
        ret, frame = self.__dev.read()
        if not ret:
            logger.error(
                f"Cannot retrieve frame data from {self.serial}, is DIGIT device open?"
            )
            raise Exception(
                f"Unable to grab frame from {self.serial} - {self.dev_name}!"
            )
        if not transpose:
            # FIX IMMAGINE SPEZZATA E MEMORIA: Niente 'frame' come secondo parametro
            frame = cv2.transpose(frame)
            frame = cv2.flip(frame, 0)
            # Forza l'allineamento in RAM per Windows
            frame = np.ascontiguousarray(frame)

        return frame

    def save_frame(self, path: str) -> np.ndarray:
        """
        Saves a single image frame to host
        :param path: Path and file name where the frame shall be saved to
        :return: Image frame
        """
        frame = self.get_frame()
        logger.debug(f"Saving frame to {path}")
        cv2.imwrite(path, frame)
        return frame

    def get_diff(self, ref_frame: np.ndarray) -> np.ndarray:
        """
        Returns the difference between two frames
        :param ref_frame: Original frame
        :return: Frame difference
        """
        diff = self.get_frame() - ref_frame
        return diff

    def show_view(self, ref_frame: np.ndarray = None) -> None:
        """
        Creates OpenCV named window with live view of DIGIT device, ESC to close window
        :param ref_frame: Specify reference frame to show image difference
        :return: None
        """
        while True:
            frame = self.get_frame()
            if ref_frame is not None:
                frame = self.get_diff(ref_frame)
            cv2.imshow(f"Digit View {self.serial}", frame)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()

    def disconnect(self) -> None:
        logger.debug(f"{self.serial}:Closing DIGIT device")
        if self.__dev is not None:
            self.__dev.release()

    def info(self) -> str:
        """
        Returns DIGIT device info
        :return: String representation of DIGIT device
        """
        has_dev = self.__dev is not None
        is_connected = False
        if has_dev:
            is_connected = self.__dev.isOpened()
        info_string = (
            f"Name: {self.name} {self.dev_name}"
            f"\n\t- Model: {self.model}"
            f"\n\t- Revision: {self.revision}"
            f"\n\t- Connected?: {is_connected}"
        )
        if is_connected:
            info_string += (
                f"\nStream Info:"
                f"\n\t- Resolution: {self.resolution['width']} x {self.resolution['height']}"
                f"\n\t- FPS: {self.fps}"
                f"\n\t- LED Intensity: {self.intensity}"
            )
        return info_string

    def populate(self, serial: str) -> None:
        """
        Find the connected DIGIT based on the serial number and populate device parameters into the class
        :param serial: DIGIT serial number
        :return:
        """
        digit = DigitHandler.find_digit(serial)
        if digit is None:
            raise Exception(f"Cannot find DIGIT with serial {self.serial}")
        self.dev_name = digit["dev_name"]
        self.manufacturer = digit["manufacturer"]
        self.model = digit["model"]
        self.revision = int(digit["revision"])
        self.serial = digit["serial"]
        self.device_index = int(digit["dev_index"])

    def populate_by_index(self, device_index: int) -> None:
        """
        Populate device info based on camera index (Windows convenience method)
        :param device_index: Camera device index
        :return:
        """
        digit = DigitHandler.get_device_by_index(device_index)
        if digit is None:
            raise Exception(f"Cannot find DIGIT at device index {device_index}")
        self.dev_name = digit["dev_name"]
        self.manufacturer = digit["manufacturer"]
        self.model = digit["model"]
        self.revision = int(digit["revision"])
        self.serial = digit["serial"]
        self.device_index = int(digit["dev_index"])

    def __repr__(self) -> str:
        return f"Digit(serial={self.serial}, name={self.name}, device_index={self.device_index})"
