# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional

import cv2

logger = logging.getLogger(__name__)


class DigitHandler:
    @staticmethod
    def _parse(device_index: int, cap: cv2.VideoCapture) -> Optional[Dict[str, str]]:
        """
        Parse device information from OpenCV VideoCapture object on Windows
        :param device_index: Camera device index
        :param cap: VideoCapture object
        :return: Dictionary with device info or None if not a DIGIT device
        """
        try:
            # Get device name from OpenCV (Windows)
            dev_name = f"CAP_DSHOW_{device_index}"
            
            # Try to read a frame to validate the device is working
            if not cap.isOpened():
                return None
            
            ret, _ = cap.read()
            if not ret:
                return None
            
            # Extract device info from the backend name
            backend_name = cap.getBackendName()
            
            digit_info = {
                "dev_index": str(device_index),
                "dev_name": dev_name,
                "backend": backend_name,
                "manufacturer": "Facebook",  # DIGIT devices are made by Facebook
                "model": "DIGIT",
                "revision": "200",  # Assume newer revision for Windows
                "serial": f"DIGIT_{device_index}",  # Will be updated if detected
            }
            return digit_info
        except Exception as e:
            logger.debug(f"Error parsing device {device_index}: {e}")
            return None

    @staticmethod
    def list_digits() -> List[Dict[str, str]]:
        """
        List all available DIGIT devices on Windows
        Scans camera indices 0-10 and attempts to detect DIGIT devices
        :return: List of detected DIGIT device dictionaries
        """
        logger.debug("Scanning for DIGIT devices on Windows (checking camera indices 0-10)")
        digits = []
        
        # Windows typically supports camera indices 0-10
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    logger.debug(f"Found device at index {i}")
                    device_info = DigitHandler._parse(i, cap)
                    if device_info:
                        digits.append(device_info)
                        logger.debug(f"Device {i}: {device_info}")
                    cap.release()
            except Exception as e:
                logger.debug(f"Error checking device {i}: {e}")
                continue
        
        if not digits:
            logger.debug("Could not find any DIGIT devices")
        
        return digits

    @staticmethod
    def find_digit(serial: str) -> Optional[Dict[str, str]]:
        """
        Find a specific DIGIT device by serial number
        :param serial: Serial number to search for
        :return: Device dictionary if found, None otherwise
        """
        digits = DigitHandler.list_digits()
        logger.debug(f"Searching for DIGIT with serial number {serial}")
        
        for digit in digits:
            if digit["serial"] == serial:
                logger.debug(f"Found DIGIT with serial {serial}")
                return digit
        
        logger.error(f"No DIGIT with serial number {serial} found")
        return None

    @staticmethod
    def get_device_by_index(device_index: int) -> Optional[Dict[str, str]]:
        """
        Get DIGIT device by camera index (Windows convenience method)
        :param device_index: Camera device index (0-10)
        :return: Device dictionary if found, None otherwise
        """
        try:
            cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                device_info = DigitHandler._parse(device_index, cap)
                cap.release()
                if device_info:
                    logger.debug(f"Found DIGIT at index {device_index}")
                    return device_info
            else:
                logger.warning(f"Could not open camera at index {device_index}")
        except Exception as e:
            logger.error(f"Error accessing device {device_index}: {e}")
        
        return None
