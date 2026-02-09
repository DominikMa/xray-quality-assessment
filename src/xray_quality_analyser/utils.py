import logging
from collections import namedtuple
from pathlib import Path
from typing import Optional

import coloredlogs
import numpy as np
from pydicom import FileDataset

ImageData = namedtuple("ImageData", ["shm_name", "pixel_array_shape", "pixel_array_dtype", "metadata"])
PredictionData = namedtuple("PredictionData", ["image_data", "body_part", "view", "laterality", "quality"])

SOP_Class_UIDS = [
    "1.2.840.10008.5.1.4.1.1.1",
    "1.2.840.10008.5.1.4.1.1.1.1",
    "1.2.840.10008.5.1.4.1.1.1.1.1",
]


def to_uint8_image(image: np.ndarray):
    image = image.astype(float)
    image = image - image.min()
    image = image / (image.max() / 255)
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    return image


def ensure_sop_class_uid(dicom: FileDataset):
    if dicom.SOPClassUID not in SOP_Class_UIDS:
        raise ValueError()
    return dicom


def ensure_photometric_interpretation(dicom: FileDataset):
    if dicom.PhotometricInterpretation == "MONOCHROME2":
        return dicom
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        # "The minimum sample value is intended to be displayed as white after any VOI gray scale transformations have been performed."
        # Monochrome 1 means more x-rays -> higher value, less x-rays -> lower value (with no transformation this means bones black, air white)
        #  Therefore "inverse" the image
        pixel_array = dicom.pixel_array
        pixel_array_org = pixel_array.copy()

        # Try to remove background around actual x-ray by assuming it is uniform and the largest (whitest) color
        maximum = pixel_array.max()
        second_largest_value = np.unique(pixel_array_org)[-2]
        pixel_array[pixel_array == maximum] = second_largest_value

        # Inverse color
        pixel_array = pixel_array.max() - pixel_array
        dicom.PixelData = pixel_array.tobytes()
        dicom.PhotometricInterpretation = "MONOCHROME2"

        return dicom

    raise ValueError()


def ensure_15bit_depth(dicom: FileDataset):
    bits_stored: int = dicom.BitsStored
    if bits_stored == 15:
        return dicom

    pixel_array = dicom.pixel_array
    assert pixel_array.dtype == np.uint16

    pixel_array = pixel_array.astype(float)
    # shifting bits so always 15 bits are used
    if bits_stored > 15:
        pixel_array = pixel_array / (2 ** (bits_stored - 15))
    elif bits_stored < 15:
        pixel_array = pixel_array * (2 ** (15 - bits_stored))

    pixel_array = pixel_array.astype(np.uint16)
    dicom.PixelData = pixel_array.tobytes()
    dicom.BitsStored = 15
    return dicom


def dicom_to_png(dicom: FileDataset):
    dicom = ensure_sop_class_uid(dicom)
    dicom = ensure_photometric_interpretation(dicom)
    dicom = ensure_15bit_depth(dicom)

    pixel_array = dicom.pixel_array
    assert pixel_array.dtype == np.uint16

    return pixel_array


def setup_logger(
    logger_name: str,
    console_log_level: int = logging.INFO,
    file_log_level: int = logging.DEBUG,
    file_log_path: Optional[Path] = None,
) -> None:
    min_log_level = min(console_log_level, file_log_level)
    logger = logging.getLogger(logger_name)
    logger.setLevel(min_log_level)

    coloredlogs.install(level=logging.getLevelName(console_log_level), logger=logger)
    logging.getLogger().setLevel(0)
    logger.setLevel(min_log_level)

    if file_log_path is not None:
        file_log_handler = logging.FileHandler(file_log_path)
        file_log_handler.setLevel(file_log_level)

        coloredlogs.HostNameFilter.install(file_log_handler, fmt=coloredlogs.DEFAULT_LOG_FORMAT)
        coloredlogs.ProgramNameFilter.install(file_log_handler, fmt=coloredlogs.DEFAULT_LOG_FORMAT)
        coloredlogs.UserNameFilter.install(handler=file_log_handler, fmt=coloredlogs.DEFAULT_LOG_FORMAT)
        formatter = coloredlogs.BasicFormatter(
            fmt=coloredlogs.DEFAULT_LOG_FORMAT, datefmt=coloredlogs.DEFAULT_DATE_FORMAT
        )
        file_log_handler.setFormatter(formatter)

        logger.addHandler(file_log_handler)
