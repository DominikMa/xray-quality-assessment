import cv2
import numpy as np

# Points in (x, y) format from -1 to 1
# LeftTop, RightTop, RightBottom, LeftBottom
NORMALIZED_TARGETS = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])


def longest_max_size(image: np.ndarray, max_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    longest_size = max(h, w)
    factor = max_size / longest_size
    new_height, new_width = tuple(int(round(dim * factor)) for dim in (h, w))
    # output_image = cv2.resize(image, dsize=None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
    output_image = cv2.resize(image, dsize=(new_width, new_height), interpolation=cv2.INTER_LINEAR)
    output_image = cv2.resize(image, dsize=(new_width, new_height), interpolation=cv2.INTER_AREA)
    return output_image


def pad_if_needed(image: np.ndarray, min_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    new_h = max(h, min_size)
    new_w = max(w, min_size)
    new_shape = new_h, new_w, *image.shape[2:]
    offset_h = int((new_h - h) / 2)
    offset_w = int((new_w - w) / 2)

    output_image = np.zeros(new_shape, dtype=image.dtype)
    output_image[offset_h : offset_h + h, offset_w : offset_w + w] = image
    return output_image


def pad_channels_if_needed(image: np.ndarray, channels: int) -> np.ndarray:
    if len(image.shape) >= channels:
        return image
    empty_channels = [1] * (channels - len(image.shape))
    return image.reshape((*empty_channels, *image.shape))


def prepare_input(image: np.ndarray, size: int, channels: int) -> np.ndarray:
    image = image.copy()
    image = pad_if_needed(longest_max_size(image, size), size)
    image = pad_channels_if_needed(image.astype(np.float32), channels)
    image = image - 0.5 * 2**15
    return image


def prepare_input_roi(image: np.ndarray) -> np.ndarray:
    image = prepare_input(image, 512, 2)

    # Add spatial encoding
    input_h, input_w = image.shape[-2:]
    encoding = np.mgrid[0:input_h, 0:input_w].astype(np.float32)
    encoding = encoding / [[[input_h - 1]], [[input_w - 1]]] - 0.5
    image = np.concatenate([image[None, :], encoding], axis=0)
    image = image.astype(np.float32)
    image = pad_channels_if_needed(image, 4)
    return image


def get_roi_for_theta(image: np.array, theta: np.ndarray):
    max_size = max(image.shape[-2:])
    roi_cut_image = pad_if_needed(image, max_size)
    h, w = roi_cut_image.shape[-2:]

    theta = np.concatenate((np.squeeze(theta), np.array([[0, 0, 1]])))
    normalized_sources = (theta @ NORMALIZED_TARGETS.T)[:2, :].T

    sources = ((normalized_sources + 1) * (np.array([w, h]) / 2)).astype(np.float32)
    targets = np.array([[0, 0], [224, 0], [224, 224], [0, 224]]).astype(np.float32)

    transform = cv2.getAffineTransform(sources[:3], targets[:3])
    roi = cv2.warpAffine(roi_cut_image, transform, (224, 224))
    return roi
