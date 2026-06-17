import cv2
import numpy as np
from PIL import Image
from typing import Any

def normalize_rotation(degrees: int) -> int:
    return int(degrees) % 360

def pil_to_cv_gray(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def prepare_gray(image: Image.Image, max_side: int = 1200) -> np.ndarray:
    gray = pil_to_cv_gray(image)
    h, w = gray.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(gray)

def rotate_pil_clockwise(image: Image.Image, degrees: int) -> Image.Image:
    degrees = normalize_rotation(degrees)
    if degrees == 0:
        return image.copy()
    # PIL rotates counter-clockwise for positive angles.
    return image.rotate(-degrees, expand=True)

def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)
