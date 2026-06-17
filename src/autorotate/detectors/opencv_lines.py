import cv2
import numpy as np
from PIL import Image

from autorotate.types import OrientationDecision
from autorotate.utils import pil_to_cv_gray, rotate_pil_clockwise


def line_projection_score(gray: np.ndarray) -> float:
    h, w = gray.shape[:2]
    scale = min(1.0, 1200.0 / max(h, w))
    small = (
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1
        else gray
    )
    small = cv2.GaussianBlur(small, (3, 3), 0)

    edges = cv2.Canny(small, 70, 180)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(30, small.shape[1] // 8),
        maxLineGap=10,
    )

    horizontal_weight = 0.0
    vertical_weight = 0.0
    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = float(np.hypot(dx, dy))
            if length <= 0:
                continue
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            angle = min(angle, 180 - angle)
            if angle <= 12:
                horizontal_weight += length
            elif abs(angle - 90) <= 12:
                vertical_weight += length

    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_density = binary.mean(axis=1)
    col_density = binary.mean(axis=0)
    projection_bias = float(row_density.var() - col_density.var())

    return horizontal_weight - vertical_weight + projection_bias * 20


def opencv_sideways_orientation(
    image: Image.Image, min_delta: float
) -> OrientationDecision:
    scores: dict[int, float] = {}
    for degrees in (0, 90, 270):
        candidate = rotate_pil_clockwise(image, degrees)
        scores[degrees] = line_projection_score(pil_to_cv_gray(candidate))

    best_angle, best_score = max(scores.items(), key=lambda item: item[1])
    second_score = sorted(scores.values(), reverse=True)[1]
    delta = best_score - second_score
    if best_angle != 0 and delta >= min_delta:
        return OrientationDecision(best_angle, delta, "opencv-lines")
    return OrientationDecision(
        0, max(0.0, delta), "opencv-lines", "no confident rotation", is_confident=False
    )
