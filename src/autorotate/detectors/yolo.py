import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Any

from autorotate.types import OrientationDecision, YoloOnnxModel
from autorotate.utils import rotate_pil_clockwise, tensor_to_numpy

YOLO_MODEL_CACHE: dict[Path, Any] = {}

def load_yolo_model(model_path: Path) -> YoloOnnxModel:
    resolved = model_path.expanduser().resolve()
    if resolved in YOLO_MODEL_CACHE:
        return YOLO_MODEL_CACHE[resolved]
    if not resolved.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {resolved}. Download yolo11m-pose.onnx into models/."
        )
    try:
        import onnxruntime as ort  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(
            "YOLO ONNX backend needs onnxruntime. Install it with "
            ".venv/bin/python -m pip install onnxruntime"
        ) from exc

    session = ort.InferenceSession(str(resolved), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    input_shape = model_input.shape
    input_size = 640
    for dimension in reversed(input_shape):
        if isinstance(dimension, int) and dimension > 0:
            input_size = dimension
            break
    model = YoloOnnxModel(
        session=session, input_name=model_input.name, input_size=input_size
    )
    YOLO_MODEL_CACHE[resolved] = model
    return model

def yolo_preprocess(image: Image.Image, input_size: int) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    blob = canvas.astype(np.float32) / 255.0
    return np.transpose(blob, (2, 0, 1))[None, ...]

def yolo_prediction_rows(output: np.ndarray) -> np.ndarray:
    predictions = np.squeeze(output)
    if predictions.ndim != 2:
        return np.empty((0, 0), dtype=np.float32)
    if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] <= 64:
        predictions = predictions.T
    return predictions

def yolo_pose_rows(output: np.ndarray, confidence: float) -> list[tuple[float, np.ndarray]]:
    rows = yolo_prediction_rows(output)
    if rows.size == 0:
        return []

    detections: list[tuple[float, np.ndarray]] = []
    for row in rows:
        if row.shape[0] < 56:
            continue
        box_confidence = float(row[4])
        keypoint_start = 5
        if row.shape[0] >= 57 and (row.shape[0] - 5) % 3 != 0:
            box_confidence = float(row[4] * row[5])
            keypoint_start = 6
        if box_confidence < confidence:
            continue
        keypoints = row[keypoint_start : keypoint_start + 51].reshape(17, 3)
        detections.append((box_confidence, keypoints))
    return detections

def keypoint_point(keypoints: np.ndarray, index: int, min_confidence: float) -> tuple[float, float] | None:
    if keypoints.shape[0] <= index or keypoints.shape[1] < 3:
        return None
    x, y, confidence = keypoints[index][:3]
    if float(confidence) < min_confidence:
        return None
    return float(x), float(y)

def average_points(points: list[tuple[float, float] | None]) -> tuple[float, float] | None:
    valid = [point for point in points if point is not None]
    if not valid:
        return None
    xs, ys = zip(*valid, strict=False)
    return float(np.mean(xs)), float(np.mean(ys))

def vertical_order_score(upper: tuple[float, float] | None, lower: tuple[float, float] | None, expected_gap: float, weight: float) -> float:
    if upper is None or lower is None:
        return 0.0
    gap = lower[1] - upper[1]
    if gap <= 0:
        return -weight
    return min(gap / max(expected_gap, 1.0), 1.5) * weight

def pair_level_score(left: tuple[float, float] | None, right: tuple[float, float] | None, expected_width: float, weight: float) -> float:
    if left is None or right is None:
        return 0.0
    dx = abs(right[0] - left[0])
    dy = abs(right[1] - left[1])
    if dx < expected_width * 0.15:
        return -weight * 0.5
    tilt = dy / max(dx, 1.0)
    return max(0.0, 1.0 - tilt) * weight

def yolo_person_score(keypoints: np.ndarray, image_height: int, box_confidence: float) -> float:
    min_kpt_conf = 0.25
    nose = keypoint_point(keypoints, 0, min_kpt_conf)
    left_eye = keypoint_point(keypoints, 1, min_kpt_conf)
    right_eye = keypoint_point(keypoints, 2, min_kpt_conf)
    left_shoulder = keypoint_point(keypoints, 5, min_kpt_conf)
    right_shoulder = keypoint_point(keypoints, 6, min_kpt_conf)
    left_hip = keypoint_point(keypoints, 11, min_kpt_conf)
    right_hip = keypoint_point(keypoints, 12, min_kpt_conf)
    left_knee = keypoint_point(keypoints, 13, min_kpt_conf)
    right_knee = keypoint_point(keypoints, 14, min_kpt_conf)
    left_ankle = keypoint_point(keypoints, 15, min_kpt_conf)
    right_ankle = keypoint_point(keypoints, 16, min_kpt_conf)

    head = average_points([nose, left_eye, right_eye])
    shoulders = average_points([left_shoulder, right_shoulder])
    hips = average_points([left_hip, right_hip])
    knees = average_points([left_knee, right_knee])
    ankles = average_points([left_ankle, right_ankle])

    visible = sum(float(point[2] >= min_kpt_conf) for point in keypoints if len(point) >= 3)
    expected_gap = max(image_height * 0.08, 20.0)
    expected_width = max(image_height * 0.05, 15.0)

    score = 0.0
    score += visible * 2.5
    score += float(box_confidence) * 35
    score += vertical_order_score(head, shoulders, expected_gap, 24)
    score += vertical_order_score(shoulders, hips, expected_gap, 30)
    score += vertical_order_score(hips, knees, expected_gap, 18)
    score += vertical_order_score(knees, ankles, expected_gap, 14)
    score += pair_level_score(left_shoulder, right_shoulder, expected_width, 14)
    score += pair_level_score(left_hip, right_hip, expected_width, 10)
    return score

def yolo_pose_score(image: Image.Image, model: YoloOnnxModel, confidence: float) -> float:
    blob = yolo_preprocess(image, model.input_size)
    outputs = model.session.run(None, {model.input_name: blob})
    detections = yolo_pose_rows(tensor_to_numpy(outputs[0]), confidence)
    scores = [
        yolo_person_score(person_keypoints, model.input_size, box_confidence)
        for box_confidence, person_keypoints in detections
    ]
    return float(max(scores, default=0.0))

def yolo_orientation(image: Image.Image, model_path: Path, confidence: float, min_score: float, min_margin: float) -> OrientationDecision:
    model = load_yolo_model(model_path)
    scores: dict[int, float] = {}
    for degrees in (0, 90, 180, 270):
        candidate = rotate_pil_clockwise(image, degrees)
        scores[degrees] = yolo_pose_score(candidate, model, confidence)

    best_angle, best_score = max(scores.items(), key=lambda item: item[1])
    second_score = sorted(scores.values(), reverse=True)[1]
    margin = best_score - second_score
    if best_score >= min_score and margin >= min_margin:
        return OrientationDecision(
            best_angle,
            margin,
            "yolo-pose",
            f"score={best_score:.2f}, next={second_score:.2f}",
        )
    return OrientationDecision(0, margin, "yolo-pose", "no confident YOLO orientation")
