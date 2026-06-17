import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from autorotate.types import OrientationDecision
from autorotate.utils import prepare_gray, rotate_pil_clockwise

CASCADE_CACHE: dict[str, cv2.CascadeClassifier | None] = {}
HOG_PEOPLE_DETECTOR: cv2.HOGDescriptor | None = None


def load_haar(name: str) -> cv2.CascadeClassifier | None:
    if name in CASCADE_CACHE:
        return CASCADE_CACHE[name]

    candidates: list[Path] = []
    cv2_data = getattr(cv2, "data", None)
    haarcascades = getattr(cv2_data, "haarcascades", None)
    if haarcascades:
        candidates.append(Path(haarcascades) / name)
    candidates.extend(
        [
            Path("/usr/share/opencv4/haarcascades") / name,
            Path("/usr/share/opencv/haarcascades") / name,
            Path("/usr/local/share/opencv4/haarcascades") / name,
        ]
    )

    for path in candidates:
        if not path.exists():
            continue
        classifier = cv2.CascadeClassifier(str(path))
        if not classifier.empty():
            CASCADE_CACHE[name] = classifier
            return classifier
    CASCADE_CACHE[name] = None
    return None


def detect_objects(
    gray: np.ndarray,
    cascade_name: str,
    *,
    scale_factor: float = 1.08,
    min_neighbors: int = 5,
    min_size: tuple[int, int] = (30, 30),
) -> list[tuple[int, int, int, int]]:
    detector = load_haar(cascade_name)
    if detector is None:
        return []
    objects = detector.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in objects]


def largest_area_ratio(
    boxes: list[tuple[int, int, int, int]], image_area: int
) -> float:
    if not boxes or image_area <= 0:
        return 0.0
    return max((w * h) / image_area for _, _, w, h in boxes)


def get_hog_people_detector() -> cv2.HOGDescriptor:
    global HOG_PEOPLE_DETECTOR
    if HOG_PEOPLE_DETECTOR is None:
        HOG_PEOPLE_DETECTOR = cv2.HOGDescriptor()
        HOG_PEOPLE_DETECTOR.setSVMDetector(  # type: ignore[reportAttributeAccessIssue]
            cv2.HOGDescriptor_getDefaultPeopleDetector()
        )
    return HOG_PEOPLE_DETECTOR


def hog_people(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    detector = get_hog_people_detector()
    boxes, weights = detector.detectMultiScale(  # type: ignore[reportCallIssue]
        gray,
        hitThreshold=0,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.05,
        finalThreshold=2,
    )

    detections: list[tuple[int, int, int, int]] = []
    for box, weight in zip(boxes, weights, strict=False):
        if float(weight) >= 0.35:
            x, y, w, h = box
            detections.append((int(x), int(y), int(w), int(h)))
    return detections


def eye_bonus(gray: np.ndarray, faces: list[tuple[int, int, int, int]]) -> float:
    eye_detector = load_haar("haarcascade_eye.xml") or load_haar(
        "haarcascade_eye_tree_eyeglasses.xml"
    )
    if eye_detector is None:
        return 0.0

    bonus = 0.0
    for x, y, fw, fh in faces:
        upper_face = gray[y : y + max(1, fh // 2), x : x + fw]
        eyes = eye_detector.detectMultiScale(
            upper_face,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(8, 8),
        )
        bonus += min(len(eyes), 2) * 8
    return bonus


def profile_faces(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    direct = detect_objects(
        gray,
        "haarcascade_profileface.xml",
        scale_factor=1.08,
        min_neighbors=5,
        min_size=(35, 35),
    )
    flipped = cv2.flip(gray, 1)
    mirrored = detect_objects(
        flipped,
        "haarcascade_profileface.xml",
        scale_factor=1.08,
        min_neighbors=5,
        min_size=(35, 35),
    )
    width = gray.shape[1]
    mirrored = [(width - x - w, y, w, h) for x, y, w, h in mirrored]
    return direct + mirrored


def human_score(gray: np.ndarray) -> float:
    image_area = gray.shape[0] * gray.shape[1]
    frontal_faces: list[tuple[int, int, int, int]] = []
    for cascade in (
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt.xml",
        "haarcascade_frontalface_alt2.xml",
    ):
        frontal_faces.extend(
            detect_objects(
                gray,
                cascade,
                scale_factor=1.07,
                min_neighbors=5,
                min_size=(35, 35),
            )
        )

    profiles = profile_faces(gray)
    upper_bodies = detect_objects(
        gray,
        "haarcascade_upperbody.xml",
        scale_factor=1.05,
        min_neighbors=4,
        min_size=(60, 80),
    )
    full_bodies = detect_objects(
        gray,
        "haarcascade_fullbody.xml",
        scale_factor=1.05,
        min_neighbors=3,
        min_size=(45, 90),
    )
    hog_bodies = hog_people(gray)

    score = 0.0
    score += len(frontal_faces) * 32
    score += len(profiles) * 20
    score += len(upper_bodies) * 7
    score += len(full_bodies) * 5
    score += len(hog_bodies) * 12
    score += largest_area_ratio(frontal_faces, image_area) * 180
    score += largest_area_ratio(profiles, image_area) * 120
    score += largest_area_ratio(upper_bodies, image_area) * 45
    score += largest_area_ratio(hog_bodies, image_area) * 75
    score += eye_bonus(gray, frontal_faces)
    return score


def opencv_human_orientation(
    image: Image.Image, min_score: float, min_margin: float
) -> OrientationDecision:
    scores: dict[int, float] = {}
    for degrees in (0, 90, 180, 270):
        candidate = rotate_pil_clockwise(image, degrees)
        scores[degrees] = human_score(prepare_gray(candidate))

    best_angle, best_score = max(scores.items(), key=lambda item: item[1])
    second_score = sorted(scores.values(), reverse=True)[1]
    margin = best_score - second_score
    if best_score >= min_score and margin >= min_margin:
        return OrientationDecision(
            best_angle,
            margin,
            "opencv-human",
            f"score={best_score:.2f}, next={second_score:.2f}",
        )
    return OrientationDecision(
        0, margin, "opencv-human", "no confident human orientation", is_confident=False
    )
