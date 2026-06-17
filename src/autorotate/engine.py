import dataclasses
from pathlib import Path
from PIL import Image

from autorotate.types import OrientationDecision
from autorotate.detectors.tesseract import tesseract_osd
from autorotate.detectors.yolo import yolo_orientation
from autorotate.detectors.opencv_human import opencv_human_orientation
from autorotate.detectors.opencv_lines import opencv_sideways_orientation


def decide_orientation(
    image: Image.Image,
    mode: str,
    backend: str,
    yolo_model: Path,
    yolo_confidence: float,
    yolo_min_score: float,
    yolo_min_margin: float,
    human_min_score: float,
    human_min_margin: float,
    tesseract_min_confidence: float,
    line_min_delta: float,
) -> OrientationDecision:
    yolo_note = "not attempted"
    if mode in {"photo", "auto"}:
        if backend in {"auto", "yolo"}:
            try:
                decision = yolo_orientation(
                    image,
                    yolo_model,
                    confidence=yolo_confidence,
                    min_score=yolo_min_score,
                    min_margin=yolo_min_margin,
                )
                if backend == "yolo" or decision.is_confident:
                    return decision
            except (FileNotFoundError, RuntimeError) as exc:
                if backend == "yolo":
                    raise
                yolo_note = str(exc)
            else:
                yolo_note = "no confident YOLO orientation"

        decision = opencv_human_orientation(image, human_min_score, human_min_margin)
        if decision.is_confident:
            return decision
        if mode == "photo":
            if backend == "auto":
                return dataclasses.replace(
                    decision, note=f"{decision.note}; YOLO skipped: {yolo_note}"
                )
            return decision

    if mode in {"document", "auto"}:
        decision = tesseract_osd(image, tesseract_min_confidence)
        if decision is not None:
            return decision
        return opencv_sideways_orientation(image, line_min_delta)

    return OrientationDecision(0, 0, "none", "no detector enabled", is_confident=False)
