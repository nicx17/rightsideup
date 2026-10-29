import dataclasses
import os
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}
EXIF_ORIENTATION_TAG = 274


def _find_default_yolo_model() -> Path:
    env_model = os.environ.get("AUTOROTATE_YOLO_MODEL")
    if env_model:
        return Path(env_model)
    # Check repo directory relative to this package file
    repo_model = (
        Path(__file__).resolve().parent.parent.parent / "models" / "yolo11m-pose.onnx"
    )
    if repo_model.is_file():
        return repo_model
    # Check standard user cache directory
    cache_model = Path.home() / ".cache" / "autorotate" / "models" / "yolo11m-pose.onnx"
    if cache_model.is_file():
        return cache_model
    return repo_model


DEFAULT_YOLO_MODEL = _find_default_yolo_model()


@dataclasses.dataclass(frozen=True)
class OrientationDecision:
    rotate_clockwise: int
    confidence: float
    method: str
    note: str = ""
    is_confident: bool = True


@dataclasses.dataclass(frozen=True)
class YoloOnnxModel:
    session: Any
    input_name: str
    input_size: int
