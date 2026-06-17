import dataclasses
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
DEFAULT_YOLO_MODEL = Path("models/yolo11m-pose.onnx")

@dataclasses.dataclass(frozen=True)
class OrientationDecision:
    rotate_clockwise: int
    confidence: float
    method: str
    note: str = ""

@dataclasses.dataclass(frozen=True)
class YoloOnnxModel:
    session: Any
    input_name: str
    input_size: int
