import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from autorotate.types import OrientationDecision
from autorotate.utils import normalize_rotation


def tesseract_osd(
    image: Image.Image, min_confidence: float
) -> OrientationDecision | None:
    tesseract_bin = shutil.which("tesseract")
    if tesseract_bin is None:
        return None

    fd, temp_path_str = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    temp_path = Path(temp_path_str)

    try:
        image.convert("RGB").save(temp_path)
        proc = subprocess.run(
            [tesseract_bin, str(temp_path), "stdout", "--psm", "0"],
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    output = f"{proc.stdout}\n{proc.stderr}"
    rotate_match = re.search(r"Rotate:\s*(\d+)", output)
    conf_match = re.search(r"Orientation confidence:\s*([0-9.]+)", output)
    if rotate_match is None:
        return None

    rotation = normalize_rotation(int(rotate_match.group(1)))
    confidence = float(conf_match.group(1)) if conf_match else 0.0
    if confidence < min_confidence:
        return None

    return OrientationDecision(rotation, confidence, "tesseract-osd")
