import re
import shutil
import subprocess
import tempfile
from PIL import Image
from autorotate.types import OrientationDecision
from autorotate.utils import normalize_rotation

def tesseract_osd(image: Image.Image, min_confidence: float) -> OrientationDecision | None:
    if shutil.which("tesseract") is None:
        return None

    try:
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image.convert("RGB").save(tmp.name)
            proc = subprocess.run(
                ["tesseract", tmp.name, "stdout", "--psm", "0"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
    except OSError:
        return None

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
