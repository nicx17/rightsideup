import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable
from PIL import Image

from autorotate.types import SUPPORTED_EXTENSIONS, JPEG_EXTENSIONS, EXIF_ORIENTATION_TAG
from autorotate.utils import normalize_rotation


def iter_images(paths: Iterable[Path], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            files.extend(
                item
                for item in iterator
                if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(dict.fromkeys(files))


def output_path_for(source: Path, output_dir: Path | None, roots: list[Path]) -> Path:
    if output_dir is None:
        return source

    parent_root = next(
        (root for root in roots if root.is_dir() and source.is_relative_to(root)), None
    )
    if parent_root is None:
        return output_dir / source.name
    return output_dir / source.relative_to(parent_root)


def ensure_can_write(destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; pass --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)


def exif_orientation(image: Image.Image) -> int:
    try:
        return int(image.getexif().get(EXIF_ORIENTATION_TAG, 1))
    except (AttributeError, TypeError, ValueError):
        return 1


def reset_exif_orientation(path: Path) -> None:
    if shutil.which("exiftool") is None:
        return
    subprocess.run(
        ["exiftool", "-overwrite_original", "-Orientation=1", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def lossless_jpeg_rotate(
    source: Path, destination: Path, degrees: int, overwrite: bool
) -> bool:
    if source.suffix.lower() not in JPEG_EXTENSIONS or shutil.which("jpegtran") is None:
        return False
    degrees = normalize_rotation(degrees)
    if degrees not in {90, 180, 270}:
        return False

    ensure_can_write(destination, overwrite)
    output_target = destination
    temp_target: Path | None = None
    if source.resolve() == destination.resolve():
        fd, temp_name = tempfile.mkstemp(suffix=source.suffix, dir=str(source.parent))
        os.close(fd)
        temp_target = Path(temp_name)
        output_target = temp_target

    try:
        subprocess.run(
            [
                "jpegtran",
                "-copy",
                "all",
                "-perfect",
                "-rotate",
                str(degrees),
                "-outfile",
                str(output_target),
                str(source),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        reset_exif_orientation(output_target)
        if temp_target is not None:
            temp_target.replace(destination)
        return True
    except (OSError, subprocess.CalledProcessError):
        if temp_target is not None and temp_target.exists():
            temp_target.unlink()
        return False


def copy_original(source: Path, destination: Path, overwrite: bool) -> None:
    if source.resolve() == destination.resolve():
        return
    ensure_can_write(destination, overwrite)
    shutil.copy2(source, destination)


def save_image(
    image: Image.Image, destination: Path, overwrite: bool, quality: int
) -> None:
    ensure_can_write(destination, overwrite)
    save_kwargs: dict[str, Any] = {}
    suffix = destination.suffix.lower()
    if suffix in JPEG_EXTENSIONS:
        save_kwargs.update(quality=quality, subsampling=0, optimize=True)
    elif suffix == ".png":
        save_kwargs.update(compress_level=0)
    image.save(destination, **save_kwargs)
