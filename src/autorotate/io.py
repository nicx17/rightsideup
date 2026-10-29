import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image

from autorotate.types import EXIF_ORIENTATION_TAG, JPEG_EXTENSIONS, SUPPORTED_EXTENSIONS
from autorotate.utils import normalize_rotation


def iter_images(paths: Iterable[Path], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            resolved_dir = path.resolve()
            iterator = path.rglob("*") if recursive else path.iterdir()
            for item in iterator:
                if (
                    not item.is_file()
                    or item.suffix.lower() not in SUPPORTED_EXTENSIONS
                ):
                    continue
                # If item is a symlink, prevent symlink traversal outside the search root
                if item.is_symlink():
                    try:
                        resolved_item = item.resolve()
                        if not resolved_item.is_relative_to(resolved_dir):
                            continue
                    except (OSError, RuntimeError, ValueError):
                        continue
                files.append(item)
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
        destination = output_dir / source.name
    else:
        destination = output_dir / source.relative_to(parent_root)

    # Security check: ensure destination cannot escape output_dir
    resolved_output_dir = output_dir.resolve()
    resolved_destination = destination.resolve()
    if not (
        resolved_destination == resolved_output_dir
        or resolved_destination.is_relative_to(resolved_output_dir)
    ):
        raise ValueError(
            f"Path traversal detected: {destination} escapes output directory {output_dir}"
        )

    return destination


def ensure_can_write(destination: Path, overwrite: bool) -> None:
    if (destination.exists() or destination.is_symlink()) and not overwrite:
        raise FileExistsError(f"{destination} already exists; pass --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        destination.unlink()


def exif_orientation(image: Image.Image) -> int:
    try:
        return int(image.getexif().get(EXIF_ORIENTATION_TAG, 1))
    except (AttributeError, TypeError, ValueError):
        return 1


def reset_exif_orientation(path: Path) -> None:
    exiftool_bin = shutil.which("exiftool")
    if exiftool_bin is None:
        return
    try:
        subprocess.run(
            [
                exiftool_bin,
                "-overwrite_original",
                "-Orientation=1",
                "--",
                str(path.resolve()),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def lossless_jpeg_rotate(
    source: Path, destination: Path, degrees: int, overwrite: bool
) -> bool:
    jpegtran_bin = shutil.which("jpegtran")
    if source.suffix.lower() not in JPEG_EXTENSIONS or jpegtran_bin is None:
        return False
    degrees = normalize_rotation(degrees)
    if degrees not in {90, 180, 270}:
        return False

    ensure_can_write(destination, overwrite)
    output_parent = destination.parent
    fd, temp_name = tempfile.mkstemp(suffix=destination.suffix, dir=str(output_parent))
    os.close(fd)
    temp_target = Path(temp_name)

    try:
        subprocess.run(
            [
                jpegtran_bin,
                "-copy",
                "all",
                "-perfect",
                "-rotate",
                str(degrees),
                "-outfile",
                str(temp_target.resolve()),
                str(source.resolve()),
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
        reset_exif_orientation(temp_target)
        if destination.exists() and not destination.is_symlink():
            try:
                shutil.copymode(destination, temp_target)
            except OSError:
                pass
        elif source.exists() and not source.is_symlink():
            try:
                shutil.copymode(source, temp_target)
            except OSError:
                pass
        temp_target.replace(destination)
        return True
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        if temp_target.exists():
            try:
                temp_target.unlink()
            except OSError:
                pass


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

    output_parent = destination.parent
    fd, temp_name = tempfile.mkstemp(suffix=destination.suffix, dir=str(output_parent))
    os.close(fd)
    temp_target = Path(temp_name)
    try:
        image.save(temp_target, **save_kwargs)
        if destination.exists() and not destination.is_symlink():
            try:
                shutil.copymode(destination, temp_target)
            except OSError:
                pass
        temp_target.replace(destination)
    finally:
        if temp_target.exists():
            try:
                temp_target.unlink()
            except OSError:
                pass
