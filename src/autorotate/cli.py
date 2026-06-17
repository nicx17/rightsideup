import sys
from pathlib import Path
from typing import Annotated, Optional
from enum import Enum
import typer
from rich.console import Console
from PIL import Image, ImageOps

from autorotate.types import DEFAULT_YOLO_MODEL, OrientationDecision
from autorotate.engine import decide_orientation
from autorotate.io import (
    iter_images,
    output_path_for,
    exif_orientation,
    lossless_jpeg_rotate,
    copy_original,
    save_image,
)
from autorotate.utils import rotate_pil_clockwise

app = typer.Typer(
    help="Auto-rotate batches of images using local OpenCV/Tesseract analysis."
)
console = Console()


def process_file(
    source: Path,
    destination: Path,
    mode: str,
    backend: str,
    yolo_model: Path,
    yolo_confidence: float,
    yolo_score: float,
    yolo_margin: float,
    human_score: float,
    human_margin: float,
    tesseract_confidence: float,
    line_delta: float,
    dry_run: bool,
    overwrite: bool,
    in_place: bool,
    quality: int,
) -> tuple[Path, OrientationDecision, str | None]:
    try:
        with Image.open(source) as opened:
            source_exif_orientation = exif_orientation(opened)
            image = ImageOps.exif_transpose(opened)
            decision = decide_orientation(
                image,
                mode=mode,
                backend=backend,
                yolo_model=yolo_model,
                yolo_confidence=yolo_confidence,
                yolo_min_score=yolo_score,
                yolo_min_margin=yolo_margin,
                human_min_score=human_score,
                human_min_margin=human_margin,
                tesseract_min_confidence=tesseract_confidence,
                line_min_delta=line_delta,
            )

            if dry_run:
                return source, decision, None

            do_overwrite = overwrite or in_place
            if decision.rotate_clockwise == 0 and source_exif_orientation == 1:
                copy_original(source, destination, do_overwrite)
                return source, decision, None

            if source_exif_orientation == 1 and lossless_jpeg_rotate(
                source,
                destination,
                decision.rotate_clockwise,
                do_overwrite,
            ):
                return source, decision, None

            rotated = rotate_pil_clockwise(image, decision.rotate_clockwise)
            save_image(rotated, destination, overwrite=do_overwrite, quality=quality)
            return source, decision, None
    except Exception as exc:  # noqa: BLE001
        return source, OrientationDecision(0, 0, "error"), str(exc)


class Mode(str, Enum):
    photo = "photo"
    document = "document"
    auto = "auto"


class Backend(str, Enum):
    auto = "auto"
    yolo = "yolo"
    opencv = "opencv"


@app.command()
def main(
    inputs: Annotated[
        list[Path], typer.Argument(help="Image files or directories to process.")
    ],
    output_dir: Annotated[
        Optional[Path],
        typer.Option("-o", "--output-dir", help="Directory for rotated images."),
    ] = None,
    in_place: Annotated[
        bool, typer.Option("--in-place", help="Overwrite input files.")
    ] = False,
    recursive: Annotated[
        bool, typer.Option("-r", "--recursive", help="Recurse into input directories.")
    ] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite files in --output-dir.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print decisions without writing files.")
    ] = False,
    mode: Annotated[
        Mode,
        typer.Option(
            "--mode",
            help="Detection strategy. photo is conservative and optimized for people.",
        ),
    ] = Mode.photo,
    backend: Annotated[
        Backend,
        typer.Option(
            "--backend",
            help="Photo-analysis backend. auto tries YOLO first, then OpenCV.",
        ),
    ] = Backend.auto,
    yolo_model: Annotated[
        Path, typer.Option("--yolo-model", help="Path to a YOLO pose model.")
    ] = DEFAULT_YOLO_MODEL,
    yolo_confidence: Annotated[
        float,
        typer.Option(
            "--yolo-confidence", help="Minimum YOLO person detection confidence."
        ),
    ] = 0.35,
    yolo_score: Annotated[
        float, typer.Option("--yolo-score", help="Minimum YOLO pose-orientation score.")
    ] = 70.0,
    yolo_margin: Annotated[
        float, typer.Option("--yolo-margin", help="Minimum YOLO winning score margin.")
    ] = 18.0,
    human_score: Annotated[
        float, typer.Option("--human-score", help="Minimum human detector score.")
    ] = 52.0,
    human_margin: Annotated[
        float, typer.Option("--human-margin", help="Minimum winning score margin.")
    ] = 22.0,
    tesseract_confidence: Annotated[
        float, typer.Option("--tesseract-confidence", help="Minimum OSD confidence.")
    ] = 2.0,
    line_delta: Annotated[
        float, typer.Option("--line-delta", help="Minimum OpenCV line-score margin.")
    ] = 500.0,
    quality: Annotated[
        int, typer.Option("--quality", help="JPEG output quality.")
    ] = 100,
) -> None:
    if not in_place and output_dir is None and not dry_run:
        console.print(
            "[bold red]error:[/bold red] use --in-place, --output-dir, or --dry-run",
            style="red",
        )
        raise typer.Exit(code=2)

    roots = [path.resolve() for path in inputs]
    files = iter_images(roots, recursive=recursive)
    if not files:
        console.print("No supported images found.", style="yellow")
        raise typer.Exit(code=1)

    failures = 0
    for source in files:
        destination = output_path_for(
            source, output_dir.resolve() if output_dir else None, roots
        )
        _, decision, error = process_file(
            source,
            destination,
            mode=mode.value,
            backend=backend.value,
            yolo_model=yolo_model,
            yolo_confidence=yolo_confidence,
            yolo_score=yolo_score,
            yolo_margin=yolo_margin,
            human_score=human_score,
            human_margin=human_margin,
            tesseract_confidence=tesseract_confidence,
            line_delta=line_delta,
            dry_run=dry_run,
            overwrite=overwrite,
            in_place=in_place,
            quality=quality,
        )
        if error:
            failures += 1
            console.print(f"[bold red]FAIL[/bold red] {source}: {error}")
            continue

        action = "dry-run" if dry_run else str(destination)
        note = f" ({decision.note})" if decision.note else ""
        console.print(
            f"{source} -> rotate [cyan]{decision.rotate_clockwise:3d}[/cyan] deg "
            f"[[green]{decision.method}[/green], confidence={decision.confidence:.2f}]{note} -> {action}"
        )

    if failures:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
