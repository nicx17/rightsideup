<p align="center">
  <img src="assets/round-icons-9r1ibje4J5A-unsplash.svg" alt="RightSideUp Icon" width="128"/>
</p>

# RightSideUp

This project provides a local Python CLI that analyzes images and rotates them to the most likely upright orientation. It does not call any external API.

The default detector is optimized for photos with people. It tries a local YOLO pose model first, then falls back to conservative OpenCV detection if YOLO is unavailable.

The detector can use:

- YOLO11 pose keypoints for person, face, and body orientation.
- OpenCV face/eye detection for photos with upright people.
- OpenCV profile-face, upper-body, and full-body detection for human subjects.
- OpenCV HOG person detection for full-body people photos.
- Tesseract OSD and OpenCV line/projection heuristics only when using `--mode document` or `--mode auto`.

JPEG files are rotated losslessly with `jpegtran` when possible. Files that do not need rotation are copied without recompression when writing to an output directory.

## Install

First, install [uv](https://github.com/astral-sh/uv) if you haven't already. Then, run the following command to sync the project dependencies:

```bash
uv sync
```

Download a YOLO11 pose ONNX model:

```bash
mkdir -p models
curl -L https://huggingface.co/MikeLud/ObjectDetectionYOLO11-ONNX/resolve/main/yolo11m-pose.onnx -o models/yolo11m-pose.onnx
```

For higher accuracy and slower CPU runtime, use `yolo11l-pose.onnx` or `yolo11x-pose.onnx` from the same model repository and pass `--yolo-model`.

For best text/document results, install Tesseract too:

```bash
sudo apt install tesseract-ocr
```

## Usage

Write rotated files to a separate directory:

```bash
uv run autorotate ./images --recursive --output-dir ./rotated
```

Preview decisions without writing files:

```bash
uv run autorotate ./images --recursive --dry-run
```

The default photo thresholds are tuned for precision over recall:

```bash
uv run autorotate ./images --recursive --backend yolo --yolo-score 70 --yolo-margin 18 --dry-run
```

Higher values reduce wrong rotations but leave more images unchanged. Lower values rotate more images but risk more mistakes.

Overwrite the original files:

```bash
uv run autorotate ./images --recursive --in-place
```

For scans, receipts, or screenshots with text:

```bash
uv run autorotate ./images --recursive --mode document --output-dir ./rotated
```

Supported formats include JPEG, PNG, BMP, TIFF, and WebP.

### CLI Options

```
 Usage: autorotate [OPTIONS] INPUTS...

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    inputs      INPUTS...  Image files or directories to process.           │
│                             [required]                                       │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --output-dir            -o      PATH                   Directory for rotated │
│                                                        images.               │
│ --in-place                                             Overwrite input       │
│                                                        files.                │
│ --recursive             -r                             Recurse into input    │
│                                                        directories.          │
│ --overwrite                                            Overwrite files in    │
│                                                        --output-dir.         │
│ --dry-run                                              Print decisions       │
│                                                        without writing       │
│                                                        files.                │
│ --mode                          [photo|document|auto]  Detection strategy.   │
│                                                        photo is conservative │
│                                                        and optimized for     │
│                                                        people.               │
│                                                        [default: photo]      │
│ --backend                       [auto|yolo|opencv]     Photo-analysis        │
│                                                        backend. auto tries   │
│                                                        YOLO first, then      │
│                                                        OpenCV.               │
│                                                        [default: auto]       │
│ --yolo-model                    PATH                   Path to a YOLO pose   │
│                                                        model.                │
│                                                        [default:             │
│                                                        models/yolo11m-pose.… │
│ --yolo-confidence               FLOAT                  Minimum YOLO person   │
│                                                        detection confidence. │
│                                                        [default: 0.35]       │
│ --yolo-score                    FLOAT                  Minimum YOLO          │
│                                                        pose-orientation      │
│                                                        score.                │
│                                                        [default: 70.0]       │
│ --yolo-margin                   FLOAT                  Minimum YOLO winning  │
│                                                        score margin.         │
│                                                        [default: 18.0]       │
│ --human-score                   FLOAT                  Minimum human         │
│                                                        detector score.       │
│                                                        [default: 52.0]       │
│ --human-margin                  FLOAT                  Minimum winning score │
│                                                        margin.               │
│                                                        [default: 22.0]       │
│ --tesseract-confidence          FLOAT                  Minimum OSD           │
│                                                        confidence.           │
│                                                        [default: 2.0]        │
│ --line-delta                    FLOAT                  Minimum OpenCV        │
│                                                        line-score margin.    │
│                                                        [default: 500.0]      │
│ --quality                       INTEGER                JPEG output quality.  │
│                                                        [default: 100]        │
│ --install-completion                                   Install completion    │
│                                                        for the current       │
│                                                        shell.                │
│ --show-completion                                      Show completion for   │
│                                                        the current shell, to │
│                                                        copy it or customize  │
│                                                        the installation.     │
│ --help                                                 Show this message and │
│                                                        exit.                 │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Notes

No local visual model can perfectly infer orientation for every photo. If a person is too small, turned away, heavily occluded, or absent, the photo mode may leave the image unchanged instead of guessing.

There is no fixed accuracy percentage without a labeled test set. To measure it, run `--dry-run` on a folder where you know the correct rotation for each image, then compare the reported `rotate` value against the expected value.

## Attribution

<a href="https://unsplash.com/illustrations/a-black-circle-with-a-pink-cone-on-it-9r1ibje4J5A?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Illustration</a> by <a href="https://unsplash.com/@roundicons/illustrations?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Round Icons</a> on <a href="https://unsplash.com/illustrations?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
