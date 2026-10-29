import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from autorotate.detectors.yolo import yolo_person_score, yolo_pose_rows
from autorotate.io import (
    ensure_can_write,
    iter_images,
    output_path_for,
    reset_exif_orientation,
    save_image,
)
from autorotate.types import _find_default_yolo_model


class TestSecurity(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name).resolve()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_output_path_for_prevents_directory_escape(self) -> None:
        output_dir = self.base_path / "out"
        output_dir.mkdir(parents=True, exist_ok=True)

        root = self.base_path / "in"
        root.mkdir(parents=True, exist_ok=True)

        source = root / "image.jpg"
        source.touch()

        # Normal usage inside root
        dest = output_path_for(source, output_dir, [root])
        self.assertTrue(dest.resolve().is_relative_to(output_dir.resolve()))

    def test_output_path_for_rejects_path_traversal(self) -> None:
        output_dir = self.base_path / "out"
        output_dir.mkdir(parents=True, exist_ok=True)

        outside_dir = self.base_path / "outside"
        outside_dir.mkdir(parents=True, exist_ok=True)

        # Create a directory symlink in output_dir that targets an external directory
        symlink_sub = output_dir / "escaped_sub"
        symlink_sub.symlink_to(outside_dir)

        root = self.base_path / "in"
        root.mkdir(parents=True, exist_ok=True)
        source = root / "escaped_sub" / "escaped.jpg"

        with self.assertRaises(ValueError):
            output_path_for(source, output_dir, [root])

    def test_ensure_can_write_rejects_existing_symlink_without_overwrite(self) -> None:
        target_file = self.base_path / "target.txt"
        target_file.write_text("protected content")

        symlink_file = self.base_path / "link.jpg"
        symlink_file.symlink_to(target_file)

        # Without overwrite, symlink must raise FileExistsError
        with self.assertRaises(FileExistsError):
            ensure_can_write(symlink_file, overwrite=False)

        # Target content should remain unchanged
        self.assertEqual(target_file.read_text(), "protected content")

    def test_ensure_can_write_unlinks_symlink_with_overwrite(self) -> None:
        target_file = self.base_path / "target.txt"
        target_file.write_text("protected content")

        symlink_file = self.base_path / "link.jpg"
        symlink_file.symlink_to(target_file)

        # With overwrite, symlink should be unlinked before write
        ensure_can_write(symlink_file, overwrite=True)

        self.assertFalse(symlink_file.exists())
        self.assertFalse(symlink_file.is_symlink())
        self.assertEqual(target_file.read_text(), "protected content")

    def test_save_image_does_not_overwrite_symlink_target(self) -> None:
        target_file = self.base_path / "sensitive.txt"
        target_file.write_text("sensitive data")

        symlink_file = self.base_path / "image.png"
        symlink_file.symlink_to(target_file)

        img = Image.new("RGB", (10, 10), color="blue")
        save_image(img, symlink_file, overwrite=True, quality=90)

        # The symlink should be replaced by a regular file, target intact
        self.assertEqual(target_file.read_text(), "sensitive data")
        self.assertFalse(symlink_file.is_symlink())
        self.assertTrue(symlink_file.is_file())

    def test_iter_images_skips_symlinks_pointing_outside_root(self) -> None:
        scan_dir = self.base_path / "scandir"
        scan_dir.mkdir()

        outside_dir = self.base_path / "outside"
        outside_dir.mkdir()
        outside_image = outside_dir / "secret.jpg"
        outside_image.touch()

        inside_image = scan_dir / "normal.jpg"
        inside_image.touch()

        # Symlink pointing outside
        symlink = scan_dir / "external_link.jpg"
        symlink.symlink_to(outside_image)

        discovered = iter_images([scan_dir], recursive=True)
        discovered_names = [f.name for f in discovered]

        self.assertIn("normal.jpg", discovered_names)
        self.assertNotIn("external_link.jpg", discovered_names)

    def test_yolo_pose_rows_filters_nan_and_inf(self) -> None:
        # Construct row with NaN confidence
        bad_row = np.zeros((56,), dtype=np.float32)
        bad_row[4] = float("nan")

        rows = np.array([bad_row])
        detections = yolo_pose_rows(rows, confidence=0.35)
        self.assertEqual(len(detections), 0)

        # Construct row with Inf confidence
        inf_row = np.zeros((56,), dtype=np.float32)
        inf_row[4] = float("inf")
        rows_inf = np.array([inf_row])
        detections_inf = yolo_pose_rows(rows_inf, confidence=0.35)
        self.assertEqual(len(detections_inf), 0)

    def test_yolo_person_score_handles_nan_gracefully(self) -> None:
        keypoints = np.full((17, 3), float("nan"), dtype=np.float32)
        score = yolo_person_score(keypoints, image_height=640, box_confidence=0.5)
        self.assertTrue(math.isfinite(score))

    def test_default_yolo_model_env_override(self) -> None:
        custom_path = "/custom/model/path.onnx"
        old_env = os.environ.get("AUTOROTATE_YOLO_MODEL")
        try:
            os.environ["AUTOROTATE_YOLO_MODEL"] = custom_path
            resolved = _find_default_yolo_model()
            self.assertEqual(str(resolved), custom_path)
        finally:
            if old_env is None:
                os.environ.pop("AUTOROTATE_YOLO_MODEL", None)
            else:
                os.environ["AUTOROTATE_YOLO_MODEL"] = old_env

    def test_reset_exif_orientation_safe_with_dash_filename(self) -> None:
        # Verify that filenames starting with dash do not trigger option injection
        dash_file = self.base_path / "--config.jpg"
        dash_file.touch()
        # Should not raise any exception or fail
        reset_exif_orientation(dash_file)


if __name__ == "__main__":
    unittest.main()
