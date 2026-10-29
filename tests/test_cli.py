import tempfile
import unittest
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from autorotate.cli import app


class TestCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name).resolve()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_cli_help(self) -> None:
        result = self.runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Auto-rotate batches of images", result.output)

    def test_cli_requires_action_flag(self) -> None:
        img_path = self.base_path / "test.jpg"
        img = Image.new("RGB", (20, 20), color="red")
        img.save(img_path)

        result = self.runner.invoke(app, [str(img_path)])
        self.assertEqual(result.exit_code, 2)
        self.assertIn("error", result.output)

    def test_cli_dry_run(self) -> None:
        img_path = self.base_path / "sample.jpg"
        img = Image.new("RGB", (20, 20), color="green")
        img.save(img_path)

        result = self.runner.invoke(
            app, [str(img_path), "--dry-run", "--mode", "photo", "--backend", "opencv"]
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("dry-run", result.output)

    def test_cli_refuses_in_place_symlink(self) -> None:
        target_path = self.base_path / "target.jpg"
        img = Image.new("RGB", (20, 20), color="blue")
        img.save(target_path)

        symlink_path = self.base_path / "symlink.jpg"
        symlink_path.symlink_to(target_path)

        result = self.runner.invoke(
            app,
            [
                str(symlink_path),
                "--in-place",
                "--mode",
                "photo",
                "--backend",
                "opencv",
            ],
        )
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Refusing to modify symlink in-place", result.output)


if __name__ == "__main__":
    unittest.main()
