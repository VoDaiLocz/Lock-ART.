"""Tests for security fixes: file extension validation and input handling."""

import pytest
import torch
from pathlib import Path
from PIL import Image

from auralock.core.image import (
    load_image,
    save_image,
    SUPPORTED_EXTENSIONS,
    _validate_image_extension,
)


class TestFileExtensionValidation:
    """Tests for file extension validation in image I/O."""

    def test_validate_supported_extensions(self):
        """All supported extensions should pass validation."""
        for ext in SUPPORTED_EXTENSIONS:
            _validate_image_extension(Path(f"image{ext}"))

    def test_validate_rejects_unsupported_extension(self):
        """Unsupported file extensions should raise ValueError."""
        for ext in [".exe", ".py", ".sh", ".txt", ".html", ".svg", ".pdf"]:
            with pytest.raises(ValueError, match="Unsupported image format"):
                _validate_image_extension(Path(f"file{ext}"))

    def test_validate_rejects_no_extension(self):
        """Files without extension should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported image format"):
            _validate_image_extension(Path("no_extension"))

    def test_load_image_rejects_unsupported(self, tmp_path: Path):
        """load_image should reject unsupported file extensions."""
        bad_file = tmp_path / "data.txt"
        bad_file.write_text("not an image")

        with pytest.raises(ValueError, match="Unsupported image format"):
            load_image(bad_file)

    def test_save_image_rejects_unsupported(self, tmp_path: Path):
        """save_image should reject unsupported file extensions."""
        tensor = torch.rand(3, 32, 32)
        bad_path = tmp_path / "output.txt"

        with pytest.raises(ValueError, match="Unsupported image format"):
            save_image(tensor, bad_path)

    def test_save_image_accepts_supported(self, tmp_path: Path):
        """save_image should accept supported file extensions."""
        tensor = torch.rand(3, 32, 32)
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            out_path = tmp_path / f"output{ext}"
            result = save_image(tensor, out_path)
            assert result.exists()
