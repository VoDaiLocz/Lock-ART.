"""Tests for image utilities."""

from pathlib import Path

import torch
from PIL import Image

from auralock.core.image import load_image, save_image


def test_load_image(tmp_path: Path):
    """Test loading a PNG image."""
    img = Image.new("RGB", (100, 100), color="red")
    img_path = tmp_path / "test.png"
    img.save(img_path)

    tensor = load_image(img_path)

    assert tensor.shape == (3, 100, 100)
    assert tensor.dtype == torch.float32


def test_save_image(tmp_path: Path):
    """Test saving as PNG."""
    tensor = torch.rand(3, 100, 100)
    out_path = tmp_path / "output.png"

    result = save_image(tensor, out_path)

    assert result.exists()


def test_roundtrip(tmp_path: Path):
    """Test load -> save -> load preserves content."""
    original = torch.rand(3, 64, 64)
    path = tmp_path / "test.png"

    save_image(original, path)
    loaded = load_image(path)

    assert torch.allclose(original, loaded, atol=0.01)
