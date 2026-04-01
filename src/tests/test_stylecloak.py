"""Tests for style-aware protection and readability metrics."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .test_pipeline import RecordingClassifier


class DummyStyleFeatureExtractor(nn.Module):
    """Small feature extractor used to test style-aware metrics without downloads."""

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        block1 = F.avg_pool2d(images, kernel_size=2)
        block2 = F.avg_pool2d(block1, kernel_size=2)
        embedding = block2.mean(dim=(2, 3))
        return {
            "layer1": block1,
            "layer2": block2,
            "embedding": embedding,
        }


def test_stylecloak_generates_bounded_output():
    """StyleCloak should preserve image bounds and the epsilon budget."""
    from auralock.attacks import StyleCloak

    images = torch.rand(1, 3, 64, 64)
    attack = StyleCloak(
        DummyStyleFeatureExtractor(),
        epsilon=0.03,
        alpha=0.01,
        num_steps=3,
    )

    adversarial = attack.generate(images)
    perturbation = adversarial - images

    assert adversarial.shape == images.shape
    assert torch.max(torch.abs(perturbation)).item() <= 0.03 + 1e-6
    assert adversarial.min().item() >= 0.0
    assert adversarial.max().item() <= 1.0


def test_high_frequency_energy_prefers_smoother_patterns():
    """Smooth perturbations should carry less high-frequency energy than noisy ones."""
    from auralock.core.style import gaussian_blur, high_frequency_energy

    torch.manual_seed(7)
    noisy = torch.rand(1, 3, 64, 64)
    smooth = gaussian_blur(noisy, kernel_size=9, sigma=2.0)

    assert high_frequency_energy(smooth) < high_frequency_energy(noisy)


def test_resnet_style_feature_extractor_exposes_layer4_features():
    """The default style extractor should retain deep semantic feature maps."""
    from torchvision.models import resnet18

    from auralock.core.style import ResNetStyleFeatureExtractor

    extractor = ResNetStyleFeatureExtractor(resnet18(weights=None), device="cpu")
    bundle = extractor(torch.rand(1, 3, 64, 64))

    assert "layer4" in bundle
    assert "embedding" in bundle


def test_protection_readability_report_rewards_larger_feature_shift():
    """Stronger feature drift should score as stronger protection."""
    from auralock.core.metrics import get_protection_readability_report

    extractor = DummyStyleFeatureExtractor()
    original = torch.rand(1, 3, 64, 64)
    small_shift = torch.clamp(original + torch.randn_like(original) * 0.002, 0.0, 1.0)
    large_shift = torch.clamp(
        original + torch.sign(torch.randn_like(original)) * 0.03,
        0.0,
        1.0,
    )

    small_report = get_protection_readability_report(
        original,
        small_shift,
        feature_extractor=extractor,
    )
    large_report = get_protection_readability_report(
        original,
        large_shift,
        feature_extractor=extractor,
    )

    assert (
        large_report["robust_style_similarity"]
        < small_report["robust_style_similarity"]
    )
    assert large_report["protection_score"] > small_report["protection_score"]


def test_protection_readability_report_does_not_collapse_large_shift_to_perfect_style_similarity():
    """Meaningful image shifts should not still read as nearly identical style."""
    from auralock.core.metrics import get_protection_readability_report

    extractor = DummyStyleFeatureExtractor()
    torch.manual_seed(0)
    original = torch.rand(1, 3, 64, 64)
    large_shift = torch.clamp(
        original + torch.sign(torch.randn_like(original)) * 0.03,
        0.0,
        1.0,
    )

    report = get_protection_readability_report(
        original,
        large_shift,
        feature_extractor=extractor,
    )

    assert report["robust_style_similarity"] < 0.995


def test_protection_service_stylecloak_returns_protection_report():
    """The protection service should expose style-aware metrics for stylecloak."""
    from auralock.core.pipeline import ImageNetModelAdapter
    from auralock.services import ProtectionService

    service = ProtectionService(
        model=ImageNetModelAdapter(RecordingClassifier()),
        style_feature_extractor=DummyStyleFeatureExtractor(),
    )
    image = Image.new("RGB", (192, 128), color="#557799")

    result = service.protect_image(
        image,
        epsilon=0.02,
        method="stylecloak",
        num_steps=2,
        alpha=0.01,
    )

    assert result.method == "stylecloak"
    assert result.protected_image.size == image.size
    assert "embedding_similarity" in result.protection_report
    assert "robust_style_similarity" in result.protection_report
    assert "protection_score" in result.protection_report
    assert result.original_prediction is None
    assert result.adversarial_prediction is None
    assert result.attack_success is None


def test_jpeg_compress_decompress_preserves_shape_and_bounds():
    """JPEG compression should preserve image shape and valid pixel range."""
    from auralock.core.style import jpeg_compress_decompress

    images = torch.rand(2, 3, 64, 64)

    for quality in [95, 85, 75, 50]:
        compressed = jpeg_compress_decompress(images, quality=quality)
        assert compressed.shape == images.shape
        assert compressed.min().item() >= 0.0
        assert compressed.max().item() <= 1.0


def test_jpeg_compress_decompress_introduces_artifacts():
    """JPEG compression at lower quality should introduce noticeable differences."""
    from auralock.core.style import jpeg_compress_decompress

    torch.manual_seed(42)
    images = torch.rand(1, 3, 64, 64)

    compressed_high = jpeg_compress_decompress(images, quality=95)
    compressed_low = jpeg_compress_decompress(images, quality=50)

    # Lower quality should introduce more artifacts
    diff_high = (images - compressed_high).abs().mean()
    diff_low = (images - compressed_low).abs().mean()

    assert diff_low > diff_high


def test_center_crop_and_resize_preserves_shape():
    """Center crop should restore original dimensions after cropping."""
    from auralock.core.style import center_crop_and_resize

    images = torch.rand(2, 3, 64, 64)

    for crop_ratio in [0.9, 0.8, 0.5]:
        cropped = center_crop_and_resize(images, crop_ratio=crop_ratio)
        assert cropped.shape == images.shape
        assert cropped.min().item() >= 0.0
        assert cropped.max().item() <= 1.0


def test_center_crop_removes_border_information():
    """Center crop should remove border pixels and resize back."""
    from auralock.core.style import center_crop_and_resize

    # Create image with distinct border
    images = torch.zeros(1, 3, 64, 64)
    images[:, :, 10:54, 10:54] = 1.0  # White center, black border

    cropped = center_crop_and_resize(images, crop_ratio=0.8)

    # After center crop at 0.8, the border should be mostly removed
    # The reconstructed image should have more white than the original
    assert cropped.mean() > images.mean()


def test_random_crop_and_resize_preserves_shape():
    """Random crop should restore original dimensions after cropping."""
    from auralock.core.style import random_crop_and_resize

    images = torch.rand(2, 3, 64, 64)

    for crop_ratio in [0.9, 0.8]:
        cropped = random_crop_and_resize(images, crop_ratio=crop_ratio)
        assert cropped.shape == images.shape
        assert cropped.min().item() >= 0.0
        assert cropped.max().item() <= 1.0


def test_color_jitter_preserves_shape_and_bounds():
    """Color jitter should preserve image shape and valid pixel range."""
    from auralock.core.style import color_jitter

    torch.manual_seed(42)
    images = torch.rand(2, 3, 64, 64)

    jittered = color_jitter(
        images, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    )
    assert jittered.shape == images.shape
    assert jittered.min().item() >= 0.0
    assert jittered.max().item() <= 1.0


def test_color_jitter_modifies_image():
    """Color jitter should produce different output than input."""
    from auralock.core.style import color_jitter

    torch.manual_seed(42)
    images = torch.rand(1, 3, 64, 64)

    jittered = color_jitter(images, brightness=0.2, contrast=0.2)

    # Should modify the image
    diff = (images - jittered).abs().mean()
    assert diff > 0.001


def test_add_gaussian_noise_preserves_shape_and_bounds():
    """Gaussian noise injection should preserve image shape and valid pixel range."""
    from auralock.core.style import add_gaussian_noise

    torch.manual_seed(42)
    images = torch.rand(2, 3, 64, 64)

    for std in [0.01, 0.03, 0.05]:
        noisy = add_gaussian_noise(images, std=std)
        assert noisy.shape == images.shape
        assert noisy.min().item() >= 0.0
        assert noisy.max().item() <= 1.0


def test_add_gaussian_noise_increases_variance():
    """Gaussian noise should increase image variance."""
    from auralock.core.style import add_gaussian_noise

    torch.manual_seed(42)
    images = torch.ones(1, 3, 64, 64) * 0.5  # Constant image

    noisy = add_gaussian_noise(images, std=0.03)

    # Noisy image should have higher variance
    assert noisy.var() > images.var()


def test_build_style_transform_suite_includes_new_transforms():
    """Transform suite should include critical preprocessing transforms."""
    from auralock.core.style import build_style_transform_suite

    suite = build_style_transform_suite()
    transform_names = [name for name, _ in suite]

    # Check for new critical transforms
    assert "jpeg_quality_95" in transform_names
    assert "jpeg_quality_85" in transform_names
    assert "jpeg_quality_75" in transform_names
    assert "center_crop_90" in transform_names
    assert "center_crop_80" in transform_names
    assert "gaussian_blur_medium" in transform_names
    assert "color_jitter_mild" in transform_names
    assert "gaussian_noise_small" in transform_names

    # Check backward compatibility - old names should still exist
    assert "gaussian_blur_mild" in transform_names
    assert "resize_restore_75" in transform_names
    assert "resize_restore_50" in transform_names


def test_all_transforms_in_suite_are_callable():
    """All transforms in the suite should be callable and process images correctly."""
    from auralock.core.style import build_style_transform_suite

    suite = build_style_transform_suite()
    images = torch.rand(1, 3, 64, 64)

    for name, transform in suite:
        result = transform(images)
        assert result.shape == images.shape, f"Transform {name} changed shape"
        assert result.min().item() >= 0.0, f"Transform {name} produced negative values"
        assert result.max().item() <= 1.0, f"Transform {name} produced values > 1.0"
