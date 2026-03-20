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
