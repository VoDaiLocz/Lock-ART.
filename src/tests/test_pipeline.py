"""Tests for preprocessing adapters and protection service."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class RecordingClassifier(nn.Module):
    """Small classifier that records the last tensor it received."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.last_input: torch.Tensor | None = None
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TinyStyleFeatureExtractor(nn.Module):
    """Small style extractor to keep pipeline tests fast and deterministic."""

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        layer1 = F.avg_pool2d(images, kernel_size=2)
        layer2 = F.avg_pool2d(layer1, kernel_size=2)
        layer4 = F.avg_pool2d(layer2, kernel_size=2)
        embedding = layer4.mean(dim=(2, 3))
        return {
            "layer1": layer1,
            "layer2": layer2,
            "layer4": layer4,
            "embedding": embedding,
        }


def test_imagenet_model_adapter_resizes_and_normalizes_inputs():
    """The wrapped model should always receive ImageNet-ready tensors."""
    from auralock.core.pipeline import ImageNetModelAdapter

    model = RecordingClassifier()
    adapter = ImageNetModelAdapter(
        model,
        input_size=(224, 224),
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
    )

    images = torch.full((2, 3, 64, 128), 0.5)
    logits = adapter(images)

    assert logits.shape == (2, 4)
    assert model.last_input is not None
    assert model.last_input.shape == (2, 3, 224, 224)
    assert torch.allclose(
        model.last_input, torch.zeros_like(model.last_input), atol=1e-6
    )


def test_protection_service_preserves_original_resolution():
    """Protection should keep the original output resolution for users."""
    from auralock.core.pipeline import ImageNetModelAdapter
    from auralock.services import ProtectionService

    model = ImageNetModelAdapter(RecordingClassifier())
    service = ProtectionService(model=model)
    image = Image.new("RGB", (321, 197), color="#4455aa")

    result = service.protect_image(image, epsilon=0.03, method="fgsm")

    assert result.original_size == image.size
    assert result.protected_image.size == image.size
    assert result.original_tensor.shape == (1, 3, 197, 321)
    assert result.protected_tensor.shape == (1, 3, 197, 321)
    assert "psnr_db" in result.quality_report
    assert "ssim" in result.quality_report


def test_protection_service_supports_blindfold_profile():
    """The service should expose the aggressive blindfold mode as a first-class option."""
    from auralock.core.pipeline import ImageNetModelAdapter
    from auralock.services import ProtectionService

    model = ImageNetModelAdapter(RecordingClassifier())
    service = ProtectionService(
        model=model,
        style_feature_extractor=TinyStyleFeatureExtractor(),
    )
    image = Image.new("RGB", (64, 48), color="#8a7344")

    result = service.protect_image(
        image,
        profile="blindfold",
        num_steps=1,
        alpha=0.01,
    )

    assert result.profile == "blindfold"
    assert result.method == "blindfold"
    assert result.protected_image.size == image.size
    assert "protection_score" in result.protection_report


def test_protection_service_reports_metrics_for_exported_image(tmp_path):
    """Protect results should reflect the quantized image users actually receive."""
    from auralock.core.image import save_image
    from auralock.core.pipeline import ImageNetModelAdapter
    from auralock.services import ProtectionService

    original_path = tmp_path / "original.png"
    protected_path = tmp_path / "protected.png"
    image = Image.effect_noise((96, 64), 64).convert("RGB")
    image.save(original_path)

    service = ProtectionService(model=ImageNetModelAdapter(RecordingClassifier()))
    result = service.protect_image(
        image,
        epsilon=0.02,
        method="stylecloak",
        num_steps=2,
        alpha=0.01,
    )
    save_image(result.protected_tensor, protected_path)
    analyzed = service.analyze_files(str(original_path), str(protected_path))

    assert result.quality_report["psnr_db"] == pytest.approx(
        analyzed["quality_report"]["psnr_db"],
        abs=1e-6,
    )
    assert result.quality_report["ssim"] == pytest.approx(
        analyzed["quality_report"]["ssim"],
        abs=1e-6,
    )


def test_ui_module_defers_gradio_import_until_runtime():
    """Importing the optional UI package should not require gradio immediately."""
    import importlib

    module = importlib.import_module("auralock.ui")

    assert hasattr(module, "create_ui")
    assert hasattr(module, "launch_app")


@pytest.mark.parametrize("epsilon", [0.0, -0.1, 1.1])
def test_attacks_reject_invalid_epsilon(epsilon: float):
    """Invalid epsilon values should be rejected early."""
    from auralock.attacks import FGSM

    with pytest.raises(ValueError, match="epsilon"):
        FGSM(RecordingClassifier(), epsilon=epsilon)


def test_pgd_rejects_invalid_step_configuration():
    """PGD should validate step count and step size."""
    from auralock.attacks import PGD

    with pytest.raises(ValueError, match="num_steps"):
        PGD(RecordingClassifier(), epsilon=0.03, num_steps=0)

    with pytest.raises(ValueError, match="alpha"):
        PGD(RecordingClassifier(), epsilon=0.03, alpha=0.0)
