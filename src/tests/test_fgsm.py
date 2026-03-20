"""Tests for FGSM attack."""

import pytest
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TestFGSM:
    """Tests for FGSM attack."""

    def test_fgsm_init(self):
        """Test FGSM initialization."""
        from auralock.attacks import FGSM

        model = SimpleModel()
        attack = FGSM(model, epsilon=0.03)

        assert attack.epsilon == 0.03
        assert attack.model is not None

    def test_fgsm_generate_output_shape(self):
        """Test that FGSM generates output with correct shape."""
        from auralock.attacks import FGSM

        model = SimpleModel()
        attack = FGSM(model, epsilon=0.03)

        images = torch.rand(2, 3, 32, 32)
        labels = torch.tensor([0, 1])

        adversarial = attack.generate(images, labels)

        assert adversarial.shape == images.shape

    def test_fgsm_perturbation_bounded(self):
        """Test that perturbation is bounded by epsilon."""
        from auralock.attacks import FGSM

        model = SimpleModel()
        epsilon = 0.03
        attack = FGSM(model, epsilon=epsilon)

        images = torch.rand(2, 3, 32, 32)
        labels = torch.tensor([0, 1])

        adversarial = attack.generate(images, labels)
        perturbation = adversarial - images

        assert perturbation.max() <= epsilon + 1e-6
        assert perturbation.min() >= -epsilon - 1e-6

    def test_fgsm_output_clamped(self):
        """Test that output values are in [0, 1]."""
        from auralock.attacks import FGSM

        model = SimpleModel()
        attack = FGSM(model, epsilon=0.1)

        images = torch.rand(2, 3, 32, 32)
        labels = torch.tensor([0, 1])

        adversarial = attack.generate(images, labels)

        assert adversarial.min() >= 0.0
        assert adversarial.max() <= 1.0

    def test_fgsm_callable(self):
        """Test that FGSM can be called directly."""
        from auralock.attacks import FGSM

        model = SimpleModel()
        attack = FGSM(model, epsilon=0.03)

        images = torch.rand(1, 3, 32, 32)

        # Should work without labels
        adversarial = attack(images)

        assert adversarial.shape == images.shape

    def test_fgsm_generate_with_info(self):
        """Test generate_with_info returns expected keys."""
        from auralock.attacks import FGSM

        model = SimpleModel()
        attack = FGSM(model, epsilon=0.03)

        images = torch.rand(1, 3, 32, 32)
        labels = torch.tensor([0])

        result = attack.generate_with_info(images, labels)

        assert "adversarial" in result
        assert "perturbation" in result
        assert "success_rate" in result
        assert "original_preds" in result
        assert "adversarial_preds" in result

    def test_fgsm_targeted_attack(self):
        """Test targeted attack mode."""
        from auralock.attacks import FGSM

        model = SimpleModel()
        attack = FGSM(model, epsilon=0.1)

        images = torch.rand(1, 3, 32, 32)
        target_labels = torch.tensor([5])

        # Targeted attack should not raise error
        adversarial = attack.generate(images, target_labels, targeted=True)

        assert adversarial.shape == images.shape


class TestBaseAttack:
    """Tests for BaseAttack class."""

    def test_base_attack_is_abstract(self):
        """Test that BaseAttack cannot be instantiated directly."""
        from auralock.attacks.base import BaseAttack

        model = SimpleModel()

        with pytest.raises(TypeError):
            BaseAttack(model)

    def test_clamp_method(self):
        """Test _clamp helper method."""
        from auralock.attacks import FGSM

        model = SimpleModel()
        attack = FGSM(model)

        tensor = torch.tensor([-0.5, 0.5, 1.5])
        clamped = attack._clamp(tensor)

        assert clamped.min() >= 0.0
        assert clamped.max() <= 1.0

    def test_project_method(self):
        """Test _project helper method."""
        from auralock.attacks import FGSM

        model = SimpleModel()
        attack = FGSM(model, epsilon=0.1)

        perturbation = torch.tensor([-0.5, 0.05, 0.5])
        projected = attack._project(perturbation)

        assert projected.max() <= 0.1
        assert projected.min() >= -0.1

    def test_get_info(self):
        """Test get_info method."""
        from auralock.attacks import FGSM

        model = SimpleModel()
        attack = FGSM(model, epsilon=0.03)

        info = attack.get_info()

        assert info["name"] == "FGSM"
        assert info["epsilon"] == 0.03
        assert "device" in info
