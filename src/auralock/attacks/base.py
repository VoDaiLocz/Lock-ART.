"""
Base class for all adversarial attacks.

All attack implementations should inherit from BaseAttack
and implement the `generate` method.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseAttack(ABC):
    """
    Abstract base class for adversarial attacks.

    An adversarial attack generates perturbations that:
    1. Are imperceptible to humans (small magnitude)
    2. Cause AI models to misclassify or misinterpret the image

    Attributes:
        model: The target model to attack
        device: Device to run computations on (CPU/CUDA)
        epsilon: Maximum perturbation magnitude
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        device: str | torch.device | None = None,
    ):
        """
        Initialize the attack.

        Args:
            model: Target neural network model
            epsilon: Maximum L∞ norm of perturbation (default: 0.03 ≈ 8/255)
                     Higher values = stronger attack but more visible
            device: Device for computation. If None, auto-detect.
        """
        if model is None:
            raise ValueError("model must not be None.")
        if not 0.0 < epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1.")

        self.model = model
        self.epsilon = float(epsilon)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

    @abstractmethod
    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate adversarial perturbations for the given images.

        Args:
            images: Input images tensor of shape (B, C, H, W)
            labels: Ground truth labels (required for some attacks)
            **kwargs: Attack-specific parameters

        Returns:
            Adversarial images of the same shape as input
        """
        pass

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Allow calling attack instance directly."""
        return self.generate(images, labels, **kwargs)

    def _clamp(self, tensor: torch.Tensor) -> torch.Tensor:
        """Clamp tensor values to valid image range [0, 1]."""
        return torch.clamp(tensor, 0.0, 1.0)

    def _project(
        self,
        perturbation: torch.Tensor,
        epsilon: float | None = None,
    ) -> torch.Tensor:
        """
        Project perturbation to L∞ ball of radius epsilon.

        This ensures the perturbation magnitude doesn't exceed epsilon.
        """
        eps = epsilon if epsilon is not None else self.epsilon
        return torch.clamp(perturbation, -eps, eps)

    def get_perturbation(
        self,
        original: torch.Tensor,
        adversarial: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the perturbation from adversarial example."""
        return adversarial - original

    def get_info(self) -> dict:
        """Return information about the attack configuration."""
        return {
            "name": self.__class__.__name__,
            "epsilon": self.epsilon,
            "device": str(self.device),
            "model": self.model.__class__.__name__,
        }
