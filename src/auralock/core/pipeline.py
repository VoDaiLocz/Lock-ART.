"""Differentiable preprocessing and model loading utilities."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve the requested device with a sensible default."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but no CUDA device is available.")
    return resolved


def _normalize_stats(values: Iterable[float], name: str) -> tuple[float, float, float]:
    normalized = tuple(float(value) for value in values)
    if len(normalized) != 3:
        raise ValueError(f"{name} must contain exactly 3 channel values.")
    return normalized


class ImageNetModelAdapter(nn.Module):
    """Wrap a classifier with differentiable resize and normalization."""

    def __init__(
        self,
        model: nn.Module,
        input_size: tuple[int, int] = (224, 224),
        mean: Iterable[float] = IMAGENET_MEAN,
        std: Iterable[float] = IMAGENET_STD,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if model is None:
            raise ValueError("model must not be None.")

        self.model = model
        self.input_size = tuple(int(dim) for dim in input_size)
        if len(self.input_size) != 2 or min(self.input_size) <= 0:
            raise ValueError("input_size must contain two positive integers.")

        self.device = resolve_device(device)
        mean_values = _normalize_stats(mean, "mean")
        std_values = _normalize_stats(std, "std")

        if any(value <= 0 for value in std_values):
            raise ValueError("std values must be strictly positive.")

        self.register_buffer(
            "mean",
            torch.tensor(mean_values, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor(std_values, dtype=torch.float32).view(1, 3, 1, 1),
        )

        self.model = self.model.to(self.device)
        self.to(self.device)
        self.eval()

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Resize and normalize user images for ImageNet classifiers."""
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.dim() != 4:
            raise ValueError("images must have shape (C, H, W) or (B, C, H, W).")

        images = images.to(self.device, dtype=torch.float32)
        resized = F.interpolate(
            images,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return (resized - self.mean) / self.std

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(self.preprocess(images))

    def get_info(self) -> dict[str, object]:
        return {
            "adapter": self.__class__.__name__,
            "model": self.model.__class__.__name__,
            "device": str(self.device),
            "input_size": self.input_size,
        }


@lru_cache(maxsize=2)
def _load_default_model_cached(device_name: str) -> ImageNetModelAdapter:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    return ImageNetModelAdapter(model=model, device=device_name)


def load_default_model(
    device: str | torch.device | None = None,
) -> ImageNetModelAdapter:
    """Load the default production classifier once per device."""
    resolved = resolve_device(device)
    return _load_default_model_cached(str(resolved))
