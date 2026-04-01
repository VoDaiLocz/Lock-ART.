"""Style feature extraction and robust transform utilities."""

from __future__ import annotations

import io
from collections.abc import Callable, Iterable
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor

from auralock.core.pipeline import IMAGENET_MEAN, IMAGENET_STD, resolve_device

StyleTransform = Callable[[torch.Tensor], torch.Tensor]


def _normalize_stats(values: Iterable[float], name: str) -> tuple[float, float, float]:
    normalized = tuple(float(value) for value in values)
    if len(normalized) != 3:
        raise ValueError(f"{name} must contain exactly 3 channel values.")
    return normalized


class ResNetStyleFeatureExtractor(nn.Module):
    """Extract multi-layer ImageNet features for style-aware protection."""

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

        self.feature_extractor = create_feature_extractor(
            model,
            return_nodes={
                "relu": "stem",
                "layer1": "layer1",
                "layer2": "layer2",
                "layer3": "layer3",
                "layer4": "layer4",
            },
        )
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.to(self.device)
        self.eval()

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Resize and normalize images before feature extraction."""
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

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.feature_extractor(self.preprocess(images))
        bundle = {
            name: tensor
            for name, tensor in features.items()
            if name in {"stem", "layer1", "layer2", "layer3", "layer4"}
        }
        layer4 = features["layer4"]
        bundle["embedding"] = F.adaptive_avg_pool2d(layer4, output_size=1).flatten(1)
        return bundle


@lru_cache(maxsize=2)
def _load_default_style_feature_extractor_cached(
    device_name: str,
) -> ResNetStyleFeatureExtractor:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    return ResNetStyleFeatureExtractor(model=model, device=device_name)


def load_default_style_feature_extractor(
    device: str | torch.device | None = None,
) -> ResNetStyleFeatureExtractor:
    """Load the default style feature extractor once per device."""
    resolved = resolve_device(device)
    return _load_default_style_feature_extractor_cached(str(resolved))


def ensure_feature_bundle(
    feature_extractor: nn.Module,
    images: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Normalize feature extractor outputs into a common dictionary format."""
    outputs = feature_extractor(images)
    if isinstance(outputs, torch.Tensor):
        return {"embedding": outputs.flatten(1)}
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("feature_extractor must return a tensor or a non-empty dict.")

    bundle = {
        str(name): value
        for name, value in outputs.items()
        if isinstance(value, torch.Tensor)
    }
    if not bundle:
        raise ValueError("feature_extractor returned no tensor outputs.")

    if "embedding" not in bundle:
        last_name = next(reversed(bundle))
        last_tensor = bundle[last_name]
        if last_tensor.dim() == 4:
            bundle["embedding"] = F.adaptive_avg_pool2d(last_tensor, 1).flatten(1)
        else:
            bundle["embedding"] = last_tensor.flatten(1)

    return bundle


def gaussian_blur(
    images: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Apply differentiable Gaussian blur to a batch of images."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    coords = torch.arange(kernel_size, device=images.device, dtype=images.dtype)
    coords = coords - kernel_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(images.shape[1], 1, kernel_size, kernel_size)

    return F.conv2d(images, kernel, padding=kernel_size // 2, groups=images.shape[1])


def high_frequency_residual(
    images: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Return the high-frequency component of an image tensor."""
    return images - gaussian_blur(images, kernel_size=kernel_size, sigma=sigma)


def high_frequency_energy(
    images: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Measure how much high-frequency energy a tensor carries."""
    residual = high_frequency_residual(
        images,
        kernel_size=kernel_size,
        sigma=sigma,
    )
    return residual.pow(2).mean()


def resize_restore(images: torch.Tensor, scale: float = 0.75) -> torch.Tensor:
    """Downscale and restore an image batch to emulate common purification steps."""
    if not 0.0 < scale < 1.0:
        raise ValueError("scale must be between 0 and 1.")

    height, width = images.shape[-2:]
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))
    downscaled = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return F.interpolate(
        downscaled,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def jpeg_compress_decompress(
    images: torch.Tensor,
    quality: int = 85,
) -> torch.Tensor:
    """Apply JPEG compression and decompression to test robustness against compression artifacts."""
    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100.")

    batch_size = images.shape[0]
    device = images.device
    results = []

    for i in range(batch_size):
        # Convert tensor to PIL Image (0-255 range)
        img_np = (
            (images[i].detach().permute(1, 2, 0).cpu().numpy() * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )
        img_pil = Image.fromarray(img_np)

        # JPEG compress in memory
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        # Decompress
        img_pil = Image.open(buffer)
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        results.append(img_tensor)

    return torch.stack(results).to(device)


def center_crop_and_resize(
    images: torch.Tensor,
    crop_ratio: float = 0.9,
) -> torch.Tensor:
    """Center crop to crop_ratio and resize back to original size."""
    if not 0.0 < crop_ratio <= 1.0:
        raise ValueError("crop_ratio must be in (0, 1].")

    _, _, h, w = images.shape
    crop_h = max(1, int(h * crop_ratio))
    crop_w = max(1, int(w * crop_ratio))

    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2

    cropped = images[:, :, start_h : start_h + crop_h, start_w : start_w + crop_w]

    return F.interpolate(
        cropped,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def random_crop_and_resize(
    images: torch.Tensor,
    crop_ratio: float = 0.9,
) -> torch.Tensor:
    """Random crop to crop_ratio and resize back to original size."""
    if not 0.0 < crop_ratio <= 1.0:
        raise ValueError("crop_ratio must be in (0, 1].")

    _, _, h, w = images.shape
    crop_h = max(1, int(h * crop_ratio))
    crop_w = max(1, int(w * crop_ratio))

    # Random start position
    max_start_h = h - crop_h
    max_start_w = w - crop_w
    start_h = torch.randint(0, max_start_h + 1, (1,)).item() if max_start_h > 0 else 0
    start_w = torch.randint(0, max_start_w + 1, (1,)).item() if max_start_w > 0 else 0

    cropped = images[:, :, start_h : start_h + crop_h, start_w : start_w + crop_w]

    return F.interpolate(
        cropped,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def color_jitter(
    images: torch.Tensor,
    brightness: float = 0.1,
    contrast: float = 0.1,
    saturation: float = 0.1,
    hue: float = 0.05,
) -> torch.Tensor:
    """Apply color jitter transformations (brightness, contrast, saturation, hue)."""
    if any(x < 0 for x in [brightness, contrast, saturation, hue]):
        raise ValueError("All jitter parameters must be non-negative.")

    result = images.clone()

    # Apply brightness jitter
    if brightness > 0:
        brightness_factor = (
            1.0 + torch.empty(1).uniform_(-brightness, brightness).item()
        )
        result = (result * brightness_factor).clamp(0.0, 1.0)

    # Apply contrast jitter
    if contrast > 0:
        contrast_factor = 1.0 + torch.empty(1).uniform_(-contrast, contrast).item()
        mean = result.mean(dim=(2, 3), keepdim=True)
        result = ((result - mean) * contrast_factor + mean).clamp(0.0, 1.0)

    # Apply saturation jitter (convert to grayscale and interpolate)
    if saturation > 0:
        saturation_factor = (
            1.0 + torch.empty(1).uniform_(-saturation, saturation).item()
        )
        # Convert to grayscale using standard weights
        gray = (
            0.299 * result[:, 0:1, :, :]
            + 0.587 * result[:, 1:2, :, :]
            + 0.114 * result[:, 2:3, :, :]
        )
        gray = gray.expand_as(result)
        result = (saturation_factor * result + (1 - saturation_factor) * gray).clamp(
            0.0, 1.0
        )

    # Apply hue jitter (convert to HSV and back)
    if hue > 0:
        # Simple approximation: shift each channel differently
        hue_factor = torch.empty(1).uniform_(-hue, hue).item()
        # Rotate RGB channels
        if abs(hue_factor) > 0.01:
            # This is a simplified hue shift - not perfect but preserves differentiability
            shift = int(hue_factor * 255) % 3
            if shift != 0:
                result = torch.roll(result, shifts=shift, dims=1)

    return result


def add_gaussian_noise(
    images: torch.Tensor,
    std: float = 0.01,
) -> torch.Tensor:
    """Add Gaussian noise to images for robustness testing."""
    if std < 0:
        raise ValueError("std must be non-negative.")

    noise = torch.randn_like(images) * std
    return (images + noise).clamp(0.0, 1.0)


def build_style_transform_suite() -> tuple[tuple[str, StyleTransform], ...]:
    """Transforms used for both robust optimization and benchmark reporting.

    This suite now includes critical preprocessing transformations that real-world
    mimicry pipelines actually use, including JPEG compression, cropping, and
    color augmentations.
    """

    def identity(images: torch.Tensor) -> torch.Tensor:
        return images

    return (
        # Baseline
        ("identity", identity),
        # Existing transforms (kept for backward compatibility)
        (
            "gaussian_blur_mild",
            lambda images: gaussian_blur(images, kernel_size=5, sigma=1.0),
        ),
        ("resize_restore_75", lambda images: resize_restore(images, scale=0.75)),
        ("resize_restore_50", lambda images: resize_restore(images, scale=0.5)),
        # NEW: JPEG Compression (critical - most effective purification)
        (
            "jpeg_quality_95",
            lambda images: jpeg_compress_decompress(images, quality=95),
        ),
        (
            "jpeg_quality_85",
            lambda images: jpeg_compress_decompress(images, quality=85),
        ),
        (
            "jpeg_quality_75",
            lambda images: jpeg_compress_decompress(images, quality=75),
        ),
        # NEW: Cropping (standard data augmentation in training pipelines)
        (
            "center_crop_90",
            lambda images: center_crop_and_resize(images, crop_ratio=0.9),
        ),
        (
            "center_crop_80",
            lambda images: center_crop_and_resize(images, crop_ratio=0.8),
        ),
        # NEW: Stronger blur variants
        (
            "gaussian_blur_medium",
            lambda images: gaussian_blur(images, kernel_size=7, sigma=2.0),
        ),
        # NEW: Color augmentation (common in training)
        (
            "color_jitter_mild",
            lambda images: color_jitter(
                images, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
        ),
        # NEW: Noise injection (regularization during training)
        ("gaussian_noise_small", lambda images: add_gaussian_noise(images, std=0.01)),
    )


def feature_statistics(feature_map: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return mean, std, and Gram matrix for a feature map."""
    if feature_map.dim() == 2:
        feature_map = feature_map.unsqueeze(-1).unsqueeze(-1)
    if feature_map.dim() != 4:
        raise ValueError("feature_map must have shape (B, C, H, W) or (B, C).")

    flattened = feature_map.flatten(2)
    mean = flattened.mean(dim=-1)
    std = flattened.std(dim=-1, correction=0)
    gram = torch.bmm(flattened, flattened.transpose(1, 2)) / flattened.shape[-1]
    return mean, std, gram


def compute_style_distance(
    original_bundle: dict[str, torch.Tensor],
    modified_bundle: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Measure style drift between two feature bundles."""
    layer_names = sorted(
        name
        for name in original_bundle
        if name != "embedding" and name in modified_bundle
    )
    if not layer_names:
        return compute_embedding_distance(original_bundle, modified_bundle)

    total = torch.zeros((), device=original_bundle["embedding"].device)
    for name in layer_names:
        original_feature = original_bundle[name]
        modified_feature = modified_bundle[name]
        original_mean, original_std, original_gram = feature_statistics(
            original_feature
        )
        modified_mean, modified_std, modified_gram = feature_statistics(
            modified_feature
        )
        flattened_original = original_feature.flatten(1)
        flattened_modified = modified_feature.flatten(1)

        feature_distance = F.l1_loss(flattened_modified, flattened_original)
        cosine_distance = (
            1.0
            - F.cosine_similarity(
                flattened_original,
                flattened_modified,
                dim=1,
            ).mean()
        )
        stats_distance = F.l1_loss(modified_mean, original_mean) + F.l1_loss(
            modified_std,
            original_std,
        )
        gram_distance = F.l1_loss(modified_gram, original_gram)
        total = total + feature_distance + cosine_distance + stats_distance
        total = total + 0.25 * gram_distance

    return total / len(layer_names)


def compute_embedding_similarity(
    original_bundle: dict[str, torch.Tensor],
    modified_bundle: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Cosine similarity in [0, 1] between feature embeddings."""
    original = original_bundle["embedding"].flatten(1)
    modified = modified_bundle["embedding"].flatten(1)
    cosine = F.cosine_similarity(original, modified, dim=1).mean()
    return ((cosine + 1.0) / 2.0).clamp(0.0, 1.0)


def compute_embedding_distance(
    original_bundle: dict[str, torch.Tensor],
    modified_bundle: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Feature divergence in [0, 1] derived from embedding similarity."""
    return 1.0 - compute_embedding_similarity(original_bundle, modified_bundle)
