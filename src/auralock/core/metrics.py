"""
Image quality metrics for measuring perturbation visibility.

This module provides functions to measure how visible the perturbations are:
- PSNR (Peak Signal-to-Noise Ratio): Higher = less visible noise
- SSIM (Structural Similarity Index): Higher = more similar structure
- LPIPS (Learned Perceptual Image Patch Similarity): Lower = more similar

Good protection should have:
- PSNR > 35 dB (imperceptible noise)
- SSIM > 0.95 (nearly identical structure)
- LPIPS < 0.1 (perceptually similar)
"""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from auralock.core.style import (
    build_style_transform_suite,
    compute_embedding_similarity,
    compute_style_distance,
    ensure_feature_bundle,
    load_default_style_feature_extractor,
)

# Protection Score Warning Constants
PROTECTION_SCORE_WARNING = (
    "⚠️  Protection Score is a proxy metric. "
    "Real-world effectiveness is NOT validated."
)
PROTECTION_SCORE_METRIC_TYPE = "proxy_unvalidated"
PROTECTION_SCORE_DISCLAIMER = (
    "This score measures drift in ResNet18 feature space, "
    "NOT actual mimicry prevention against real-world attacks like DreamBooth or LoRA. "
    "Use for relative comparisons within this repository only."
)


def calculate_psnr(
    original: torch.Tensor | np.ndarray,
    modified: torch.Tensor | np.ndarray,
    data_range: float = 1.0,
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.

    Higher PSNR means the perturbation is less visible.
    - PSNR > 40 dB: Excellent (virtually invisible)
    - PSNR 35-40 dB: Good (barely noticeable)
    - PSNR 30-35 dB: Acceptable (slight artifacts)
    - PSNR < 30 dB: Poor (visible distortion)

    Args:
        original: Original image tensor/array
        modified: Modified (perturbed) image tensor/array
        data_range: The data range of the images (1.0 for normalized, 255 for uint8)

    Returns:
        PSNR value in decibels (dB)

    Example:
        >>> psnr = calculate_psnr(original_img, protected_img)
        >>> print(f"PSNR: {psnr:.2f} dB")
    """
    # Convert tensors to numpy
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(modified, torch.Tensor):
        modified = modified.detach().cpu().numpy()

    # Handle batch dimension
    if original.ndim == 4:
        original = original[0]
    if modified.ndim == 4:
        modified = modified[0]

    # Rearrange from (C, H, W) to (H, W, C) if needed
    # Detect channel-first: first dim is a channel count (1, 3, or 4) and smaller than last dim
    if (
        original.ndim == 3
        and original.shape[0] in (1, 3, 4)
        and original.shape[0] < original.shape[2]
    ):
        original = np.transpose(original, (1, 2, 0))
        modified = np.transpose(modified, (1, 2, 0))

    return peak_signal_noise_ratio(original, modified, data_range=data_range)


def calculate_ssim(
    original: torch.Tensor | np.ndarray,
    modified: torch.Tensor | np.ndarray,
    data_range: float = 1.0,
) -> float:
    """
    Calculate Structural Similarity Index between two images.

    SSIM measures structural similarity rather than pixel-by-pixel difference.
    - SSIM > 0.98: Excellent (virtually identical)
    - SSIM 0.95-0.98: Good (minor differences)
    - SSIM 0.90-0.95: Acceptable (noticeable but not distracting)
    - SSIM < 0.90: Poor (significant structural changes)

    Args:
        original: Original image tensor/array
        modified: Modified (perturbed) image tensor/array
        data_range: The data range of the images

    Returns:
        SSIM value between 0 and 1

    Example:
        >>> ssim = calculate_ssim(original_img, protected_img)
        >>> print(f"SSIM: {ssim:.4f}")
    """
    # Convert tensors to numpy
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(modified, torch.Tensor):
        modified = modified.detach().cpu().numpy()

    # Handle batch dimension
    if original.ndim == 4:
        original = original[0]
    if modified.ndim == 4:
        modified = modified[0]

    # Rearrange from (C, H, W) to (H, W, C) if needed
    # Detect channel-first: first dim is a channel count (1, 3, or 4) and smaller than last dim
    if (
        original.ndim == 3
        and original.shape[0] in (1, 3, 4)
        and original.shape[0] < original.shape[2]
    ):
        original = np.transpose(original, (1, 2, 0))
        modified = np.transpose(modified, (1, 2, 0))

    return structural_similarity(
        original,
        modified,
        data_range=data_range,
        channel_axis=2,  # Color channel is last dimension
    )


def calculate_lpips(
    original: torch.Tensor,
    modified: torch.Tensor,
    net: str = "alex",
) -> float:
    """
    Calculate Learned Perceptual Image Patch Similarity.

    LPIPS uses deep features to measure perceptual similarity.
    Lower values mean more similar images.
    - LPIPS < 0.05: Excellent (perceptually identical)
    - LPIPS 0.05-0.1: Good (barely perceptible difference)
    - LPIPS 0.1-0.2: Acceptable (noticeable but minor)
    - LPIPS > 0.2: Poor (clearly different)

    Note: Requires 'lpips' package. Install with: pip install lpips

    Args:
        original: Original image tensor (C, H, W) or (B, C, H, W)
        modified: Modified image tensor
        net: Network to use ('alex', 'vgg', 'squeeze')

    Returns:
        LPIPS distance (lower = more similar)

    Example:
        >>> lpips_dist = calculate_lpips(original_img, protected_img)
        >>> print(f"LPIPS: {lpips_dist:.4f}")
    """
    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            "LPIPS requires the 'lpips' package. " "Install it with: pip install lpips"
        ) from exc

    # Initialize LPIPS model (cached per network type)
    if not hasattr(calculate_lpips, "_models"):
        calculate_lpips._models = {}

    if net not in calculate_lpips._models:
        calculate_lpips._models[net] = lpips.LPIPS(net=net)

    model = calculate_lpips._models[net]

    # Add batch dimension if needed
    if original.dim() == 3:
        original = original.unsqueeze(0)
    if modified.dim() == 3:
        modified = modified.unsqueeze(0)

    # LPIPS expects values in [-1, 1] range
    # Our images are in [0, 1], so we need to normalize
    original_normalized = original * 2 - 1
    modified_normalized = modified * 2 - 1

    with torch.no_grad():
        distance = model(original_normalized, modified_normalized)

    return distance.item()


def get_quality_report(
    original: torch.Tensor,
    modified: torch.Tensor,
) -> dict:
    """
    Generate a comprehensive quality report comparing two images.

    Args:
        original: Original image tensor
        modified: Modified image tensor

    Returns:
        Dictionary with all metrics and quality assessment

    Example:
        >>> report = get_quality_report(original, protected)
        >>> print(report['overall_quality'])
    """
    psnr = calculate_psnr(original, modified)
    ssim = calculate_ssim(original, modified)

    # Calculate L2 distance (perturbation magnitude)
    l2_distance = torch.norm(original - modified).item()
    linf_distance = torch.max(torch.abs(original - modified)).item()

    # Determine quality level
    if psnr > 40 and ssim > 0.98:
        quality = "Excellent"
    elif psnr > 35 and ssim > 0.95:
        quality = "Good"
    elif psnr > 30 and ssim > 0.90:
        quality = "Acceptable"
    else:
        quality = "Poor"

    return {
        "psnr_db": psnr,
        "ssim": ssim,
        "l2_distance": l2_distance,
        "linf_distance": linf_distance,
        "overall_quality": quality,
        "thresholds": {
            "psnr_excellent": 40,
            "psnr_good": 35,
            "ssim_excellent": 0.98,
            "ssim_good": 0.95,
        },
    }


def get_protection_readability_report(
    original: torch.Tensor,
    modified: torch.Tensor,
    *,
    feature_extractor: torch.nn.Module | None = None,
) -> dict[str, object]:
    """
    Measure how readable the protected image remains to style/embedding extractors.

    Lower similarity values indicate stronger protection.
    """
    if original.dim() == 3:
        original = original.unsqueeze(0)
    if modified.dim() == 3:
        modified = modified.unsqueeze(0)
    if original.dim() != 4 or modified.dim() != 4:
        raise ValueError("original and modified must be image tensors.")
    if original.shape != modified.shape:
        raise ValueError("original and modified must have the same shape.")

    original = original.detach().to(torch.float32)
    modified = modified.detach().to(torch.float32)

    extractor = feature_extractor or load_default_style_feature_extractor()
    if isinstance(extractor, torch.nn.Module):
        target_device = next(extractor.parameters(), None)
        target_device = (
            target_device.device if target_device is not None else original.device
        )
        extractor = extractor.to(target_device)
        extractor.eval()
        original = original.to(target_device)
        modified = modified.to(target_device)

    transform_style_similarities: dict[str, float] = {}
    transform_embedding_similarities: dict[str, float] = {}
    robust_style_values: list[float] = []
    robust_embedding_values: list[float] = []

    with torch.no_grad():
        for name, transform in build_style_transform_suite():
            original_bundle = ensure_feature_bundle(extractor, transform(original))
            modified_bundle = ensure_feature_bundle(extractor, transform(modified))

            style_distance = compute_style_distance(original_bundle, modified_bundle)
            style_similarity = float((1.0 / (1.0 + style_distance)).item())
            embedding_similarity = float(
                compute_embedding_similarity(original_bundle, modified_bundle).item()
            )

            transform_style_similarities[name] = style_similarity
            transform_embedding_similarities[name] = embedding_similarity
            robust_style_values.append(style_similarity)
            robust_embedding_values.append(embedding_similarity)

    style_similarity = transform_style_similarities["identity"]
    embedding_similarity = transform_embedding_similarities["identity"]
    robust_style_similarity = sum(robust_style_values) / len(robust_style_values)
    robust_embedding_similarity = sum(robust_embedding_values) / len(
        robust_embedding_values
    )
    protection_score = 100.0 * (
        0.65 * (1.0 - robust_style_similarity)
        + 0.35 * (1.0 - robust_embedding_similarity)
    )

    if protection_score >= 45:
        assessment = "Strong (proxy space)"
    elif protection_score >= 25:
        assessment = "Moderate (proxy space)"
    else:
        assessment = "Weak (proxy space)"

    return {
        "style_similarity": style_similarity,
        "embedding_similarity": embedding_similarity,
        "robust_style_similarity": robust_style_similarity,
        "robust_embedding_similarity": robust_embedding_similarity,
        "protection_score": protection_score,
        "assessment": assessment,
        "metric_type": PROTECTION_SCORE_METRIC_TYPE,
        "warning": PROTECTION_SCORE_WARNING,
        "disclaimer": PROTECTION_SCORE_DISCLAIMER,
        "transform_style_similarities": transform_style_similarities,
        "transform_embedding_similarities": transform_embedding_similarities,
    }


def print_quality_report(report: dict) -> None:
    """Pretty print a quality report."""
    print("\n" + "=" * 50)
    print("📊 Image Quality Report")
    print("=" * 50)
    print(f"PSNR:          {report['psnr_db']:.2f} dB")
    print(f"SSIM:          {report['ssim']:.4f}")
    print(f"L2 Distance:   {report['l2_distance']:.4f}")
    print(f"L∞ Distance:   {report['linf_distance']:.4f}")
    print("-" * 50)
    print(f"Overall:       {report['overall_quality']}")
    print("=" * 50 + "\n")
