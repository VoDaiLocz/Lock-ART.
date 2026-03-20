"""
AuraLock protects artwork from AI style mimicry.

This package provides reusable protection, preprocessing, and analysis
utilities for building a stable image-cloaking workflow.
"""

__version__ = "0.1.0"
__author__ = "locfaker"

from auralock.core.image import load_image, save_image
from auralock.core.metrics import calculate_psnr, calculate_ssim
from auralock.services import ProtectionService

__all__ = [
    "load_image",
    "save_image",
    "calculate_psnr",
    "calculate_ssim",
    "ProtectionService",
    "__version__",
]
