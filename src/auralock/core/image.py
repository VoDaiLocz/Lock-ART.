"""
Image loading, saving, and conversion utilities.

This module provides functions to:
- Load images from disk into PyTorch tensors
- Save tensors back to image files
- Convert between PIL Images, NumPy arrays, and PyTorch tensors
"""

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

# Supported image file extensions
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def _validate_image_extension(path: Path) -> None:
    """Validate that the file has a supported image extension."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: '{suffix}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def load_image(
    path: Union[str, Path],
    size: tuple[int, int] | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Load an image from disk and convert to PyTorch tensor.
    
    Args:
        path: Path to the image file (supports PNG, JPG, WEBP)
        size: Optional (width, height) to resize the image
        normalize: If True, normalize pixel values to [0, 1] range
        
    Returns:
        Tensor of shape (C, H, W) where C=3 for RGB images
        
    Example:
        >>> img = load_image("artwork.png", size=(512, 512))
        >>> print(img.shape)  # torch.Size([3, 512, 512])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    _validate_image_extension(path)
    
    # Load image and convert to RGB (handles RGBA, grayscale, etc.)
    image = Image.open(path).convert("RGB")
    
    # Resize if specified
    if size is not None:
        image = image.resize(size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array: (H, W, C)
    array = np.array(image, dtype=np.float32)
    
    # Normalize to [0, 1] if requested
    if normalize:
        array = array / 255.0
    
    # Convert to tensor and rearrange to (C, H, W)
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    
    return tensor


def save_image(
    tensor: torch.Tensor,
    path: Union[str, Path],
    quality: int = 95,
) -> Path:
    """
    Save a PyTorch tensor as an image file.
    
    Args:
        tensor: Image tensor of shape (C, H, W) or (B, C, H, W)
                Values should be in [0, 1] range
        path: Output path (format determined by extension)
        quality: JPEG/WEBP quality (1-100)
        
    Returns:
        Path to the saved image
        
    Example:
        >>> save_image(protected_img, "output/protected.png")
    """
    path = Path(path)
    _validate_image_extension(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image from batch
    
    # Ensure tensor is on CPU and convert to numpy
    array = tensor.detach().cpu().numpy()
    
    # Rearrange from (C, H, W) to (H, W, C)
    array = np.transpose(array, (1, 2, 0))
    
    # Clip values to [0, 1] and convert to uint8
    array = np.clip(array, 0.0, 1.0)
    array = (array * 255).astype(np.uint8)
    
    # Create PIL image and save
    image = Image.fromarray(array)
    
    # Set quality for formats that support it
    save_kwargs = {}
    suffix = path.suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        save_kwargs["quality"] = quality
    elif suffix == ".webp":
        save_kwargs["quality"] = quality
    elif suffix == ".png":
        save_kwargs["compress_level"] = 6
    
    image.save(path, **save_kwargs)
    return path


def image_to_tensor(
    image: Image.Image,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Convert a PIL Image to PyTorch tensor.
    
    Args:
        image: PIL Image object
        normalize: If True, normalize to [0, 1] range
        
    Returns:
        Tensor of shape (C, H, W)
    """
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    array = np.array(image, dtype=np.float32)
    
    if normalize:
        array = array / 255.0
    
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a PyTorch tensor to PIL Image.
    
    Args:
        tensor: Tensor of shape (C, H, W) or (B, C, H, W)
                Values should be in [0, 1] range
                
    Returns:
        PIL Image object
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    array = tensor.detach().cpu().numpy()
    array = np.transpose(array, (1, 2, 0))
    array = np.clip(array, 0.0, 1.0)
    array = (array * 255).astype(np.uint8)
    
    return Image.fromarray(array)


def add_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Add batch dimension if not present."""
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    return tensor


def remove_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Remove batch dimension if present."""
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor
