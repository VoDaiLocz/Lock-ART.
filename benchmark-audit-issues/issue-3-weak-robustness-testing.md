# Issue: Weak Robustness Testing - Missing Critical Preprocessing Transformations

## Labels
`benchmark`, `robustness`, `security`, `technical-debt`, `enhancement`

## Problem Description

The robustness testing suite includes only **4 basic transforms** (identity, gaussian blur, 2 resize operations), missing critical preprocessing steps that real-world mimicry pipelines actually use. This creates a false sense of robustness—protection may survive gaussian blur but fail against JPEG compression, center crops, or CLIP preprocessing that DreamBooth/LoRA models apply.

## What is Wrong with the Benchmark

### 1. Limited Transform Suite

**Current Implementation** (`/src/auralock/core/style.py:217-231`):
```python
def build_style_transform_suite() -> tuple[tuple[str, StyleTransform], ...]:
    return (
        ("identity", identity),
        ("gaussian_blur", lambda images: gaussian_blur(images, kernel_size=5, sigma=1.0)),
        ("resize_restore_75", lambda images: resize_restore(images, scale=0.75)),
        ("resize_restore_50", lambda images: resize_restore(images, scale=0.5)),
    )
```

**What's Missing**:
1. **JPEG Compression**: DreamBooth training often involves JPEG-compressed images from web scraping or dataset curation
2. **Center/Random Crop**: Standard data augmentation in training pipelines
3. **CLIP Preprocessing**: Mimicry models use CLIP for guidance, which applies specific normalization and resizing
4. **VAE Encode/Decode**: Latent diffusion models encode images through VAE, which introduces information loss
5. **Color Jitter**: Common augmentation (brightness, contrast, saturation, hue shifts)
6. **Rotation**: Small angle rotations are standard augmentation
7. **Noise Injection**: Training often adds small Gaussian noise for regularization

### 2. Proxy for Real Transformations

**Resize-Restore is Not JPEG**:
```python
# Current (line 193-214)
def resize_restore(images: torch.Tensor, scale: float = 0.75):
    # Bilinear downscale → bilinear upscale
    # Used as proxy for compression
```

**Problem**: Bilinear resize ≠ JPEG compression artifacts
- JPEG introduces block artifacts (DCT quantization)
- JPEG has chroma subsampling (4:2:0)
- JPEG quality levels vary (10-100), each with different artifact patterns
- Protection robust to bilinear resize may **fail completely** against JPEG artifacts

### 3. Gaussian Blur Parameter Space Unexplored

**Current** (line 226-227):
```python
("gaussian_blur", lambda images: gaussian_blur(images, kernel_size=5, sigma=1.0))
```

**Single configuration tested**:
- Only `kernel_size=5, sigma=1.0`
- No exploration of stronger blurs (sigma=2.0, 3.0)
- No exploration of different kernel sizes (3, 7, 9)

**Real-world variation**: Images may be blurred with various strengths. Testing only one configuration doesn't validate robustness.

### 4. No Adversarial Purification Defenses

Common purification techniques attackers use to remove perturbations:
- **JPEG compression at various quality levels** (most effective against adversarial perturbations)
- **Denoising autoencoders**
- **Total variation minimization**
- **Median filtering**
- **Bilateral filtering**

**None of these are tested in the robustness suite.**

## Why This Can Mislead Users

### Scenario 1: False Robustness Claims

User sees in report:
```json
{
  "robust_style_similarity": 0.72,
  "robust_embedding_similarity": 0.68,
  "protection_score": 42.1,
  "assessment": "Strong"
}
```

User assumes:
- "My protection survives all common preprocessing"
- "42.1 score is robust to mimicry pipeline transformations"

Reality:
- Protection only tested against 4 transforms
- JPEG compression at quality=70 may **completely remove perturbations**
- Center crop may destroy spatially-localized protection patterns
- Assessment is "Strong" in a limited test environment, not real-world

### Scenario 2: Purification Attack Vulnerability

Attacker applies simple purification before training:
```python
# Attacker's preprocessing (NOT tested by AuraLock)
def purify_protected_image(img):
    # JPEG compression at quality=85
    img = apply_jpeg_compression(img, quality=85)
    # Center crop + resize
    img = center_crop_and_resize(img, crop_size=0.9)
    # Mild denoising
    img = cv2.fastNlMeansDenoisingColored(img)
    return img

# Now train DreamBooth on purified images
# Protection may be significantly weakened or removed
```

**AuraLock's robustness testing wouldn't catch this vulnerability.**

### Scenario 3: CLIP/VAE Feature Space Mismatch

DreamBooth uses CLIP features for guidance:
```python
# Actual mimicry pipeline
clip_features = clip_model.encode_image(
    clip_preprocessing(protected_image)  # Specific resizing + normalization
)
vae_latents = vae.encode(
    vae_preprocessing(protected_image)  # Different preprocessing
).latent_dist.sample()
```

**But AuraLock tests robustness in ResNet18 feature space with ResNet18 preprocessing.** Protection robust in one space may be fragile in another.

## Evidence from Repository

### 1. Transform Suite is Hardcoded and Minimal

Search for all transform definitions:
```bash
$ grep -n "build_style_transform_suite" /src/auralock/core/style.py
217:def build_style_transform_suite() -> tuple[tuple[str, StyleTransform], ...]:
```

Only one function, one implementation, 4 transforms. No extension mechanism, no parameterization.

### 2. No JPEG Implementation

Search for JPEG compression:
```bash
$ grep -r "jpeg" src/ --ignore-case
# (no matches in core robustness code)

$ grep -r "compression" src/
# (no matches related to robustness testing)
```

**No JPEG compression testing exists in the robustness suite.**

### 3. No Crop Testing

Search for cropping:
```bash
$ grep -r "crop" src/auralock/core/
# (no matches)
```

Center crop, random crop, and resize-crop patterns are standard in training pipelines but **not tested**.

### 4. Acknowledged in Documentation (Partially)

README line 232 mentions robustness testing:
> "Robustness testing: via blur, resize/restore transforms averaged across multiple scales"

**But doesn't mention the severe limitations**:
- No JPEG (most critical purification defense)
- No crop (standard augmentation)
- No CLIP/VAE preprocessing (actual mimicry pipeline)

### 5. Comment in Code Acknowledges Resize as Proxy

`/src/auralock/core/style.py:193-214`:
```python
def resize_restore(images: torch.Tensor, scale: float = 0.75) -> torch.Tensor:
    """Downscale and restore an image batch to emulate common purification steps."""
    # ^^^^^ "emulate" = proxy, not actual purification
```

Resize is used to **emulate** compression, not test against actual compression artifacts.

## Proposed Benchmark Upgrade

### Phase 1: Expand Transform Suite

Replace hardcoded minimal suite with comprehensive testing:

```python
def build_comprehensive_transform_suite() -> tuple[tuple[str, StyleTransform], ...]:
    """Transforms covering real-world preprocessing and purification attacks."""
    return (
        # Baseline
        ("identity", identity),

        # Existing (keep for backward compatibility)
        ("gaussian_blur_mild", lambda x: gaussian_blur(x, kernel_size=5, sigma=1.0)),
        ("resize_restore_75", lambda x: resize_restore(x, scale=0.75)),
        ("resize_restore_50", lambda x: resize_restore(x, scale=0.5)),

        # NEW: JPEG Compression (critical)
        ("jpeg_quality_95", lambda x: jpeg_compress_decompress(x, quality=95)),
        ("jpeg_quality_85", lambda x: jpeg_compress_decompress(x, quality=85)),
        ("jpeg_quality_75", lambda x: jpeg_compress_decompress(x, quality=75)),
        ("jpeg_quality_50", lambda x: jpeg_compress_decompress(x, quality=50)),

        # NEW: Cropping
        ("center_crop_90", lambda x: center_crop_and_resize(x, crop_ratio=0.9)),
        ("center_crop_80", lambda x: center_crop_and_resize(x, crop_ratio=0.8)),
        ("random_crop_90", lambda x: random_crop_and_resize(x, crop_ratio=0.9)),

        # NEW: Stronger blur variants
        ("gaussian_blur_medium", lambda x: gaussian_blur(x, kernel_size=7, sigma=2.0)),
        ("gaussian_blur_strong", lambda x: gaussian_blur(x, kernel_size=9, sigma=3.0)),

        # NEW: Color augmentation
        ("color_jitter_mild", lambda x: color_jitter(x, brightness=0.1, contrast=0.1)),
        ("color_jitter_medium", lambda x: color_jitter(x, brightness=0.2, contrast=0.2)),

        # NEW: Noise injection
        ("gaussian_noise_small", lambda x: add_gaussian_noise(x, std=0.01)),
        ("gaussian_noise_medium", lambda x: add_gaussian_noise(x, std=0.03)),

        # NEW: Mimicry-specific preprocessing
        ("clip_preprocess", lambda x: apply_clip_preprocessing(x)),
        ("vae_encode_decode", lambda x: vae_roundtrip(x)),
    )
```

### Phase 2: Implement Missing Transforms

**JPEG Compression** (most critical):
```python
def jpeg_compress_decompress(
    images: torch.Tensor,
    quality: int = 85,
) -> torch.Tensor:
    """Apply JPEG compression and decompression to test robustness."""
    from PIL import Image
    import io

    batch_size = images.shape[0]
    results = []

    for i in range(batch_size):
        # Convert tensor to PIL Image
        img_np = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        # JPEG compress in memory
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)

        # Decompress
        img_pil = Image.open(buffer)
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        results.append(img_tensor)

    return torch.stack(results).to(images.device)
```

**Center Crop and Resize**:
```python
def center_crop_and_resize(
    images: torch.Tensor,
    crop_ratio: float = 0.9,
) -> torch.Tensor:
    """Center crop to crop_ratio and resize back to original size."""
    if not 0.0 < crop_ratio <= 1.0:
        raise ValueError("crop_ratio must be in (0, 1]")

    _, _, h, w = images.shape
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)

    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2

    cropped = images[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]

    return F.interpolate(
        cropped,
        size=(h, w),
        mode='bilinear',
        align_corners=False,
        antialias=True,
    )
```

**CLIP Preprocessing**:
```python
def apply_clip_preprocessing(images: torch.Tensor) -> torch.Tensor:
    """Apply CLIP model's standard preprocessing."""
    from transformers import CLIPImageProcessor

    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # CLIP expects 224x224 with specific normalization
    resized = F.interpolate(images, size=(224, 224), mode='bicubic', antialias=True)

    # Apply CLIP normalization
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)

    normalized = (resized - mean) / std
    return normalized
```

### Phase 3: Configurable Transform Selection

Allow users to select which transforms to test:
```python
# CLI
$ auralock protect artwork.png -o protected.png \
    --robustness-suite comprehensive  # all transforms
    # or --robustness-suite mimicry    # CLIP, VAE, JPEG only
    # or --robustness-suite minimal    # current 4 transforms (backward compat)

# Programmatic
from auralock.core.style import TransformSuitePreset

result = service.protect_file(
    "artwork.png",
    profile="balanced",
    robustness_suite=TransformSuitePreset.COMPREHENSIVE,
)
```

### Phase 4: Worst-Case Robustness Reporting

Instead of only reporting mean robustness, report:
```json
{
  "protection_report": {
    "robust_style_similarity_mean": 0.72,
    "robust_style_similarity_worst": 0.89,  // weakest protection
    "robust_style_similarity_best": 0.55,   // strongest protection
    "worst_case_transform": "jpeg_quality_75",
    "best_case_transform": "identity",
    "transform_breakdown": {
      "identity": 0.55,
      "gaussian_blur": 0.68,
      "jpeg_quality_95": 0.71,
      "jpeg_quality_85": 0.82,
      "jpeg_quality_75": 0.89,  // ⚠️ Protection significantly weakened
      "center_crop_90": 0.65
    }
  }
}
```

This reveals vulnerability: "Protection looks good (mean=0.72) but JPEG at quality=75 weakens it severely (0.89)."

## Acceptance Criteria

### Phase 1: JPEG Compression (Critical, Week 1-2)
- [ ] Implement `jpeg_compress_decompress()` function
- [ ] Add JPEG quality levels: 95, 85, 75, 50 to transform suite
- [ ] Test protection against JPEG compression in robustness evaluation
- [ ] Report JPEG robustness separately in all outputs
- [ ] Document JPEG vulnerability in README if protection is weak

### Phase 2: Cropping and Augmentation (Week 2-3)
- [ ] Implement `center_crop_and_resize()` function
- [ ] Implement `random_crop_and_resize()` function
- [ ] Implement `color_jitter()` function
- [ ] Add to transform suite and test

### Phase 3: Mimicry-Aligned Preprocessing (Week 3-4)
- [ ] Implement `apply_clip_preprocessing()` function
- [ ] Implement `vae_encode_decode()` function (requires VAE model)
- [ ] Test protection against actual mimicry preprocessing
- [ ] Compare robustness in ResNet18 space vs CLIP/VAE space

### Phase 4: Comprehensive Suite and Reporting (Week 4-5)
- [ ] Create `TransformSuitePreset` enum: MINIMAL, STANDARD, COMPREHENSIVE, MIMICRY
- [ ] CLI flag: `--robustness-suite {minimal,standard,comprehensive,mimicry}`
- [ ] Report worst-case robustness alongside mean
- [ ] Highlight vulnerable transforms in warnings

### Phase 5: Validation and Documentation (Week 5-6)
- [ ] Run comprehensive robustness tests on all profiles
- [ ] Publish robustness comparison table: profile → worst_case_transform + weakness
- [ ] Update README with honest robustness limitations
- [ ] Document purification attack resistance (or lack thereof)

## Additional Context

### Why JPEG is Most Critical

From adversarial robustness literature:
- **JPEG compression is the most effective purification defense** against adversarial perturbations
- Quality level 75-85 removes most imperceptible perturbations
- But also degrades image quality noticeably

If protection doesn't survive JPEG compression at quality 85:
- Users can't share protected images on platforms that recompress (Facebook, Instagram)
- Attackers can trivially remove protection by applying JPEG compression
- Protection is effectively useless in real-world scenarios

### Resize vs JPEG: Not Equivalent

Resize-restore (bilinear) simulates information loss but not artifact patterns:
- Bilinear: smooth, continuous interpolation
- JPEG: block artifacts, ringing, chroma subsampling

Protection may survive smooth information loss but fail against structured artifacts.

### Defense in Depth

Strong protection should survive **combinations** of transforms:
```python
# Real-world scenario
def realistic_preprocessing(img):
    img = jpeg_compress(img, quality=85)  # Platform recompression
    img = center_crop(img, ratio=0.95)    # User crops before upload
    img = resize(img, target_size=512)    # Training pipeline resize
    return img
```

Current testing doesn't evaluate transform combinations.

## References

- `/src/auralock/core/style.py:217-231` - Current minimal transform suite
- `/src/auralock/core/style.py:193-214` - Resize-restore proxy for compression
- `/src/auralock/core/metrics.py:286-300` - Robustness evaluation loop
- Adversarial robustness literature on purification defenses
- DreamBooth/LoRA data preprocessing pipelines
