"""
AURALOCK Demo Script
Demonstrates the complete workflow of protecting images from AI.

This script:
1. Creates a sample artwork image
2. Applies FGSM protection
3. Shows before/after comparison
4. Tests if AI can still "understand" the protected image
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights

from auralock.core.image import load_image, save_image, tensor_to_image
from auralock.core.metrics import get_quality_report, calculate_psnr, calculate_ssim
from auralock.attacks import FGSM


def create_sample_artwork(size=(512, 512)) -> Image.Image:
    """Create a sample 'artwork' image with distinct visual style."""
    img = Image.new('RGB', size, color='#1a1a2e')
    draw = ImageDraw.Draw(img)
    
    # Draw some artistic patterns
    w, h = size
    
    # Gradient circles
    for i in range(10):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        r = np.random.randint(20, 100)
        color = (
            np.random.randint(100, 255),
            np.random.randint(50, 200),
            np.random.randint(100, 255),
        )
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline=None)
    
    # Diagonal lines
    for i in range(20):
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        x2 = x1 + np.random.randint(-100, 100)
        y2 = y1 + np.random.randint(-100, 100)
        color = (
            np.random.randint(150, 255),
            np.random.randint(100, 200),
            np.random.randint(50, 150),
        )
        draw.line([x1, y1, x2, y2], fill=color, width=3)
    
    return img


def main():
    print("=" * 60)
    print("🛡️  AURALOCK - Complete Demo")
    print("=" * 60)
    print()
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Create sample artwork
    print("📎 Step 1: Creating sample artwork...")
    artwork = create_sample_artwork()
    artwork_path = output_dir / "original_artwork.png"
    artwork.save(artwork_path)
    print(f"   Saved: {artwork_path}")
    
    # Step 2: Load and prepare image
    print("\n📎 Step 2: Loading image for protection...")
    image = load_image(artwork_path, size=(224, 224))
    image = image.unsqueeze(0)  # Add batch dimension
    print(f"   Image shape: {image.shape}")
    print(f"   Value range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Step 3: Load AI model
    print("\n📎 Step 3: Loading AI model (ResNet18)...")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()
    print("   Model loaded successfully")
    
    # Get original prediction
    with torch.no_grad():
        original_output = model(image)
        original_class = original_output.argmax(dim=1).item()
        original_conf = torch.softmax(original_output, dim=1).max().item()
    print(f"   Original prediction: class {original_class} (confidence: {original_conf:.2%})")
    
    # Step 4: Apply FGSM protection at different levels
    print("\n📎 Step 4: Applying FGSM protection...")
    
    protection_results = []
    epsilons = [0.01, 0.03, 0.05]
    
    for eps in epsilons:
        print(f"\n   Testing epsilon = {eps}...")
        
        attack = FGSM(model, epsilon=eps)
        result = attack.generate_with_info(image)
        
        protected = result['adversarial']
        
        # Get new prediction
        with torch.no_grad():
            new_output = model(protected)
            new_class = new_output.argmax(dim=1).item()
            new_conf = torch.softmax(new_output, dim=1).max().item()
        
        # Calculate quality metrics
        report = get_quality_report(image, protected)
        
        protection_results.append({
            'epsilon': eps,
            'original_class': original_class,
            'new_class': new_class,
            'success': new_class != original_class,
            'psnr': report['psnr_db'],
            'ssim': report['ssim'],
            'quality': report['overall_quality'],
            'protected_image': protected,
        })
        
        print(f"      Attack success: {'✅ Yes' if new_class != original_class else '❌ No'}")
        print(f"      New prediction: class {new_class}")
        print(f"      PSNR: {report['psnr_db']:.2f} dB")
        print(f"      SSIM: {report['ssim']:.4f}")
    
    # Step 5: Save protected images
    print("\n📎 Step 5: Saving protected images...")
    for result in protection_results:
        eps = result['epsilon']
        output_path = output_dir / f"protected_eps{eps}.png"
        save_image(result['protected_image'], output_path)
        print(f"   Saved: {output_path}")
    
    # Step 6: Create comparison visualization
    print("\n📎 Step 6: Creating comparison visualization...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Top row: Images
    original_np = image.squeeze().permute(1, 2, 0).numpy()
    axes[0, 0].imshow(original_np)
    axes[0, 0].set_title(f"Original\nClass: {original_class}")
    axes[0, 0].axis('off')
    
    for i, result in enumerate(protection_results):
        protected_np = result['protected_image'].squeeze().permute(1, 2, 0).numpy()
        axes[0, i+1].imshow(protected_np)
        status = "✓" if result['success'] else "✗"
        axes[0, i+1].set_title(f"ε={result['epsilon']} {status}\nClass: {result['new_class']}")
        axes[0, i+1].axis('off')
    
    # Bottom row: Perturbation visualization
    axes[1, 0].text(0.5, 0.5, "Perturbation\nVisualization", 
                    ha='center', va='center', fontsize=14)
    axes[1, 0].axis('off')
    
    for i, result in enumerate(protection_results):
        perturbation = result['protected_image'] - image
        # Normalize for visualization
        pert_vis = (perturbation.squeeze().permute(1, 2, 0).numpy() + result['epsilon']) / (2 * result['epsilon'])
        pert_vis = np.clip(pert_vis, 0, 1)
        axes[1, i+1].imshow(pert_vis)
        axes[1, i+1].set_title(f"Perturbation (ε={result['epsilon']})\nPSNR: {result['psnr']:.1f}dB")
        axes[1, i+1].axis('off')
    
    plt.suptitle("AURALOCK Protection Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    comparison_path = output_dir / "comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {comparison_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"\n{'Epsilon':<10} {'Success':<10} {'PSNR (dB)':<12} {'SSIM':<10} {'Quality':<12}")
    print("-" * 54)
    for result in protection_results:
        success_str = "✅ Yes" if result['success'] else "❌ No"
        print(f"{result['epsilon']:<10} {success_str:<10} {result['psnr']:<12.2f} {result['ssim']:<10.4f} {result['quality']:<12}")
    
    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print(f"   Output files saved to: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
