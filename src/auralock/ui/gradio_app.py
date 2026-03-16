"""
AI Shield Web UI - Gradio-based web interface for image protection.

Launch with:
    python -m auralock.ui.gradio_app
    
Or:
    ai-shield-webui
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.models import resnet18, ResNet18_Weights

# Global model (loaded once)
_model = None


def get_model():
    """Load model lazily."""
    global _model
    if _model is None:
        _model = resnet18(weights=ResNet18_Weights.DEFAULT)
        _model.eval()
    return _model


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to tensor."""
    image = image.convert("RGB")
    arr = np.array(image.resize((224, 224)), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor back to PIL Image."""
    arr = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def protect_image(
    image: Image.Image,
    epsilon: float,
    method: str,
    num_steps: int,
) -> tuple[Image.Image, str]:
    """
    Protect an image from AI style mimicry.
    
    Returns:
        (protected_image, report_text)
    """
    if image is None:
        return None, "⚠️ Please upload an image first."
    
    from auralock.attacks import FGSM, PGD
    from auralock.core.metrics import get_quality_report
    
    # Convert to tensor
    original_size = image.size
    tensor = image_to_tensor(image)
    
    # Get model
    model = get_model()
    
    # Get original prediction
    with torch.no_grad():
        orig_output = model(tensor)
        orig_class = orig_output.argmax(dim=1).item()
    
    # Apply attack
    if method == "FGSM (Fast)":
        attack = FGSM(model, epsilon=epsilon)
    else:
        attack = PGD(model, epsilon=epsilon, num_steps=num_steps)
    
    result = attack.generate_with_info(tensor)
    protected = result['adversarial']
    
    # Get new prediction
    with torch.no_grad():
        new_output = model(protected)
        new_class = new_output.argmax(dim=1).item()
    
    # Quality report
    report = get_quality_report(tensor, protected)
    
    # Convert back to image
    protected_img = tensor_to_image(protected)
    # Resize back to original size
    protected_img = protected_img.resize(original_size, Image.Resampling.LANCZOS)
    
    # Create report text
    success_emoji = "✅" if new_class != orig_class else "⚠️"
    quality_emoji = "🟢" if report['overall_quality'] in ['Excellent', 'Good'] else "🟡"
    
    report_text = f"""
## 📊 Protection Report

### Attack Results
- **Method**: {method}
- **Epsilon**: {epsilon}
- **Success**: {success_emoji} {'Yes' if new_class != orig_class else 'No'} (class {orig_class} → {new_class})

### Quality Metrics
- **PSNR**: {report['psnr_db']:.2f} dB
- **SSIM**: {report['ssim']:.4f}
- **Overall**: {quality_emoji} {report['overall_quality']}

### Perturbation Stats
- **L2 Distance**: {result['perturbation_l2']:.4f}
- **L∞ Distance**: {result['perturbation_linf']:.4f}

---
💡 **Tip**: Lower epsilon = less visible but may not fool AI. Higher epsilon = stronger protection but more visible changes.
"""
    
    return protected_img, report_text


def create_ui():
    """Create the Gradio UI."""
    
    with gr.Blocks(
        title="AI Shield - Protect Your Artwork",
    ) as app:
        gr.Markdown("""
        # 🛡️ AI Shield
        ### Protect your artwork from AI style mimicry
        
        Upload an image and apply adversarial protection to prevent AI from learning your artistic style.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="📤 Upload Your Artwork",
                    type="pil",
                    height=400,
                )
                
                with gr.Group():
                    gr.Markdown("### ⚙️ Protection Settings")
                    
                    method = gr.Radio(
                        choices=["FGSM (Fast)", "PGD (Stronger)"],
                        value="FGSM (Fast)",
                        label="Attack Method",
                    )
                    
                    epsilon = gr.Slider(
                        minimum=0.01,
                        maximum=0.1,
                        value=0.03,
                        step=0.01,
                        label="Epsilon (Protection Strength)",
                        info="Higher = stronger protection but more visible",
                    )
                    
                    num_steps = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=10,
                        step=5,
                        label="PGD Steps (only for PGD)",
                        visible=True,
                    )
                
                protect_btn = gr.Button(
                    "🛡️ Protect Image",
                    variant="primary",
                    size="lg",
                )
            
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="🎨 Protected Image",
                    type="pil",
                    height=400,
                )
                
                report_output = gr.Markdown(
                    label="📊 Report",
                    value="*Upload an image and click 'Protect Image' to see results*",
                )
        
        # Examples
        gr.Markdown("### 📌 How it works")
        gr.Markdown("""
        1. **Upload** your artwork
        2. **Choose** protection method (FGSM is fast, PGD is stronger)
        3. **Adjust** epsilon (0.03 is a good starting point)
        4. **Click** "Protect Image"
        5. **Download** the protected image
        
        The protected image looks nearly identical to humans, but AI models will struggle to:
        - Learn your artistic style
        - Copy your techniques
        - Classify or analyze your artwork correctly
        """)
        
        # Connect function
        protect_btn.click(
            fn=protect_image,
            inputs=[input_image, epsilon, method, num_steps],
            outputs=[output_image, report_output],
        )
    
    return app


def main():
    """Launch the web UI."""
    print("🛡️ Starting AI Shield Web UI...")
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
