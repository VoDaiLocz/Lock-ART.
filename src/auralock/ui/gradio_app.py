"""Gradio-based web UI for AuraLock."""

from __future__ import annotations

from typing import Any

from PIL import Image

from auralock.services import ProtectionResult, ProtectionService

_service: ProtectionService | None = None


def _get_service() -> ProtectionService:
    global _service
    if _service is None:
        _service = ProtectionService()
    return _service


def _require_gradio():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is not installed. Install the UI extras with: pip install 'auralock[ui]'"
        ) from exc
    return gr


def _format_report(result: ProtectionResult) -> str:
    quality = result.quality_report
    success_emoji = "✅" if result.attack_success else "⚠️"
    quality_emoji = (
        "🟢" if quality["overall_quality"] in {"Excellent", "Good"} else "🟡"
    )
    return f"""
## Protection Report

### Attack Results
- **Method**: {result.method.upper()}
- **Epsilon**: {result.epsilon}
- **Success**: {success_emoji} {'Yes' if result.attack_success else 'No'} ({result.original_prediction} → {result.adversarial_prediction})

### Quality Metrics
- **PSNR**: {quality['psnr_db']:.2f} dB
- **SSIM**: {quality['ssim']:.4f}
- **Overall**: {quality_emoji} {quality['overall_quality']}

### Perturbation Stats
- **L2 Distance**: {result.perturbation_l2:.4f}
- **L∞ Distance**: {result.perturbation_linf:.4f}

### Runtime
- **Output size**: {result.original_size[0]} x {result.original_size[1]}
- **Device**: {result.device}
"""


def protect_image(
    image: Image.Image | None,
    epsilon: float,
    method: str,
    num_steps: int,
) -> tuple[Image.Image | None, str]:
    """Protect an uploaded image and return the protected image plus a report."""
    if image is None:
        return None, "Please upload an image first."

    result = _get_service().protect_image(
        image,
        epsilon=epsilon,
        method=method,
        num_steps=num_steps,
    )
    return result.protected_image, _format_report(result)


def create_ui():
    """Create the Gradio UI."""
    gr = _require_gradio()

    with gr.Blocks(title="AuraLock") as app:
        gr.Markdown("""
# AuraLock
### Invisible shielding for artists who want production-grade control

Upload an image, apply adversarial protection, and inspect the protection report in one flow.
""")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Artwork",
                    type="pil",
                    height=400,
                )

                with gr.Group():
                    gr.Markdown("### Protection Settings")

                    method = gr.Radio(
                        choices=["fgsm", "pgd"],
                        value="fgsm",
                        label="Attack Method",
                    )

                    epsilon = gr.Slider(
                        minimum=0.01,
                        maximum=0.1,
                        value=0.03,
                        step=0.01,
                        label="Epsilon",
                        info="Higher values produce stronger but more visible perturbations.",
                    )

                    num_steps = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=10,
                        step=5,
                        label="PGD Steps",
                    )

                protect_btn = gr.Button("Protect Image", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Protected Image",
                    type="pil",
                    height=400,
                )
                report_output = gr.Markdown(
                    value="Upload an image and click **Protect Image** to generate a report.",
                )

        gr.Markdown("""
### Why this version is more stable
1. The UI uses the same protection service as the CLI.
2. Model preprocessing is consistent for every request.
3. Output resolution is preserved instead of silently shrinking to the model input size.
""")

        protect_btn.click(
            fn=protect_image,
            inputs=[input_image, epsilon, method, num_steps],
            outputs=[output_image, report_output],
        )

    return app


def main(host: str = "127.0.0.1", port: int = 7860, *_: Any, **__: Any) -> None:
    """Launch the web UI."""
    app = create_ui()
    app.launch(
        server_name=host,
        server_port=port,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
