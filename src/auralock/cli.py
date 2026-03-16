"""
AI Shield CLI - Command Line Interface for image protection.

Usage:
    ai-shield protect image.png --output protected.png --epsilon 0.03
    ai-shield analyze original.png protected.png
    ai-shield demo
"""

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

app = typer.Typer(
    name="ai-shield",
    help="🛡️ Protect your artwork from AI style mimicry",
    add_completion=False,
)
console = Console()


@app.command()
def protect(
    input_path: Path = typer.Argument(..., help="Path to input image"),
    output: Path = typer.Option(None, "--output", "-o", help="Output path"),
    epsilon: float = typer.Option(0.03, "--epsilon", "-e", help="Perturbation strength (0.01-0.1)"),
    method: str = typer.Option("fgsm", "--method", "-m", help="Attack method: fgsm, pgd"),
):
    """
    Protect an image from AI style mimicry.
    
    Example:
        ai-shield protect artwork.png -o protected.png -e 0.03
    """
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    
    from auralock.core.image import load_image, save_image
    from auralock.core.metrics import get_quality_report
    from auralock.attacks import FGSM, PGD
    
    # Validate input
    if not input_path.exists():
        console.print(f"[red]Error:[/red] File not found: {input_path}")
        raise typer.Exit(1)
    
    if output is None:
        output = input_path.parent / f"{input_path.stem}_protected{input_path.suffix}"
    
    if not 0.01 <= epsilon <= 0.1:
        console.print("[yellow]Warning:[/yellow] Epsilon outside recommended range (0.01-0.1)")
    
    if method not in ("fgsm", "pgd"):
        console.print(f"[red]Error:[/red] Unknown method: {method}. Use 'fgsm' or 'pgd'.")
        raise typer.Exit(1)
    
    console.print(f"\n[bold blue]🛡️ AI Shield - Image Protection[/bold blue]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load image
        task = progress.add_task("Loading image...", total=None)
        image = load_image(input_path)
        image = image.unsqueeze(0)  # Add batch dimension
        progress.update(task, completed=True, description="✅ Image loaded")
        
        # Load model
        task = progress.add_task("Loading AI model...", total=None)
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        progress.update(task, completed=True, description="✅ Model loaded (ResNet18)")
        
        # Apply protection
        task = progress.add_task(f"Applying {method.upper()} protection...", total=None)
        
        if method == "fgsm":
            attack = FGSM(model, epsilon=epsilon)
        elif method == "pgd":
            attack = PGD(model, epsilon=epsilon)
        
        protected = attack(image)
        progress.update(task, completed=True, description=f"✅ {method.upper()} applied")
        
        # Save result
        task = progress.add_task("Saving protected image...", total=None)
        save_image(protected, output)
        progress.update(task, completed=True, description=f"✅ Saved to {output}")
    
    # Quality report
    report = get_quality_report(image, protected)
    
    table = Table(title="\n📊 Quality Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Status", style="yellow")
    
    psnr_status = "✅ Excellent" if report['psnr_db'] > 35 else "⚠️ Visible"
    ssim_status = "✅ Excellent" if report['ssim'] > 0.95 else "⚠️ Noticeable"
    
    table.add_row("PSNR", f"{report['psnr_db']:.2f} dB", psnr_status)
    table.add_row("SSIM", f"{report['ssim']:.4f}", ssim_status)
    table.add_row("Overall", report['overall_quality'], "")
    
    console.print(table)
    console.print(f"\n[green]✅ Protected image saved to:[/green] {output}\n")


@app.command()
def analyze(
    original: Path = typer.Argument(..., help="Path to original image"),
    modified: Path = typer.Argument(..., help="Path to modified/protected image"),
):
    """
    Analyze and compare two images.
    
    Example:
        ai-shield analyze original.png protected.png
    """
    from auralock.core.image import load_image
    from auralock.core.metrics import get_quality_report
    
    if not original.exists():
        console.print(f"[red]Error:[/red] Original image not found: {original}")
        raise typer.Exit(1)
    
    if not modified.exists():
        console.print(f"[red]Error:[/red] Modified image not found: {modified}")
        raise typer.Exit(1)
    
    console.print(f"\n[bold blue]📊 Image Analysis[/bold blue]\n")
    
    img1 = load_image(original)
    img2 = load_image(modified)
    
    report = get_quality_report(img1, img2)
    
    table = Table(title="Comparison Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("PSNR", f"{report['psnr_db']:.2f} dB")
    table.add_row("SSIM", f"{report['ssim']:.4f}")
    table.add_row("L2 Distance", f"{report['l2_distance']:.4f}")
    table.add_row("L∞ Distance", f"{report['linf_distance']:.4f}")
    table.add_row("Quality", report['overall_quality'])
    
    console.print(table)


@app.command()
def demo():
    """
    Run a demonstration of AI Shield capabilities.
    """
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    from auralock.attacks import FGSM
    from auralock.core.metrics import get_quality_report
    
    console.print("\n[bold blue]🔬 AI Shield Demo[/bold blue]\n")
    
    # Create random test image
    console.print("Creating test image...")
    image = torch.rand(1, 3, 224, 224)
    
    # Load model
    console.print("Loading ResNet18 model...")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Test different epsilon values
    epsilons = [0.01, 0.03, 0.05, 0.1]
    
    table = Table(title="FGSM Attack Results at Different Epsilon Values")
    table.add_column("Epsilon", style="cyan")
    table.add_column("Attack Success", style="green")
    table.add_column("PSNR (dB)", style="yellow")
    table.add_column("SSIM", style="yellow")
    table.add_column("Quality", style="magenta")
    
    for eps in epsilons:
        attack = FGSM(model, epsilon=eps)
        result = attack.generate_with_info(image)
        report = get_quality_report(image, result['adversarial'])
        
        table.add_row(
            f"{eps}",
            f"{result['success_rate']*100:.0f}%",
            f"{report['psnr_db']:.1f}",
            f"{report['ssim']:.4f}",
            report['overall_quality'],
        )
    
    console.print(table)
    console.print("\n[green]Demo completed![/green]\n")


@app.command()
def version():
    """Show version information."""
    from auralock import __version__
    console.print(f"AI Shield version: [bold]{__version__}[/bold]")


if __name__ == "__main__":
    app()
