"""AuraLock command-line interface."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from auralock import __version__
from auralock.benchmarks import (
    DEFAULT_ANTI_DREAMBOOTH_CLASS_PROMPT,
    DEFAULT_ANTI_DREAMBOOTH_INFER_SCRIPT,
    DEFAULT_ANTI_DREAMBOOTH_INSTANCE_PROMPT,
    DEFAULT_ANTI_DREAMBOOTH_TRAIN_SCRIPT,
    DEFAULT_BENCHMARK_BASE_IMAGE,
    DEFAULT_COMPOSE_FILE,
    DEFAULT_SERVICE_NAME,
    AntiDreamBoothSubjectBenchmarkHarness,
    DockerLoraBenchmarkConfig,
    LoraBenchmarkHarness,
    build_docker_lora_benchmark_plan,
)
from auralock.core.image import save_image
from auralock.services import (
    BatchProtectionSummary,
    ProtectionResult,
    ProtectionService,
)

app = typer.Typer(
    name="auralock",
    help="Protect your artwork from AI style mimicry with a consistent production pipeline.",
    add_completion=False,
)
console = Console()


def _to_builtin(value: Any) -> Any:
    """Convert report payloads into JSON-friendly values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _write_json_report(path: Path, payload: dict[str, object]) -> None:
    """Write a structured JSON report to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_builtin(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _render_quality_table(report: dict[str, object]) -> Table:
    table = Table(title="Quality Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("PSNR", f"{report['psnr_db']:.2f} dB")
    table.add_row("SSIM", f"{report['ssim']:.4f}")
    table.add_row("L2 Distance", f"{report['l2_distance']:.4f}")
    table.add_row("L∞ Distance", f"{report['linf_distance']:.4f}")
    table.add_row("Quality", str(report["overall_quality"]))
    return table


def _render_readability_table(report: dict[str, object]) -> Table:
    table = Table(title="Protection Readability (⚠️ PROXY - NOT VALIDATED)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Embedding similarity", f"{report['embedding_similarity']:.4f}")
    table.add_row("Style similarity", f"{report['style_similarity']:.4f}")
    table.add_row(
        "Robust style similarity",
        f"{report['robust_style_similarity']:.4f}",
    )
    table.add_row(
        "Protection score (proxy)",
        f"{report['protection_score']:.1f}/100",
    )
    table.add_row("Assessment", str(report["assessment"]))
    table.add_row("Validation status", "❌ Not validated against real attacks")

    # Add warning row if present
    if "warning" in report:
        table.add_row("⚠️  Warning", str(report["warning"]))

    return table


def _render_protection_table(result: ProtectionResult) -> Table:
    table = Table(title="Protection Summary")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Profile", result.profile)
    table.add_row("Method", result.method.upper())
    table.add_row("Epsilon", f"{result.epsilon:.3f}")
    table.add_row("Num steps", str(result.num_steps))
    table.add_row(
        "Alpha", f"{result.alpha:.4f}" if result.alpha is not None else "auto"
    )
    table.add_row(
        "Original prediction",
        (
            str(result.original_prediction)
            if result.original_prediction is not None
            else "N/A"
        ),
    )
    table.add_row(
        "Protected prediction",
        (
            str(result.adversarial_prediction)
            if result.adversarial_prediction is not None
            else "N/A"
        ),
    )
    table.add_row(
        "Classifier success",
        (
            "Yes"
            if result.attack_success is True
            else "No" if result.attack_success is False else "N/A"
        ),
    )
    table.add_row(
        "Output size", f"{result.original_size[0]} x {result.original_size[1]}"
    )
    table.add_row("Device", result.device)
    return table


def _render_batch_table(summary: BatchProtectionSummary) -> Table:
    table = Table(title="Batch Summary")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Input", str(summary.input_dir))
    table.add_row("Output", str(summary.output_dir))
    table.add_row("Profile", summary.profile or "N/A")
    table.add_row("Method", summary.method.upper() if summary.method else "N/A")
    table.add_row("Mode", "Collective" if summary.collective else "Per-image")
    table.add_row(
        "Working size",
        (
            f"{summary.working_size[0]} x {summary.working_size[1]}"
            if summary.working_size is not None
            else "N/A"
        ),
    )
    table.add_row("Processed", str(summary.processed_count))
    table.add_row("Skipped unsupported", str(summary.skipped_unsupported_count))
    table.add_row("Skipped existing", str(summary.skipped_existing_count))
    table.add_row("Failed", str(summary.failed_count))
    return table


def _render_profile_summary_table(
    profile_summaries: dict[str, dict[str, object]],
) -> Table:
    table = Table(title="Profile Summary (⚠️ Protection scores are proxy metrics)")
    table.add_column("Profile", style="cyan")
    table.add_column("Images", style="green")
    table.add_column("Avg PSNR", style="yellow")
    table.add_column("Avg SSIM", style="yellow")
    table.add_column("Avg Protect (proxy)", style="magenta")
    table.add_column("Avg Runtime", style="green")

    for profile, summary in profile_summaries.items():
        table.add_row(
            profile,
            str(summary["image_count"]),
            f"{summary['avg_psnr_db']:.2f}",
            f"{summary['avg_ssim']:.4f}",
            f"{summary['avg_protection_score']:.1f}",
            f"{summary['avg_runtime_sec']:.2f}s",
        )
    return table


def _render_preflight_table(preflight: dict[str, object]) -> Table:
    table = Table(title="LoRA Preflight")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Ready", "Yes" if preflight["ready"] else "No")
    table.add_row("CUDA", "Yes" if preflight["cuda_available"] else "No")
    table.add_row(
        "Missing modules",
        ", ".join(preflight["missing_modules"]) or "None",
    )
    missing_paths = preflight.get("missing_paths", {})
    table.add_row(
        "Missing paths",
        ", ".join(f"{key}={value}" for key, value in missing_paths.items()) or "None",
    )
    invalid_paths = preflight.get("invalid_paths", {})
    table.add_row(
        "Invalid paths",
        ", ".join(f"{key}={value}" for key, value in invalid_paths.items()) or "None",
    )
    table.add_row("Notes", " | ".join(preflight["notes"]) or "None")
    return table


def _render_lora_job_table(jobs: list[dict[str, object]]) -> Table:
    table = Table(title="LoRA Jobs")
    table.add_column("Profile", style="cyan")
    table.add_column("Variant", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Executed", style="green")
    table.add_column("Output", style="magenta")

    for job in jobs:
        table.add_row(
            str(job["profile"]),
            str(job["variant"]),
            str(job["status"]),
            "Yes" if job["executed"] else "No",
            str(job["output_dir"]),
        )
    return table


def _render_subject_layout_table(subject_layout: dict[str, object]) -> Table:
    table = Table(title="Subject Split")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")

    split_counts = subject_layout["split_counts"]
    table.add_row("Subject", str(subject_layout["subject_id"]))
    table.add_row("Root", str(subject_layout["subject_root"]))
    table.add_row(
        "set_A", f"{subject_layout['set_a_dir']} ({split_counts['set_A']} images)"
    )
    table.add_row(
        "set_B", f"{subject_layout['set_b_dir']} ({split_counts['set_B']} images)"
    )
    table.add_row(
        "set_C", f"{subject_layout['set_c_dir']} ({split_counts['set_C']} images)"
    )
    return table


def _render_docker_plan_table(
    *,
    compose_file: Path,
    service_name: str,
    gpu_count: str,
    workspace_dir: Path,
) -> Table:
    table = Table(title="Docker Benchmark Runtime")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Workspace", str(workspace_dir))
    table.add_row("Compose file", str(compose_file))
    table.add_row("Service", service_name)
    table.add_row("GPU count", gpu_count)
    return table


def _parse_profile_sequence(raw_profiles: str | None) -> tuple[str, ...] | None:
    """Parse a comma-separated profile list while preserving order."""
    if raw_profiles is None:
        return None
    profiles = tuple(
        profile.strip() for profile in raw_profiles.split(",") if profile.strip()
    )
    if not profiles:
        raise ValueError("auto-profiles must include at least one profile name.")
    return profiles


def _adaptive_thresholds_met(
    result: ProtectionResult,
    *,
    min_protection_score: float,
    min_ssim: float,
    min_psnr_db: float,
) -> bool:
    """Check whether a result satisfies the requested adaptive floors."""
    return (
        float(result.protection_report.get("protection_score", float("-inf")))
        >= min_protection_score
        and float(result.quality_report.get("ssim", float("-inf"))) >= min_ssim
        and float(result.quality_report.get("psnr_db", float("-inf"))) >= min_psnr_db
    )


@app.command()
def protect(
    input_path: Path = typer.Argument(..., help="Path to input image"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output path"),
    profile: str = typer.Option(
        "balanced",
        "--profile",
        help="Protection profile: safe, balanced, strong, subject, fortress, or blindfold",
    ),
    auto_profiles: str | None = typer.Option(
        None,
        "--auto-profiles",
        help="Comma-separated adaptive profile order, for example safe,balanced,strong,subject,fortress,blindfold",
    ),
    epsilon: float | None = typer.Option(
        None, "--epsilon", "-e", help="Perturbation strength override"
    ),
    method: str | None = typer.Option(
        None,
        "--method",
        "-m",
        help="Attack method override: fgsm, pgd, stylecloak, or blindfold",
    ),
    num_steps: int | None = typer.Option(
        None,
        "--num-steps",
        help="Iteration count override for PGD and StyleCloak",
    ),
    alpha: float | None = typer.Option(
        None,
        "--alpha",
        help="Per-step update size for iterative methods like PGD and StyleCloak",
    ),
    report: Path | None = typer.Option(
        None,
        "--report",
        help="Optional JSON report output path",
    ),
    min_protection_score: float | None = typer.Option(
        None,
        "--min-protection-score",
        help="Enable adaptive mode and require at least this protection score",
    ),
    min_ssim: float | None = typer.Option(
        None,
        "--min-ssim",
        help="Enable adaptive mode and require at least this SSIM value",
    ),
    min_psnr: float | None = typer.Option(
        None,
        "--min-psnr",
        help="Enable adaptive mode and require at least this PSNR value in dB",
    ),
) -> None:
    """Protect a single image while preserving its original resolution."""
    if not input_path.exists():
        console.print(f"[red]Error:[/red] File not found: {input_path}")
        raise typer.Exit(1)

    output_path = (
        output or input_path.parent / f"{input_path.stem}_protected{input_path.suffix}"
    )
    adaptive_mode = any(
        value is not None
        for value in (
            auto_profiles,
            min_protection_score,
            min_ssim,
            min_psnr,
        )
    )
    adaptive_requirements: dict[str, float] | None = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading protection pipeline...", total=None)
        service = ProtectionService()
        progress.update(task, completed=True, description="Protection pipeline ready")

        task = progress.add_task("Generating protected image...", total=None)
        try:
            if adaptive_mode:
                parsed_profiles = _parse_profile_sequence(auto_profiles) or (
                    "safe",
                    "balanced",
                    "strong",
                    "subject",
                    "fortress",
                    "blindfold",
                )
                adaptive_requirements = {
                    "min_protection_score": (
                        25.0 if min_protection_score is None else min_protection_score
                    ),
                    "min_ssim": 0.92 if min_ssim is None else min_ssim,
                    "min_psnr_db": 35.0 if min_psnr is None else min_psnr,
                }
                result = service.protect_file_adaptive(
                    str(input_path),
                    profiles=parsed_profiles,
                    min_protection_score=adaptive_requirements["min_protection_score"],
                    min_ssim=adaptive_requirements["min_ssim"],
                    min_psnr_db=adaptive_requirements["min_psnr_db"],
                    epsilon=epsilon,
                    method=method,
                    num_steps=num_steps,
                    alpha=alpha,
                )
            else:
                result = service.protect_file(
                    str(input_path),
                    profile=profile,
                    epsilon=epsilon,
                    method=method,
                    num_steps=num_steps,
                    alpha=alpha,
                )
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc
        progress.update(task, completed=True, description="Protection generated")

        task = progress.add_task("Saving protected image...", total=None)
        save_image(result.protected_tensor, output_path)
        progress.update(task, completed=True, description=f"Saved to {output_path}")

    console.print(_render_protection_table(result))
    console.print(_render_quality_table(result.quality_report))
    console.print(_render_readability_table(result.protection_report))

    # Print prominent warning about protection score
    if "warning" in result.protection_report:
        console.print(f"\n[yellow]{result.protection_report['warning']}[/yellow]")
        if "disclaimer" in result.protection_report:
            console.print(f"[dim]{result.protection_report['disclaimer']}[/dim]\n")

    if report is not None:
        payload = result.to_report_dict(output_path=output_path)
        if adaptive_requirements is not None:
            payload["adaptive_requirements"] = adaptive_requirements
            payload["adaptive_requirements_met"] = _adaptive_thresholds_met(
                result,
                min_protection_score=adaptive_requirements["min_protection_score"],
                min_ssim=adaptive_requirements["min_ssim"],
                min_psnr_db=adaptive_requirements["min_psnr_db"],
            )
        _write_json_report(report, payload)
        console.print(f"[green]Report saved to:[/green] {report}")
    console.print(f"\n[green]Protected image saved to:[/green] {output_path}")
    if adaptive_requirements is not None and not _adaptive_thresholds_met(
        result,
        min_protection_score=adaptive_requirements["min_protection_score"],
        min_ssim=adaptive_requirements["min_ssim"],
        min_psnr_db=adaptive_requirements["min_psnr_db"],
    ):
        console.print(
            "[red]Adaptive protection requirements were not met after trying all requested profiles.[/red]"
        )
        raise typer.Exit(1)


@app.command()
def analyze(
    original: Path = typer.Argument(..., help="Path to original image"),
    modified: Path = typer.Argument(..., help="Path to modified image"),
    report_path: Path | None = typer.Option(
        None,
        "--report",
        help="Optional JSON report output path",
    ),
) -> None:
    """Analyze two images with the same dimensions."""
    if not original.exists():
        console.print(f"[red]Error:[/red] Original image not found: {original}")
        raise typer.Exit(1)
    if not modified.exists():
        console.print(f"[red]Error:[/red] Modified image not found: {modified}")
        raise typer.Exit(1)

    service = ProtectionService()
    try:
        analysis_report = service.analyze_files(str(original), str(modified))
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(_render_quality_table(analysis_report["quality_report"]))
    console.print(_render_readability_table(analysis_report["protection_report"]))

    # Print prominent warning about protection score
    if "warning" in analysis_report["protection_report"]:
        console.print(
            f"\n[yellow]{analysis_report['protection_report']['warning']}[/yellow]"
        )
        if "disclaimer" in analysis_report["protection_report"]:
            console.print(
                f"[dim]{analysis_report['protection_report']['disclaimer']}[/dim]\n"
            )

    if report_path is not None:
        _write_json_report(
            report_path,
            {
                "original_path": original,
                "modified_path": modified,
                **analysis_report,
            },
        )
        console.print(f"[green]Report saved to:[/green] {report_path}")


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., help="Directory with input images"),
    output_dir: Path = typer.Argument(..., help="Directory for protected images"),
    profile: str = typer.Option(
        "balanced",
        "--profile",
        help="Protection profile: safe, balanced, strong, subject, fortress, or blindfold",
    ),
    epsilon: float | None = typer.Option(
        None, "--epsilon", "-e", help="Perturbation strength override"
    ),
    method: str | None = typer.Option(
        None,
        "--method",
        "-m",
        help="Attack method override: fgsm, pgd, stylecloak, or blindfold",
    ),
    num_steps: int | None = typer.Option(
        None,
        "--num-steps",
        help="Iteration count override for PGD and StyleCloak",
    ),
    alpha: float | None = typer.Option(
        None,
        "--alpha",
        help="Per-step update size for iterative methods like PGD and StyleCloak",
    ),
    recursive: bool = typer.Option(
        False, "--recursive", help="Process nested directories recursively"
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing outputs"
    ),
    collective: bool = typer.Option(
        False,
        "--collective",
        help="Optimize one shared subject-set perturbation across the directory",
    ),
    working_size: int | None = typer.Option(
        None,
        "--working-size",
        help="Common square working size for collective mode",
    ),
    report: Path | None = typer.Option(
        None,
        "--report",
        help="Optional JSON report output path",
    ),
) -> None:
    """Protect all supported images in a directory."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading protection pipeline...", total=None)
        service = ProtectionService()
        progress.update(task, completed=True, description="Protection pipeline ready")

        task = progress.add_task("Processing image directory...", total=None)
        try:
            summary = service.protect_directory(
                input_dir,
                output_dir,
                profile=profile,
                epsilon=epsilon,
                method=method,
                num_steps=num_steps,
                alpha=alpha,
                recursive=recursive,
                overwrite=overwrite,
                collective=collective,
                working_size=working_size,
            )
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc
        progress.update(task, completed=True, description="Batch processing completed")

    console.print(_render_batch_table(summary))

    # Print prominent warning about protection scores in batch
    from auralock.core.metrics import (
        PROTECTION_SCORE_DISCLAIMER,
        PROTECTION_SCORE_WARNING,
    )

    console.print(f"\n[yellow]{PROTECTION_SCORE_WARNING}[/yellow]")
    console.print(f"[dim]{PROTECTION_SCORE_DISCLAIMER}[/dim]\n")

    if report is not None:
        _write_json_report(report, summary.to_report_dict())
        console.print(f"[green]Report saved to:[/green] {report}")
    if summary.failures:
        console.print("\n[red]Failures:[/red]")
        for failure in summary.failures:
            console.print(f"- {failure}")


@app.command()
def benchmark(
    input_path: Path = typer.Argument(
        ..., help="Input image or directory to benchmark"
    ),
    profiles: str = typer.Option(
        "safe,balanced,strong",
        "--profiles",
        help="Comma-separated profile list",
    ),
    recursive: bool = typer.Option(
        False, "--recursive", help="Benchmark nested directories recursively"
    ),
    report: Path | None = typer.Option(
        None,
        "--report",
        help="Optional JSON report output path",
    ),
) -> None:
    """Benchmark one or more protection profiles on an image or directory."""
    if not input_path.exists():
        console.print(f"[red]Error:[/red] Input not found: {input_path}")
        raise typer.Exit(1)

    profile_names = tuple(
        profile.strip() for profile in profiles.split(",") if profile.strip()
    )
    if not profile_names:
        console.print("[red]Error:[/red] At least one profile is required.")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading protection pipeline...", total=None)
        service = ProtectionService()
        progress.update(task, completed=True, description="Protection pipeline ready")

        task = progress.add_task("Running benchmark...", total=None)
        try:
            if input_path.is_dir():
                summary = service.benchmark_directory(
                    input_path,
                    profiles=profile_names,
                    recursive=recursive,
                )
            else:
                summary = service.benchmark_file(
                    input_path,
                    profiles=profile_names,
                )
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc
        progress.update(task, completed=True, description="Benchmark completed")

    console.print(_render_profile_summary_table(summary.profile_summaries))

    # Print prominent warning about protection scores in benchmark summary
    from auralock.core.metrics import (
        PROTECTION_SCORE_DISCLAIMER,
        PROTECTION_SCORE_WARNING,
    )

    console.print(f"\n[yellow]{PROTECTION_SCORE_WARNING}[/yellow]")
    console.print(f"[dim]{PROTECTION_SCORE_DISCLAIMER}[/dim]\n")

    if report is not None:
        _write_json_report(report, summary.to_report_dict())
        console.print(f"[green]Report saved to:[/green] {report}")


@app.command("benchmark-lora")
def benchmark_lora(
    input_path: Path = typer.Argument(
        ...,
        help="Input image or directory for real DreamBooth/LoRA benchmark planning",
    ),
    work_dir: Path = typer.Option(
        Path("benchmark_runs/lora"),
        "--work-dir",
        help="Working directory for datasets, runs, and samples",
    ),
    profiles: str = typer.Option(
        "safe,balanced,strong",
        "--profiles",
        help="Comma-separated profile list",
    ),
    recursive: bool = typer.Option(
        False, "--recursive", help="Process nested directories recursively"
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Actually run training/inference if preflight is ready",
    ),
    pretrained_model_path: Path = typer.Option(
        ...,
        "--pretrained-model-path",
        help="Stable Diffusion checkpoint path or model directory",
    ),
    script_path: Path = typer.Option(
        ...,
        "--script-path",
        help="Path to train_dreambooth_lora.py",
    ),
    infer_script_path: Path | None = typer.Option(
        None,
        "--infer-script-path",
        help="Optional inference script path for post-training sampling",
    ),
    instance_prompt: str = typer.Option(
        ...,
        "--instance-prompt",
        help="Prompt describing the protected training images",
    ),
    class_prompt: str = typer.Option(
        ...,
        "--class-prompt",
        help="Class prompt used for prior preservation",
    ),
    resolution: int = typer.Option(512, "--resolution", help="Training resolution"),
    train_batch_size: int = typer.Option(
        1, "--train-batch-size", help="DreamBooth train batch size"
    ),
    learning_rate: float = typer.Option(
        1e-4, "--learning-rate", help="DreamBooth learning rate"
    ),
    max_train_steps: int = typer.Option(
        400, "--max-train-steps", help="DreamBooth max train steps"
    ),
    report: Path | None = typer.Option(
        None,
        "--report",
        help="Optional JSON manifest/report output path",
    ),
) -> None:
    """Plan or execute a real DreamBooth/LoRA benchmark workflow."""
    if not input_path.exists():
        console.print(f"[red]Error:[/red] Input not found: {input_path}")
        raise typer.Exit(1)

    profile_names = tuple(
        profile.strip() for profile in profiles.split(",") if profile.strip()
    )
    if not profile_names:
        console.print("[red]Error:[/red] At least one profile is required.")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Preparing LoRA benchmark harness...", total=None)
        harness = LoraBenchmarkHarness()
        try:
            manifest = harness.run(
                input_path,
                work_dir=work_dir,
                profiles=profile_names,
                recursive=recursive,
                execute=execute,
                instance_prompt=instance_prompt,
                class_prompt=class_prompt,
                pretrained_model_path=pretrained_model_path,
                script_path=script_path,
                infer_script_path=infer_script_path,
                resolution=resolution,
                train_batch_size=train_batch_size,
                learning_rate=learning_rate,
                max_train_steps=max_train_steps,
            )
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc
        progress.update(task, completed=True, description="LoRA benchmark prepared")

    manifest_payload = manifest.to_report_dict()
    console.print(_render_preflight_table(manifest_payload["preflight"]))
    console.print(_render_lora_job_table(manifest_payload["jobs"]))
    console.print(
        f"[green]Prepared {len(manifest_payload['jobs'])} LoRA job(s)[/green] in {manifest.work_dir}"
    )
    if report is not None:
        _write_json_report(report, manifest_payload)
        console.print(f"[green]Report saved to:[/green] {report}")
    if execute:
        job_statuses = [str(job["status"]) for job in manifest_payload["jobs"]]
        if (not manifest.preflight.ready) or any(
            status != "completed" for status in job_statuses
        ):
            console.print(
                "[red]LoRA benchmark execution did not complete successfully.[/red]"
            )
            raise typer.Exit(1)


@app.command("benchmark-antidreambooth")
def benchmark_antidreambooth(
    subject_root: Path = typer.Argument(
        ...,
        help="Subject directory containing set_A, set_B, and set_C",
    ),
    work_dir: Path = typer.Option(
        Path("benchmark_runs/antidreambooth"),
        "--work-dir",
        help="Working directory for datasets, runs, and samples",
    ),
    profiles: str = typer.Option(
        "safe,balanced,strong",
        "--profiles",
        help="Comma-separated protection profile list for set_B",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Actually run DreamBooth training/inference if preflight is ready",
    ),
    pretrained_model_path: Path = typer.Option(
        ...,
        "--pretrained-model-path",
        help="Stable Diffusion checkpoint path or model directory",
    ),
    script_path: Path = typer.Option(
        DEFAULT_ANTI_DREAMBOOTH_TRAIN_SCRIPT,
        "--script-path",
        help="Path to Anti-DreamBooth train_dreambooth.py",
    ),
    infer_script_path: Path | None = typer.Option(
        DEFAULT_ANTI_DREAMBOOTH_INFER_SCRIPT,
        "--infer-script-path",
        help="Optional Anti-DreamBooth inference script path",
    ),
    instance_prompt: str = typer.Option(
        DEFAULT_ANTI_DREAMBOOTH_INSTANCE_PROMPT,
        "--instance-prompt",
        help="Prompt describing the protected subject images",
    ),
    class_prompt: str = typer.Option(
        DEFAULT_ANTI_DREAMBOOTH_CLASS_PROMPT,
        "--class-prompt",
        help="Class prompt used for prior preservation",
    ),
    resolution: int = typer.Option(512, "--resolution", help="Training resolution"),
    train_batch_size: int = typer.Option(
        1, "--train-batch-size", help="DreamBooth train batch size"
    ),
    learning_rate: float = typer.Option(
        1e-4, "--learning-rate", help="DreamBooth learning rate"
    ),
    max_train_steps: int = typer.Option(
        400, "--max-train-steps", help="DreamBooth max train steps"
    ),
    report: Path | None = typer.Option(
        None,
        "--report",
        help="Optional JSON manifest/report output path",
    ),
) -> None:
    """Plan or execute an Anti-DreamBooth-style subject split benchmark."""
    if not subject_root.exists():
        console.print(f"[red]Error:[/red] Input not found: {subject_root}")
        raise typer.Exit(1)

    profile_names = tuple(
        profile.strip() for profile in profiles.split(",") if profile.strip()
    )
    if not profile_names:
        console.print("[red]Error:[/red] At least one profile is required.")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Preparing Anti-DreamBooth subject benchmark...", total=None
        )
        harness = AntiDreamBoothSubjectBenchmarkHarness()
        try:
            manifest = harness.run(
                subject_root,
                work_dir=work_dir,
                profiles=profile_names,
                execute=execute,
                instance_prompt=instance_prompt,
                class_prompt=class_prompt,
                pretrained_model_path=pretrained_model_path,
                script_path=script_path,
                infer_script_path=infer_script_path,
                resolution=resolution,
                train_batch_size=train_batch_size,
                learning_rate=learning_rate,
                max_train_steps=max_train_steps,
            )
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc
        progress.update(
            task,
            completed=True,
            description="Anti-DreamBooth subject benchmark prepared",
        )

    manifest_payload = manifest.to_report_dict()
    console.print(_render_subject_layout_table(manifest_payload["subject_layout"]))
    console.print(_render_preflight_table(manifest_payload["preflight"]))
    console.print(_render_lora_job_table(manifest_payload["jobs"]))
    console.print(
        f"[green]Prepared {len(manifest_payload['jobs'])} subject benchmark job(s)[/green] in {manifest.work_dir}"
    )
    if report is not None:
        _write_json_report(report, manifest_payload)
        console.print(f"[green]Report saved to:[/green] {report}")
    if execute:
        job_statuses = [str(job["status"]) for job in manifest_payload["jobs"]]
        if (not manifest.preflight.ready) or any(
            status != "completed" for status in job_statuses
        ):
            console.print(
                "[red]Anti-DreamBooth benchmark execution did not complete successfully.[/red]"
            )
            raise typer.Exit(1)


@app.command("benchmark-lora-docker")
def benchmark_lora_docker(
    input_path: Path = typer.Argument(
        ...,
        help="Input image or directory for Dockerized DreamBooth/LoRA benchmark execution",
    ),
    work_dir: Path = typer.Option(
        Path("benchmark_runs/lora_docker"),
        "--work-dir",
        help="Working directory for datasets, runs, and samples",
    ),
    profiles: str = typer.Option(
        "safe,balanced,strong",
        "--profiles",
        help="Comma-separated profile list",
    ),
    recursive: bool = typer.Option(
        False, "--recursive", help="Process nested directories recursively"
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Actually run training/inference inside the Docker benchmark runtime",
    ),
    pretrained_model_path: Path = typer.Option(
        ...,
        "--pretrained-model-path",
        help="Host path to a Diffusers checkpoint directory",
    ),
    script_path: Path = typer.Option(
        ...,
        "--script-path",
        help="Host path to train_dreambooth_lora.py",
    ),
    infer_script_path: Path | None = typer.Option(
        None,
        "--infer-script-path",
        help="Optional host path to an inference script",
    ),
    instance_prompt: str = typer.Option(
        ...,
        "--instance-prompt",
        help="Prompt describing the protected training images",
    ),
    class_prompt: str = typer.Option(
        ...,
        "--class-prompt",
        help="Class prompt used for prior preservation",
    ),
    resolution: int = typer.Option(512, "--resolution", help="Training resolution"),
    train_batch_size: int = typer.Option(
        1, "--train-batch-size", help="DreamBooth train batch size"
    ),
    learning_rate: float = typer.Option(
        1e-4, "--learning-rate", help="DreamBooth learning rate"
    ),
    max_train_steps: int = typer.Option(
        400, "--max-train-steps", help="DreamBooth max train steps"
    ),
    report: Path | None = typer.Option(
        None,
        "--report",
        help="Optional JSON report output path inside the workspace",
    ),
    compose_file: Path = typer.Option(
        DEFAULT_COMPOSE_FILE,
        "--compose-file",
        help="Docker Compose file for the benchmark runtime",
    ),
    service_name: str = typer.Option(
        DEFAULT_SERVICE_NAME,
        "--service-name",
        help="Docker Compose service name for the benchmark runtime",
    ),
    workspace_dir: Path = typer.Option(
        Path.cwd(),
        "--workspace-dir",
        help="Workspace root mounted into the Docker benchmark runtime",
    ),
    gpu_count: str = typer.Option(
        "all",
        "--gpu-count",
        help="GPU count reserved for the container: 'all' or a positive integer",
    ),
    base_image: str = typer.Option(
        DEFAULT_BENCHMARK_BASE_IMAGE,
        "--base-image",
        help="Base image used to build the benchmark runtime",
    ),
    skip_build: bool = typer.Option(
        False,
        "--skip-build",
        help="Skip rebuilding the Docker benchmark image",
    ),
    skip_gpu_check: bool = typer.Option(
        False,
        "--skip-gpu-check",
        help="Skip host GPU smoke test before execution",
    ),
) -> None:
    """Launch the LoRA benchmark inside the Docker GPU runtime."""
    profile_names = tuple(
        profile.strip() for profile in profiles.split(",") if profile.strip()
    )
    if not profile_names:
        console.print("[red]Error:[/red] At least one profile is required.")
        raise typer.Exit(1)

    try:
        plan = build_docker_lora_benchmark_plan(
            DockerLoraBenchmarkConfig(
                workspace_dir=workspace_dir,
                input_path=input_path,
                work_dir=work_dir,
                pretrained_model_path=pretrained_model_path,
                script_path=script_path,
                infer_script_path=infer_script_path,
                instance_prompt=instance_prompt,
                class_prompt=class_prompt,
                profiles=profile_names,
                recursive=recursive,
                execute=execute,
                resolution=resolution,
                train_batch_size=train_batch_size,
                learning_rate=learning_rate,
                max_train_steps=max_train_steps,
                report=report,
                compose_file=compose_file,
                service_name=service_name,
                gpu_count=gpu_count,
                base_image=base_image,
            )
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(
        _render_docker_plan_table(
            compose_file=(
                compose_file
                if compose_file.is_absolute()
                else (workspace_dir.resolve() / compose_file).resolve()
            ),
            service_name=service_name,
            gpu_count=gpu_count,
            workspace_dir=workspace_dir.resolve(),
        )
    )

    process_env = os.environ.copy()
    process_env.update(plan.environment)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            if not skip_build:
                task = progress.add_task(
                    "Building Docker benchmark runtime...", total=None
                )
                subprocess.run(
                    plan.build_command,
                    cwd=workspace_dir,
                    env=process_env,
                    check=True,
                )
                progress.update(
                    task,
                    completed=True,
                    description="Docker benchmark runtime is ready",
                )

            if execute and not skip_gpu_check:
                task = progress.add_task("Running Docker GPU smoke test...", total=None)
                subprocess.run(
                    plan.gpu_check_command,
                    cwd=workspace_dir,
                    env=process_env,
                    check=True,
                )
                progress.update(
                    task,
                    completed=True,
                    description="Docker GPU runtime is available",
                )

            task = progress.add_task("Running LoRA benchmark in Docker...", total=None)
            subprocess.run(
                plan.run_command,
                cwd=workspace_dir,
                env=process_env,
                check=True,
            )
            progress.update(
                task,
                completed=True,
                description="Docker LoRA benchmark completed",
            )
    except FileNotFoundError as exc:
        console.print(
            "[red]Error:[/red] Docker is unavailable on PATH. Install Docker Desktop and retry."
        )
        raise typer.Exit(1) from exc
    except subprocess.CalledProcessError as exc:
        console.print(
            "[red]Error:[/red] Docker benchmark command failed "
            f"(exit code {exc.returncode})."
        )
        raise typer.Exit(exc.returncode or 1) from exc


@app.command()
def demo() -> None:
    """Run a quick demo using a synthetic image."""
    service = ProtectionService()
    image = torch.rand(1, 3, 256, 256)

    table = Table(title="StyleCloak Demo")
    table.add_column("Epsilon", style="cyan")
    table.add_column("Protection", style="green")
    table.add_column("PSNR", style="yellow")
    table.add_column("SSIM", style="yellow")

    for epsilon in (0.01, 0.02, 0.03):
        result = service.protect_tensor(image, epsilon=epsilon, method="stylecloak")
        table.add_row(
            f"{epsilon:.2f}",
            f"{result.protection_report['protection_score']:.1f}",
            f"{result.quality_report['psnr_db']:.2f} dB",
            f"{result.quality_report['ssim']:.4f}",
        )

    console.print(table)


@app.command()
def webui(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind the web UI"),
    port: int = typer.Option(7860, "--port", help="Port for the web UI"),
) -> None:
    """Launch the optional Gradio UI."""
    from auralock.ui import launch_app

    launch_app(host=host, port=port)


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"AuraLock version: [bold]{__version__}[/bold]")


def main() -> None:
    """Entry point for console scripts."""
    app()


if __name__ == "__main__":
    main()
