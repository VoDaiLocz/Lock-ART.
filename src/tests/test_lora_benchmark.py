"""Tests for DreamBooth/LoRA benchmark harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

from auralock.cli import app


def test_lora_preflight_reports_missing_dependencies_and_cuda():
    """Preflight should make missing runtime requirements explicit."""
    from auralock.benchmarks.lora import evaluate_lora_preflight

    report = evaluate_lora_preflight(
        required_modules=("diffusers", "accelerate"),
        module_probe=lambda name: False,
        cuda_available=False,
    )

    assert report.ready is False
    assert report.cuda_available is False
    assert report.missing_modules == ["diffusers", "accelerate"]
    assert report.missing_paths == {}
    assert report.invalid_paths == {}


def test_lora_preflight_reports_missing_paths(tmp_path: Path):
    """Preflight should identify missing script/model paths explicitly."""
    from auralock.benchmarks.lora import evaluate_lora_preflight

    report = evaluate_lora_preflight(
        required_modules=(),
        required_paths={"script_path": tmp_path / "missing.py"},
        cuda_available=True,
    )

    assert report.ready is False
    assert report.cuda_available is True
    assert report.missing_modules == []
    assert report.missing_paths == {"script_path": str(tmp_path / "missing.py")}
    assert report.invalid_paths == {}


def test_lora_preflight_reports_invalid_model_directory(tmp_path: Path):
    """Preflight should reject paths that exist but are not valid Diffusers checkpoints."""
    from auralock.benchmarks.lora import evaluate_lora_preflight

    invalid_model_dir = tmp_path / "not-a-diffusers-model"
    invalid_model_dir.mkdir()

    report = evaluate_lora_preflight(
        required_modules=(),
        required_paths={"pretrained_model_path": invalid_model_dir},
        cuda_available=True,
    )

    assert report.ready is False
    assert report.missing_paths == {}
    assert report.invalid_paths == {
        "pretrained_model_path": (
            f"{invalid_model_dir} must be a Diffusers model directory "
            "containing model_index.json."
        )
    }


def test_lora_command_builder_generates_expected_training_invocation(tmp_path: Path):
    """The harness should build a deterministic accelerate launch command."""
    from auralock.benchmarks.lora import LoraBenchmarkConfig, build_lora_train_command

    config = LoraBenchmarkConfig(
        script_path=tmp_path / "train_dreambooth_lora.py",
        pretrained_model_path=tmp_path / "sd15",
        instance_prompt="a sks painting",
        class_prompt="a painting",
        resolution=512,
        train_batch_size=1,
        learning_rate=1e-4,
        max_train_steps=400,
    )

    command = build_lora_train_command(
        config,
        instance_data_dir=tmp_path / "protected",
        class_data_dir=tmp_path / "class",
        output_dir=tmp_path / "runs" / "safe",
    )

    assert command[:4] == [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        str(config.script_path),
    ]
    assert "--instance_data_dir" in command
    assert "--output_dir" in command
    assert str(tmp_path / "protected") in command


def test_lora_harness_dry_run_includes_clean_baseline(tmp_path: Path):
    """Dry-run manifests should include a clean baseline alongside protected profiles."""
    from auralock.benchmarks.lora import LoraBenchmarkHarness

    class DummyProtectionService:
        def protect_file(self, *args, **kwargs):
            raise AssertionError("dry-run should not invoke protect_file")

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.png").write_bytes(b"fake")
    script_path = tmp_path / "train_dreambooth_lora.py"
    script_path.write_text("print('train')", encoding="utf-8")
    model_dir = tmp_path / "sd15"
    model_dir.mkdir()

    manifest = LoraBenchmarkHarness(protection_service=DummyProtectionService()).run(
        input_dir,
        work_dir=tmp_path / "work",
        profiles=("safe",),
        instance_prompt="a sks painting",
        class_prompt="a painting",
        pretrained_model_path=model_dir,
        script_path=script_path,
    )

    assert manifest.execute is False
    assert [job["profile"] for job in manifest.jobs] == ["clean", "safe"]
    assert manifest.jobs[0]["variant"] == "clean"
    assert manifest.jobs[1]["variant"] == "protected"


def test_benchmark_lora_cli_writes_dry_run_manifest(monkeypatch, tmp_path: Path):
    """The CLI should support a dry-run that writes a manifest even without training."""
    from auralock.benchmarks.lora import LoraBenchmarkManifest, LoraPreflightReport

    class FakeHarness:
        def run(
            self,
            input_path,
            *,
            work_dir,
            profiles,
            recursive,
            execute,
            instance_prompt,
            class_prompt,
            pretrained_model_path,
            script_path,
            infer_script_path=None,
            resolution=512,
            train_batch_size=1,
            learning_rate=1e-4,
            max_train_steps=400,
        ):
            assert execute is False
            return LoraBenchmarkManifest(
                input_path=Path(input_path),
                work_dir=Path(work_dir),
                profiles=list(profiles),
                execute=execute,
                preflight=LoraPreflightReport(
                    ready=False,
                    cuda_available=False,
                    missing_modules=["diffusers"],
                    missing_paths={},
                    invalid_paths={},
                    notes=["missing runtime"],
                ),
                jobs=[
                    {
                        "profile": "clean",
                        "variant": "clean",
                        "status": "planned",
                        "executed": False,
                        "output_dir": str(Path(work_dir) / "runs" / "clean"),
                    }
                ],
            )

    monkeypatch.setattr("auralock.cli.LoraBenchmarkHarness", FakeHarness)

    runner = CliRunner()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.png").write_bytes(b"fake")
    report_path = tmp_path / "lora-manifest.json"

    result = runner.invoke(
        app,
        [
            "benchmark-lora",
            str(input_dir),
            "--work-dir",
            str(tmp_path / "work"),
            "--pretrained-model-path",
            str(tmp_path / "sd15"),
            "--script-path",
            str(tmp_path / "train_dreambooth_lora.py"),
            "--instance-prompt",
            "a sks painting",
            "--class-prompt",
            "a painting",
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["execute"] is False
    assert payload["profiles"] == ["safe", "balanced", "strong"]
    assert payload["preflight"]["ready"] is False


def test_benchmark_lora_cli_fails_when_execute_is_blocked(
    monkeypatch,
    tmp_path: Path,
):
    """Execution mode should return a failing exit code when preflight blocks it."""
    from auralock.benchmarks.lora import LoraBenchmarkManifest, LoraPreflightReport

    class FakeHarness:
        def run(
            self,
            input_path,
            *,
            work_dir,
            profiles,
            recursive,
            execute,
            instance_prompt,
            class_prompt,
            pretrained_model_path,
            script_path,
            infer_script_path=None,
            resolution=512,
            train_batch_size=1,
            learning_rate=1e-4,
            max_train_steps=400,
        ):
            return LoraBenchmarkManifest(
                input_path=Path(input_path),
                work_dir=Path(work_dir),
                profiles=list(profiles),
                execute=execute,
                preflight=LoraPreflightReport(
                    ready=False,
                    cuda_available=False,
                    missing_modules=["diffusers"],
                    missing_paths={"script_path": str(Path(script_path))},
                    invalid_paths={},
                    notes=["missing runtime"],
                ),
                jobs=[
                    {
                        "profile": "clean",
                        "variant": "clean",
                        "status": "blocked_preflight",
                        "executed": False,
                        "output_dir": str(Path(work_dir) / "runs" / "clean"),
                    }
                ],
            )

    monkeypatch.setattr("auralock.cli.LoraBenchmarkHarness", FakeHarness)

    runner = CliRunner()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.png").write_bytes(b"fake")
    report_path = tmp_path / "lora-manifest.json"

    result = runner.invoke(
        app,
        [
            "benchmark-lora",
            str(input_dir),
            "--execute",
            "--work-dir",
            str(tmp_path / "work"),
            "--pretrained-model-path",
            str(tmp_path / "sd15"),
            "--script-path",
            str(tmp_path / "train_dreambooth_lora.py"),
            "--instance-prompt",
            "a sks painting",
            "--class-prompt",
            "a painting",
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 1
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["jobs"][0]["status"] == "blocked_preflight"
