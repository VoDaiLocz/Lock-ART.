"""Tests for Anti-DreamBooth-style subject split benchmarking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from auralock.cli import app


def _make_subject_split(subject_root: Path, *, per_split: int = 2) -> None:
    for split_name in ("set_A", "set_B", "set_C"):
        split_dir = subject_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for index in range(per_split):
            (split_dir / f"{split_name.lower()}_{index}.png").write_bytes(b"fake")


def test_resolve_subject_layout_requires_expected_split_dirs(tmp_path: Path):
    """The subject benchmark should require set_A/set_B/set_C directories."""
    from auralock.benchmarks.antidreambooth import resolve_subject_layout

    subject_root = tmp_path / "n000050"
    (subject_root / "set_A").mkdir(parents=True)
    (subject_root / "set_B").mkdir(parents=True)

    with pytest.raises(ValueError, match="set_A, set_B, set_C"):
        resolve_subject_layout(subject_root)


def test_subject_benchmark_harness_dry_run_uses_set_b_as_published_split(
    tmp_path: Path,
):
    """Dry-run manifests should align clean/protected variants to the paper split."""
    from auralock.benchmarks.antidreambooth import (
        AntiDreamBoothSubjectBenchmarkHarness,
    )

    class DummyProtectionService:
        def protect_file(self, *args, **kwargs):
            raise AssertionError("dry-run should not invoke protect_file")

    subject_root = tmp_path / "n000050"
    _make_subject_split(subject_root, per_split=2)
    script_path = tmp_path / "train_dreambooth.py"
    script_path.write_text("print('train')", encoding="utf-8")
    model_dir = tmp_path / "sd15"
    model_dir.mkdir()

    manifest = AntiDreamBoothSubjectBenchmarkHarness(
        protection_service=DummyProtectionService()
    ).run(
        subject_root,
        work_dir=tmp_path / "work",
        profiles=("safe",),
        pretrained_model_path=model_dir,
        script_path=script_path,
        infer_script_path=None,
    )

    assert manifest.execute is False
    assert manifest.subject_layout.split_counts == {
        "set_A": 2,
        "set_B": 2,
        "set_C": 2,
    }
    assert [job["profile"] for job in manifest.jobs] == ["clean", "safe"]
    assert manifest.jobs[0]["variant"] == "clean_published"
    assert str(manifest.jobs[0]["published_dir"]).endswith("datasets\\published\\clean")
    assert str(manifest.jobs[1]["published_dir"]).endswith("datasets\\published\\safe")


def test_subject_benchmark_harness_executes_collective_protection_for_set_b(
    monkeypatch,
    tmp_path: Path,
):
    """Execution mode should materialize protected set_B through collective protection."""
    from auralock.benchmarks.antidreambooth import (
        AntiDreamBoothSubjectBenchmarkHarness,
    )
    from auralock.benchmarks.lora import LoraPreflightReport

    calls: list[dict[str, object]] = []

    class FakeProtectionService:
        def protect_directory(self, input_dir, output_dir, **kwargs):
            calls.append(
                {
                    "input_dir": Path(input_dir),
                    "output_dir": Path(output_dir),
                    "kwargs": kwargs,
                }
            )
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for image_path in Path(input_dir).glob("*.png"):
                (output_dir / image_path.name).write_bytes(b"protected")

    def fake_run(*args, **kwargs):
        return None

    monkeypatch.setattr("auralock.benchmarks.antidreambooth.subprocess.run", fake_run)
    monkeypatch.setattr(
        "auralock.benchmarks.antidreambooth.evaluate_lora_preflight",
        lambda required_paths: LoraPreflightReport(
            ready=True,
            cuda_available=False,
            missing_modules=[],
            missing_paths={},
            invalid_paths={},
            notes=[],
        ),
    )

    subject_root = tmp_path / "n000050"
    _make_subject_split(subject_root, per_split=2)
    script_path = tmp_path / "train_dreambooth.py"
    script_path.write_text("print('train')", encoding="utf-8")
    model_dir = tmp_path / "sd15"
    model_dir.mkdir()

    manifest = AntiDreamBoothSubjectBenchmarkHarness(
        protection_service=FakeProtectionService()
    ).run(
        subject_root,
        work_dir=tmp_path / "work",
        profiles=("subject",),
        execute=True,
        pretrained_model_path=model_dir,
        script_path=script_path,
        infer_script_path=None,
        resolution=384,
    )

    assert manifest.jobs[1]["status"] == "completed"
    assert len(calls) == 1
    assert calls[0]["input_dir"] == subject_root / "set_B"
    assert calls[0]["kwargs"]["profile"] == "subject"
    assert calls[0]["kwargs"]["collective"] is True
    assert calls[0]["kwargs"]["working_size"] == (384, 384)


def test_benchmark_antidreambooth_cli_writes_manifest(monkeypatch, tmp_path: Path):
    """The CLI should emit a subject benchmark manifest in dry-run mode."""
    from auralock.benchmarks.antidreambooth import (
        AntiDreamBoothBenchmarkManifest,
        AntiDreamBoothSubjectLayout,
    )
    from auralock.benchmarks.lora import LoraPreflightReport

    class FakeHarness:
        def run(
            self,
            subject_root,
            *,
            work_dir,
            profiles,
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
            subject_root = Path(subject_root)
            return AntiDreamBoothBenchmarkManifest(
                subject_layout=AntiDreamBoothSubjectLayout(
                    subject_root=subject_root,
                    subject_id=subject_root.name,
                    set_a_dir=subject_root / "set_A",
                    set_b_dir=subject_root / "set_B",
                    set_c_dir=subject_root / "set_C",
                    set_a_images=[subject_root / "set_A" / "a.png"],
                    set_b_images=[subject_root / "set_B" / "b.png"],
                    set_c_images=[subject_root / "set_C" / "c.png"],
                ),
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
                        "variant": "clean_published",
                        "status": "planned",
                        "executed": False,
                        "output_dir": str(Path(work_dir) / "runs" / "clean"),
                    }
                ],
                notes=["paper-style split"],
            )

    monkeypatch.setattr(
        "auralock.cli.AntiDreamBoothSubjectBenchmarkHarness", FakeHarness
    )

    runner = CliRunner()
    subject_root = tmp_path / "n000050"
    _make_subject_split(subject_root, per_split=1)
    report_path = tmp_path / "subject-manifest.json"

    result = runner.invoke(
        app,
        [
            "benchmark-antidreambooth",
            str(subject_root),
            "--work-dir",
            str(tmp_path / "work"),
            "--pretrained-model-path",
            str(tmp_path / "sd15"),
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["benchmark_mode"] == "antidreambooth_subject_split"
    assert payload["subject_layout"]["split_counts"] == {
        "set_A": 1,
        "set_B": 1,
        "set_C": 1,
    }
    assert payload["profiles"] == ["safe", "balanced", "strong"]


def test_benchmark_antidreambooth_cli_fails_when_execute_is_blocked(
    monkeypatch,
    tmp_path: Path,
):
    """Execution mode should fail when subject benchmark preflight is blocked."""
    from auralock.benchmarks.antidreambooth import (
        AntiDreamBoothBenchmarkManifest,
        AntiDreamBoothSubjectLayout,
    )
    from auralock.benchmarks.lora import LoraPreflightReport

    class FakeHarness:
        def run(
            self,
            subject_root,
            *,
            work_dir,
            profiles,
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
            subject_root = Path(subject_root)
            return AntiDreamBoothBenchmarkManifest(
                subject_layout=AntiDreamBoothSubjectLayout(
                    subject_root=subject_root,
                    subject_id=subject_root.name,
                    set_a_dir=subject_root / "set_A",
                    set_b_dir=subject_root / "set_B",
                    set_c_dir=subject_root / "set_C",
                    set_a_images=[subject_root / "set_A" / "a.png"],
                    set_b_images=[subject_root / "set_B" / "b.png"],
                    set_c_images=[subject_root / "set_C" / "c.png"],
                ),
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
                        "variant": "clean_published",
                        "status": "blocked_preflight",
                        "executed": False,
                        "output_dir": str(Path(work_dir) / "runs" / "clean"),
                    }
                ],
                notes=["paper-style split"],
            )

    monkeypatch.setattr(
        "auralock.cli.AntiDreamBoothSubjectBenchmarkHarness", FakeHarness
    )

    runner = CliRunner()
    subject_root = tmp_path / "n000050"
    _make_subject_split(subject_root, per_split=1)
    report_path = tmp_path / "subject-manifest.json"

    result = runner.invoke(
        app,
        [
            "benchmark-antidreambooth",
            str(subject_root),
            "--execute",
            "--work-dir",
            str(tmp_path / "work"),
            "--pretrained-model-path",
            str(tmp_path / "sd15"),
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 1
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["jobs"][0]["status"] == "blocked_preflight"
