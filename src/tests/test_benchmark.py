"""Tests for benchmark summaries and CLI benchmark flow."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from auralock.cli import app

from .test_pipeline import RecordingClassifier
from .test_stylecloak import DummyStyleFeatureExtractor


def _create_image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color=color).save(path)


def test_protection_service_benchmark_directory_summarizes_profiles(tmp_path: Path):
    """Benchmark mode should produce per-profile aggregates over image inputs."""
    from auralock.core.pipeline import ImageNetModelAdapter
    from auralock.services import ProtectionService

    input_dir = tmp_path / "input"
    _create_image(input_dir / "a.png", "#335577")
    _create_image(input_dir / "b.png", "#775533")

    service = ProtectionService(
        model=ImageNetModelAdapter(RecordingClassifier()),
        style_feature_extractor=DummyStyleFeatureExtractor(),
    )
    summary = service.benchmark_directory(
        input_dir,
        profiles=("safe", "balanced"),
    )

    assert summary.image_count == 2
    assert len(summary.entries) == 4
    assert set(summary.profile_summaries) == {"safe", "balanced"}
    assert summary.profile_summaries["safe"]["image_count"] == 2
    assert "avg_psnr_db" in summary.profile_summaries["balanced"]


def test_benchmark_cli_writes_report(monkeypatch, tmp_path: Path):
    """The CLI should accept benchmark arguments and write a structured report."""
    calls: dict[str, object] = {}

    class FakeSummary:
        profile_summaries = {
            "safe": {
                "image_count": 1,
                "avg_psnr_db": 42.0,
                "avg_ssim": 0.96,
                "avg_protection_score": 8.5,
                "avg_runtime_sec": 1.2,
            }
        }

        def to_report_dict(self):
            return {
                "input_path": str(tmp_path / "input"),
                "image_count": 1,
                "entries": [],
                "profile_summaries": self.profile_summaries,
            }

    class FakeService:
        def benchmark_directory(self, input_path, **kwargs):
            calls["input_path"] = input_path
            calls["kwargs"] = kwargs
            return FakeSummary()

        def benchmark_file(self, input_path, **kwargs):
            calls["input_path"] = input_path
            calls["kwargs"] = kwargs
            return FakeSummary()

    monkeypatch.setattr("auralock.cli.ProtectionService", FakeService)

    runner = CliRunner()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    report_path = tmp_path / "benchmark.json"

    result = runner.invoke(
        app,
        [
            "benchmark",
            str(input_dir),
            "--profiles",
            "safe,balanced",
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert "Profile Summary" in result.output
    assert report_path.exists()
    assert calls["input_path"] == input_dir
    assert calls["kwargs"]["profiles"] == ("safe", "balanced")
