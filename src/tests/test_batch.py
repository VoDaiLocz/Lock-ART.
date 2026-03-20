"""Tests for batch processing and CLI integration."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from auralock.cli import app

from .test_pipeline import RecordingClassifier, TinyStyleFeatureExtractor


def _create_image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 64), color=color).save(path)


def test_protection_service_batch_preserves_tree_and_skips_unsupported(tmp_path: Path):
    """Batch mode should preserve directory structure and ignore unsupported files."""
    from auralock.core.pipeline import ImageNetModelAdapter
    from auralock.services import ProtectionService

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _create_image(input_dir / "cover.png", "#224488")
    _create_image(input_dir / "nested" / "poster.jpg", "#992244")
    (input_dir / "notes.txt").write_text("skip me", encoding="utf-8")

    service = ProtectionService(model=ImageNetModelAdapter(RecordingClassifier()))
    summary = service.protect_directory(
        input_dir,
        output_dir,
        method="fgsm",
        recursive=True,
    )

    assert summary.processed_count == 2
    assert summary.skipped_unsupported_count == 1
    assert summary.skipped_existing_count == 0
    assert (output_dir / "cover.png").exists()
    assert (output_dir / "nested" / "poster.jpg").exists()


def test_protection_service_collective_batch_preserves_original_sizes(
    tmp_path: Path,
):
    """Collective mode should optimize the set jointly but still export original sizes."""
    from auralock.core.pipeline import ImageNetModelAdapter
    from auralock.services import ProtectionService

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    Image.new("RGB", (96, 64), color="#224488").save(input_dir / "cover.png")
    Image.new("RGB", (72, 80), color="#992244").save(input_dir / "poster.jpg")

    service = ProtectionService(
        model=ImageNetModelAdapter(RecordingClassifier()),
        style_feature_extractor=TinyStyleFeatureExtractor(),
    )
    summary = service.protect_directory(
        input_dir,
        output_dir,
        profile="subject",
        collective=True,
        working_size=32,
        num_steps=1,
        alpha=0.01,
    )

    assert summary.collective is True
    assert summary.working_size == (32, 32)
    assert summary.processed_count == 2
    assert Image.open(output_dir / "cover.png").size == (96, 64)
    assert Image.open(output_dir / "poster.jpg").size == (72, 80)


def test_batch_cli_reports_summary(monkeypatch, tmp_path: Path):
    """The CLI should parse batch arguments and render a summary."""
    from auralock.services.protection import BatchProtectionSummary

    calls: dict[str, object] = {}

    class FakeService:
        def protect_directory(self, input_dir, output_dir, **kwargs):
            calls["input_dir"] = input_dir
            calls["output_dir"] = output_dir
            calls["kwargs"] = kwargs
            return BatchProtectionSummary(
                input_dir=Path(input_dir),
                output_dir=Path(output_dir),
                processed_count=3,
                skipped_unsupported_count=1,
                skipped_existing_count=2,
                failed_count=0,
                outputs=[
                    Path(output_dir) / "a.png",
                    Path(output_dir) / "b.png",
                    Path(output_dir) / "c.png",
                ],
                failures=[],
            )

    monkeypatch.setattr("auralock.cli.ProtectionService", FakeService)

    runner = CliRunner()
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "batch",
            str(input_dir),
            str(output_dir),
            "--method",
            "pgd",
            "--num-steps",
            "15",
            "--recursive",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0
    assert "Processed" in result.output
    assert "Skipped unsupported" in result.output
    assert calls["input_dir"] == input_dir
    assert calls["output_dir"] == output_dir
    assert calls["kwargs"]["method"] == "pgd"
    assert calls["kwargs"]["num_steps"] == 15
    assert calls["kwargs"]["recursive"] is True
    assert calls["kwargs"]["overwrite"] is True


def test_batch_cli_supports_collective_subject_mode(monkeypatch, tmp_path: Path):
    """The CLI should expose collective subject-set protection options."""
    from auralock.services.protection import BatchProtectionSummary

    calls: dict[str, object] = {}

    class FakeService:
        def protect_directory(self, input_dir, output_dir, **kwargs):
            calls["input_dir"] = input_dir
            calls["output_dir"] = output_dir
            calls["kwargs"] = kwargs
            return BatchProtectionSummary(
                input_dir=Path(input_dir),
                output_dir=Path(output_dir),
                processed_count=2,
                skipped_unsupported_count=0,
                skipped_existing_count=0,
                failed_count=0,
                outputs=[
                    Path(output_dir) / "a.png",
                    Path(output_dir) / "b.png",
                ],
                failures=[],
                collective=True,
                working_size=(48, 48),
            )

    monkeypatch.setattr("auralock.cli.ProtectionService", FakeService)

    runner = CliRunner()
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "batch",
            str(input_dir),
            str(output_dir),
            "--profile",
            "subject",
            "--collective",
            "--working-size",
            "48",
        ],
    )

    assert result.exit_code == 0
    assert calls["kwargs"]["collective"] is True
    assert calls["kwargs"]["working_size"] == 48
