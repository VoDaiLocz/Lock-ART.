"""Tests for protection profiles and report serialization."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .test_pipeline import RecordingClassifier


def test_profile_resolution_uses_presets_and_allows_overrides():
    """Profiles should map to stable defaults while still allowing overrides."""
    from auralock.core.profiles import resolve_protection_config

    safe = resolve_protection_config(profile="safe")
    strong = resolve_protection_config(profile="strong")
    subject = resolve_protection_config(profile="subject")
    fortress = resolve_protection_config(profile="fortress")
    blindfold = resolve_protection_config(profile="blindfold")
    antidreambooth = resolve_protection_config(profile="antidreambooth")
    custom = resolve_protection_config(profile="safe", epsilon=0.015, num_steps=9)

    assert safe.profile == "safe"
    assert safe.method == "stylecloak"
    assert strong.epsilon > safe.epsilon
    assert strong.num_steps > safe.num_steps
    assert subject.profile == "subject"
    assert subject.method == "stylecloak"
    assert subject.epsilon > strong.epsilon
    assert subject.num_steps > strong.num_steps
    assert fortress.profile == "fortress"
    assert fortress.epsilon > subject.epsilon
    assert fortress.num_steps > subject.num_steps
    assert blindfold.profile == "blindfold"
    assert blindfold.method == "blindfold"
    assert blindfold.epsilon > fortress.epsilon
    assert blindfold.num_steps > fortress.num_steps
    assert antidreambooth.profile == "subject"
    assert custom.profile == "safe"
    assert custom.epsilon == 0.015
    assert custom.num_steps == 9


def test_protection_result_serializes_profile_and_reports(tmp_path: Path):
    """Protection results should be serializable for JSON report export."""
    from auralock.core.pipeline import ImageNetModelAdapter
    from auralock.services import ProtectionService

    service = ProtectionService(model=ImageNetModelAdapter(RecordingClassifier()))
    image = Image.new("RGB", (96, 64), color="#446688")

    result = service.protect_image(image, profile="safe")
    payload = result.to_report_dict(output_path=tmp_path / "protected.png")

    assert payload["profile"] == "safe"
    assert payload["method"] == "stylecloak"
    assert payload["quality_report"]["psnr_db"] >= 0
    assert payload["protection_report"]["protection_score"] >= 0
    assert payload["output_path"].endswith("protected.png")


def test_batch_summary_serializes_profile_and_outputs(tmp_path: Path):
    """Batch summaries should produce machine-readable report payloads."""
    from auralock.services.protection import BatchProtectionSummary

    summary = BatchProtectionSummary(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        profile="balanced",
        method="stylecloak",
        epsilon=0.02,
        num_steps=8,
        alpha=0.004,
        processed_count=2,
        skipped_unsupported_count=1,
        skipped_existing_count=0,
        failed_count=1,
        outputs=[tmp_path / "output" / "a.png", tmp_path / "output" / "b.png"],
        failures=["bad.png: failed"],
    )

    payload = summary.to_report_dict()

    assert payload["profile"] == "balanced"
    assert payload["method"] == "stylecloak"
    assert payload["processed_count"] == 2
    assert payload["outputs"][0].endswith("a.png")
    assert payload["failures"] == ["bad.png: failed"]
