"""Tests for adaptive protection profile selection."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from typer.testing import CliRunner

from auralock.cli import app
from auralock.services.protection import ProtectionResult


def _make_result(
    *,
    profile: str,
    protection_score: float,
    ssim: float,
    psnr_db: float,
) -> ProtectionResult:
    tensor = torch.zeros(1, 3, 16, 16)
    return ProtectionResult(
        profile=profile,
        method="stylecloak",
        epsilon=0.01,
        num_steps=4,
        alpha=0.002,
        original_size=(16, 16),
        original_tensor=tensor,
        protected_tensor=tensor,
        protected_image=Image.new("RGB", (16, 16), color="#000000"),
        quality_report={
            "psnr_db": psnr_db,
            "ssim": ssim,
            "l2_distance": 0.0,
            "linf_distance": 0.0,
            "overall_quality": "Good",
        },
        protection_report={
            "style_similarity": 0.9,
            "embedding_similarity": 0.8,
            "robust_style_similarity": 0.9,
            "robust_embedding_similarity": 0.8,
            "protection_score": protection_score,
            "assessment": "Weak",
        },
        original_prediction=None,
        adversarial_prediction=None,
        attack_success=None,
        perturbation_l2=0.0,
        perturbation_linf=0.0,
        device="cpu",
        model_name="DummyModel",
    )


def test_protection_service_adaptive_selection_escalates_until_threshold_met():
    """Adaptive protection should escalate through profiles until a candidate qualifies."""
    from auralock.services import ProtectionService

    service = ProtectionService.__new__(ProtectionService)
    calls: list[str] = []
    results = {
        "safe": _make_result(
            profile="safe",
            protection_score=8.0,
            ssim=0.98,
            psnr_db=43.0,
        ),
        "balanced": _make_result(
            profile="balanced",
            protection_score=18.0,
            ssim=0.95,
            psnr_db=38.0,
        ),
        "subject": _make_result(
            profile="subject",
            protection_score=31.0,
            ssim=0.93,
            psnr_db=35.5,
        ),
    }

    def fake_protect_file(
        self,
        path: str,
        *,
        profile: str = "balanced",
        epsilon=None,
        method=None,
        num_steps=None,
        alpha=None,
    ):
        calls.append(profile)
        return results[profile]

    service.protect_file = fake_protect_file.__get__(service, ProtectionService)

    result = ProtectionService.protect_file_adaptive(
        service,
        "artwork.png",
        profiles=("safe", "balanced", "subject"),
        min_protection_score=25.0,
        min_ssim=0.92,
        min_psnr_db=35.0,
    )

    assert calls == ["safe", "balanced", "subject"]
    assert result.profile == "subject"


def test_protect_cli_routes_to_adaptive_mode(monkeypatch, tmp_path: Path):
    """CLI protect should call the adaptive service path when constraints are provided."""
    calls: dict[str, object] = {}

    class FakeService:
        def protect_file(self, *args, **kwargs):
            raise AssertionError("standard protect_file should not be used")

        def protect_file_adaptive(
            self,
            input_path,
            *,
            profiles,
            min_protection_score,
            min_ssim,
            min_psnr_db,
            epsilon=None,
            method=None,
            num_steps=None,
            alpha=None,
        ):
            calls["input_path"] = input_path
            calls["profiles"] = profiles
            calls["min_protection_score"] = min_protection_score
            calls["min_ssim"] = min_ssim
            calls["min_psnr_db"] = min_psnr_db
            return _make_result(
                profile="subject",
                protection_score=31.0,
                ssim=0.93,
                psnr_db=35.5,
            )

    monkeypatch.setattr("auralock.cli.ProtectionService", FakeService)

    runner = CliRunner()
    input_path = tmp_path / "art.png"
    Image.new("RGB", (16, 16), color="#123456").save(input_path)
    report_path = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "protect",
            str(input_path),
            "--auto-profiles",
            "safe,subject",
            "--min-protection-score",
            "25",
            "--min-ssim",
            "0.92",
            "--min-psnr",
            "35",
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert calls["input_path"] == str(input_path)
    assert calls["profiles"] == ("safe", "subject")
    assert calls["min_protection_score"] == 25.0
    assert calls["min_ssim"] == 0.92
    assert calls["min_psnr_db"] == 35.0


def test_protect_cli_fails_when_adaptive_constraints_are_not_met(
    monkeypatch, tmp_path: Path
):
    """Adaptive CLI should fail clearly when no candidate meets the requested floors."""

    class FakeService:
        def protect_file(self, *args, **kwargs):
            raise AssertionError("standard protect_file should not be used")

        def protect_file_adaptive(
            self,
            input_path,
            *,
            profiles,
            min_protection_score,
            min_ssim,
            min_psnr_db,
            epsilon=None,
            method=None,
            num_steps=None,
            alpha=None,
        ):
            return _make_result(
                profile="balanced",
                protection_score=10.0,
                ssim=0.93,
                psnr_db=35.5,
            )

    monkeypatch.setattr("auralock.cli.ProtectionService", FakeService)

    runner = CliRunner()
    input_path = tmp_path / "art.png"
    Image.new("RGB", (16, 16), color="#123456").save(input_path)

    result = runner.invoke(
        app,
        [
            "protect",
            str(input_path),
            "--auto-profiles",
            "safe,subject",
            "--min-protection-score",
            "25",
            "--min-ssim",
            "0.92",
            "--min-psnr",
            "35",
        ],
    )

    assert result.exit_code == 1
    assert "Adaptive protection requirements were not met" in result.stdout
