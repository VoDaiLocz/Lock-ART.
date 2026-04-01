"""Shared protection pipeline used by CLI and UI."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from auralock.attacks import FGSM, PGD, StyleCloak
from auralock.benchmarks.splits import SplitMetadata, SplitType
from auralock.core.image import (
    SUPPORTED_EXTENSIONS,
    image_to_tensor,
    load_image,
    quantize_image_tensor,
    save_image,
    tensor_to_image,
)
from auralock.core.metrics import (
    get_protection_readability_report,
    get_quality_report,
)
from auralock.core.pipeline import load_default_model, resolve_device
from auralock.core.profiles import ProtectionConfig, resolve_protection_config
from auralock.core.style import load_default_style_feature_extractor


def _to_builtin(value: Any) -> Any:
    """Convert tensors, paths, and numpy scalars into JSON-friendly values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, SplitMetadata):
        return value.to_dict()
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


@dataclass(slots=True)
class ProtectionResult:
    """Structured result returned by the protection service."""

    profile: str
    method: str
    epsilon: float
    num_steps: int
    alpha: float | None
    original_size: tuple[int, int]
    original_tensor: torch.Tensor
    protected_tensor: torch.Tensor
    protected_image: Image.Image
    quality_report: dict[str, object]
    protection_report: dict[str, object]
    original_prediction: int | None
    adversarial_prediction: int | None
    attack_success: bool | None
    perturbation_l2: float
    perturbation_linf: float
    device: str
    model_name: str

    def to_report_dict(
        self,
        *,
        output_path: str | Path | None = None,
    ) -> dict[str, object]:
        """Serialize the user-facing result into a JSON-friendly payload."""
        payload = {
            "profile": self.profile,
            "method": self.method,
            "epsilon": self.epsilon,
            "num_steps": self.num_steps,
            "alpha": self.alpha,
            "original_size": self.original_size,
            "quality_report": self.quality_report,
            "protection_report": self.protection_report,
            "original_prediction": self.original_prediction,
            "adversarial_prediction": self.adversarial_prediction,
            "attack_success": self.attack_success,
            "perturbation_l2": self.perturbation_l2,
            "perturbation_linf": self.perturbation_linf,
            "device": self.device,
            "model_name": self.model_name,
            "validation_metadata": {
                "is_validated": False,
                "validation_status": "not_validated",
                "validation_method": None,
                "validation_date": None,
                "notes": "Protection metrics are proxy measurements not validated against real attacks like DreamBooth or LoRA.",
            },
        }
        if output_path is not None:
            payload["output_path"] = output_path
        return _to_builtin(payload)


@dataclass(slots=True)
class BatchProtectionSummary:
    """Summary returned after processing a directory of images."""

    input_dir: Path
    output_dir: Path
    processed_count: int
    skipped_unsupported_count: int
    skipped_existing_count: int
    failed_count: int
    outputs: list[Path]
    failures: list[str]
    profile: str | None = None
    method: str | None = None
    epsilon: float | None = None
    num_steps: int | None = None
    alpha: float | None = None
    collective: bool = False
    working_size: tuple[int, int] | None = None

    def to_report_dict(self) -> dict[str, object]:
        """Serialize the batch summary into a JSON-friendly payload."""
        return _to_builtin(
            {
                "input_dir": self.input_dir,
                "output_dir": self.output_dir,
                "profile": self.profile,
                "method": self.method,
                "epsilon": self.epsilon,
                "num_steps": self.num_steps,
                "alpha": self.alpha,
                "collective": self.collective,
                "working_size": self.working_size,
                "processed_count": self.processed_count,
                "skipped_unsupported_count": self.skipped_unsupported_count,
                "skipped_existing_count": self.skipped_existing_count,
                "failed_count": self.failed_count,
                "outputs": self.outputs,
                "failures": self.failures,
                "validation_metadata": {
                    "is_validated": False,
                    "validation_status": "not_validated",
                    "validation_method": None,
                    "validation_date": None,
                    "notes": "Protection metrics are proxy measurements not validated against real attacks like DreamBooth or LoRA.",
                },
            }
        )


@dataclass(slots=True)
class BenchmarkEntry:
    """One benchmark run for a given image/profile pair."""

    input_path: Path
    profile: str
    method: str
    epsilon: float
    num_steps: int
    alpha: float | None
    runtime_sec: float
    quality_report: dict[str, object]
    protection_report: dict[str, object]

    def to_report_dict(self) -> dict[str, object]:
        """Serialize the benchmark entry into a JSON-friendly payload."""
        return _to_builtin(
            {
                "input_path": self.input_path,
                "profile": self.profile,
                "method": self.method,
                "epsilon": self.epsilon,
                "num_steps": self.num_steps,
                "alpha": self.alpha,
                "runtime_sec": self.runtime_sec,
                "quality_report": self.quality_report,
                "protection_report": self.protection_report,
                "validation_metadata": {
                    "is_validated": False,
                    "validation_status": "not_validated",
                    "validation_method": None,
                    "validation_date": None,
                    "notes": "Protection metrics are proxy measurements not validated against real attacks like DreamBooth or LoRA.",
                },
            }
        )


@dataclass(slots=True)
class BenchmarkSummary:
    """Aggregate benchmark report across one or more profiles/images."""

    input_path: Path
    image_count: int
    entries: list[BenchmarkEntry]
    profile_summaries: dict[str, dict[str, object]]
    split_metadata: SplitMetadata | None = None

    def to_report_dict(self) -> dict[str, object]:
        """Serialize the full benchmark report."""
        payload: dict[str, object] = {
            "input_path": self.input_path,
            "image_count": self.image_count,
            "entries": [entry.to_report_dict() for entry in self.entries],
            "profile_summaries": self.profile_summaries,
            "validation_metadata": {
                "is_validated": False,
                "validation_status": "not_validated",
                "validation_method": None,
                "validation_date": None,
                "notes": "Protection metrics are proxy measurements not validated against real attacks like DreamBooth or LoRA.",
            },
        }
        if self.split_metadata is not None:
            payload["split_metadata"] = self.split_metadata
            payload["split_type"] = self.split_metadata.split_type.value
            payload["split_hash"] = self.split_metadata.split_hash
            payload["dataset_name"] = self.split_metadata.dataset_name
            payload["dataset_version"] = self.split_metadata.dataset_version
        return _to_builtin(payload)


class ProtectionService:
    """High-level service for protecting and analyzing images."""

    def __init__(
        self,
        model: nn.Module | None = None,
        device: str | torch.device | None = None,
        style_feature_extractor: nn.Module | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.model = model if model is not None else load_default_model(self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.style_feature_extractor = (
            style_feature_extractor
            if style_feature_extractor is not None
            else load_default_style_feature_extractor(self.device)
        )
        self.style_feature_extractor = self.style_feature_extractor.to(self.device)
        self.style_feature_extractor.eval()

    def _normalize_method(self, method: str) -> str:
        normalized = method.strip().lower()
        if normalized.startswith("fgsm"):
            return "fgsm"
        if normalized.startswith("pgd"):
            return "pgd"
        if normalized in {"blindfold", "obfuscate", "blind"}:
            return "blindfold"
        if normalized in {"stylecloak", "style", "style-guard", "styleguard"}:
            return "stylecloak"
        raise ValueError("method must be 'fgsm', 'pgd', 'stylecloak', or 'blindfold'.")

    def _build_attack(
        self,
        method: str,
        epsilon: float,
        num_steps: int = 10,
        alpha: float | None = None,
    ) -> FGSM | PGD | StyleCloak:
        normalized = self._normalize_method(method)
        if normalized == "fgsm":
            return FGSM(self.model, epsilon=epsilon, device=self.device)
        if normalized == "pgd":
            return PGD(
                self.model,
                epsilon=epsilon,
                alpha=alpha,
                num_steps=num_steps,
                device=self.device,
            )
        if normalized == "blindfold":
            return StyleCloak(
                self.style_feature_extractor,
                epsilon=epsilon,
                alpha=alpha,
                num_steps=num_steps,
                random_start=True,
                style_weight=1.2,
                embedding_weight=1.8,
                pixel_weight=0.08,
                high_frequency_weight=0.14,
                worst_case_weight=1.15,
                coarse_gradient_weight=0.9,
                coarse_gradient_scale=0.16,
                device=self.device,
            )
        return StyleCloak(
            self.style_feature_extractor,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
            device=self.device,
        )

    def _resolve_config(
        self,
        *,
        profile: str = "balanced",
        method: str | None = None,
        epsilon: float | None = None,
        num_steps: int | None = None,
        alpha: float | None = None,
    ) -> ProtectionConfig:
        """Resolve the final runtime config from a profile plus explicit overrides."""
        config = resolve_protection_config(
            profile=profile,
            method=method,
            epsilon=epsilon,
            num_steps=num_steps,
            alpha=alpha,
        )
        self._normalize_method(config.method)
        return config

    def _resolve_working_size(
        self,
        working_size: int | tuple[int, int] | None,
    ) -> tuple[int, int]:
        """Normalize the collective working size into a concrete (width, height)."""
        if working_size is None:
            return (512, 512)
        if isinstance(working_size, int):
            if working_size <= 0:
                raise ValueError("working_size must be positive.")
            return (working_size, working_size)
        width, height = working_size
        if width <= 0 or height <= 0:
            raise ValueError("working_size dimensions must be positive.")
        return (int(width), int(height))

    def _scan_directory_jobs(
        self,
        *,
        input_path: Path,
        output_path: Path,
        recursive: bool,
        overwrite: bool,
    ) -> tuple[list[tuple[Path, Path]], int, int]:
        """Collect supported image jobs while accounting for skipped files."""
        iterator = input_path.rglob("*") if recursive else input_path.glob("*")
        jobs: list[tuple[Path, Path]] = []
        skipped_unsupported_count = 0
        skipped_existing_count = 0

        for candidate in sorted(iterator):
            if not candidate.is_file():
                continue

            if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
                skipped_unsupported_count += 1
                continue

            relative_path = candidate.relative_to(input_path)
            destination = output_path / relative_path

            if destination.exists() and not overwrite:
                skipped_existing_count += 1
                continue

            jobs.append((candidate, destination))

        return jobs, skipped_unsupported_count, skipped_existing_count

    def _protect_directory_collectively(
        self,
        jobs: list[tuple[Path, Path]],
        *,
        config: ProtectionConfig,
        input_path: Path,
        output_path: Path,
        skipped_unsupported_count: int,
        skipped_existing_count: int,
        working_size: int | tuple[int, int] | None,
    ) -> BatchProtectionSummary:
        """Protect a whole image set using one collective perturbation."""
        if not jobs:
            return BatchProtectionSummary(
                input_dir=input_path,
                output_dir=output_path,
                profile=config.profile,
                method=config.method,
                epsilon=config.epsilon,
                num_steps=config.num_steps,
                alpha=config.alpha,
                collective=True,
                working_size=self._resolve_working_size(working_size),
                processed_count=0,
                skipped_unsupported_count=skipped_unsupported_count,
                skipped_existing_count=skipped_existing_count,
                failed_count=0,
                outputs=[],
                failures=[],
            )

        normalized_size = self._resolve_working_size(working_size)
        width, height = normalized_size

        loaded: list[tuple[Path, Path, torch.Tensor]] = []
        failures: list[str] = []
        for candidate, destination in jobs:
            try:
                loaded.append((candidate, destination, load_image(candidate)))
            except Exception as exc:
                failures.append(f"{candidate}: {exc}")

        if not loaded:
            return BatchProtectionSummary(
                input_dir=input_path,
                output_dir=output_path,
                profile=config.profile,
                method=config.method,
                epsilon=config.epsilon,
                num_steps=config.num_steps,
                alpha=config.alpha,
                collective=True,
                working_size=normalized_size,
                processed_count=0,
                skipped_unsupported_count=skipped_unsupported_count,
                skipped_existing_count=skipped_existing_count,
                failed_count=len(failures),
                outputs=[],
                failures=failures,
            )

        batch = torch.cat(
            [
                F.interpolate(
                    tensor.unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                for _, _, tensor in loaded
            ],
            dim=0,
        )
        attack = self._build_attack(
            method=config.method,
            epsilon=config.epsilon,
            num_steps=config.num_steps,
            alpha=config.alpha,
        )
        collective_result = attack.generate_with_info(batch)
        batch_delta = (
            collective_result["adversarial"].detach().cpu().to(torch.float32) - batch
        )
        shared_delta = batch_delta.mean(dim=0, keepdim=True)
        blended_delta = 0.7 * batch_delta + 0.3 * shared_delta.expand_as(batch_delta)

        outputs: list[Path] = []
        for index, (candidate, destination, tensor) in enumerate(loaded):
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                upscaled_delta = F.interpolate(
                    blended_delta[index : index + 1],
                    size=tensor.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                protected_tensor = quantize_image_tensor(
                    torch.clamp(tensor.unsqueeze(0) + upscaled_delta, 0.0, 1.0)
                ).to(torch.float32)
                save_image(protected_tensor, destination)
                outputs.append(destination)
            except Exception as exc:
                failures.append(f"{candidate}: {exc}")

        failed_count = len(failures)
        return BatchProtectionSummary(
            input_dir=input_path,
            output_dir=output_path,
            profile=config.profile,
            method=config.method,
            epsilon=config.epsilon,
            num_steps=config.num_steps,
            alpha=config.alpha,
            collective=True,
            working_size=normalized_size,
            processed_count=len(outputs),
            skipped_unsupported_count=skipped_unsupported_count,
            skipped_existing_count=skipped_existing_count,
            failed_count=failed_count,
            outputs=outputs,
            failures=failures,
        )

    def protect_tensor(
        self,
        tensor: torch.Tensor,
        *,
        profile: str = "balanced",
        epsilon: float | None = None,
        method: str | None = None,
        num_steps: int | None = None,
        alpha: float | None = None,
    ) -> ProtectionResult:
        config = self._resolve_config(
            profile=profile,
            epsilon=epsilon,
            method=method,
            num_steps=num_steps,
            alpha=alpha,
        )
        original_tensor = tensor.detach().clone().to(torch.float32)
        if original_tensor.dim() == 3:
            original_tensor = original_tensor.unsqueeze(0)
        if original_tensor.dim() != 4:
            raise ValueError("tensor must have shape (C, H, W) or (B, C, H, W).")

        original_tensor = original_tensor.cpu()
        attack = self._build_attack(
            method=config.method,
            epsilon=config.epsilon,
            num_steps=config.num_steps,
            alpha=config.alpha,
        )
        result = attack.generate_with_info(original_tensor)
        protected_tensor = quantize_image_tensor(
            result["adversarial"].detach().cpu()
        ).to(torch.float32)
        quality_report = get_quality_report(original_tensor, protected_tensor)
        protection_report = get_protection_readability_report(
            original_tensor,
            protected_tensor,
            feature_extractor=self.style_feature_extractor,
        )

        original_prediction = None
        adversarial_prediction = None
        attack_success = None
        if "original_preds" in result and "adversarial_preds" in result:
            original_prediction = int(result["original_preds"][0].item())
            adversarial_prediction = int(result["adversarial_preds"][0].item())
            attack_success = bool(original_prediction != adversarial_prediction)
        original_size = (
            int(original_tensor.shape[-1]),
            int(original_tensor.shape[-2]),
        )

        return ProtectionResult(
            profile=config.profile,
            method=self._normalize_method(config.method),
            epsilon=config.epsilon,
            num_steps=config.num_steps,
            alpha=config.alpha,
            original_size=original_size,
            original_tensor=original_tensor,
            protected_tensor=protected_tensor,
            protected_image=tensor_to_image(protected_tensor),
            quality_report=quality_report,
            protection_report=protection_report,
            original_prediction=original_prediction,
            adversarial_prediction=adversarial_prediction,
            attack_success=attack_success,
            perturbation_l2=float(result["perturbation_l2"]),
            perturbation_linf=float(result["perturbation_linf"]),
            device=str(self.device),
            model_name=self.model.__class__.__name__,
        )

    def protect_image(
        self,
        image: Image.Image,
        *,
        profile: str = "balanced",
        epsilon: float | None = None,
        method: str | None = None,
        num_steps: int | None = None,
        alpha: float | None = None,
    ) -> ProtectionResult:
        tensor = image_to_tensor(image).unsqueeze(0)
        return self.protect_tensor(
            tensor,
            profile=profile,
            epsilon=epsilon,
            method=method,
            num_steps=num_steps,
            alpha=alpha,
        )

    def protect_file(
        self,
        path: str,
        *,
        profile: str = "balanced",
        epsilon: float | None = None,
        method: str | None = None,
        num_steps: int | None = None,
        alpha: float | None = None,
    ) -> ProtectionResult:
        tensor = load_image(path).unsqueeze(0)
        return self.protect_tensor(
            tensor,
            profile=profile,
            epsilon=epsilon,
            method=method,
            num_steps=num_steps,
            alpha=alpha,
        )

    def _result_meets_adaptive_constraints(
        self,
        result: ProtectionResult,
        *,
        min_protection_score: float,
        min_ssim: float,
        min_psnr_db: float,
    ) -> bool:
        """Check whether a protection result meets the requested strength floors."""
        protection_score = float(
            result.protection_report.get("protection_score", float("-inf"))
        )
        ssim = float(result.quality_report.get("ssim", float("-inf")))
        psnr_db = float(result.quality_report.get("psnr_db", float("-inf")))
        return (
            protection_score >= min_protection_score
            and ssim >= min_ssim
            and psnr_db >= min_psnr_db
        )

    def _adaptive_candidate_rank(
        self,
        result: ProtectionResult,
        *,
        min_ssim: float,
        min_psnr_db: float,
    ) -> tuple[int, float, float, float]:
        """Rank fallback candidates when no profile clears every adaptive floor."""
        ssim = float(result.quality_report.get("ssim", 0.0))
        psnr_db = float(result.quality_report.get("psnr_db", 0.0))
        protection_score = float(result.protection_report.get("protection_score", 0.0))
        quality_ok = int(ssim >= min_ssim and psnr_db >= min_psnr_db)
        return (quality_ok, protection_score, ssim, psnr_db)

    def protect_file_adaptive(
        self,
        path: str,
        *,
        profiles: tuple[str, ...] = (
            "safe",
            "balanced",
            "strong",
            "subject",
            "fortress",
            "blindfold",
        ),
        min_protection_score: float = 25.0,
        min_ssim: float = 0.92,
        min_psnr_db: float = 35.0,
        epsilon: float | None = None,
        method: str | None = None,
        num_steps: int | None = None,
        alpha: float | None = None,
    ) -> ProtectionResult:
        """Escalate through profiles until the requested protection floors are met."""
        if not profiles:
            raise ValueError("profiles must contain at least one profile name.")

        normalized_profiles: tuple[str, ...] = tuple(
            self._resolve_config(profile=profile).profile for profile in profiles
        )

        attempted: list[ProtectionResult] = []
        for profile_name in normalized_profiles:
            result = self.protect_file(
                path,
                profile=profile_name,
                epsilon=epsilon,
                method=method,
                num_steps=num_steps,
                alpha=alpha,
            )
            attempted.append(result)
            if self._result_meets_adaptive_constraints(
                result,
                min_protection_score=min_protection_score,
                min_ssim=min_ssim,
                min_psnr_db=min_psnr_db,
            ):
                return result

        return max(
            attempted,
            key=lambda candidate: self._adaptive_candidate_rank(
                candidate,
                min_ssim=min_ssim,
                min_psnr_db=min_psnr_db,
            ),
        )

    def analyze_files(
        self, original_path: str, modified_path: str
    ) -> dict[str, object]:
        """Load and compare two image files."""
        original = load_image(original_path)
        modified = load_image(modified_path)
        if original.shape != modified.shape:
            raise ValueError("Images must have the same dimensions for analysis.")
        quality_report = get_quality_report(original, modified)
        protection_report = get_protection_readability_report(
            original,
            modified,
            feature_extractor=self.style_feature_extractor,
        )
        return {
            "quality_report": quality_report,
            "protection_report": protection_report,
        }

    def _validate_split_membership(
        self, image_paths: list[Path], split_metadata: SplitMetadata
    ) -> None:
        """Ensure benchmark targets belong to the declared split."""
        missing = split_metadata.contains_all(image_paths)
        if missing:
            raise ValueError(
                "Benchmark inputs are missing from the declared split manifest: "
                + ", ".join(missing)
            )
        if (
            split_metadata.dataset_root is not None
            and split_metadata.dataset_root.strip() != ""
        ):
            root = Path(split_metadata.dataset_root).resolve()
            outside_root = [
                str(path)
                for path in image_paths
                if not Path(path).resolve().is_relative_to(root)
            ]
            if outside_root:
                raise ValueError(
                    "Benchmark inputs must be within dataset_root to avoid leakage: "
                    + ", ".join(outside_root)
                )
        if split_metadata.split_type != SplitType.TEST:
            warnings.warn(
                f"Benchmarking on non-test split '{split_metadata.split_type.value}'. "
                "Use TEST split for final evaluation to avoid optimistic bias.",
                stacklevel=2,
            )

    def _collect_benchmark_entries(
        self,
        image_paths: list[Path],
        *,
        input_path: Path,
        profiles: tuple[str, ...],
        split_metadata: SplitMetadata,
    ) -> BenchmarkSummary:
        """Benchmark the requested profiles on a list of images."""
        if not image_paths:
            raise ValueError("No supported images were found to benchmark.")

        self._validate_split_membership(image_paths, split_metadata)

        entries: list[BenchmarkEntry] = []
        for image_path in image_paths:
            for profile in profiles:
                start = perf_counter()
                result = self.protect_file(str(image_path), profile=profile)
                runtime_sec = perf_counter() - start
                entries.append(
                    BenchmarkEntry(
                        input_path=image_path,
                        profile=result.profile,
                        method=result.method,
                        epsilon=result.epsilon,
                        num_steps=result.num_steps,
                        alpha=result.alpha,
                        runtime_sec=runtime_sec,
                        quality_report=result.quality_report,
                        protection_report=result.protection_report,
                    )
                )

        profile_summaries: dict[str, dict[str, object]] = {}
        for profile in profiles:
            profile_entries = [entry for entry in entries if entry.profile == profile]
            if not profile_entries:
                continue
            profile_summaries[profile] = {
                "image_count": len(profile_entries),
                "avg_psnr_db": sum(
                    float(entry.quality_report["psnr_db"]) for entry in profile_entries
                )
                / len(profile_entries),
                "avg_ssim": sum(
                    float(entry.quality_report["ssim"]) for entry in profile_entries
                )
                / len(profile_entries),
                "avg_protection_score": sum(
                    float(entry.protection_report["protection_score"])
                    for entry in profile_entries
                )
                / len(profile_entries),
                "avg_runtime_sec": sum(
                    float(entry.runtime_sec) for entry in profile_entries
                )
                / len(profile_entries),
            }

        return BenchmarkSummary(
            input_path=input_path,
            image_count=len(image_paths),
            entries=entries,
            profile_summaries=profile_summaries,
            split_metadata=split_metadata,
        )

    def benchmark_file(
        self,
        input_path: str | Path,
        *,
        profiles: tuple[str, ...] = ("safe", "balanced", "strong"),
        split_metadata: SplitMetadata,
    ) -> BenchmarkSummary:
        """Benchmark one image against multiple named profiles with split tracking."""
        candidate = Path(input_path)
        if not candidate.exists() or not candidate.is_file():
            raise ValueError("input_path must be an existing image file.")
        if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError("input_path must point to a supported image file.")
        return self._collect_benchmark_entries(
            [candidate],
            input_path=candidate,
            profiles=profiles,
            split_metadata=split_metadata,
        )

    def benchmark_directory(
        self,
        input_dir: str | Path,
        *,
        profiles: tuple[str, ...] = ("safe", "balanced", "strong"),
        recursive: bool = False,
        split_metadata: SplitMetadata,
    ) -> BenchmarkSummary:
        """Benchmark all supported images in a directory across profiles with split tracking."""
        input_path = Path(input_dir)
        if not input_path.exists() or not input_path.is_dir():
            raise ValueError("input_dir must be an existing directory.")

        iterator = input_path.rglob("*") if recursive else input_path.glob("*")
        image_paths = [
            candidate
            for candidate in sorted(iterator)
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return self._collect_benchmark_entries(
            image_paths,
            input_path=input_path,
            profiles=profiles,
            split_metadata=split_metadata,
        )

    def protect_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        *,
        profile: str = "balanced",
        epsilon: float | None = None,
        method: str | None = None,
        num_steps: int | None = None,
        alpha: float | None = None,
        recursive: bool = False,
        overwrite: bool = False,
        collective: bool = False,
        working_size: int | tuple[int, int] | None = None,
    ) -> BatchProtectionSummary:
        """Protect all supported images in a directory."""
        config = self._resolve_config(
            profile=profile,
            epsilon=epsilon,
            method=method,
            num_steps=num_steps,
            alpha=alpha,
        )
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.exists() or not input_path.is_dir():
            raise ValueError("input_dir must be an existing directory.")
        if output_path == input_path:
            raise ValueError("output_dir must be different from input_dir.")
        if output_path.resolve().is_relative_to(input_path.resolve()):
            raise ValueError("output_dir must not be inside input_dir.")

        output_path.mkdir(parents=True, exist_ok=True)
        jobs, skipped_unsupported_count, skipped_existing_count = (
            self._scan_directory_jobs(
                input_path=input_path,
                output_path=output_path,
                recursive=recursive,
                overwrite=overwrite,
            )
        )

        if collective:
            return self._protect_directory_collectively(
                jobs,
                config=config,
                input_path=input_path,
                output_path=output_path,
                skipped_unsupported_count=skipped_unsupported_count,
                skipped_existing_count=skipped_existing_count,
                working_size=working_size,
            )

        processed_count = 0
        failed_count = 0
        outputs: list[Path] = []
        failures: list[str] = []

        for candidate, destination in jobs:
            try:
                result = self.protect_file(
                    str(candidate),
                    profile=config.profile,
                    epsilon=config.epsilon,
                    method=config.method,
                    num_steps=config.num_steps,
                    alpha=config.alpha,
                )
                save_image(result.protected_tensor, destination)
            except Exception as exc:
                failed_count += 1
                failures.append(f"{candidate}: {exc}")
                continue

            processed_count += 1
            outputs.append(destination)

        return BatchProtectionSummary(
            input_dir=input_path,
            output_dir=output_path,
            profile=config.profile,
            method=config.method,
            epsilon=config.epsilon,
            num_steps=config.num_steps,
            alpha=config.alpha,
            collective=False,
            working_size=None,
            processed_count=processed_count,
            skipped_unsupported_count=skipped_unsupported_count,
            skipped_existing_count=skipped_existing_count,
            failed_count=failed_count,
            outputs=outputs,
            failures=failures,
        )
