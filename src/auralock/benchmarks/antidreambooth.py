"""Anti-DreamBooth-style subject split benchmark harness."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auralock.benchmarks.lora import (
    LoraBenchmarkConfig,
    LoraPreflightReport,
    build_lora_infer_command,
    build_lora_train_command,
    evaluate_lora_preflight,
)
from auralock.core.image import SUPPORTED_EXTENSIONS
from auralock.core.profiles import normalize_profile
from auralock.services import ProtectionService

DEFAULT_ANTI_DREAMBOOTH_TRAIN_SCRIPT = Path(
    ".cache_ref/Anti-DreamBooth/train_dreambooth.py"
)
DEFAULT_ANTI_DREAMBOOTH_INFER_SCRIPT = Path(".cache_ref/Anti-DreamBooth/infer.py")
DEFAULT_ANTI_DREAMBOOTH_INSTANCE_PROMPT = "a photo of sks person"
DEFAULT_ANTI_DREAMBOOTH_CLASS_PROMPT = "a photo of person"


def _to_builtin(value: Any) -> Any:
    """Convert paths and scalar-like objects into JSON-friendly values."""
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


@dataclass(slots=True)
class AntiDreamBoothSubjectLayout:
    """Resolved paper-style split layout for one subject."""

    subject_root: Path
    subject_id: str
    set_a_dir: Path
    set_b_dir: Path
    set_c_dir: Path
    set_a_images: list[Path]
    set_b_images: list[Path]
    set_c_images: list[Path]

    @property
    def split_counts(self) -> dict[str, int]:
        return {
            "set_A": len(self.set_a_images),
            "set_B": len(self.set_b_images),
            "set_C": len(self.set_c_images),
        }

    def to_report_dict(self) -> dict[str, object]:
        return _to_builtin(
            {
                "subject_root": self.subject_root,
                "subject_id": self.subject_id,
                "set_a_dir": self.set_a_dir,
                "set_b_dir": self.set_b_dir,
                "set_c_dir": self.set_c_dir,
                "set_a_images": self.set_a_images,
                "set_b_images": self.set_b_images,
                "set_c_images": self.set_c_images,
                "split_counts": self.split_counts,
            }
        )


@dataclass(slots=True)
class AntiDreamBoothBenchmarkManifest:
    """Manifest describing a subject split benchmark workflow."""

    subject_layout: AntiDreamBoothSubjectLayout
    work_dir: Path
    profiles: list[str]
    execute: bool
    preflight: LoraPreflightReport
    jobs: list[dict[str, object]]
    notes: list[str]

    def to_report_dict(self) -> dict[str, object]:
        return _to_builtin(
            {
                "benchmark_mode": "antidreambooth_subject_split",
                "subject_layout": self.subject_layout.to_report_dict(),
                "work_dir": self.work_dir,
                "profiles": self.profiles,
                "execute": self.execute,
                "preflight": self.preflight.to_report_dict(),
                "jobs": self.jobs,
                "notes": self.notes,
            }
        )


def resolve_subject_layout(subject_root: str | Path) -> AntiDreamBoothSubjectLayout:
    """Resolve and validate a paper-style set_A/set_B/set_C layout."""
    subject_root = Path(subject_root)
    if not subject_root.exists() or not subject_root.is_dir():
        raise ValueError("subject_root must be an existing subject directory.")

    split_dirs = {
        "set_A": subject_root / "set_A",
        "set_B": subject_root / "set_B",
        "set_C": subject_root / "set_C",
    }
    missing_splits = [name for name, path in split_dirs.items() if not path.is_dir()]
    if missing_splits:
        missing = ", ".join(missing_splits)
        raise ValueError(
            f"subject_root must contain split directories: set_A, set_B, set_C. Missing: {missing}."
        )

    def collect_images(directory: Path) -> list[Path]:
        image_paths = [
            candidate
            for candidate in sorted(directory.iterdir())
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not image_paths:
            raise ValueError(
                f"No supported images were found in split directory: {directory}"
            )
        return image_paths

    return AntiDreamBoothSubjectLayout(
        subject_root=subject_root,
        subject_id=subject_root.name,
        set_a_dir=split_dirs["set_A"],
        set_b_dir=split_dirs["set_B"],
        set_c_dir=split_dirs["set_C"],
        set_a_images=collect_images(split_dirs["set_A"]),
        set_b_images=collect_images(split_dirs["set_B"]),
        set_c_images=collect_images(split_dirs["set_C"]),
    )


class AntiDreamBoothSubjectBenchmarkHarness:
    """Benchmark AuraLock on an Anti-DreamBooth-style subject split."""

    def __init__(self, protection_service: ProtectionService | None = None) -> None:
        self.protection_service = protection_service or ProtectionService()

    def _copy_images(self, image_paths: list[Path], target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        for image_path in image_paths:
            shutil.copy2(image_path, target_dir / image_path.name)

    def _prepare_protected_split(
        self,
        source_dir: Path,
        *,
        target_dir: Path,
        profile: str,
        working_size: tuple[int, int] | None = None,
    ) -> None:
        summary = self.protection_service.protect_directory(
            source_dir,
            target_dir,
            profile=profile,
            overwrite=True,
            collective=True,
            working_size=working_size,
        )
        if summary is not None and summary.failed_count:
            raise ValueError(
                "Failed to materialize protected subject split: "
                + "; ".join(summary.failures)
            )

    def run(
        self,
        subject_root: str | Path,
        *,
        work_dir: str | Path,
        profiles: tuple[str, ...] = ("safe", "balanced", "strong"),
        execute: bool = False,
        instance_prompt: str = DEFAULT_ANTI_DREAMBOOTH_INSTANCE_PROMPT,
        class_prompt: str = DEFAULT_ANTI_DREAMBOOTH_CLASS_PROMPT,
        pretrained_model_path: str | Path,
        script_path: str | Path = DEFAULT_ANTI_DREAMBOOTH_TRAIN_SCRIPT,
        infer_script_path: str | Path | None = DEFAULT_ANTI_DREAMBOOTH_INFER_SCRIPT,
        resolution: int = 512,
        train_batch_size: int = 1,
        learning_rate: float = 1e-4,
        max_train_steps: int = 400,
    ) -> AntiDreamBoothBenchmarkManifest:
        """Prepare or execute a subject split benchmark workflow."""
        layout = resolve_subject_layout(subject_root)
        work_dir = Path(work_dir)
        normalized_profiles = [normalize_profile(profile) for profile in profiles]

        required_paths = {
            "script_path": Path(script_path),
            "pretrained_model_path": Path(pretrained_model_path),
        }
        if infer_script_path is not None:
            required_paths["infer_script_path"] = Path(infer_script_path)
        preflight = evaluate_lora_preflight(required_paths=required_paths)

        config = LoraBenchmarkConfig(
            script_path=Path(script_path),
            pretrained_model_path=Path(pretrained_model_path),
            infer_script_path=Path(infer_script_path) if infer_script_path else None,
            instance_prompt=instance_prompt,
            class_prompt=class_prompt,
            resolution=resolution,
            train_batch_size=train_batch_size,
            learning_rate=learning_rate,
            max_train_steps=max_train_steps,
        )

        reference_dir = work_dir / "datasets" / "reference" / "set_A"
        clean_published_dir = work_dir / "datasets" / "published" / "clean"
        holdout_dir = work_dir / "datasets" / "holdout" / "set_C"
        jobs: list[dict[str, object]] = []
        variants: list[tuple[str, str | None]] = [("clean", None)]
        variants.extend((profile, profile) for profile in normalized_profiles)

        if execute and preflight.ready:
            self._copy_images(layout.set_a_images, reference_dir)
            self._copy_images(layout.set_b_images, clean_published_dir)
            self._copy_images(layout.set_c_images, holdout_dir)

        for profile_name, protection_profile in variants:
            published_dir = (
                clean_published_dir
                if protection_profile is None
                else work_dir / "datasets" / "published" / profile_name
            )
            class_dir = work_dir / "datasets" / "class" / profile_name
            output_dir = work_dir / "runs" / profile_name
            infer_output_dir = work_dir / "samples" / profile_name
            train_command = build_lora_train_command(
                config,
                instance_data_dir=published_dir,
                class_data_dir=class_dir,
                output_dir=output_dir,
            )
            infer_command = (
                build_lora_infer_command(
                    config.infer_script_path,
                    model_path=output_dir,
                    output_dir=infer_output_dir,
                )
                if config.infer_script_path is not None
                else None
            )

            job = {
                "subject_id": layout.subject_id,
                "profile": profile_name,
                "variant": (
                    "clean_published"
                    if protection_profile is None
                    else "protected_published"
                ),
                "reference_dir": reference_dir,
                "published_dir": published_dir,
                "holdout_dir": holdout_dir,
                "class_dir": class_dir,
                "output_dir": output_dir,
                "infer_output_dir": infer_output_dir,
                "split_counts": layout.split_counts,
                "train_command": train_command,
                "infer_command": infer_command,
                "executed": False,
                "status": "planned",
                "error": None,
            }

            if execute:
                if not preflight.ready:
                    job["status"] = "blocked_preflight"
                else:
                    try:
                        if protection_profile is not None:
                            self._prepare_protected_split(
                                layout.set_b_dir,
                                target_dir=published_dir,
                                profile=protection_profile,
                                working_size=(resolution, resolution),
                            )
                        class_dir.mkdir(parents=True, exist_ok=True)
                        subprocess.run(train_command, cwd=work_dir, check=True)
                        if infer_command is not None:
                            subprocess.run(infer_command, cwd=work_dir, check=True)
                        job["executed"] = True
                        job["status"] = "completed"
                    except subprocess.CalledProcessError as exc:
                        job["status"] = "failed"
                        job["error"] = (
                            f"Command exited with code {exc.returncode}: "
                            f"{' '.join(str(part) for part in exc.cmd)}"
                        )
                    except ValueError as exc:
                        job["status"] = "failed"
                        job["error"] = str(exc)

            jobs.append(_to_builtin(job))

        notes = [
            "This benchmark follows the Anti-DreamBooth paper-style set_A/set_B/set_C split.",
            "set_A is retained as a clean reference split, set_B is treated as the published (training) split, "
            "and set_C is the held-out validation split used to measure out-of-sample protection effectiveness.",
            "Evaluate mimicry success on set_C (holdout) images—never on set_B—to avoid in-sample bias.",
            "AuraLock still uses its own protection pipeline; this workflow is a benchmark alignment layer, not an ASPL/FSMG reproduction.",
        ]

        return AntiDreamBoothBenchmarkManifest(
            subject_layout=layout,
            work_dir=work_dir,
            profiles=normalized_profiles,
            execute=execute,
            preflight=preflight,
            jobs=jobs,
            notes=notes,
        )
