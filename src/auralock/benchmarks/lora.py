"""DreamBooth/LoRA benchmark harness with preflight and manifest support."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from auralock.core.image import SUPPORTED_EXTENSIONS
from auralock.core.profiles import normalize_profile
from auralock.services import ProtectionService

ModuleProbe = Callable[[str], bool]


def _module_exists(module_name: str) -> bool:
    """Check whether an importable module exists in the current environment."""
    return importlib.util.find_spec(module_name) is not None


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
class LoraPreflightReport:
    """Runtime readiness report for real LoRA/DreamBooth benchmark execution."""

    ready: bool
    cuda_available: bool
    missing_modules: list[str]
    missing_paths: dict[str, str]
    invalid_paths: dict[str, str]
    notes: list[str]

    def to_report_dict(self) -> dict[str, object]:
        """Serialize the preflight report."""
        return _to_builtin(asdict(self))


@dataclass(slots=True)
class LoraBenchmarkConfig:
    """Training config for one DreamBooth/LoRA benchmark run."""

    script_path: Path
    pretrained_model_path: Path
    instance_prompt: str
    class_prompt: str
    resolution: int = 512
    train_batch_size: int = 1
    learning_rate: float = 1e-4
    max_train_steps: int = 400
    with_prior_preservation: bool = True
    mixed_precision: str = "bf16"
    seed: int = 42
    infer_script_path: Path | None = None

    def to_report_dict(self) -> dict[str, object]:
        """Serialize the config."""
        return _to_builtin(asdict(self))


@dataclass(slots=True)
class LoraBenchmarkManifest:
    """Manifest describing LoRA benchmark jobs and their preflight state."""

    input_path: Path
    work_dir: Path
    profiles: list[str]
    execute: bool
    preflight: LoraPreflightReport
    jobs: list[dict[str, object]]

    def to_report_dict(self) -> dict[str, object]:
        """Serialize the manifest."""
        return _to_builtin(
            {
                "input_path": self.input_path,
                "work_dir": self.work_dir,
                "profiles": self.profiles,
                "execute": self.execute,
                "preflight": self.preflight.to_report_dict(),
                "jobs": self.jobs,
            }
        )


def evaluate_lora_preflight(
    *,
    required_modules: tuple[str, ...] = (
        "diffusers",
        "accelerate",
        "transformers",
        "peft",
        "safetensors",
    ),
    required_paths: dict[str, Path] | None = None,
    module_probe: ModuleProbe | None = None,
    cuda_available: bool | None = None,
) -> LoraPreflightReport:
    """Check whether the current machine can run a real LoRA benchmark."""
    probe = module_probe or _module_exists
    missing_modules = [module for module in required_modules if not probe(module)]
    expected_paths = required_paths or {}
    missing_paths = {
        label: str(path)
        for label, path in expected_paths.items()
        if not Path(path).exists()
    }
    invalid_paths: dict[str, str] = {}
    for label, path in expected_paths.items():
        path = Path(path)
        if not path.exists():
            continue
        if label in {"script_path", "infer_script_path"} and (
            not path.is_file() or path.suffix.lower() != ".py"
        ):
            invalid_paths[label] = (
                f"{path} must point to an existing Python script (.py)."
            )
        if label == "pretrained_model_path" and (
            not path.is_dir() or not (path / "model_index.json").exists()
        ):
            invalid_paths[label] = (
                f"{path} must be a Diffusers model directory containing model_index.json."
            )
    has_cuda = torch.cuda.is_available() if cuda_available is None else cuda_available
    notes: list[str] = []

    if missing_modules:
        notes.append(
            "Missing required runtime modules for DreamBooth/LoRA benchmarking."
        )
    if missing_paths:
        notes.append(
            "Required scripts or model paths are missing for DreamBooth/LoRA benchmarking."
        )
    if invalid_paths:
        notes.append(
            "Required scripts or model paths are present but invalid for DreamBooth/LoRA benchmarking."
        )
    if not has_cuda:
        notes.append(
            "CUDA is unavailable; real LoRA training is blocked on this machine."
        )

    return LoraPreflightReport(
        ready=not missing_modules
        and not missing_paths
        and not invalid_paths
        and has_cuda,
        cuda_available=has_cuda,
        missing_modules=missing_modules,
        missing_paths=missing_paths,
        invalid_paths=invalid_paths,
        notes=notes,
    )


def build_lora_train_command(
    config: LoraBenchmarkConfig,
    *,
    instance_data_dir: Path,
    class_data_dir: Path,
    output_dir: Path,
) -> list[str]:
    """Build an accelerate launch command for DreamBooth LoRA training."""
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        str(config.script_path),
        "--pretrained_model_name_or_path",
        str(config.pretrained_model_path),
        "--instance_data_dir",
        str(instance_data_dir),
        "--class_data_dir",
        str(class_data_dir),
        "--instance_prompt",
        config.instance_prompt,
        "--class_prompt",
        config.class_prompt,
        "--output_dir",
        str(output_dir),
        "--resolution",
        str(config.resolution),
        "--train_batch_size",
        str(config.train_batch_size),
        "--learning_rate",
        str(config.learning_rate),
        "--max_train_steps",
        str(config.max_train_steps),
        "--mixed_precision",
        config.mixed_precision,
        "--seed",
        str(config.seed),
    ]
    if config.with_prior_preservation:
        command.append("--with_prior_preservation")
    return command


def build_lora_infer_command(
    infer_script_path: Path,
    *,
    model_path: Path,
    output_dir: Path,
) -> list[str]:
    """Build an inference command for a trained DreamBooth/LoRA model."""
    return [
        sys.executable,
        str(infer_script_path),
        "--model_path",
        str(model_path),
        "--output_dir",
        str(output_dir),
    ]


class LoraBenchmarkHarness:
    """Prepare and optionally execute a real DreamBooth/LoRA benchmark workflow."""

    def __init__(
        self,
        protection_service: ProtectionService | None = None,
    ) -> None:
        self.protection_service = protection_service or ProtectionService()

    def _collect_images(
        self,
        input_path: Path,
        *,
        recursive: bool,
    ) -> list[Path]:
        if input_path.is_file():
            if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError("input_path must be a supported image file.")
            return [input_path]
        if not input_path.exists() or not input_path.is_dir():
            raise ValueError("input_path must be an existing image file or directory.")

        iterator = input_path.rglob("*") if recursive else input_path.glob("*")
        image_paths = [
            candidate
            for candidate in sorted(iterator)
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not image_paths:
            raise ValueError("No supported images were found in input_path.")
        return image_paths

    def _prepare_clean_dataset(
        self,
        image_paths: list[Path],
        *,
        input_path: Path,
        target_dir: Path,
    ) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        for image_path in image_paths:
            relative = (
                image_path.name
                if input_path.is_file()
                else image_path.relative_to(input_path)
            )
            destination = target_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, destination)

    def _prepare_protected_dataset(
        self,
        image_paths: list[Path],
        *,
        input_path: Path,
        target_dir: Path,
        profile: str,
    ) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        for image_path in image_paths:
            relative = (
                image_path.name
                if input_path.is_file()
                else image_path.relative_to(input_path)
            )
            destination = target_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            result = self.protection_service.protect_file(
                str(image_path), profile=profile
            )
            from auralock.core.image import save_image

            save_image(result.protected_tensor, destination)

    def run(
        self,
        input_path: str | Path,
        *,
        work_dir: str | Path,
        profiles: tuple[str, ...] = ("safe", "balanced", "strong"),
        recursive: bool = False,
        execute: bool = False,
        instance_prompt: str,
        class_prompt: str,
        pretrained_model_path: str | Path,
        script_path: str | Path,
        infer_script_path: str | Path | None = None,
        resolution: int = 512,
        train_batch_size: int = 1,
        learning_rate: float = 1e-4,
        max_train_steps: int = 400,
    ) -> LoraBenchmarkManifest:
        """Prepare a real LoRA benchmark workflow and optionally execute it."""
        input_path = Path(input_path)
        work_dir = Path(work_dir)
        normalized_profiles = [normalize_profile(profile) for profile in profiles]
        image_paths = self._collect_images(input_path, recursive=recursive)
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

        jobs: list[dict[str, object]] = []
        clean_dir = work_dir / "datasets" / "clean"
        variants: list[tuple[str, str | None]] = [("clean", None)]
        variants.extend((profile, profile) for profile in normalized_profiles)

        if execute and preflight.ready:
            self._prepare_clean_dataset(
                image_paths, input_path=input_path, target_dir=clean_dir
            )

        for profile, protection_profile in variants:
            instance_data_dir = (
                clean_dir
                if protection_profile is None
                else work_dir / "datasets" / "protected" / profile
            )
            class_dir = work_dir / "datasets" / "class" / profile
            output_dir = work_dir / "runs" / profile
            infer_output_dir = work_dir / "samples" / profile
            train_command = build_lora_train_command(
                config,
                instance_data_dir=instance_data_dir,
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
                "profile": profile,
                "variant": "clean" if protection_profile is None else "protected",
                "instance_data_dir": instance_data_dir,
                "class_dir": class_dir,
                "output_dir": output_dir,
                "infer_output_dir": infer_output_dir,
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
                            self._prepare_protected_dataset(
                                image_paths,
                                input_path=input_path,
                                target_dir=instance_data_dir,
                                profile=protection_profile,
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

        return LoraBenchmarkManifest(
            input_path=input_path,
            work_dir=work_dir,
            profiles=normalized_profiles,
            execute=execute,
            preflight=preflight,
            jobs=jobs,
        )
