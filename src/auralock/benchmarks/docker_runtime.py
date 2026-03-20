"""Docker runtime helpers for LoRA benchmark execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

DEFAULT_BENCHMARK_BASE_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
DEFAULT_GPU_SMOKE_IMAGE = "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04"
DEFAULT_COMPOSE_FILE = Path("docker-compose.benchmark.yml")
DEFAULT_SERVICE_NAME = "auralock-benchmark"
DEFAULT_CONTAINER_WORKSPACE = PurePosixPath("/workspace")


def _normalize_gpu_count(gpu_count: str) -> str:
    """Validate the GPU count string accepted by Docker Compose."""
    normalized = gpu_count.strip().lower()
    if normalized == "all":
        return "all"
    if normalized.isdigit() and int(normalized) > 0:
        return normalized
    raise ValueError("gpu_count must be 'all' or a positive integer string.")


def _to_container_path(path: Path, *, workspace_dir: Path) -> str:
    """Map a workspace-local host path to its mounted container path."""
    resolved_workspace = workspace_dir.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_workspace)
    except ValueError as exc:
        raise ValueError(
            f"Path must stay inside the workspace root: {resolved_path}"
        ) from exc
    return str(DEFAULT_CONTAINER_WORKSPACE / PurePosixPath(relative.as_posix()))


@dataclass(slots=True)
class DockerLoraBenchmarkPlan:
    """Commands and environment needed to run the benchmark via Docker."""

    build_command: list[str]
    gpu_check_command: list[str]
    run_command: list[str]
    environment: dict[str, str]


@dataclass(slots=True)
class DockerLoraBenchmarkConfig:
    """Host-side config for launching LoRA benchmarking in Docker."""

    workspace_dir: Path
    input_path: Path
    work_dir: Path
    pretrained_model_path: Path
    script_path: Path
    infer_script_path: Path | None
    instance_prompt: str
    class_prompt: str
    profiles: tuple[str, ...] = ("safe", "balanced", "strong")
    recursive: bool = False
    execute: bool = False
    resolution: int = 512
    train_batch_size: int = 1
    learning_rate: float = 1e-4
    max_train_steps: int = 400
    report: Path | None = None
    compose_file: Path = DEFAULT_COMPOSE_FILE
    service_name: str = DEFAULT_SERVICE_NAME
    gpu_count: str = "all"
    base_image: str = DEFAULT_BENCHMARK_BASE_IMAGE
    gpu_smoke_image: str = DEFAULT_GPU_SMOKE_IMAGE


def build_docker_lora_benchmark_plan(
    config: DockerLoraBenchmarkConfig,
) -> DockerLoraBenchmarkPlan:
    """Build the Docker commands required for LoRA benchmark execution."""
    if not config.input_path.exists():
        raise ValueError(f"Input not found: {config.input_path}")
    if not config.script_path.exists():
        raise ValueError(f"Training script not found: {config.script_path}")
    if config.script_path.suffix.lower() != ".py":
        raise ValueError("Training script must be a Python file ending in .py.")
    if config.infer_script_path is not None and not config.infer_script_path.exists():
        raise ValueError(f"Inference script not found: {config.infer_script_path}")
    if (
        config.infer_script_path is not None
        and config.infer_script_path.suffix.lower() != ".py"
    ):
        raise ValueError("Inference script must be a Python file ending in .py.")
    if not config.pretrained_model_path.exists():
        raise ValueError(
            f"Pretrained model path not found: {config.pretrained_model_path}"
        )
    if (
        not config.pretrained_model_path.is_dir()
        or not (config.pretrained_model_path / "model_index.json").exists()
    ):
        raise ValueError(
            "pretrained_model_path must be a Diffusers model directory "
            "containing model_index.json."
        )

    workspace_dir = config.workspace_dir.resolve()
    compose_file = (
        config.compose_file
        if config.compose_file.is_absolute()
        else workspace_dir / config.compose_file
    ).resolve()
    if not compose_file.exists():
        raise ValueError(f"Docker Compose file not found: {compose_file}")
    gpu_count = _normalize_gpu_count(config.gpu_count)

    benchmark_args = [
        "auralock",
        "benchmark-lora",
        _to_container_path(config.input_path, workspace_dir=workspace_dir),
        "--work-dir",
        _to_container_path(config.work_dir, workspace_dir=workspace_dir),
        "--profiles",
        ",".join(config.profiles),
        "--pretrained-model-path",
        _to_container_path(config.pretrained_model_path, workspace_dir=workspace_dir),
        "--script-path",
        _to_container_path(config.script_path, workspace_dir=workspace_dir),
        "--instance-prompt",
        config.instance_prompt,
        "--class-prompt",
        config.class_prompt,
        "--resolution",
        str(config.resolution),
        "--train-batch-size",
        str(config.train_batch_size),
        "--learning-rate",
        str(config.learning_rate),
        "--max-train-steps",
        str(config.max_train_steps),
    ]
    if config.recursive:
        benchmark_args.append("--recursive")
    if config.execute:
        benchmark_args.append("--execute")
    if config.infer_script_path is not None:
        benchmark_args.extend(
            [
                "--infer-script-path",
                _to_container_path(
                    config.infer_script_path,
                    workspace_dir=workspace_dir,
                ),
            ]
        )
    if config.report is not None:
        benchmark_args.extend(
            [
                "--report",
                _to_container_path(config.report, workspace_dir=workspace_dir),
            ]
        )

    compose_base = [
        "docker",
        "compose",
        "-f",
        str(compose_file),
    ]
    environment = {
        "AURALOCK_GPU_COUNT": gpu_count,
        "AURALOCK_BENCHMARK_BASE_IMAGE": config.base_image,
    }

    return DockerLoraBenchmarkPlan(
        build_command=[*compose_base, "build", config.service_name],
        gpu_check_command=[
            "docker",
            "run",
            "--rm",
            "--gpus=all",
            config.gpu_smoke_image,
            "nvidia-smi",
        ],
        run_command=[
            *compose_base,
            "run",
            "--rm",
            config.service_name,
            *benchmark_args,
        ],
        environment=environment,
    )
