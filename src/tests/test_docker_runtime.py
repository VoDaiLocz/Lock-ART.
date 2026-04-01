"""Tests for Docker-based LoRA benchmark orchestration."""

from __future__ import annotations

import subprocess
from pathlib import Path

from typer.testing import CliRunner

from auralock.cli import app


def test_build_docker_lora_benchmark_plan_maps_workspace_paths(tmp_path: Path):
    """Docker planner should convert workspace-local paths into container paths."""
    from auralock.benchmarks.docker_runtime import (
        DockerLoraBenchmarkConfig,
        build_docker_lora_benchmark_plan,
    )

    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    compose_file = workspace_dir / "docker-compose.benchmark.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    input_path = workspace_dir / "input" / "art.png"
    input_path.parent.mkdir()
    input_path.write_bytes(b"fake")
    work_dir = workspace_dir / "runs"
    script_path = workspace_dir / "train_dreambooth_lora.py"
    script_path.write_text("print('train')\n", encoding="utf-8")
    infer_script_path = workspace_dir / "infer.py"
    infer_script_path.write_text("print('infer')\n", encoding="utf-8")
    model_dir = workspace_dir / "sd15"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    report_path = workspace_dir / "output" / "report.json"

    plan = build_docker_lora_benchmark_plan(
        DockerLoraBenchmarkConfig(
            workspace_dir=workspace_dir,
            input_path=input_path,
            work_dir=work_dir,
            pretrained_model_path=model_dir,
            script_path=script_path,
            infer_script_path=infer_script_path,
            instance_prompt="a sks painting",
            class_prompt="a painting",
            profiles=("safe", "strong"),
            recursive=True,
            execute=True,
            report=report_path,
            gpu_count="2",
        )
    )

    assert plan.build_command[:4] == [
        "docker",
        "compose",
        "-f",
        str(compose_file),
    ]
    assert plan.run_command[:5] == [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "run",
    ]
    assert "/workspace/input/art.png" in plan.run_command
    assert "/workspace/runs" in plan.run_command
    assert "/workspace/train_dreambooth_lora.py" in plan.run_command
    assert "/workspace/infer.py" in plan.run_command
    assert "/workspace/sd15" in plan.run_command
    assert "/workspace/output/report.json" in plan.run_command
    assert "--execute" in plan.run_command
    assert "--recursive" in plan.run_command
    assert plan.environment["AURALOCK_GPU_COUNT"] == "2"


def test_build_docker_lora_benchmark_plan_rejects_paths_outside_workspace(
    tmp_path: Path,
):
    """Docker planner should reject paths that are not covered by the workspace mount."""
    from auralock.benchmarks.docker_runtime import (
        DockerLoraBenchmarkConfig,
        build_docker_lora_benchmark_plan,
    )

    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    compose_file = workspace_dir / "docker-compose.benchmark.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    input_path = workspace_dir / "input.png"
    input_path.write_bytes(b"fake")
    script_path = workspace_dir / "train_dreambooth_lora.py"
    script_path.write_text("print('train')\n", encoding="utf-8")
    model_dir = workspace_dir / "sd15"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    outside_report = tmp_path / "outside-report.json"

    try:
        build_docker_lora_benchmark_plan(
            DockerLoraBenchmarkConfig(
                workspace_dir=workspace_dir,
                input_path=input_path,
                work_dir=workspace_dir / "runs",
                pretrained_model_path=model_dir,
                script_path=script_path,
                infer_script_path=None,
                instance_prompt="a sks painting",
                class_prompt="a painting",
                report=outside_report,
            )
        )
    except ValueError as exc:
        assert "Path must stay inside the workspace root" in str(exc)
    else:
        raise AssertionError("Expected workspace path validation to fail")


def test_build_docker_lora_benchmark_plan_requires_diffusers_model_dir(
    tmp_path: Path,
):
    """Docker planner should reject model paths that are not Diffusers directories."""
    from auralock.benchmarks.docker_runtime import (
        DockerLoraBenchmarkConfig,
        build_docker_lora_benchmark_plan,
    )

    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    compose_file = workspace_dir / "docker-compose.benchmark.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    input_path = workspace_dir / "input.png"
    input_path.write_bytes(b"fake")
    script_path = workspace_dir / "train_dreambooth_lora.py"
    script_path.write_text("print('train')\n", encoding="utf-8")
    invalid_model_dir = workspace_dir / "sd15"
    invalid_model_dir.mkdir()

    try:
        build_docker_lora_benchmark_plan(
            DockerLoraBenchmarkConfig(
                workspace_dir=workspace_dir,
                input_path=input_path,
                work_dir=workspace_dir / "runs",
                pretrained_model_path=invalid_model_dir,
                script_path=script_path,
                infer_script_path=None,
                instance_prompt="a sks painting",
                class_prompt="a painting",
            )
        )
    except ValueError as exc:
        assert "model_index.json" in str(exc)
    else:
        raise AssertionError("Expected Diffusers checkpoint validation to fail")


def test_benchmark_lora_docker_cli_runs_build_and_execute(monkeypatch, tmp_path: Path):
    """CLI should orchestrate Docker build, GPU check, and benchmark execution."""
    from auralock.benchmarks.docker_runtime import DockerLoraBenchmarkPlan

    recorded: list[list[str]] = []

    def fake_plan_builder(config):
        return DockerLoraBenchmarkPlan(
            build_command=["docker", "compose", "build"],
            gpu_check_command=["docker", "run", "nvidia-smi"],
            run_command=["docker", "compose", "run", "auralock", "benchmark-lora"],
            environment={"AURALOCK_GPU_COUNT": "all"},
        )

    def fake_run(command, cwd=None, env=None, check=None):
        recorded.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "auralock.cli.build_docker_lora_benchmark_plan", fake_plan_builder
    )
    monkeypatch.setattr("auralock.cli.subprocess.run", fake_run)

    compose_file = tmp_path / "docker-compose.benchmark.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"fake")
    script_path = tmp_path / "train_dreambooth_lora.py"
    script_path.write_text("print('train')\n", encoding="utf-8")
    model_dir = tmp_path / "sd15"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark-lora-docker",
            str(input_path),
            "--execute",
            "--workspace-dir",
            str(tmp_path),
            "--compose-file",
            str(compose_file),
            "--pretrained-model-path",
            str(model_dir),
            "--script-path",
            str(script_path),
            "--instance-prompt",
            "a sks painting",
            "--class-prompt",
            "a painting",
        ],
    )

    assert result.exit_code == 0
    assert recorded == [
        ["docker", "compose", "build"],
        ["docker", "run", "nvidia-smi"],
        ["docker", "compose", "run", "auralock", "benchmark-lora"],
    ]
