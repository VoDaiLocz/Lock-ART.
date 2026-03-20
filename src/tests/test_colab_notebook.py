"""Tests for the Colab LoRA benchmark notebook."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/AuraLock_LoRA_Benchmark_Colab.ipynb")


def _load_notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_colab_notebook_exists_and_uses_python3_kernel():
    """The Colab benchmark notebook should exist and declare a Python 3 kernel."""
    assert NOTEBOOK_PATH.exists()

    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"


def test_colab_notebook_contains_runtime_setup_guidance():
    """The notebook should guide users through GPU runtime setup in Colab."""
    notebook = _load_notebook()
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )

    assert "Runtime > Change runtime type > GPU" in markdown
    assert "Google Drive" in markdown
    assert "dry-run" in markdown


def test_colab_notebook_contains_install_clone_and_benchmark_commands():
    """The notebook should install benchmark deps and run the LoRA CLI flow."""
    notebook = _load_notebook()
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert 'pip install -e ".[benchmark]"' in code
    assert "git clone https://github.com/VinAIResearch/Anti-DreamBooth.git" in code
    assert "auralock benchmark-lora" in code
    assert "--execute" in code
