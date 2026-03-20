"""Optional UI helpers with lazy imports."""

from __future__ import annotations

from typing import Any


def create_ui(*args: Any, **kwargs: Any):
    from auralock.ui.gradio_app import create_ui as _create_ui

    return _create_ui(*args, **kwargs)


def launch_app(*args: Any, **kwargs: Any):
    from auralock.ui.gradio_app import main as _main

    return _main(*args, **kwargs)


__all__ = ["create_ui", "launch_app"]
