"""Dataset split methodology utilities for reproducible benchmark evaluation.

This module re-exports from :mod:`auralock.core.splits` for convenience.
The canonical implementation lives in the core package to avoid circular imports.
"""

from auralock.core.splits import (
    SplitMetadata,
    SplitType,
    compute_split_hash,
    create_random_split,
    load_split_manifest,
    save_split_manifest,
    validate_split_manifest,
    warn_non_test_split,
)

__all__ = [
    "SplitMetadata",
    "SplitType",
    "compute_split_hash",
    "create_random_split",
    "load_split_manifest",
    "save_split_manifest",
    "validate_split_manifest",
    "warn_non_test_split",
]
