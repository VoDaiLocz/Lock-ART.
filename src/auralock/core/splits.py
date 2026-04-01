"""Dataset split methodology utilities for reproducible benchmark evaluation."""

from __future__ import annotations

import hashlib
import json
import random
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class SplitType(Enum):
    """Dataset split role used in reproducible benchmark evaluation."""

    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"
    DEVELOPMENT = "dev"


@dataclass
class SplitMetadata:
    """Metadata describing one split of a dataset.

    Tracks image membership, split method, and a deterministic hash so that
    split assignments can be audited and reproduced across benchmark runs.
    """

    split_type: SplitType
    dataset_name: str
    dataset_version: str
    split_hash: str
    image_ids: list[str]
    split_method: str
    split_ratio: dict[str, float]
    random_seed: int | None = None

    def verify_no_leakage(self, other: SplitMetadata) -> bool:
        """Return True when no images are shared between this split and *other*."""
        return set(self.image_ids).isdisjoint(set(other.image_ids))

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "split_type": self.split_type.value,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "split_hash": self.split_hash,
            "image_ids": self.image_ids,
            "split_method": self.split_method,
            "split_ratio": self.split_ratio,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SplitMetadata:
        """Reconstruct from a serialized dictionary."""
        return cls(
            split_type=SplitType(data["split_type"]),
            dataset_name=str(data["dataset_name"]),
            dataset_version=str(data["dataset_version"]),
            split_hash=str(data["split_hash"]),
            image_ids=list(data["image_ids"]),
            split_method=str(data["split_method"]),
            split_ratio=dict(data["split_ratio"]),
            random_seed=data.get("random_seed"),
        )


def compute_split_hash(
    image_ids: list[str],
    split_type: SplitType,
    seed: int | None,
) -> str:
    """Return a short deterministic SHA-256 hash for a split assignment."""
    content = json.dumps(
        {
            "split_type": split_type.value,
            "image_ids": sorted(image_ids),
            "seed": seed,
        },
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def create_random_split(
    image_paths: list[Path],
    *,
    dataset_name: str = "dataset",
    dataset_version: str = "1.0",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> dict[SplitType, SplitMetadata]:
    """Create deterministic random train/val/test splits with full metadata.

    Parameters
    ----------
    image_paths:
        Paths to all images in the dataset.
    dataset_name:
        Human-readable name for the dataset (stored in metadata).
    dataset_version:
        Version string for the dataset (stored in metadata).
    train_ratio, val_ratio, test_ratio:
        Fractions for each split; must sum to 1.0.
    random_seed:
        Seed for reproducible shuffling.

    Returns
    -------
    dict mapping each :class:`SplitType` to its :class:`SplitMetadata`.
    """
    if not image_paths:
        raise ValueError("image_paths must not be empty.")

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio} = {total:.6f}."
        )

    rng = random.Random(random_seed)
    shuffled = rng.sample(image_paths, len(image_paths))
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    partitions: list[tuple[SplitType, list[Path]]] = [
        (SplitType.TRAIN, shuffled[:n_train]),
        (SplitType.VALIDATION, shuffled[n_train : n_train + n_val]),
        (SplitType.TEST, shuffled[n_train + n_val :]),
    ]

    split_ratio = {
        SplitType.TRAIN.value: train_ratio,
        SplitType.VALIDATION.value: val_ratio,
        SplitType.TEST.value: test_ratio,
    }

    splits: dict[SplitType, SplitMetadata] = {}
    for split_type, images in partitions:
        ids = [str(p) for p in images]
        splits[split_type] = SplitMetadata(
            split_type=split_type,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            split_hash=compute_split_hash(ids, split_type, random_seed),
            image_ids=ids,
            split_method="random",
            split_ratio=split_ratio,
            random_seed=random_seed,
        )

    return splits


def save_split_manifest(
    splits: dict[SplitType, SplitMetadata],
    output_path: Path,
) -> None:
    """Persist split assignments to a JSON manifest for reproducibility.

    Parameters
    ----------
    splits:
        Mapping returned by :func:`create_random_split` (or built manually).
    output_path:
        Destination path for the JSON manifest.
    """
    manifest = {
        split_type.value: split_meta.to_dict()
        for split_type, split_meta in splits.items()
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_split_manifest(manifest_path: Path) -> dict[SplitType, SplitMetadata]:
    """Load a previously saved split manifest.

    Parameters
    ----------
    manifest_path:
        Path to a JSON manifest written by :func:`save_split_manifest`.

    Returns
    -------
    Mapping of :class:`SplitType` to :class:`SplitMetadata`.

    Raises
    ------
    FileNotFoundError
        If *manifest_path* does not exist.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        SplitType(key): SplitMetadata.from_dict(value) for key, value in data.items()
    }


def validate_split_manifest(
    splits: dict[SplitType, SplitMetadata],
) -> list[str]:
    """Validate splits for data leakage and hash integrity.

    Parameters
    ----------
    splits:
        Mapping of split types to their metadata (e.g. from
        :func:`load_split_manifest` or :func:`create_random_split`).

    Returns
    -------
    A list of human-readable issue descriptions.  An empty list means no
    problems were detected.
    """
    issues: list[str] = []
    split_list = list(splits.values())

    for i, split_a in enumerate(split_list):
        for split_b in split_list[i + 1 :]:
            if not split_a.verify_no_leakage(split_b):
                overlap = set(split_a.image_ids) & set(split_b.image_ids)
                issues.append(
                    f"Data leakage detected between '{split_a.split_type.value}' and "
                    f"'{split_b.split_type.value}' splits: "
                    f"{len(overlap)} shared image(s)."
                )

    for split_meta in splits.values():
        computed = compute_split_hash(
            split_meta.image_ids, split_meta.split_type, split_meta.random_seed
        )
        if computed != split_meta.split_hash:
            issues.append(
                f"Hash mismatch for '{split_meta.split_type.value}' split. "
                "Split assignment may have been modified after creation."
            )

    return issues


def warn_non_test_split(split_type: SplitType) -> None:
    """Emit a :class:`UserWarning` when benchmarking on a non-test split.

    The warning is attributed to the direct caller of this function
    (``stacklevel=2``), which is typically a benchmark method.
    """
    if split_type != SplitType.TEST:
        warnings.warn(
            f"Benchmarking on '{split_type.value}' split. "
            "Results may be overfit to this split. "
            "Use the TEST split for final evaluation to avoid bias.",
            UserWarning,
            stacklevel=2,
        )
