"""Dataset split utilities to prevent benchmark data leakage."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from auralock.core.image import SUPPORTED_EXTENSIONS


def _normalize_image_id(path: str | Path) -> str:
    """Normalize paths for stable manifest hashing and comparisons."""
    return str(Path(path).resolve())


class SplitType(Enum):
    """Canonical split types used across benchmarks."""

    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"
    DEVELOPMENT = "dev"


@dataclass(slots=True)
class SplitMetadata:
    """Reproducible manifest for one split of a dataset."""

    split_type: SplitType
    dataset_name: str
    dataset_version: str
    split_method: str
    split_ratio: dict[str, float]
    image_ids: list[str]
    random_seed: int | None = None
    split_hash: str | None = None
    dataset_root: str | None = None

    def __post_init__(self) -> None:
        self.image_ids = [_normalize_image_id(path) for path in self.image_ids]
        if len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError("image_ids must be unique for each split.")
        if self.split_hash is None:
            self.split_hash = self._compute_split_hash()

    def _compute_split_hash(self) -> str:
        payload = {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "dataset_root": self.dataset_root,
            "image_ids": sorted(self.image_ids),
            "random_seed": self.random_seed,
            "split_method": self.split_method,
            "split_ratio": self.split_ratio,
            "split_type": self.split_type.value,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()[:16]

    @property
    def normalized_image_ids(self) -> set[str]:
        """Normalized image identifiers for membership validation."""
        return set(self.image_ids)

    def verify_no_leakage(self, other: SplitMetadata) -> bool:
        """Check that no images overlap between two splits."""
        return self.normalized_image_ids.isdisjoint(other.normalized_image_ids)

    def contains_all(self, paths: Iterable[str | Path]) -> list[str]:
        """Return any paths missing from the split."""
        normalized = {_normalize_image_id(path) for path in paths}
        return sorted(normalized - self.normalized_image_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "split_type": self.split_type.value,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "dataset_root": self.dataset_root,
            "split_method": self.split_method,
            "split_ratio": self.split_ratio,
            "random_seed": self.random_seed,
            "split_hash": self.split_hash,
            "image_ids": list(self.image_ids),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> SplitMetadata:
        split_type = SplitType(str(payload["split_type"]))
        return cls(
            split_type=split_type,
            dataset_name=str(payload["dataset_name"]),
            dataset_version=str(payload.get("dataset_version", "unknown")),
            dataset_root=(
                str(payload["dataset_root"])
                if payload.get("dataset_root") is not None
                else None
            ),
            split_method=str(payload.get("split_method", "manual")),
            split_ratio=dict(payload.get("split_ratio", {})),
            random_seed=payload.get("random_seed"),  # type: ignore[arg-type]
            split_hash=str(payload.get("split_hash") or ""),
            image_ids=list(payload.get("image_ids", [])),  # type: ignore[list-item]
        )


def _assert_ratio_sum(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not abs(total - 1.0) < 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")


def collect_supported_images(dataset_root: Path) -> list[Path]:
    """Collect supported images under a dataset root."""
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise ValueError("dataset_root must be an existing directory.")
    return [
        candidate
        for candidate in sorted(dataset_root.rglob("*"))
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def create_random_split(
    image_paths: list[Path],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    dataset_name: str = "dataset",
    dataset_version: str = "v1",
    split_method: str = "random",
    dataset_root: Path | None = None,
) -> dict[SplitType, SplitMetadata]:
    """Create a reproducible random split manifest."""
    if not image_paths:
        raise ValueError("image_paths must contain at least one image.")
    _assert_ratio_sum(train_ratio, val_ratio, test_ratio)

    rng = random.Random(random_seed)
    shuffled = list(image_paths)
    rng.shuffle(shuffled)

    n_train = int(len(shuffled) * train_ratio)
    n_val = int(len(shuffled) * val_ratio)
    train_images = shuffled[:n_train]
    val_images = shuffled[n_train : n_train + n_val]
    test_images = shuffled[n_train + n_val :]

    ratio = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    root_str = str(dataset_root.resolve()) if dataset_root is not None else None
    splits = {
        SplitType.TRAIN: SplitMetadata(
            split_type=SplitType.TRAIN,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            dataset_root=root_str,
            split_method=split_method,
            split_ratio=ratio,
            random_seed=random_seed,
            image_ids=[str(path.resolve()) for path in train_images],
        ),
        SplitType.VALIDATION: SplitMetadata(
            split_type=SplitType.VALIDATION,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            dataset_root=root_str,
            split_method=split_method,
            split_ratio=ratio,
            random_seed=random_seed,
            image_ids=[str(path.resolve()) for path in val_images],
        ),
        SplitType.TEST: SplitMetadata(
            split_type=SplitType.TEST,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            dataset_root=root_str,
            split_method=split_method,
            split_ratio=ratio,
            random_seed=random_seed,
            image_ids=[str(path.resolve()) for path in test_images],
        ),
    }
    if not splits[SplitType.TEST].image_ids:
        raise ValueError("test split would be empty; adjust ratios or dataset size.")
    return splits


def save_split_manifest(
    splits: dict[SplitType, SplitMetadata], output_path: Path
) -> None:
    """Persist split metadata to a JSON manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {split_type.value: meta.to_dict() for split_type, meta in splits.items()}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_manifest(manifest_path: Path) -> dict[SplitType, SplitMetadata]:
    """Load a split manifest from disk."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits: dict[SplitType, SplitMetadata] = {}
    for key, value in payload.items():
        split_type = SplitType(key)
        splits[split_type] = SplitMetadata.from_dict(value)
    return splits


def validate_no_overlap(splits: dict[SplitType, SplitMetadata]) -> None:
    """Raise when any split pair overlaps."""
    split_items = list(splits.items())
    for idx, (split_type, split_meta) in enumerate(split_items):
        for other_type, other_meta in split_items[idx + 1 :]:
            if not split_meta.verify_no_leakage(other_meta):
                raise ValueError(
                    f"Split {split_type.value} overlaps with {other_type.value}."
                )
