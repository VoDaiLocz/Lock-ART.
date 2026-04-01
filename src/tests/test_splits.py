"""Tests for dataset split utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from auralock.benchmarks.splits import (
    SplitMetadata,
    SplitType,
    collect_supported_images,
    create_random_split,
    load_split_manifest,
    save_split_manifest,
    validate_no_overlap,
)


def _write_fake_images(root: Path, count: int) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(count):
        path = root / f"img_{index}.png"
        path.write_bytes(b"fake")
        paths.append(path)
    return paths


def test_create_random_split_generates_hash_and_manifest(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    image_paths = _write_fake_images(dataset_root, 6)

    splits = create_random_split(
        image_paths,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        random_seed=0,
        dataset_name="demo",
        dataset_version="v1",
        dataset_root=dataset_root,
    )
    validate_no_overlap(splits)

    assert set(splits) == {
        SplitType.TRAIN,
        SplitType.VALIDATION,
        SplitType.TEST,
    }
    assert splits[SplitType.TEST].split_hash is not None
    assert splits[SplitType.TRAIN].verify_no_leakage(splits[SplitType.TEST])

    manifest_path = tmp_path / "split.json"
    save_split_manifest(splits, manifest_path)
    loaded = load_split_manifest(manifest_path)

    assert loaded[SplitType.TEST].dataset_name == "demo"
    assert loaded[SplitType.TEST].split_hash == splits[SplitType.TEST].split_hash
    assert len(collect_supported_images(dataset_root)) == 6


def test_validate_no_overlap_rejects_duplicate_images(tmp_path: Path):
    img = tmp_path / "x.png"
    img.write_bytes(b"fake")
    train_meta = SplitMetadata(
        split_type=SplitType.TRAIN,
        dataset_name="demo",
        dataset_version="v1",
        split_method="manual",
        split_ratio={"train": 1.0},
        random_seed=None,
        dataset_root=str(tmp_path),
        image_ids=[str(img)],
    )
    test_meta = SplitMetadata(
        split_type=SplitType.TEST,
        dataset_name="demo",
        dataset_version="v1",
        split_method="manual",
        split_ratio={"test": 1.0},
        random_seed=None,
        dataset_root=str(tmp_path),
        image_ids=[str(img)],
    )

    with pytest.raises(ValueError):
        validate_no_overlap({SplitType.TRAIN: train_meta, SplitType.TEST: test_meta})
