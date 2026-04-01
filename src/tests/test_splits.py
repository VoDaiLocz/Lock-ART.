"""Tests for dataset split methodology utilities."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
from typer.testing import CliRunner

from auralock.benchmarks.splits import (
    SplitMetadata,
    SplitType,
    compute_split_hash,
    create_random_split,
    load_split_manifest,
    save_split_manifest,
    validate_split_manifest,
    warn_non_test_split,
)
from auralock.cli import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_images(base: Path, names: list[str]) -> list[Path]:
    """Create empty stub files at *base* and return their paths."""
    base.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        p = base / name
        p.write_bytes(b"stub")
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# SplitType
# ---------------------------------------------------------------------------


def test_split_type_values():
    assert SplitType.TRAIN.value == "train"
    assert SplitType.VALIDATION.value == "val"
    assert SplitType.TEST.value == "test"
    assert SplitType.DEVELOPMENT.value == "dev"


# ---------------------------------------------------------------------------
# SplitMetadata
# ---------------------------------------------------------------------------


def test_split_metadata_verify_no_leakage_disjoint():
    a = SplitMetadata(
        split_type=SplitType.TRAIN,
        dataset_name="d",
        dataset_version="1",
        split_hash="abc",
        image_ids=["a.png", "b.png"],
        split_method="random",
        split_ratio={"train": 0.7},
    )
    b = SplitMetadata(
        split_type=SplitType.TEST,
        dataset_name="d",
        dataset_version="1",
        split_hash="def",
        image_ids=["c.png", "d.png"],
        split_method="random",
        split_ratio={"test": 0.15},
    )
    assert a.verify_no_leakage(b) is True


def test_split_metadata_verify_no_leakage_overlap():
    a = SplitMetadata(
        split_type=SplitType.TRAIN,
        dataset_name="d",
        dataset_version="1",
        split_hash="abc",
        image_ids=["a.png", "b.png"],
        split_method="random",
        split_ratio={"train": 0.7},
    )
    b = SplitMetadata(
        split_type=SplitType.TEST,
        dataset_name="d",
        dataset_version="1",
        split_hash="def",
        image_ids=["b.png", "c.png"],
        split_method="random",
        split_ratio={"test": 0.15},
    )
    assert a.verify_no_leakage(b) is False


def test_split_metadata_to_dict_and_from_dict():
    meta = SplitMetadata(
        split_type=SplitType.VALIDATION,
        dataset_name="mydata",
        dataset_version="2.0",
        split_hash="deadbeef",
        image_ids=["x.png"],
        split_method="random",
        split_ratio={"val": 0.15},
        random_seed=99,
    )
    roundtripped = SplitMetadata.from_dict(meta.to_dict())
    assert roundtripped.split_type == SplitType.VALIDATION
    assert roundtripped.dataset_name == "mydata"
    assert roundtripped.dataset_version == "2.0"
    assert roundtripped.split_hash == "deadbeef"
    assert roundtripped.image_ids == ["x.png"]
    assert roundtripped.split_method == "random"
    assert roundtripped.split_ratio == {"val": 0.15}
    assert roundtripped.random_seed == 99


# ---------------------------------------------------------------------------
# create_random_split
# ---------------------------------------------------------------------------


def test_create_random_split_basic(tmp_path: Path):
    images = _make_images(tmp_path, [f"{i}.png" for i in range(10)])
    splits = create_random_split(images, random_seed=0)

    assert set(splits) == {SplitType.TRAIN, SplitType.VALIDATION, SplitType.TEST}
    total = sum(len(s.image_ids) for s in splits.values())
    assert total == 10

    train_meta = splits[SplitType.TRAIN]
    assert train_meta.split_type == SplitType.TRAIN
    assert train_meta.split_method == "random"
    assert train_meta.random_seed == 0


def test_create_random_split_ratios_must_sum_to_one(tmp_path: Path):
    images = _make_images(tmp_path, ["a.png", "b.png"])
    with pytest.raises(ValueError, match="sum to 1.0"):
        create_random_split(images, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


def test_create_random_split_empty_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="must not be empty"):
        create_random_split([])


def test_create_random_split_no_leakage_across_splits(tmp_path: Path):
    images = _make_images(tmp_path, [f"{i}.png" for i in range(20)])
    splits = create_random_split(images)
    split_list = list(splits.values())
    for i, a in enumerate(split_list):
        for b in split_list[i + 1 :]:
            assert a.verify_no_leakage(
                b
            ), f"Leakage between {a.split_type.value} and {b.split_type.value}"


def test_create_random_split_deterministic(tmp_path: Path):
    images = _make_images(tmp_path, [f"{i}.png" for i in range(10)])
    splits1 = create_random_split(images, random_seed=42)
    splits2 = create_random_split(images, random_seed=42)
    assert splits1[SplitType.TEST].image_ids == splits2[SplitType.TEST].image_ids


def test_create_random_split_hash_stored_in_metadata(tmp_path: Path):
    images = _make_images(tmp_path, [f"{i}.png" for i in range(6)])
    splits = create_random_split(images, random_seed=7)
    for split_type, meta in splits.items():
        expected = compute_split_hash(meta.image_ids, split_type, 7)
        assert meta.split_hash == expected


# ---------------------------------------------------------------------------
# save_split_manifest / load_split_manifest
# ---------------------------------------------------------------------------


def test_save_and_load_split_manifest_roundtrip(tmp_path: Path):
    images = _make_images(tmp_path / "data", [f"{i}.png" for i in range(8)])
    splits = create_random_split(images, random_seed=1)

    manifest_path = tmp_path / "splits.json"
    save_split_manifest(splits, manifest_path)

    assert manifest_path.exists()
    loaded = load_split_manifest(manifest_path)

    assert set(loaded) == {SplitType.TRAIN, SplitType.VALIDATION, SplitType.TEST}
    for split_type in splits:
        assert (
            splits[split_type].image_ids == loaded[split_type].image_ids
        ), f"image_ids mismatch for {split_type}"
        assert splits[split_type].split_hash == loaded[split_type].split_hash


def test_load_split_manifest_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_split_manifest(tmp_path / "nonexistent.json")


def test_save_split_manifest_creates_parent_dirs(tmp_path: Path):
    images = _make_images(tmp_path, ["a.png"])
    splits = create_random_split(images, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)
    output = tmp_path / "deep" / "nested" / "splits.json"
    save_split_manifest(splits, output)
    assert output.exists()


# ---------------------------------------------------------------------------
# validate_split_manifest
# ---------------------------------------------------------------------------


def test_validate_split_manifest_clean(tmp_path: Path):
    images = _make_images(tmp_path, [f"{i}.png" for i in range(10)])
    splits = create_random_split(images)
    issues = validate_split_manifest(splits)
    assert issues == []


def test_validate_split_manifest_detects_leakage():
    shared_id = "shared.png"
    meta_train = SplitMetadata(
        split_type=SplitType.TRAIN,
        dataset_name="d",
        dataset_version="1",
        split_hash=compute_split_hash([shared_id, "a.png"], SplitType.TRAIN, None),
        image_ids=[shared_id, "a.png"],
        split_method="manual",
        split_ratio={"train": 0.7},
    )
    meta_test = SplitMetadata(
        split_type=SplitType.TEST,
        dataset_name="d",
        dataset_version="1",
        split_hash=compute_split_hash([shared_id, "b.png"], SplitType.TEST, None),
        image_ids=[shared_id, "b.png"],
        split_method="manual",
        split_ratio={"test": 0.15},
    )
    issues = validate_split_manifest(
        {SplitType.TRAIN: meta_train, SplitType.TEST: meta_test}
    )
    assert any("leakage" in issue.lower() for issue in issues)


def test_validate_split_manifest_detects_hash_mismatch():
    meta = SplitMetadata(
        split_type=SplitType.TEST,
        dataset_name="d",
        dataset_version="1",
        split_hash="tampered_hash",
        image_ids=["a.png", "b.png"],
        split_method="random",
        split_ratio={"test": 0.15},
        random_seed=42,
    )
    issues = validate_split_manifest({SplitType.TEST: meta})
    assert any("hash mismatch" in issue.lower() for issue in issues)


# ---------------------------------------------------------------------------
# warn_non_test_split
# ---------------------------------------------------------------------------


def test_warn_non_test_split_emits_warning_for_train():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_non_test_split(SplitType.TRAIN)
    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)
    assert "train" in str(caught[0].message).lower()


def test_warn_non_test_split_silent_for_test():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_non_test_split(SplitType.TEST)
    assert len(caught) == 0


# ---------------------------------------------------------------------------
# ProtectionService.benchmark_file / benchmark_directory with split_metadata
# ---------------------------------------------------------------------------


def test_benchmark_file_with_test_split_metadata(tmp_path: Path):
    """benchmark_file should succeed silently when image is in the TEST split."""
    from PIL import Image as PILImage

    from auralock.core.pipeline import ImageNetModelAdapter
    from auralock.services import ProtectionService

    from .test_pipeline import RecordingClassifier
    from .test_stylecloak import DummyStyleFeatureExtractor

    img_path = tmp_path / "art.png"
    PILImage.new("RGB", (32, 32), color="blue").save(img_path)

    # Manually build a TEST split with this image
    test_meta = SplitMetadata(
        split_type=SplitType.TEST,
        dataset_name="d",
        dataset_version="1",
        split_hash=compute_split_hash([str(img_path)], SplitType.TEST, None),
        image_ids=[str(img_path)],
        split_method="manual",
        split_ratio={"test": 1.0},
    )

    service = ProtectionService(
        model=ImageNetModelAdapter(RecordingClassifier()),
        style_feature_extractor=DummyStyleFeatureExtractor(),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = service.benchmark_file(
            img_path, profiles=("safe",), split_metadata=test_meta
        )

    assert summary.split_metadata is not None
    assert summary.split_metadata.split_type == SplitType.TEST
    # No warnings should fire for TEST split
    assert all(not issubclass(w.category, UserWarning) for w in caught)


def test_benchmark_file_warns_on_train_split(tmp_path: Path):
    """benchmark_file should emit a UserWarning when using a TRAIN split."""
    from PIL import Image as PILImage

    from auralock.core.pipeline import ImageNetModelAdapter
    from auralock.services import ProtectionService

    from .test_pipeline import RecordingClassifier
    from .test_stylecloak import DummyStyleFeatureExtractor

    img_path = tmp_path / "art.png"
    PILImage.new("RGB", (32, 32), color="red").save(img_path)

    train_meta = SplitMetadata(
        split_type=SplitType.TRAIN,
        dataset_name="d",
        dataset_version="1",
        split_hash=compute_split_hash([str(img_path)], SplitType.TRAIN, None),
        image_ids=[str(img_path)],
        split_method="manual",
        split_ratio={"train": 1.0},
    )

    service = ProtectionService(
        model=ImageNetModelAdapter(RecordingClassifier()),
        style_feature_extractor=DummyStyleFeatureExtractor(),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = service.benchmark_file(
            img_path, profiles=("safe",), split_metadata=train_meta
        )

    assert summary.split_metadata is not None
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert "train" in str(user_warnings[0].message).lower()


def test_benchmark_file_raises_when_image_not_in_split(tmp_path: Path):
    """benchmark_file should raise ValueError if the image is not in the split."""
    from PIL import Image as PILImage

    from auralock.core.pipeline import ImageNetModelAdapter
    from auralock.services import ProtectionService

    from .test_pipeline import RecordingClassifier
    from .test_stylecloak import DummyStyleFeatureExtractor

    img_path = tmp_path / "art.png"
    PILImage.new("RGB", (32, 32), color="green").save(img_path)

    test_meta = SplitMetadata(
        split_type=SplitType.TEST,
        dataset_name="d",
        dataset_version="1",
        split_hash="x",
        image_ids=["other_image.png"],  # does NOT contain img_path
        split_method="manual",
        split_ratio={"test": 1.0},
    )

    service = ProtectionService(
        model=ImageNetModelAdapter(RecordingClassifier()),
        style_feature_extractor=DummyStyleFeatureExtractor(),
    )

    with pytest.raises(ValueError, match="not in the declared"):
        service.benchmark_file(img_path, profiles=("safe",), split_metadata=test_meta)


def test_benchmark_summary_to_report_dict_includes_split_metadata(tmp_path: Path):
    """BenchmarkSummary.to_report_dict should include split_metadata when set."""
    from PIL import Image as PILImage

    from auralock.core.pipeline import ImageNetModelAdapter
    from auralock.services import ProtectionService

    from .test_pipeline import RecordingClassifier
    from .test_stylecloak import DummyStyleFeatureExtractor

    img_path = tmp_path / "art.png"
    PILImage.new("RGB", (32, 32), color="white").save(img_path)

    test_meta = SplitMetadata(
        split_type=SplitType.TEST,
        dataset_name="d",
        dataset_version="1",
        split_hash=compute_split_hash([str(img_path)], SplitType.TEST, None),
        image_ids=[str(img_path)],
        split_method="manual",
        split_ratio={"test": 1.0},
    )

    service = ProtectionService(
        model=ImageNetModelAdapter(RecordingClassifier()),
        style_feature_extractor=DummyStyleFeatureExtractor(),
    )
    summary = service.benchmark_file(
        img_path, profiles=("safe",), split_metadata=test_meta
    )
    report = summary.to_report_dict()

    assert report["split_metadata"] is not None
    assert report["split_metadata"]["split_type"] == "test"
    assert report["split_metadata"]["dataset_name"] == "d"


# ---------------------------------------------------------------------------
# CLI: split create
# ---------------------------------------------------------------------------


def test_split_create_cli_writes_manifest(tmp_path: Path):
    """CLI 'split create' should write a valid JSON manifest."""
    from PIL import Image as PILImage

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    for i in range(10):
        PILImage.new("RGB", (8, 8), color="red").save(dataset_dir / f"{i}.png")

    manifest_path = tmp_path / "splits.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "split",
            "create",
            str(dataset_dir),
            "--output",
            str(manifest_path),
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0, result.output
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "train" in data
    assert "val" in data
    assert "test" in data
    total_images = sum(len(v["image_ids"]) for v in data.values())
    assert total_images == 10


def test_split_create_cli_fails_on_missing_dir(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "split",
            "create",
            str(tmp_path / "nonexistent"),
            "--output",
            str(tmp_path / "splits.json"),
        ],
    )
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "Error" in result.output


def test_split_create_cli_fails_on_bad_ratios(tmp_path: Path):
    from PIL import Image as PILImage

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    PILImage.new("RGB", (8, 8)).save(dataset_dir / "a.png")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "split",
            "create",
            str(dataset_dir),
            "--output",
            str(tmp_path / "splits.json"),
            "--train-ratio",
            "0.5",
            "--val-ratio",
            "0.3",
            "--test-ratio",
            "0.3",
        ],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI: split validate
# ---------------------------------------------------------------------------


def test_split_validate_cli_clean_manifest(tmp_path: Path):
    """CLI 'split validate' should exit 0 for a valid manifest."""
    from PIL import Image as PILImage

    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    images = []
    for i in range(6):
        p = dataset_dir / f"{i}.png"
        PILImage.new("RGB", (8, 8)).save(p)
        images.append(p)

    splits = create_random_split(images)
    manifest_path = tmp_path / "splits.json"
    save_split_manifest(splits, manifest_path)

    runner = CliRunner()
    result = runner.invoke(app, ["split", "validate", str(manifest_path)])

    assert result.exit_code == 0, result.output
    assert "valid" in result.output.lower() or "No issues" in result.output


def test_split_validate_cli_fails_on_leakage(tmp_path: Path):
    """CLI 'split validate' should exit 1 when leakage is detected."""
    shared = "shared.png"
    meta_train = SplitMetadata(
        split_type=SplitType.TRAIN,
        dataset_name="d",
        dataset_version="1",
        split_hash=compute_split_hash([shared], SplitType.TRAIN, None),
        image_ids=[shared],
        split_method="manual",
        split_ratio={"train": 0.7},
    )
    meta_test = SplitMetadata(
        split_type=SplitType.TEST,
        dataset_name="d",
        dataset_version="1",
        split_hash=compute_split_hash([shared], SplitType.TEST, None),
        image_ids=[shared],
        split_method="manual",
        split_ratio={"test": 0.15},
    )
    manifest_path = tmp_path / "bad_splits.json"
    save_split_manifest(
        {SplitType.TRAIN: meta_train, SplitType.TEST: meta_test}, manifest_path
    )

    runner = CliRunner()
    result = runner.invoke(app, ["split", "validate", str(manifest_path)])

    assert result.exit_code != 0
    assert "leakage" in result.output.lower()


def test_split_validate_cli_fails_on_missing_file(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(app, ["split", "validate", str(tmp_path / "missing.json")])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI: benchmark with --split-manifest / --split-type
# ---------------------------------------------------------------------------


def test_benchmark_cli_with_split_manifest(monkeypatch, tmp_path: Path):
    """The benchmark CLI should pass split metadata to the service when a manifest is given."""
    from PIL import Image as PILImage

    captured: dict[str, object] = {}

    class FakeSummary:
        profile_summaries = {
            "safe": {
                "image_count": 1,
                "avg_psnr_db": 38.0,
                "avg_ssim": 0.95,
                "avg_protection_score": 10.0,
                "avg_runtime_sec": 0.5,
            }
        }
        split_metadata = None

        def to_report_dict(self):
            return {
                "input_path": str(tmp_path),
                "image_count": 1,
                "entries": [],
                "profile_summaries": self.profile_summaries,
                "split_metadata": None,
            }

    class FakeService:
        def benchmark_file(self, input_path, **kwargs):
            captured["split_metadata"] = kwargs.get("split_metadata")
            return FakeSummary()

        def benchmark_directory(self, input_path, **kwargs):
            captured["split_metadata"] = kwargs.get("split_metadata")
            return FakeSummary()

    monkeypatch.setattr("auralock.cli.ProtectionService", FakeService)

    # Build manifest with the image in the test split
    img_path = tmp_path / "img" / "a.png"
    img_path.parent.mkdir(parents=True)
    PILImage.new("RGB", (8, 8)).save(img_path)

    test_meta = SplitMetadata(
        split_type=SplitType.TEST,
        dataset_name="d",
        dataset_version="1",
        split_hash=compute_split_hash([str(img_path)], SplitType.TEST, 42),
        image_ids=[str(img_path)],
        split_method="manual",
        split_ratio={"test": 1.0},
        random_seed=42,
    )
    manifest_path = tmp_path / "splits.json"
    save_split_manifest({SplitType.TEST: test_meta}, manifest_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            str(img_path),
            "--profiles",
            "safe",
            "--split-manifest",
            str(manifest_path),
            "--split-type",
            "test",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured.get("split_metadata") is not None
    assert captured["split_metadata"].split_type == SplitType.TEST


def test_benchmark_cli_warns_on_train_split(monkeypatch, tmp_path: Path):
    """The benchmark CLI should print a warning when using a train split."""
    from PIL import Image as PILImage

    class FakeSummary:
        profile_summaries = {
            "safe": {
                "image_count": 1,
                "avg_psnr_db": 38.0,
                "avg_ssim": 0.95,
                "avg_protection_score": 10.0,
                "avg_runtime_sec": 0.5,
            }
        }
        split_metadata = None

        def to_report_dict(self):
            return {
                "input_path": "",
                "image_count": 1,
                "entries": [],
                "profile_summaries": {},
                "split_metadata": None,
            }

    class FakeService:
        def benchmark_file(self, input_path, **kwargs):
            return FakeSummary()

        def benchmark_directory(self, input_path, **kwargs):
            return FakeSummary()

    monkeypatch.setattr("auralock.cli.ProtectionService", FakeService)

    img_path = tmp_path / "a.png"
    PILImage.new("RGB", (8, 8)).save(img_path)

    train_meta = SplitMetadata(
        split_type=SplitType.TRAIN,
        dataset_name="d",
        dataset_version="1",
        split_hash=compute_split_hash([str(img_path)], SplitType.TRAIN, 1),
        image_ids=[str(img_path)],
        split_method="manual",
        split_ratio={"train": 1.0},
        random_seed=1,
    )
    manifest_path = tmp_path / "splits.json"
    save_split_manifest({SplitType.TRAIN: train_meta}, manifest_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            str(img_path),
            "--profiles",
            "safe",
            "--split-manifest",
            str(manifest_path),
            "--split-type",
            "train",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "WARNING" in result.output or "overfit" in result.output.lower()


def test_benchmark_cli_invalid_split_type(tmp_path: Path):
    """The benchmark CLI should exit with error for invalid --split-type."""
    from PIL import Image as PILImage

    img_path = tmp_path / "a.png"
    PILImage.new("RGB", (8, 8)).save(img_path)

    splits = create_random_split(
        [img_path], train_ratio=1.0, val_ratio=0.0, test_ratio=0.0
    )
    manifest_path = tmp_path / "splits.json"
    save_split_manifest(splits, manifest_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            str(img_path),
            "--profiles",
            "safe",
            "--split-manifest",
            str(manifest_path),
            "--split-type",
            "invalid_type",
        ],
    )
    assert result.exit_code != 0
