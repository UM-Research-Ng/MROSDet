"""用合成数据检查抽样可重复性、配对规则及源文件保护。"""

import json
from pathlib import Path

import pytest
from PIL import Image

from tools.prepare_sample import (
    MODALITIES,
    SPLITS,
    inspect_source,
    label_classes,
    prepare,
    sha256,
)


@pytest.fixture
def source_dataset(tmp_path):
    source = tmp_path / "source"
    serial = 1
    for split, count in (("train", 75), ("val", 24), ("test", 12)):
        for modality in MODALITIES:
            images = source / modality / split / "images"
            labels = source / modality / split / "labels"
            images.mkdir(parents=True)
            labels.mkdir()
            for index in range(count):
                stem = f"{split}_{index:03d}"
                color = (serial % 256, (serial // 256) % 256, (serial * 17) % 256)
                Image.new("RGB", (4, 3), color).save(images / f"{stem}.png")
                (labels / f"{stem}.txt").write_text(
                    f"{index % 9} 0.5 0.5 0.25 0.5\n", encoding="utf-8"
                )
                serial += 1
    return source


def test_default_sample_preserves_splits_hashes_and_source(source_dataset, tmp_path):
    source = source_dataset
    original = {path.relative_to(source): sha256(path) for path in source.rglob("*") if path.is_file()}
    destination = tmp_path / "first"
    summary = prepare(source, destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert len(manifest["samples"]) == 100
    assert summary["selected_pairs"] == 100
    assert summary["all_nine_classes_in_each_modality"] is True
    assert str(source) not in json.dumps(summary)
    assert str(source) not in json.dumps(manifest)
    image_hashes = set()
    for split, count in SPLITS.items():
        assert summary["selected"][split]["pairs"] == count
        assert sum(sample["split"] == split for sample in manifest["samples"]) == count
    for sample in manifest["samples"]:
        assert sample["id"].startswith(sample["split"] + "_")
        for modality in MODALITIES:
            view = sample[modality]
            for kind in ("image", "label"):
                relative = Path(view[kind])
                assert not relative.is_absolute()
                assert relative.parts[:2] == (modality, sample["split"])
                assert sha256(destination / relative) == view[f"sha256_{kind}"]
                assert view[f"sha256_{kind}"] == original[relative]
            assert view["sha256_image"] not in image_hashes
            image_hashes.add(view["sha256_image"])
    assert {path.relative_to(source): sha256(path) for path in source.rglob("*") if path.is_file()} == original
    repeated = tmp_path / "second"
    prepare(source, repeated)
    assert (repeated / "manifest.json").read_bytes() == (destination / "manifest.json").read_bytes()


def test_missing_and_bad_source_pairs_are_excluded_without_repair(source_dataset, tmp_path):
    source = source_dataset
    missing = source / "Sonar/train/labels/train_000.txt"
    missing.unlink()
    corrupt = source / "RGB/train/images/train_001.png"
    corrupt.write_bytes(b"not a decodable image")
    bad_label = source / "RGB/train/labels/train_002.txt"
    bad_label.write_text("9 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    empty = source / "RGB/train/labels/train_003.txt"
    empty.write_text("", encoding="utf-8")
    summary = prepare(source, tmp_path / "subset")
    excluded = {entry["id"]: entry for entry in summary["source_hash_audit"]["excluded_pairs"]}
    assert set(excluded) == {"train_000", "train_001", "train_002"}
    assert excluded["train_000"]["issues"][0]["reason"] == "missing_label"
    assert excluded["train_001"]["issues"][0]["reason"] == "invalid_image"
    assert excluded["train_002"]["issues"][0]["reason"] == "invalid_label"
    assert not missing.exists()
    assert corrupt.read_bytes() == b"not a decodable image"
    assert bad_label.read_text().startswith("9 ")
    assert label_classes(empty) == ()


def test_duplicate_content_is_audited_and_not_published_twice(source_dataset, tmp_path):
    source = source_dataset
    original = source / "RGB/test/images/test_000.png"
    duplicate = source / "RGB/train/images/train_000.png"
    duplicate.write_bytes(original.read_bytes())
    summary = prepare(source, tmp_path / "subset")
    audit = summary["source_hash_audit"]
    assert audit["cross_split_image_content_duplicate_groups"] == 1
    assert "cross_split_duplicate_paths" not in audit
    _, private_audit = inspect_source(source)
    assert set(private_audit["cross_split_duplicate_paths"][0]) == {
        "RGB/train/images/train_000.png", "RGB/test/images/test_000.png"
    }
    manifest = json.loads((tmp_path / "subset/manifest.json").read_text())
    hashes = [sample[modality]["sha256_image"] for sample in manifest["samples"] for modality in MODALITIES]
    assert len(hashes) == len(set(hashes))


def test_insufficient_valid_data_does_not_create_output(source_dataset, tmp_path):
    source = source_dataset
    for index in range(6):
        (source / "RGB/train/labels" / f"train_{index:03d}.txt").write_text("invalid\n")
    output = tmp_path / "not_created"
    with pytest.raises(ValueError, match="only 69 valid pairs"):
        prepare(source, output)
    assert not output.exists()


def test_existing_destination_is_never_overwritten(source_dataset, tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "preserve.txt"
    marker.write_text("keep me")
    with pytest.raises(FileExistsError, match="must not already exist"):
        prepare(source_dataset, destination)
    assert marker.read_text() == "keep me"


def test_destination_cannot_be_inside_source(source_dataset):
    with pytest.raises(ValueError, match="must not be inside"):
        prepare(source_dataset, source_dataset / "subset")


@pytest.mark.parametrize("row", [
    "0 0.5 0.5 0.1",       # 列数不足
    "0.5 0.5 0.5 0.1 0.1", # 类别编号不是整数
    "9 0.5 0.5 0.1 0.1",   # 类别编号超出九类范围
    "0 nan 0.5 0.1 0.1",   # 坐标不是有限数值
    "0 0.5 0.5 0 0.1",     # 框面积为零
    "0 0.01 0.5 0.5 0.1",  # 框超出图像边界
])
def test_illegal_labels_are_rejected(tmp_path, row):
    path = tmp_path / "label.txt"
    path.write_text(row + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        label_classes(path)


def test_empty_label_is_valid_but_missing_label_is_not(tmp_path):
    label = tmp_path / "empty.txt"
    label.write_text("\n \n", encoding="utf-8")
    assert label_classes(label) == ()
    with pytest.raises(FileNotFoundError):
        label_classes(tmp_path / "missing.txt")
