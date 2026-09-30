#!/usr/bin/env python3
"""从 UMOD 抽取固定的 100 对示例，保留原有 train/val/test 归属。

先检查源图像与标签，再写入尚不存在的目标目录；记录并排除不合法样本，
不修改源数据。抽样使用固定随机种子，兼顾两路类别覆盖并排除完全重复图像。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

SPLITS = {"train": 70, "val": 20, "test": 10}
MODALITIES = ("RGB", "Sonar")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NUM_CLASSES = 9


@dataclass(frozen=True)
class View:
    image: Path
    label: Path
    image_hash: str
    label_hash: str
    width: int
    height: int
    classes: tuple[int, ...]


@dataclass(frozen=True)
class Pair:
    split: str
    stem: str
    views: tuple[View, View]

    @property
    def tokens(self) -> set[tuple[str, int]]:
        return {(modality, cls) for modality, view in zip(MODALITIES, self.views)
                for cls in view.classes}

    @property
    def hashes(self) -> set[str]:
        return {view.image_hash for view in self.views}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def indexed_files(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"Required directory is missing: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.name.startswith("."):
            continue
        if path.is_dir():
            raise ValueError(f"Nested image/label directories are unsupported: {path}")
        if path.suffix.lower() not in suffixes:
            continue
        if path.stem in result:
            raise ValueError(f"Ambiguous duplicate stem: {result[path.stem]} and {path}")
        result[path.stem] = path
    return result


def label_classes(path: Path) -> tuple[int, ...]:
    classes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected five normalized detection label fields: {path}:{line_number}")
        try:
            values = [float(value) for value in fields]
        except ValueError as exc:
            raise ValueError(f"Non-numeric label: {path}:{line_number}") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite label: {path}:{line_number}")
        cls, x, y, width, height = values
        if not cls.is_integer() or not 0 <= cls < NUM_CLASSES:
            raise ValueError(f"Class must be an integer in [0, 8]: {path}:{line_number}")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"Invalid normalized box: {path}:{line_number}")
        tolerance = 1e-5
        if (x - width / 2 < -tolerance or y - height / 2 < -tolerance
                or x + width / 2 > 1 + tolerance or y + height / 2 > 1 + tolerance):
            raise ValueError(f"Box extends outside image: {path}:{line_number}")
        classes.append(int(cls))
    # 空标签文件表示背景，与标签文件缺失分别处理。
    return tuple(classes)


def inspect_source(source: Path) -> tuple[dict[str, list[Pair]], dict]:
    candidates: dict[str, list[Pair]] = {}
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    exclusions = []
    inspected_counts = {}
    for split in SPLITS:
        image_maps = []
        label_maps = []
        for modality in MODALITIES:
            images = indexed_files(source / modality / split / "images", IMAGE_SUFFIXES)
            labels = indexed_files(source / modality / split / "labels", {".txt"})
            image_maps.append(images)
            label_maps.append(labels)
        inspected_counts[split] = {
            modality: {"images": len(images), "labels": len(labels)}
            for modality, images, labels in zip(MODALITIES, image_maps, label_maps)
        }
        pairs = []
        # 检查所有已识别的图像和标签，包括无法配对的文件。
        stems = set().union(*(set(files) for files in image_maps + label_maps))
        for stem in sorted(stems):
            views = []
            issues = []
            for modality, images, labels in zip(MODALITIES, image_maps, label_maps):
                image_path, label_path = images.get(stem), labels.get(stem)
                classes = None
                label_hash = None
                image_hash = None
                width = height = 0
                if label_path is None:
                    issues.append({"modality": modality, "reason": "missing_label"})
                else:
                    try:
                        classes = label_classes(label_path)
                        label_hash = sha256(label_path)
                    except (OSError, ValueError) as exc:
                        # 诊断只保留相对路径，不写入本机源数据目录。
                        issues.append({"modality": modality, "reason": "invalid_label",
                                       "detail": str(exc).replace(str(source) + "/", "")})
                if image_path is None:
                    issues.append({"modality": modality, "reason": "missing_image"})
                else:
                    try:
                        image_hash = sha256(image_path)
                        hashes[image_hash].append((split, image_path.relative_to(source).as_posix()))
                        with Image.open(image_path) as image:
                            image.verify()
                        with Image.open(image_path) as image:
                            image.load()
                            width, height = image.size
                        if min(width, height) <= 0:
                            raise ValueError("Image dimensions must be positive")
                    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
                        issues.append({"modality": modality, "reason": "invalid_image",
                                       "detail": str(exc).replace(str(source) + "/", "")})
                if (image_path is not None and label_path is not None
                        and image_hash is not None and label_hash is not None
                        and classes is not None and min(width, height) > 0):
                    views.append(View(image_path, label_path, image_hash, label_hash,
                                      width, height, classes))
            if not issues and len({view.image_hash for view in views}) != len(MODALITIES):
                issues.append({"reason": "identical_optical_sonar_image_content"})
            if issues:
                exclusions.append({"split": split, "id": stem, "issues": issues})
            else:
                pairs.append(Pair(split, stem, tuple(views)))
        if len(pairs) < SPLITS[split]:
            raise ValueError(f"{split} has only {len(pairs)} valid pairs; need {SPLITS[split]}. "
                             f"Excluded {sum(item['split'] == split for item in exclusions)} source stems.")
        candidates[split] = pairs
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    cross_split_groups = [paths for paths in duplicate_groups if len({s for s, _ in paths}) > 1]
    report = {
        "inspected_files": inspected_counts,
        "excluded_pair_count": len(exclusions),
        "excluded_pairs": exclusions,
        "image_content_duplicate_groups": len(duplicate_groups),
        "cross_split_image_content_duplicate_groups": len(cross_split_groups),
        "cross_split_duplicate_paths": [[path for _, path in group] for group in cross_split_groups],
    }
    return candidates, report


def select_pairs(candidates: dict[str, list[Pair]], seed: int) -> dict[str, list[Pair]]:
    rng = random.Random(seed)
    selected: dict[str, list[Pair]] = {}
    used_hashes: set[str] = set()
    # 先选最小的测试划分，为其中的稀有类别留出样本；不跨划分移动数据。
    for split in ("test", "val", "train"):
        pool = [pair for pair in candidates[split] if not pair.hashes & used_hashes]
        rng.shuffle(pool)
        frequency = Counter(token for pair in pool for token in pair.tokens)
        covered: set[tuple[str, int]] = set()
        chosen = []
        while len(chosen) < SPLITS[split]:
            if not pool:
                raise ValueError(f"Cannot select {SPLITS[split]} non-overlapping {split} pairs")
            # 优先补齐两路尚未覆盖的类别，其次考虑稀有程度；同分时沿用随机顺序。
            best = max(range(len(pool)), key=lambda index: (
                len(pool[index].tokens - covered),
                sum(1 / frequency[token] for token in sorted(pool[index].tokens - covered)),
            ))
            pair = pool.pop(best)
            chosen.append(pair)
            covered.update(pair.tokens)
            used_hashes.update(pair.hashes)
            # 任一路图像文件的 SHA-256 相同即排除，包括当前划分内的重复。
            pool = [other for other in pool if not other.hashes & pair.hashes]
        selected[split] = sorted(chosen, key=lambda pair: pair.stem)
    expected = {(modality, cls) for modality in MODALITIES for cls in range(NUM_CLASSES)}
    actual = set().union(*(pair.tokens for pairs in selected.values() for pair in pairs))
    if actual != expected:
        raise ValueError(f"The 100-pair subset does not cover all nine classes in both modalities: {sorted(expected - actual)}")
    return selected


def summarize(pairs_by_split: dict[str, list[Pair]]) -> dict:
    result = {}
    for split, pairs in pairs_by_split.items():
        modalities = {}
        for index, modality in enumerate(MODALITIES):
            instances = Counter(cls for pair in pairs for cls in pair.views[index].classes)
            images = Counter(cls for pair in pairs for cls in set(pair.views[index].classes))
            modalities[modality] = {
                "classes_present": sorted(instances),
                "instances_per_class": {str(cls): instances[cls] for cls in range(NUM_CLASSES)},
                "images_per_class": {str(cls): images[cls] for cls in range(NUM_CLASSES)},
                "empty_labels": sum(not pair.views[index].classes for pair in pairs),
            }
        result[split] = {"pairs": len(pairs), "modalities": modalities}
    return result


def prepare(source: Path, destination: Path, seed: int = 0) -> dict:
    source = source.expanduser().resolve(strict=True)
    destination = destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Destination must not already exist: {destination}")
    if destination.resolve().is_relative_to(source):
        raise ValueError("Destination must not be inside the source dataset")
    if not destination.parent.is_dir():
        raise ValueError(f"Destination parent must already exist: {destination.parent}")
    candidates, source_hash_report = inspect_source(source)
    selected = select_pairs(candidates, seed)
    summary = {
        "schema_version": 1,
        "dataset": "UMOD demonstration subset",
        "license": "CC-BY-4.0",
        "seed": seed,
        "source_valid_pairs": summarize(candidates),
        "source_hash_audit": {k: v for k, v in source_hash_report.items() if k != "cross_split_duplicate_paths"},
        "selected": summarize({split: selected[split] for split in SPLITS}),
        "selected_pairs": sum(map(len, selected.values())),
        "selected_exact_image_duplicates": 0,
        "all_nine_classes_in_each_modality": True,
    }
    destination.mkdir(exist_ok=False)
    manifest = []
    for split in SPLITS:
        for pair in selected[split]:
            entry = {"id": pair.stem, "split": split}
            for modality, view in zip(MODALITIES, pair.views):
                image_relative = view.image.relative_to(source)
                label_relative = view.label.relative_to(source)
                for source_path, relative, expected_hash in (
                    (view.image, image_relative, view.image_hash),
                    (view.label, label_relative, view.label_hash),
                ):
                    output = destination / relative
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with source_path.open("rb") as src, output.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
                    if sha256(output) != expected_hash:
                        raise RuntimeError(f"Source changed while copying: {relative.as_posix()}")
                entry[modality] = {
                    "image": image_relative.as_posix(), "label": label_relative.as_posix(),
                    "sha256_image": view.image_hash, "sha256_label": view.label_hash,
                    "width": view.width, "height": view.height,
                    "classes": sorted(set(view.classes)), "instances": len(view.classes),
                    "empty_label": not view.classes,
                }
            manifest.append(entry)
    with (destination / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": 1, "license": "CC-BY-4.0", "seed": seed,
                   "samples": manifest}, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    with (destination / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Existing UMOD_v2 dataset root")
    parser.add_argument("--destination", required=True, type=Path, help="New, nonexistent output directory")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        summary = prepare(args.source, args.destination, args.seed)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Sample preparation failed: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
