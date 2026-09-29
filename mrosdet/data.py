# Adapted from Ultralytics, AGPL-3.0. See THIRD_PARTY_NOTICES.md.
# MROSDet contributors: portable paired loading and strict validation, 2026-09-29.
"""Paired optical/sonar images with independent normalized detection labels."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_project_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(config):
    if isinstance(config, dict):
        data = dict(config)
        base = ROOT
    else:
        path = resolve_project_path(config)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset configuration not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        base = path.parent
    if not isinstance(data, dict):
        raise ValueError("Dataset configuration must be a mapping")
    names = data.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError("Dataset names must contain at least one class")
    names = {int(k): str(v) for k, v in names.items()}
    if sorted(names) != list(range(len(names))):
        raise ValueError("Class ids must be contiguous, starting at zero")
    if int(data.get("nc", len(names))) != len(names):
        raise ValueError("nc and names disagree")
    root = Path(data.get("path", ".")).expanduser()
    data.update(names=names, nc=len(names), path=str((base / root).resolve()))
    if not all(k in data.get("modalities", {}) for k in ("rgb", "sonar")):
        raise ValueError("Both rgb and sonar modalities must be configured")
    return data


def image_index(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    result = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name.startswith(".") or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = path.relative_to(directory).with_suffix("").as_posix()
        if key in result:
            raise ValueError(f"Duplicate image key: {key}")
        result[key] = path
    if not result:
        raise FileNotFoundError(f"No supported images in {directory}")
    return result


def paired_images(rgb_dir, sonar_dir):
    rgb, sonar = image_index(rgb_dir), image_index(sonar_dir)
    if rgb.keys() != sonar.keys():
        missing = sorted(rgb.keys() ^ sonar.keys())[:10]
        raise ValueError(f"Unpaired RGB/Sonar image keys: {missing}")
    return [(key, rgb[key], sonar[key]) for key in sorted(rgb)]


def read_labels(path, nc):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing label (create an empty file for background): {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return torch.zeros((0, 1), dtype=torch.float32), torch.zeros((0, 4), dtype=torch.float32)
    try:
        rows = np.array([[float(x) for x in line.split()] for line in text.splitlines()], dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Malformed detection labels: {path}") from exc
    if rows.ndim != 2 or rows.shape[1] != 5 or not np.isfinite(rows).all():
        raise ValueError(f"Labels must have five finite columns: {path}")
    cls, boxes = rows[:, 0], rows[:, 1:]
    if np.any(cls != np.floor(cls)) or np.any(cls < 0) or np.any(cls >= nc):
        raise ValueError(f"Class id outside [0, {nc - 1}]: {path}")
    if np.any(boxes < 0) or np.any(boxes > 1) or np.any(boxes[:, 2:] <= 0):
        raise ValueError(f"Invalid normalized box: {path}")
    corners = np.concatenate((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)
    if np.any(corners < -1e-5) or np.any(corners > 1 + 1e-5):
        raise ValueError(f"Normalized box extends beyond image: {path}")
    return torch.from_numpy(rows[:, :1].copy()), torch.from_numpy(boxes.copy())


class PairedDataset(Dataset):
    def __init__(self, config, split="train", imgsz=640, augment=False):
        self.data = load_config(config)
        self.names, self.nc = self.data["names"], self.data["nc"]
        self.split, self.imgsz, self.augment = split, int(imgsz), bool(augment)
        if self.imgsz < 32 or self.imgsz % 32:
            raise ValueError("imgsz must be a positive multiple of 32")
        root = Path(self.data["path"])
        try:
            self.rgb_dir = root / self.data["modalities"]["rgb"][split]
            self.sonar_dir = root / self.data["modalities"]["sonar"][split]
        except KeyError as exc:
            raise ValueError(f"Missing modality paths for split: {split}") from exc
        self.pairs = paired_images(self.rgb_dir, self.sonar_dir)
        self.label_pairs = []
        for key, _, _ in self.pairs:
            paths = [d.parent / "labels" / f"{key}.txt" for d in (self.rgb_dir, self.sonar_dir)]
            self.label_pairs.append(tuple(read_labels(p, self.nc) for p in paths))

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path):
        im = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f"Image cannot be decoded: {path}")
        shape = im.shape[:2]
        im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(im).permute(2, 0, 1).contiguous(), shape

    def __getitem__(self, index):
        key, rp, sp = self.pairs[index]
        rgb, rgb_shape = self._load_image(rp)
        sonar, sonar_shape = self._load_image(sp)
        condition = 0
        if self.augment:
            r = random.random()
            condition = 0 if r < 0.70 else (1 if r < 0.85 else 2)
            factor = random.uniform(0.55, 0.85)
            if condition == 2:
                rgb = (rgb.float() * factor).to(rgb.dtype)
            elif condition == 1:
                sonar = (sonar.float() * factor).to(sonar.dtype)
        (cr, br), (cs, bs) = self.label_pairs[index]
        return {
            "img_rgb": rgb, "img_sonar": sonar, "img": rgb,
            "cls_rgb": cr.clone(), "bboxes_rgb": br.clone(),
            "cls_sonar": cs.clone(), "bboxes_sonar": bs.clone(),
            "batch_idx_rgb": torch.zeros(len(cr)), "batch_idx_sonar": torch.zeros(len(cs)),
            "cls": cr.clone(), "bboxes": br.clone(), "batch_idx": torch.zeros(len(cr)),
            "im_file_rgb": str(rp), "im_file_sonar": str(sp), "im_file": str(rp), "pair_key": key,
            "ori_shape_rgb": rgb_shape, "ori_shape_sonar": sonar_shape, "ori_shape": rgb_shape,
            "ratio_pad": (1.0, 1.0), "condition": torch.tensor(condition, dtype=torch.long),
            "modality_valid": torch.tensor([float(condition != 2), float(condition != 1)]),
        }

    @staticmethod
    def collate_fn(batch):
        stack = {"img_rgb", "img_sonar", "img", "condition", "modality_valid"}
        cat = {"cls_rgb", "bboxes_rgb", "batch_idx_rgb", "cls_sonar", "bboxes_sonar", "batch_idx_sonar", "cls", "bboxes", "batch_idx"}
        out = {}
        for key in batch[0]:
            values = [b[key] for b in batch]
            if key in stack:
                out[key] = torch.stack(values)
            elif key in cat:
                if key.startswith("batch_idx"):
                    values = [v + i for i, v in enumerate(values)]
                out[key] = torch.cat(values)
            else:
                out[key] = values
        return out
