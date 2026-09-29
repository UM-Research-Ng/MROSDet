"""Single-device training, evaluation and paired inference for MROSDet.

The learning-rate, warmup and parameter-group rules follow the upstream
Ultralytics trainer (AGPL-3.0). This small engine does not import that package.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

LOGGER = logging.getLogger("mrosdet")
LOSS_NAMES = (
    "rgb_box", "rgb_cls", "rgb_dfl", "sonar_box", "sonar_cls", "sonar_dfl",
    "semantic", "reliability", "uncertainty",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def select_device(value="auto"):
    """Resolve CPU or one CUDA device, rejecting unsupported multi-device input."""
    value = str(value)
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    if value.isdecimal():
        value = f"cuda:{value}"
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This release supports CPU or a single CUDA device.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this environment.")
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} does not exist.")
        device = torch.device("cuda", index)
    return device


def seed_everything(seed=0, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def _seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def read_data_config(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    names = config["names"]
    config["names"] = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    if sorted(config["names"]) != list(range(len(config["names"]))):
        raise ValueError("Class IDs must be consecutive integers starting at zero.")
    config["nc"] = int(config.get("nc", len(config["names"])))
    if config["nc"] != len(config["names"]):
        raise ValueError("nc and the number of class names differ.")
    root = Path(config.get("path", ".")).expanduser()
    config["path"] = str((root if root.is_absolute() else path.parent / root).resolve())
    return config


def make_loader(data, split, imgsz, batch, workers, augment=False, seed=0):
    from mrosdet.data import PairedDataset

    dataset = PairedDataset(data, split=split, imgsz=imgsz, augment=augment)
    if not len(dataset):
        raise ValueError(f"The {split} split is empty.")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch, shuffle=augment, num_workers=workers,
        collate_fn=dataset.collate_fn, generator=generator,
        worker_init_fn=_seed_worker, pin_memory=False, drop_last=False,
    )


def preprocess(batch, device):
    result = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    for key in ("img_rgb", "img_sonar"):
        result[key] = result[key].float() / 255.0
    result["img"] = result["img_rgb"]
    return result


def unique_output(path):
    """Create a new directory without overwriting previous runs."""
    base = Path(path).expanduser().resolve()
    base.parent.mkdir(parents=True, exist_ok=True)
    for index in range(1, 100000):
        candidate = base if index == 1 else base.with_name(f"{base.name}{index}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Unable to allocate a fresh output directory beside {base}")


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(_plain(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _flatten(value, prefix=""):
    output = {}
    for key, item in value.items():
        key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(item, dict):
            output.update(_flatten(item, key))
        elif isinstance(item, (int, float, np.number)):
            output[key] = float(item)
    return output


def build_optimizer(model, lr=0.000769, weight_decay=0.0005):
    """AdamW with the original weight / normalization / bias grouping."""
    norm_types = tuple(v for k, v in vars(torch.nn).items() if "Norm" in k and isinstance(v, type))
    weights, norms, biases = [], [], []
    for module_name, module in model.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            full_name = f"{module_name}.{name}"
            if "bias" in full_name:
                biases.append(parameter)
            elif isinstance(module, norm_types) or "logit_scale" in full_name:
                norms.append(parameter)
            else:
                weights.append(parameter)
    groups = [
        {"params": weights, "weight_decay": weight_decay, "param_group": "weight"},
        {"params": norms, "weight_decay": 0.0, "param_group": "bn"},
        {"params": biases, "weight_decay": 0.0, "param_group": "bias"},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999))


def linear_lr_factor(epoch, epochs, lrf=0.01):
    return max(1.0 - epoch / epochs, 0.0) * (1.0 - lrf) + lrf


class ModelEMA:
    """Exponential moving average for evaluation and published training outputs."""

    def __init__(self, model, decay=0.9999, tau=2000.0):
        self.model = deepcopy(model).eval().requires_grad_(False)
        self.updates = 0
        self.decay = decay
        self.tau = tau

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = self.decay * (1.0 - math.exp(-self.updates / self.tau))
        source = model.state_dict()
        for key, value in self.model.state_dict().items():
            if value.is_floating_point():
                value.mul_(decay).add_(source[key].detach(), alpha=1.0 - decay)


def _check_names(model, data):
    names = getattr(model, "names", None)
    if names is not None:
        names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
        if names != data["names"]:
            raise ValueError("Checkpoint class names/IDs do not match the dataset; choose a matching checkpoint or train from scratch.")


def train(args):
    from mrosdet.checkpoint import load_model, save_checkpoint
    from mrosdet.loss import DualModalLoss
    from mrosdet.metrics import evaluate
    from mrosdet.nn.model import MROSDet

    device = select_device(args.device)
    seed_everything(args.seed, args.deterministic)
    data = read_data_config(args.data)
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["nc"] = data["nc"]
    if args.weights is not None:
        if not Path(args.weights).is_file():
            raise FileNotFoundError(f"Weights not found: {args.weights}")
        model = load_model(args.weights, device=device)
        _check_names(model, data)
        config = deepcopy(getattr(model, "config", getattr(model, "yaml", config)))
    else:
        model = MROSDet(config).to(device)
    model = model.float().requires_grad_(True)
    model.names = data["names"]
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, dual_sem=0.0, dual_rel=0.02, dual_unc=0.002)
    train_loader = make_loader(data, "train", args.imgsz, args.batch, args.workers, True, args.seed)
    val_loader = make_loader(data, "val", args.imgsz, args.batch, args.workers)
    nominal_accumulate = max(round(args.nbs / args.batch), 1)
    decay = args.weight_decay * args.batch * nominal_accumulate / args.nbs
    optimizer = build_optimizer(model, args.lr0, decay)
    criterion = DualModalLoss(model)
    ema = ModelEMA(model)
    amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    output = unique_output(args.output)
    (output / "weights").mkdir()
    settings = {**vars(args), "device": str(device), "amp": amp, "optimizer": "AdamW", "betas": [0.9, 0.999], "effective_weight_decay": decay}
    with (output / "args.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_plain(settings), handle, sort_keys=False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    best_fitness = -float("inf")
    batch_count = len(train_loader)
    warmup_steps = max(round(args.warmup_epochs * batch_count), 100) if args.warmup_epochs > 0 else -1
    last_optimizer_step = -1
    first_parameter = next(p for p in model.parameters() if p.requires_grad)
    initial_parameter = first_parameter.detach().cpu().clone()
    nonzero_gradient_steps = 0
    optimizer.zero_grad(set_to_none=True)
    rows = []
    LOGGER.info("Training on %s; output: %s", device, output)
    for epoch in range(args.epochs):
        model.train()
        running = torch.zeros(len(LOSS_NAMES), device=device)
        for index, original_batch in enumerate(train_loader):
            step = epoch * batch_count + index
            lr = args.lr0 * linear_lr_factor(epoch, args.epochs, args.lrf)
            accumulate = nominal_accumulate
            if 0 <= step <= warmup_steps:
                fraction = step / warmup_steps
                lr *= fraction
                accumulate = max(1, round(1 + fraction * (args.nbs / args.batch - 1)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = preprocess(original_batch, device)
            with torch.autocast(device_type=device.type, enabled=amp):
                predictions = model({"img_rgb": batch["img_rgb"], "img_sonar": batch["img_sonar"]})
                loss, components = criterion(predictions, batch)
                loss = loss.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}, batch {index + 1}")
            scaler.scale(loss).backward()
            if step - last_optimizer_step >= accumulate:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                if torch.isfinite(gradient_norm) and gradient_norm > 0:
                    nonzero_gradient_steps += 1
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)
                last_optimizer_step = step
            running += components.detach().reshape(-1)
        metrics = evaluate(ema.model, val_loader, device, conf=0.001, iou=0.7, max_det=300)
        fitness = float(metrics["fitness"])
        if not math.isfinite(fitness):
            raise FloatingPointError("Validation produced a non-finite fitness value.")
        row = {"epoch": epoch + 1, "time_seconds": time.perf_counter() - start, "lr": optimizer.param_groups[0]["lr"]}
        row.update({f"train/{name}": float(value) for name, value in zip(LOSS_NAMES, running.cpu() / batch_count)})
        row.update(_flatten(metrics))
        rows.append(row)
        with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save_checkpoint(ema.model, output / "weights" / "last.pt", config, data["names"])
        if fitness >= best_fitness:
            best_fitness = fitness
            save_checkpoint(ema.model, output / "weights" / "best.pt", config, data["names"])
        LOGGER.info("Epoch %d/%d: fitness=%.5f, lr=%.7g", epoch + 1, args.epochs, fitness, row["lr"])
    summary = {
        "epochs": args.epochs, "elapsed_seconds": time.perf_counter() - start,
        "best_fitness": best_fitness, "output": str(output),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "initialization": "checkpoint_finetuning" if args.weights else "random",
        "optimizer_updates": ema.updates,
        "finite_nonzero_gradient_steps": nonzero_gradient_steps,
        "first_parameter_max_change": float((first_parameter.detach().cpu() - initial_parameter).abs().max()),
    }
    _write_json(output / "summary.json", summary)
    LOGGER.info("Completed: %s", output)
    return summary


def validate(args):
    from mrosdet.checkpoint import load_model
    from mrosdet.metrics import evaluate

    if not Path(args.weights).is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    device = select_device(args.device)
    data = read_data_config(args.data)
    model = load_model(args.weights, device=device).float().eval()
    _check_names(model, data)
    loader = make_loader(data, args.split, args.imgsz, args.batch, args.workers)
    metrics = evaluate(model, loader, device, conf=args.conf, iou=args.iou, max_det=args.max_det)
    output = unique_output(args.output)
    _write_json(output / "metrics.json", metrics)
    LOGGER.info("Metrics: %s\nOutput: %s", json.dumps(_plain(metrics), ensure_ascii=False), output)
    return metrics


def pair_image_files(rgb_root, sonar_root):
    from mrosdet.data import paired_images

    try:
        pairs = paired_images(rgb_root, sonar_root)
    except ValueError as exc:
        raise ValueError(f"RGB/Sonar images are not paired unambiguously: {exc}") from exc
    return [(rgb.relative_to(rgb_root), rgb, sonar) for _, rgb, sonar in pairs]


def scale_detections(detections, original_shape, imgsz):
    """Undo independent square resizing for one modality; retain confidence/class."""
    result = detections.detach().cpu().clone()
    height, width = original_shape[:2]
    if len(result):
        result[:, [0, 2]] *= width / imgsz
        result[:, [1, 3]] *= height / imgsz
        result[:, [0, 2]] = result[:, [0, 2]].clamp(0, width)
        result[:, [1, 3]] = result[:, [1, 3]].clamp(0, height)
    return result


def _draw(image, detections, names):
    image = image.copy()
    for x1, y1, x2, y2, score, cls in detections.tolist():
        start, end = (round(x1), round(y1)), (round(x2), round(y2))
        cv2.rectangle(image, start, end, (0, 220, 80), 2)
        cv2.putText(image, f"{names[int(cls)]} {score:.2f}", (start[0], max(15, start[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 1, cv2.LINE_AA)
    return image


@torch.inference_mode()
def predict(args):
    from mrosdet.checkpoint import load_model
    from mrosdet.metrics import postprocess

    if not Path(args.weights).is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if bool(args.rgb) != bool(args.sonar):
        raise ValueError("Provide both --rgb and --sonar directories.")
    device = select_device(args.device)
    model = load_model(args.weights, device=device).float().eval()
    names = model.names
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    if args.rgb:
        rgb_root, sonar_root = args.rgb, args.sonar
    else:
        data = read_data_config(args.data)
        _check_names(model, data)
        rgb_root = Path(data["path"]) / data["modalities"]["rgb"][args.split]
        sonar_root = Path(data["path"]) / data["modalities"]["sonar"][args.split]
    pairs = pair_image_files(rgb_root, sonar_root)
    if args.limit:
        pairs = pairs[:args.limit]
    output = unique_output(args.output)
    records = []
    for relative, rgb_path, sonar_path in pairs:
        images = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in (rgb_path, sonar_path)]
        if any(image is None for image in images):
            raise ValueError(f"Unable to decode paired image {relative}")
        tensors = [torch.from_numpy(cv2.cvtColor(cv2.resize(image, (args.imgsz, args.imgsz)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).contiguous().unsqueeze(0).to(device).float() / 255 for image in images]
        raw = model({"img_rgb": tensors[0], "img_sonar": tensors[1]})
        detections = postprocess(raw, nc=len(names), conf=args.conf, iou=args.iou, max_det=args.max_det)
        record = {"pair": relative.as_posix()}
        for branch, image, source_path, source_root in zip(("rgb", "sonar"), images, (rgb_path, sonar_path), (rgb_root, sonar_root)):
            scaled = scale_detections(detections[branch][0], image.shape, args.imgsz)
            record[branch] = [
                {"xyxy": box[:4], "confidence": box[4], "class_id": int(box[5]), "class_name": names[int(box[5])]}
                for box in scaled.tolist()
            ]
            target = output / branch / source_path.relative_to(source_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), _draw(image, scaled, names)):
                raise OSError(f"Unable to write prediction image: {target}")
        record["reliability"] = _plain(raw.get("reliability"))
        records.append(record)
    _write_json(output / "predictions.json", records)
    LOGGER.info("Predicted %d pairs; output: %s", len(records), output)
    return records
