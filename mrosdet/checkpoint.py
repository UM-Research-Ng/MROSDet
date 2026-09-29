"""Portable, weights-only checkpoints for MROSDet."""

from copy import deepcopy
from pathlib import Path

import torch

FORMAT_VERSION = 1


def model_config(config):
    """Keep only portable architecture fields, never machine-specific paths."""
    allowed = {"nc", "task", "scale", "scales", "end2end", "reg_max", "modalities", "backbone", "head", "channels", "depth_multiple", "width_multiple"}
    return {k: deepcopy(v) for k, v in config.items() if k in allowed}


def extra_buffers(model):
    """Nonpersistent fusion priors must survive export despite not being state_dict entries."""
    result = {}
    for prefix, module in model.named_modules():
        for key in sorted(module._non_persistent_buffers_set):
            value = module._buffers.get(key)
            if value is not None:
                result[f"{prefix}.{key}" if prefix else key] = value.detach().cpu().clone()
    return result


def restore_buffers(model, buffers):
    expected = extra_buffers(model)
    if expected.keys() != buffers.keys():
        raise ValueError(f"Checkpoint nonpersistent buffers differ: missing={expected.keys() - buffers.keys()}, extra={buffers.keys() - expected.keys()}")
    for path, value in buffers.items():
        prefix, _, key = path.rpartition(".")
        module = model.get_submodule(prefix) if prefix else model
        target = module._buffers[key]
        if not isinstance(value, torch.Tensor) or target.shape != value.shape:
            raise ValueError(f"Checkpoint buffer shape mismatch: {path}")
        target.copy_(value.to(device=target.device, dtype=target.dtype))


def save_checkpoint(model, path, config, names, *, source_sha256=None):
    """Export only plain configuration and tensors; no Python model pickle or training history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(k): str(v) for k, v in names.items()}
    config = model_config(config)
    if sorted(names) != list(range(int(config["nc"]))):
        raise ValueError("Checkpoint class names must match model nc")
    payload = {
        "format_version": FORMAT_VERSION, "architecture": "MROSDet",
        "config": config, "names": names,
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "extra_buffers": extra_buffers(model),
        "preprocess": {"resize": "stretch", "color": "RGB", "scale": 255.0},
    }
    if source_sha256 is not None:
        if len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
            raise ValueError("Invalid source SHA256")
        payload["source_sha256"] = source_sha256
    torch.save(payload, path)


def load_model(weights, device="cpu"):
    from mrosdet.nn.model import MROSDet

    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}. Download the release checkpoint first.")
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != FORMAT_VERSION or checkpoint.get("architecture") != "MROSDet":
        raise ValueError("Unsupported checkpoint. Use an independent MROSDet release checkpoint.")
    config = model_config(checkpoint["config"])
    model = MROSDet(config)
    names = {int(k): str(v) for k, v in checkpoint["names"].items()}
    if sorted(names) != list(range(int(config["nc"]))):
        raise ValueError("Checkpoint class names do not match architecture")
    if checkpoint.get("preprocess") != {"resize": "stretch", "color": "RGB", "scale": 255.0}:
        raise ValueError("Unsupported checkpoint preprocessing")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    restore_buffers(model, checkpoint["extra_buffers"])
    model.names = names
    model.config = config
    return model.to(device).eval()
