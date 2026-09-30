# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
# 基于项目 nn/tasks.py 整理为独立模型，2026-09-29。
"""按论文配置构建光学–声纳双流模型，无需原框架运行时。"""
from __future__ import annotations

import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import yaml

from .blocks import C3k2, Conv, Index
from .dual import ARC2PSA, ARGConv, BCDHead, CPRSPPF, LDRC3k2, MRE, OSIR, RGCF, SDPN


_MODULES = {
    c.__name__: c for c in
    (Conv, C3k2, Index, ARC2PSA, ARGConv, BCDHead, CPRSPPF, LDRC3k2, MRE, OSIR, RGCF, SDPN)
}


def parse_model(config: dict, channels: int = 3) -> tuple[nn.Sequential, list[int]]:
    """按原始深度/通道缩放规则解析 MROSDet 配置图。

    仅接受本模型所需的模块白名单；字符串参数用字面量解析，不执行任意 Python 表达式。
    """
    nc = int(config["nc"])
    scale = config.get("scale", "n")
    depth, width, max_channels = config["scales"][scale]
    reg_max = int(config.get("reg_max", 1))
    end2end = bool(config.get("end2end", False))
    if end2end:
        raise ValueError("The published MROSDet graph uses end2end=False.")
    if config.get("activation") not in (None, "nn.SiLU()"):
        raise ValueError("The published MROSDet graph uses SiLU activation.")
    ch = [channels]
    layers, save = [], []
    base_modules = {Conv, ARGConv, C3k2, LDRC3k2, CPRSPPF, ARC2PSA}
    repeat_modules = {C3k2, LDRC3k2, ARC2PSA}
    for i, (f, n, name, raw_args) in enumerate(config["backbone"] + config["head"]):
        if name not in _MODULES:
            raise ValueError(f"Unsupported MROSDet layer {name!r} at index {i}.")
        module = _MODULES[name]
        args = copy.deepcopy(raw_args)
        for j, arg in enumerate(args):
            if isinstance(arg, str):
                if arg == "nc":
                    args[j] = nc
                else:
                    try:
                        args[j] = ast.literal_eval(arg)
                    except (ValueError, SyntaxError):
                        raise ValueError(f"Unsupported argument {arg!r} at layer {i}.") from None
        n = max(round(n * depth), 1) if n > 1 else n
        if module in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = math.ceil(min(c2, max_channels) * width / 8) * 8
            args = [c1, c2, *args[1:]]
            if module in repeat_modules:
                args.insert(2, n)
                n = 1
            if module in {C3k2, LDRC3k2} and scale in "mlx":
                args[3] = True
        elif module is OSIR:
            c2 = [args[0], args[1]]
        elif module is MRE:
            c2 = [ch[x] for x in f]
            args = [[ch[x] for x in f[:3]], *args[1:]]
        elif module is RGCF:
            c2 = [ch[f[0]], ch[f[0]]]
            args = [ch[f[0]], args[1]]
        elif module is SDPN:
            in_ch = [ch[x] for x in f[:3]]
            c2 = [*args[1], *args[1]]
            args = [in_ch, *args[1:]]
        elif module is BCDHead:
            args.extend([reg_max, end2end, [ch[x] for x in f[:6]]])
            c2 = [ch[x] for x in f[:6]]
        elif module is Index:
            c1 = ch[f]
            c2 = c1[args[1]] if isinstance(c1, list) and len(args) > 1 else args[0]
            args = [*args[1:]]
        else:
            raise ValueError(f"Unsupported MROSDet layer {name!r}.")
        layer = nn.Sequential(*(module(*args) for _ in range(n))) if n > 1 else module(*args)
        layer.i, layer.f, layer.type = i, f, name
        layer.np = sum(p.numel() for p in layer.parameters())
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(layer)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class MROSDet(nn.Module):
    """由 configs/mrosdet.yaml 定义的 43 层光学–声纳模型。

    输入字典包含已归一化的 img_rgb/img_sonar，张量形状均为 BCHW。
    输出字典含 rgb、sonar、reliability；训练时各检测分支返回 boxes/scores/feats，
    评估时各分支返回 (解码预测, 原始预测)，可靠性信息保持为字典。
    """

    def __init__(self, config: str | Path | dict | None = None, *, nc: int | None = None, ch: int = 3):
        super().__init__()
        if config is None:
            config = Path(__file__).resolve().parents[2] / "configs" / "mrosdet.yaml"
        if isinstance(config, dict):
            cfg = copy.deepcopy(config)
        else:
            with Path(config).open(encoding="utf-8") as stream:
                cfg = yaml.safe_load(stream)
        if not isinstance(cfg, dict):
            raise ValueError("Model configuration must contain a YAML mapping.")
        if nc is not None:
            cfg["nc"] = int(nc)
        cfg.pop("yaml_file", None)
        cfg["channels"] = ch
        self.config = cfg
        self.yaml = self.config
        self.nc = int(cfg["nc"])
        self.names = {i: str(i) for i in range(self.nc)}
        self.task = "dual_detect"
        self.inplace = bool(cfg.get("inplace", True))
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, dual_sem=0.0, dual_rel=0.02, dual_unc=0.002)
        self.model, self.save = parse_model(copy.deepcopy(cfg), channels=ch)
        head = self.model[-1]
        if not isinstance(head, BCDHead):
            raise ValueError("MROSDet requires BCDHead as its final layer.")
        head.inplace = self.inplace
        self.model.eval()
        head.training = True
        # 保持原始步长探测方式，同时避免更新批归一化的运行统计量。
        with torch.no_grad():
            probe = torch.zeros(1, ch, 256, 256)
            output = self.forward({"img_rgb": probe, "img_sonar": probe})["rgb"]
            raw = output[1] if isinstance(output, tuple) else output
            head.stride = torch.tensor([256 / feature.shape[-2] for feature in raw["feats"]])
        head.rgb.stride = head.stride
        head.sonar.stride = head.stride
        self.stride = head.stride
        # 探测时步长尚未赋值，需让后续推理重建网格点缓存。
        head.rgb.shape = head.sonar.shape = None
        self.model.train()
        head.bias_init()
        for module in self.modules():
            if type(module) is nn.BatchNorm2d:
                module.eps = 1e-3
                module.momentum = 0.03
            elif type(module) in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
                module.inplace = True

    @property
    def end2end(self) -> bool:
        return self.model[-1].end2end

    def forward(self, x):
        """按层索引执行计算图，仅生成预测；损失由外部 DualModalLoss 计算。"""
        saved = []
        for layer in self.model:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [
                    x if j == -1 else saved[j] for j in layer.f
                ]
            x = layer(x)
            saved.append(x if layer.i in self.save else None)
        return x

    def predict(self, x):
        """直接成对图像预测的别名；是否解码取决于模型的 train/eval 状态。"""
        return self.forward(x)

    def _apply(self, fn):
        """同步迁移步长及非持久化推理缓存，并触发网格点缓存重建。"""
        super()._apply(fn)
        if hasattr(self, "model"):
            head = self.model[-1]
            for branch in (head, head.rgb, head.sonar):
                branch.stride = fn(branch.stride)
                branch.anchors = fn(branch.anchors)
                branch.strides = fn(branch.strides)
                if hasattr(branch, "shape"):
                    branch.shape = None
            self.stride = head.stride
        return self
