# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
# Adapted for the standalone MROSDet release, 2026-09-29; imports only.
"""MROSDet modules for modality-robust optical-sonar detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import C2PSA, C3k2, SPPF
from .blocks import Conv
from .head import Detect


class OSIR(nn.Module):
    """Optical-Sonar Input Router for paired or concatenated modality tensors."""

    def __init__(self, rgb_channels: int = 3, sonar_channels: int = 3):
        super().__init__()
        self.rgb_channels = rgb_channels
        self.sonar_channels = sonar_channels

    def forward(self, x):
        """Return [rgb, sonar] from a dict, tuple/list, or concatenated tensor."""
        if isinstance(x, dict):
            return [x["img_rgb"], x["img_sonar"]]
        if isinstance(x, (list, tuple)):
            return [x[0], x[1]]
        if x.shape[1] == self.rgb_channels + self.sonar_channels:
            return list(torch.split(x, [self.rgb_channels, self.sonar_channels], dim=1))
        raise ValueError("OSIR expects a dict with img_rgb/img_sonar, a pair, or a concatenated RGB+Sonar tensor.")


class _ResidualGate(nn.Module):
    """Small channel-spatial residual gate initialized close to identity."""

    def __init__(self, channels: int, init: float = -4.0):
        super().__init__()
        hidden = max(16, channels // 8)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(1, keepdim=True)
        mx = x.amax(1, keepdim=True)
        gate = self.channel(x) * self.spatial(torch.cat((avg, mx), dim=1))
        return x + self.alpha.sigmoid() * (gate * x - x)


class ARGConv(Conv):
    """Adaptive Residual-Gated Convolution for reliability-aware feature calibration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_gate = _ResidualGate(self.bn.num_features, init=-4.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual_gate(super().forward(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual_gate(super().forward_fuse(x))


class LDRC3k2(C3k2):
    """Local Detail Recalibrated C3k2 block for small underwater targets."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.detail_refine = nn.Sequential(Conv(c2, c2, 3, 1, g=c2), Conv(c2, c2, 1, 1, act=False))
        self.detail_gate = _ResidualGate(c2, init=-4.2)
        self.detail_alpha = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        y = y + self.detail_alpha.sigmoid() * self.detail_refine(y)
        return self.detail_gate(y)


class CPRSPPF(SPPF):
    """Contextual Pyramid Recalibrated SPPF block."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        super().__init__(c1, c2, k, n, shortcut)
        self.context_gate = _ResidualGate(c2, init=-4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.context_gate(super().forward(x))


class ARC2PSA(C2PSA):
    """Attention-Recalibrated C2PSA block for sharper localization."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, e)
        self.attn_refine = Conv(c2, c2, 3, 1, g=c2)
        self.attn_alpha = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return y + self.attn_alpha.sigmoid() * self.attn_refine(y)


class MRE(nn.Module):
    """Modality Reliability Estimator for per-scale optical and sonar confidence."""

    def __init__(self, channels: list[int], num_conditions: int = 4, hidden: int = 128):
        super().__init__()
        self.channels = channels
        self.num_conditions = num_conditions
        in_features = sum(channels) * 2
        hidden = max(32, min(hidden, in_features))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, len(channels) * 2 + num_conditions),
        )

    def forward(self, x: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        rgb_feats, sonar_feats = x[:3], x[3:6]
        pooled = [self.pool(f).flatten(1) for f in (*rgb_feats, *sonar_feats)]
        logits = self.mlp(torch.cat(pooled, dim=1))
        rel_logits = logits[:, : len(self.channels) * 2].view(-1, len(self.channels), 2)
        condition_logits = logits[:, len(self.channels) * 2 :]
        weights = rel_logits.softmax(dim=-1)
        uncertainty = 1.0 - weights.max(dim=-1).values
        return {
            "weights": weights,
            "condition_logits": condition_logits,
            "uncertainty": uncertainty,
        }


class RGCF(nn.Module):
    """Reliability-Guided Cross-modal Fusion for one pyramid scale."""

    def __init__(self, channels: int, scale_index: int):
        super().__init__()
        self.scale_index = scale_index
        self.rgb_from_sonar = Conv(channels, channels, 1, 1)
        self.sonar_from_rgb = Conv(channels, channels, 1, 1)
        hidden = max(32, channels // 4)
        self.rgb_channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.sonar_channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.rgb_spatial_gate = nn.Sequential(
            Conv(channels * 2, hidden, 1, 1),
            nn.Conv2d(hidden, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.sonar_spatial_gate = nn.Sequential(
            Conv(channels * 2, hidden, 1, 1),
            nn.Conv2d(hidden, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.rgb_alpha = nn.Parameter(torch.tensor(-4.0))
        self.sonar_alpha = nn.Parameter(torch.tensor(-3.0))
        self.register_buffer("scale_gain", torch.tensor([0.0, 0.35, 0.65]), persistent=False)

    def _attention(self, target: torch.Tensor, auxiliary: torch.Tensor, branch: str) -> torch.Tensor:
        """Return channel-spatial gate for auxiliary residual injection."""
        pair = torch.cat((target, auxiliary), dim=1)
        if branch == "rgb":
            return self.rgb_channel_gate(pair) * self.rgb_spatial_gate(pair)
        return self.sonar_channel_gate(pair) * self.sonar_spatial_gate(pair)

    def forward(self, x: list) -> list[torch.Tensor]:
        rgb, sonar, reliability = x
        if reliability and "weights" in reliability:
            weights = reliability["weights"][:, self.scale_index]
        else:
            weights = rgb.new_full((rgb.shape[0], 2), 0.5)
        w_rgb = weights[:, 0].view(-1, 1, 1, 1)
        w_sonar = weights[:, 1].view(-1, 1, 1, 1)
        scale_gain = self.scale_gain[min(self.scale_index, self.scale_gain.numel() - 1)].to(rgb)

        sonar_residual = self.rgb_from_sonar(sonar)
        rgb_residual = self.sonar_from_rgb(rgb)
        rgb_gate = self._attention(rgb, sonar_residual, "rgb")
        sonar_gate = self._attention(sonar, rgb_residual, "sonar")
        rgb_out = rgb + scale_gain * self.rgb_alpha.sigmoid() * w_sonar * rgb_gate * sonar_residual
        sonar_out = sonar + scale_gain * self.sonar_alpha.sigmoid() * w_rgb * sonar_gate * rgb_residual
        return [rgb_out, sonar_out]


class _PANNeck(nn.Module):
    """Lightweight PAN/FPN neck for one modality."""

    def __init__(self, in_channels: list[int], out_channels: list[int]):
        super().__init__()
        c3, c4, c5 = in_channels
        o3, o4, o5 = out_channels
        self.lat3 = Conv(c3, o3, 1, 1)
        self.lat4 = Conv(c4, o4, 1, 1)
        self.lat5 = Conv(c5, o5, 1, 1)
        self.fuse4 = Conv(o5 + o4, o4, 3, 1)
        self.fuse3 = Conv(o4 + o3, o3, 3, 1)
        self.down3 = Conv(o3, o3, 3, 2)
        self.out4 = Conv(o3 + o4, o4, 3, 1)
        self.down4 = Conv(o4, o4, 3, 2)
        self.out5 = Conv(o4 + o5, o5, 3, 1)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        p3, p4, p5 = self.lat3(feats[0]), self.lat4(feats[1]), self.lat5(feats[2])
        n4 = self.fuse4(torch.cat((F.interpolate(p5, size=p4.shape[-2:], mode="nearest"), p4), dim=1))
        n3 = self.fuse3(torch.cat((F.interpolate(n4, size=p3.shape[-2:], mode="nearest"), p3), dim=1))
        o4 = self.out4(torch.cat((self.down3(n3), n4), dim=1))
        o5 = self.out5(torch.cat((self.down4(o4), p5), dim=1))
        return [n3, o4, o5]


class SDPN(nn.Module):
    """Synergistic Dual-Pyramid Neck with post-neck cross-modal adaptation."""

    def __init__(self, in_channels: list[int], out_channels: list[int]):
        super().__init__()
        self.rgb_neck = _PANNeck(in_channels, out_channels)
        self.sonar_neck = _PANNeck(in_channels, out_channels)
        self.rgb_from_sonar = nn.ModuleList(Conv(c, c, 1, 1) for c in out_channels)
        self.sonar_from_rgb = nn.ModuleList(Conv(c, c, 1, 1) for c in out_channels)
        self.rgb_alpha = nn.Parameter(torch.full((len(out_channels),), -4.0))
        self.sonar_alpha = nn.Parameter(torch.full((len(out_channels),), -3.2))
        self.register_buffer("post_gain", torch.tensor([0.0, 0.15, 0.25]), persistent=False)

    def forward(self, x: list) -> list[torch.Tensor]:
        rel = x[6] if len(x) > 6 else None
        rgb = self.rgb_neck(x[:3])
        sonar = self.sonar_neck(x[3:6])
        if rel and "weights" in rel:
            weights = rel["weights"]
        else:
            weights = rgb[0].new_full((rgb[0].shape[0], len(rgb), 2), 0.5)

        rgb_out, sonar_out = [], []
        for i, (r, s) in enumerate(zip(rgb, sonar)):
            gain = self.post_gain[min(i, self.post_gain.numel() - 1)].to(r)
            w_rgb = weights[:, i, 0].view(-1, 1, 1, 1)
            w_sonar = weights[:, i, 1].view(-1, 1, 1, 1)
            rgb_out.append(r + gain * self.rgb_alpha[i].sigmoid() * w_sonar * self.rgb_from_sonar[i](s))
            sonar_out.append(s + gain * self.sonar_alpha[i].sigmoid() * w_rgb * self.sonar_from_rgb[i](r))
        rgb, sonar = rgb_out, sonar_out
        return [*rgb, *sonar]


class BCDHead(nn.Module):
    """Bimodal Collaborative Detection Head for optical and sonar predictions."""

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = ()):
        super().__init__()
        if len(ch) < 6:
            raise ValueError("BCDHead expects six feature channel entries: RGB P3/P4/P5 and Sonar P3/P4/P5.")
        self.nc = nc
        self.reg_max = reg_max
        self.nl = 3
        self.rgb = Detect(nc, reg_max, end2end, ch[:3])
        self.sonar = Detect(nc, reg_max, end2end, ch[3:6])
        self.stride = torch.zeros(self.nl)
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)

    @property
    def end2end(self):
        return self.rgb.end2end

    @end2end.setter
    def end2end(self, value):
        self.rgb.end2end = value
        self.sonar.end2end = value

    def bias_init(self):
        self.rgb.bias_init()
        self.sonar.bias_init()

    @staticmethod
    def _sync_head_cache(head: Detect, feats: list[torch.Tensor]) -> None:
        """Force anchor cache rebuild when a reloaded head crosses device or dtype boundaries."""
        if head.training or not hasattr(head, "anchors") or head.anchors.numel() == 0:
            return
        ref = feats[0]
        if head.anchors.device != ref.device or head.anchors.dtype != ref.dtype:
            head.shape = None

    def forward(self, x: list):
        rel = x[6] if len(x) > 6 else None
        self._sync_head_cache(self.rgb, x[:3])
        self._sync_head_cache(self.sonar, x[3:6])
        rgb = self.rgb(x[:3])
        sonar = self.sonar(x[3:6])
        return {"rgb": rgb, "sonar": sonar, "reliability": rel}
