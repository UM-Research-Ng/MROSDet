# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
# 为 MROSDet 抽取检测头，2026-09-29；保留原计算逻辑。
"""MROSDet 光学与声纳分支共用的检测头实现。"""
from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn

from ..box_ops import dist2bbox, make_anchors
from .blocks import Conv, DWConv, DFL


class Detect(nn.Module):
    """单模态多尺度检测头，分别预测边框距离与类别 logits。

    当前 MROSDet 使用 legacy=False、end2end=False：训练返回原始预测字典，
    评估返回解码预测及原始字典。保留的 one2one、导出和 fuse 路径是内部兼容接口，
    并非当前发布模型的运行路径。
    """

    dynamic = False  # 强制重建网格点缓存
    export = False  # 内部导出开关
    format = None  # 内部导出格式
    max_det = 300  # 内部端到端路径的候选数上限
    agnostic_nms = False
    shape = None
    anchors = torch.empty(0)  # 推理缓存，首次解码时填充
    strides = torch.empty(0)  # 推理缓存，首次解码时填充
    legacy = False  # 内部旧分类分支开关，当前配置关闭
    xyxy = False  # False 默认输出 xywh，True 输出 xyxy

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()):
        """按 nc、reg_max 和各尺度通道数建立回归与分类分支。

        end2end=True 时额外复制 one2one 分支；当前 MROSDet 不启用该路径。
        """
        super().__init__()
        self.nc = nc  # 类别数
        self.nl = len(ch)  # 检测尺度数
        self.reg_max = reg_max  # 每条边的回归通道数，由 reg_max 显式指定
        self.no = nc + self.reg_max * 4  # 每个网格点的原始输出通道数
        self.stride = torch.zeros(self.nl)  # 构建模型时通过探测前向确定
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # 回归与分类分支的隐藏通道数
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @property
    def one2many(self):
        """返回当前模型使用的回归与分类分支。"""
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        """返回仅在 end2end=True 构造时建立的 one2one 分支。"""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    @property
    def end2end(self):
        """检查内部开关及 one2one 分支是否同时可用。"""
        return getattr(self, "_end2end", True) and hasattr(self, "one2one")

    @end2end.setter
    def end2end(self, value):
        """设置内部端到端路径开关，不负责创建 one2one 分支。"""
        self._end2end = value

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """返回原始预测：boxes 为 (B, 4*reg_max, A)，scores 为 (B, nc, A)。

        boxes 为未解码的回归输出，scores 为类别 logits；feats 保留输入特征，
        A 为所有尺度的网格点总数。
        """
        if box_head is None or cls_head is None:  # 兼容仅保留 one2one 的内部路径
            return dict()
        bs = x[0].shape[0]  # 批量大小
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """训练时返回原始预测，评估时返回 (解码预测, 原始预测)。

        当前配置的解码张量为 (B, 4+nc, A)，包含 xywh 边框与 sigmoid 类别分数；
        此处不执行 NMS。export 与 end2end 是当前发布模型未启用的内部路径。
        """
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """将各尺度边框解码到输入图像坐标，并拼接 sigmoid 类别分数。"""
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """按特征形状维护网格点缓存，用对应步长将距离解码为像素坐标。"""
        shape = x["feats"][0].shape  # 批量、通道、高、宽
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape

        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def bias_init(self):
        """初始化回归与分类偏置；必须先设置各尺度 stride。"""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):
            a[-1].bias.data[:] = 2.0  # 回归分支偏置
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )  # 以 640 像素图像中 5 个目标的密度先验初始化分类偏置
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):
                a[-1].bias.data[:] = 2.0  # 回归分支偏置
                b[-1].bias.data[: self.nc] = math.log(
                    5 / self.nc / (640 / self.stride[i]) ** 2
                )  # 以 640 像素图像中 5 个目标的密度先验初始化分类偏置

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """根据网格点及四边距离还原边框；当前配置默认输出 xywh。"""
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """内部端到端路径的 Top-K 筛选，不是 NMS，当前发布模型不调用。

        输入为 (B, A, 4+nc)，包含 xyxy 边框及类别分数；普通非导出路径输出
        (B, min(max_det, A), 6)，末维为 [x1, y1, x2, y2, 分数, 类别]。
        """
        boxes, scores = preds.split([4, self.nc], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    def get_topk_index(self, scores: torch.Tensor, max_det: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """为内部端到端路径选择最高分候选，返回分数、类别编号和网格点索引。

        输入 scores 为 (B, A, nc)；当前发布模型不调用此路径。
        """
        batch_size, anchors, nc = scores.shape  # 批量、网格点数、类别数
        # 内部导出路径固定 k，以满足 TensorRT 对常量的要求；
        # 普通推理路径限制 k 不超过网格点数。当前发布模型不调用此路径。
        k = max_det if self.export else min(max_det, anchors)
        if self.agnostic_nms:
            scores, labels = scores.max(dim=-1, keepdim=True)
            scores, indices = scores.topk(k, dim=1)
            labels = labels.gather(1, indices)
            return scores, labels, indices
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size)[..., None], index // nc]  # 映射回原网格点索引
        return scores[..., None], (index % nc)[..., None].float(), idx

    def fuse(self) -> None:
        """仅供 one2one 推理路径移除 one2many 分支。

        当前 MROSDet 依赖 one2many，不能调用此方法；它也不是卷积/批归一化融合。
        """
        self.cv2 = self.cv3 = None
