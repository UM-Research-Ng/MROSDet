# Derived from Ultralytics (AGPL-3.0), https://github.com/ultralytics/ultralytics.
# Copyright (c) Ultralytics. See LICENSE and THIRD_PARTY_NOTICES.md.
"""MROSDet 的双路检测、模态可靠性与可选一致性损失。"""
from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .assigner import TaskAlignedAssigner
from .box_ops import bbox_iou, bbox2dist, dist2bbox, make_anchors, xywh2xyxy


class DFLoss(nn.Module):
    """离散分布焦点损失；仅在回归分箱数 reg_max 大于 1 时使用。"""

    def __init__(self, reg_max: int = 16) -> None:
        """设置每个边界距离的离散分箱数 reg_max。"""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """按相邻两个分箱加权交叉熵，再对四个边界距离取均值。

        算法参考：https://ieeexplore.ieee.org/document/9792391。
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # 左侧分箱
        tr = tl + 1  # 右侧分箱
        wl = tr - target  # 左侧权重
        wr = 1 - wl  # 右侧权重
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """按分配分数加权的 CIoU 损失与边界距离回归损失。"""

    def __init__(self, reg_max: int = 16):
        """reg_max > 1 时使用 DFL，否则使用按图像尺寸归一化的 L1。"""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 CIoU 与距离回归两项标量。

        当前公开模型 reg_max=1，第二项为归一化 L1，而非 DFL；
        为保持训练记录接口不变，变量和日志仍沿用 dfl 命名。
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # 分箱回归使用 DFL；单值回归走下方的归一化 L1 分支。
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            # 将左、上、右、下距离换算为像素，再按图像宽、高归一化。
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class DetectionLoss:
    """单模态检测分支的目标分配与三项损失：box、cls、dfl。"""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # 传入未套并行包装器的模型
        """从模型读取检测头设置，并建立任务对齐分配器及损失权重。"""
        device = next(model.parameters()).device  # 模型所在设备
        h = model.args  # 损失超参数

        m = model.model[-1]  # 双路检测头提供的公共类别数、步长及回归分箱数
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # 检测层步长
        self.nc = m.nc  # 类别数
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        # 仅当调用方在模型上提供 class_weights 时使用逐类别权重。
        self.class_weights = getattr(model, "class_weights", None)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device).view(1, 1, -1)

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """将按图像分组的标签补齐为批次张量，并把归一化 xywh 转成像素 xyxy。"""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # 标签所属的批内图像索引
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = targets[:, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """由网格点和四边距离解码 xyxy 框；多分箱回归先取分布期望。"""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # 批量、候选点数、回归通道数
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """返回分配结果、加权三项损失向量及其脱离计算图的副本。

        分配结果依次为前景掩码、目标索引、像素目标框、网格点及对应步长。
        三项向量按 box、cls、dfl 排列；本方法不求和，也不乘批量大小。
        """
        loss = torch.zeros(3, device=self.device)  # CIoU、分类、距离回归
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # 各图像的独立标签合并后，换算到当前输入图像的像素坐标。
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # 类别、xyxy 框
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # 解码预测框，此处仍处于各特征层的网格坐标系。
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy，形状 [批量,全部候选点数,4]

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # 分类 BCE，可选逐类别加权。
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # [批量,全部候选点数,类别数]
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum  # BCE

        # 仅前景候选点参与边界框回归。
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box  # CIoU 权重
        loss[1] *= self.hyp.cls  # 分类权重
        loss[2] *= self.hyp.dfl  # 距离回归权重，当前配置对应归一化 L1
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )  # 三项损失顺序：box、cls、dfl

    def parse_output(
        self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """取得原始预测字典；验证模式的 (解码结果, 原始预测) 取第二项。"""
        return preds[1] if isinstance(preds, tuple) else preds

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回乘批量大小的三项训练向量，以及未乘批量大小的三项日志值。"""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """计算分配后的三项损失；仅可求导向量乘以批量大小，不在此求和。"""
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss * batch_size, loss_detach


class DualModalLoss:
    """组合 RGB、Sonar 各自标签的检测损失和三个辅助项。

    返回九项可求导向量与九项日志向量，顺序为 RGB 的 box/cls/dfl、
    Sonar 的 box/cls/dfl、sem、rel、unc；训练循环对可求导向量求和。
    前六项训练损失乘批量大小，后三项仅乘各自系数；日志向量中的前六项
    不乘批量大小，后三项则记录未乘辅助系数的原值。
    当前默认辅助系数为 sem=0、rel=0.02、unc=0.002。
    """

    def __init__(self, model):
        self.rgb_loss = DetectionLoss(model)
        self.sonar_loss = DetectionLoss(model)
        self.device = next(model.parameters()).device
        self.lambda_sem = float(getattr(model.args, "dual_sem", 0.0))
        self.mu_rel = float(getattr(model.args, "dual_rel", 0.02))
        self.nu_unc = float(getattr(model.args, "dual_unc", 0.002))

    @staticmethod
    def _branch_batch(batch: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
        """提取指定模态的标签，保持两路目标框及类别彼此独立。"""
        return {
            "batch_idx": batch[f"batch_idx_{prefix}"],
            "cls": batch[f"cls_{prefix}"],
            "bboxes": batch[f"bboxes_{prefix}"],
        }

    @staticmethod
    def _presence_scores(preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]) -> torch.Tensor:
        """每张图像在所有类别和候选点中取最大概率，不是逐类别语义向量。"""
        preds = preds[1] if isinstance(preds, tuple) else preds
        return preds["scores"].sigmoid().amax(dim=(1, 2))

    def __call__(self, preds: dict[str, Any], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """分别计算检测分支，再组合条件分类、权重监督及可选一致性损失。"""
        rgb_loss, rgb_items = self.rgb_loss(preds["rgb"], self._branch_batch(batch, "rgb"))
        sonar_loss, sonar_items = self.sonar_loss(preds["sonar"], self._branch_batch(batch, "sonar"))

        rel = preds.get("reliability") or {}
        rel_loss = torch.zeros(1, device=self.device)
        unc_loss = torch.zeros(1, device=self.device)
        # rel 项监督四类模态条件 logits；实际目标条件由数据加载器提供。
        if rel and "condition_logits" in rel and "condition" in batch:
            rel_loss = F.cross_entropy(rel["condition_logits"], batch["condition"].to(self.device).long()).view(1)
        if rel and "uncertainty" in rel:
            weights = rel["weights"]
            valid = batch.get("modality_valid")
            if valid is not None:
                # 将有效模态指示归一化为相对权重，监督各尺度的两路权重。
                valid = valid.to(self.device, dtype=weights.dtype).unsqueeze(1).expand_as(weights)
                valid = valid / valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
                unc_loss = F.mse_loss(weights, valid).view(1)
            else:
                # 未提供有效性标签时，沿用模型 uncertainty 的均值作为辅助项。
                unc_loss = rel["uncertainty"].mean().view(1)

        # sem 项默认关闭；启用时仅比较每张图像的两路全局最大检测概率。
        if self.lambda_sem:
            sem_loss = F.mse_loss(self._presence_scores(preds["rgb"]), self._presence_scores(preds["sonar"])).view(1)
        else:
            sem_loss = torch.zeros(1, device=self.device)
        loss = torch.cat(
            (
                rgb_loss,
                sonar_loss,
                sem_loss * self.lambda_sem,
                rel_loss * self.mu_rel,
                unc_loss * self.nu_unc,
            )
        )
        detached = torch.cat((rgb_items, sonar_items, sem_loss.detach(), rel_loss.detach(), unc_loss.detach()))
        return loss, detached
