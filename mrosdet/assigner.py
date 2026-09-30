# Derived from Ultralytics (AGPL-3.0), https://github.com/ultralytics/ultralytics.
# Copyright (c) Ultralytics. See LICENSE and THIRD_PARTY_NOTICES.md.
"""结合分类分数与定位质量的任务对齐目标分配，保留来源实现的计算规则。"""
from __future__ import annotations

import logging
import torch
import torch.nn as nn

from .box_ops import bbox_iou, xywh2xyxy, xyxy2xywh

LOGGER = logging.getLogger("mrosdet")


class TaskAlignedAssigner(nn.Module):
    """根据分类分数和截断到非负值的 CIoU，为网格候选点分配真实目标。

    对齐度量为分类分数的 alpha 次幂乘定位质量的 beta 次幂；每个目标先取
    topk 个候选点，再消解多目标冲突。topk2 可用于冲突消解后的二次筛选。
    stride_val 取第二层步长（只有一层时取该层），用于小框候选区域扩展。
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        stride: list = [8, 16, 32],
        eps: float = 1e-9,
        topk2=None,
    ):
        """设置候选数、类别数、分类/定位幂次、各层步长与数值稳定常数。

        当前 MROSDet 损失显式传入 topk=10、num_classes=9、alpha=0.5、
        beta=6.0 和模型步长；这些设置不同于此通用构造器的部分默认值。
        """
        super().__init__()
        self.topk = topk
        self.topk2 = topk2 or topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride = stride
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """在无梯度上下文中分配目标，空标签批次直接返回背景结果。

        记 B 为批量、A 为全部网格点数、G 为批内最大目标数、C 为类别数。
        输入 pd_scores 为 [B,A,C] 概率，pd_bboxes 为 [B,A,4] 像素 xyxy，
        anc_points 为 [A,2] 像素坐标；gt_labels、gt_bboxes、mask_gt 分别为
        [B,G,1]、[B,G,4]、[B,G,1]，mask_gt 用于排除补齐标签。
        返回目标类别 [B,A]、目标框 [B,A,4]、软目标分数 [B,A,C]、
        前景掩码 [B,A] 和所分配的目标索引 [B,A]。

        算法参考：
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # 仅目标分配临时在 CPU 重试，结果随后返回原设备，不迁移整个训练。
                LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
                cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
                result = self._forward(*cpu_tensors)
                return tuple(t.to(device) for t in result)
            raise

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """筛选候选点、消解目标冲突并归一化软目标分数；形状约定同 forward。"""
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, align_metric
        )

        # 根据分配索引取得真实类别和未经候选区域扩展的真实框。
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # 按各目标的最大对齐度量和定位质量归一化分类软目标。
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # [B,G,1]
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # [B,G,1]
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """结合目标有效性、候选区域与 topk，生成每个目标的正候选点掩码。

        返回掩码、对齐度量、定位质量，三者形状均为 [B,G,A]。
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        # 计算每个真实目标与候选点的对齐度量，形状 [B,G,A]。
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # 每个有效目标选取 topk 候选点，得到 [B,G,A] 掩码。
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # 合并三个筛选条件。
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """计算 [B,G,A] 对齐度量与截断到非负值的 CIoU。

        分类分数取各真实目标所属的类别；仅 mask_gt 指定的目标—候选点组合
        参与计算，其余位置为零。
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # [B,G,A]
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # [2,B,G]
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # [B,G]
        ind[1] = gt_labels.squeeze(-1)  # [B,G]
        # 取每个网格点对各真实目标类别的预测概率。
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # [B,G,A]

        # 将预测框与真实框广播到 [B,G,A,4]，再按有效组合取值。
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """逐对计算轴对齐 xyxy 框的 CIoU，并将负值截断为零。"""
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, topk_mask=None):
        """按 [B,G,A] 度量选择 topk，返回同形状的候选掩码。

        topk_mask 可指定 [B,G,topk] 有效性；未提供时，仅保留最大度量大于
        eps 的目标。无效位置填入零索引后，通过计数清除重复占位索引。
        """
        # 每个目标的 topk 个度量与索引：[B,G,topk]。
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # 无效候选位置用零索引占位。
        topk_idxs.masked_fill_(~topk_mask, 0)

        # 将候选索引累加为 [B,G,A] 计数。
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # 逐个候选索引累加计数。
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # 清除无效填充产生的重复索引。
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """按 [B,A] 分配索引读取真实类别与框，并构造前景的独热分类目标。

        返回类别 [B,A]、真实框 [B,A,4] 与分类目标 [B,A,num_classes]。
        背景位置的分类目标置零；类别和框只有在前景掩码下才作为监督使用。
        """
        # 为每张图像加上展平标签数组的批次偏移。
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # [B,A]
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # [B,A]

        # 按分配索引读取真实框：[B,G,4] -> [B,A,4]。
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # 构造分类目标前，保证类别索引非负。
        target_labels.clamp_(0)

        # 用 scatter_ 写入独热类别值。
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # [B,A,num_classes]
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # [B,A,num_classes]
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        """筛选落在目标候选区域内部的网格点，返回 [B,G,A] 布尔掩码。

        输入为像素网格点 [A,2]、真实 xyxy 框 [B,G,4] 与有效性 [B,G,1]。
        对有效目标，宽或高小于最小步长时，仅将该维候选区域扩展到 stride_val，
        中心不变。当前步长为 [8,16,32]，对应阈值 8、扩展尺寸 16。
        扩展只用于候选点筛选，不修改后续回归的真实框；边界距离须大于 eps。
        """
        gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
        wh_mask = gt_bboxes_xywh[..., 2:] < self.stride[0]  # 最小特征层步长
        gt_bboxes_xywh[..., 2:] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(self.stride_val, dtype=gt_bboxes_xywh.dtype, device=gt_bboxes_xywh.device),
            gt_bboxes_xywh[..., 2:],
        )
        gt_bboxes = xywh2xyxy(gt_bboxes_xywh)

        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # 左上角、右下角
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps)

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        """多个目标占用同一候选点时，按定位质量选择唯一目标。

        mask_pos、overlaps、align_metric 均为 [B,G,A]；返回目标索引 [B,A]、
        前景计数 [B,A] 和更新后的 [B,G,A] 掩码。topk2 不等于 topk 时，
        再按剩余正候选点的对齐度量筛选。
        """
        # 沿目标维统计每个候选点被分配的次数：[B,G,A] -> [B,A]。
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # 同一候选点被多个真实目标占用
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # [B,G,A]

            max_overlaps_idx = overlaps.argmax(1)  # [B,A]
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # [B,G,A]

            fg_mask = mask_pos.sum(-2)

        if self.topk2 != self.topk:
            align_metric = align_metric * mask_pos  # 仅保留当前正候选点的度量
            max_overlaps_idx = torch.topk(align_metric, self.topk2, dim=-1, largest=True).indices  # [B,G,topk2]
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)  # 二次筛选掩码
            topk_idx.scatter_(-1, max_overlaps_idx, 1.0)
            mask_pos *= topk_idx
            fg_mask = mask_pos.sum(-2)
        # 返回每个候选点分配到的目标索引。
        target_gt_idx = mask_pos.argmax(-2)  # [B,A]
        return target_gt_idx, fg_mask, mask_pos
