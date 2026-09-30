# Derived from Ultralytics (AGPL-3.0), https://github.com/ultralytics/ultralytics.
# Copyright (c) Ultralytics. See LICENSE and THIRD_PARTY_NOTICES.md.
"""两路检测共用的轴对齐边界框几何计算与非极大值抑制。"""
from __future__ import annotations

import math
import numpy as np
import torch


def empty_like(x):
    """创建与输入形状、dtype 相同的未初始化 Tensor 或 NumPy 数组。"""
    return torch.empty_like(x, dtype=x.dtype) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=x.dtype)


def xywh2xyxy(x):
    """将中心点、宽、高 (x,y,w,h) 转为左上角、右下角 (x1,y1,x2,y2)。

    输入为最后一维长度为 4 的 Tensor 或 NumPy 数组，返回同形状、同 dtype
    的新对象，不改变输入。
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # 四个坐标均在下方赋值，无需复制原数据。
    xy = x[..., :2]  # 中心坐标
    wh = x[..., 2:] / 2  # 半宽、半高
    y[..., :2] = xy - wh  # 左上角
    y[..., 2:] = xy + wh  # 右下角
    return y


def xyxy2xywh(x):
    """将左上角、右下角 (x1,y1,x2,y2) 转为中心点、宽、高 (x,y,w,h)。

    输入为最后一维长度为 4 的 Tensor 或 NumPy 数组，返回同形状、同 dtype
    的新对象，不改变输入。
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # 为输出坐标分配新空间。
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = (x1 + x2) / 2  # 中心 x
    y[..., 1] = (y1 + y2) / 2  # 中心 y
    y[..., 2] = x2 - x1  # 宽
    y[..., 3] = y2 - y1  # 高
    return y


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """按各特征层尺寸生成网格点 [A,2] 及对应步长 [A,1]。

    当前检测头传入各层 BCHW 特征列表；网格坐标以特征格为单位，默认偏移
    0.5，表示格点中心。乘对应步长后才转换为输入图像像素坐标。
    """
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i in range(len(feats)):  # 用层索引读取步长，不直接遍历步长张量。
        stride = strides[i]
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # x 方向格内偏移
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # y 方向格内偏移
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """将网格点到左、上、右、下边界的距离转换为 xywh 或 xyxy 框。"""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # 中心点、宽、高
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: int | None = None) -> torch.Tensor:
    """由 xyxy 框求网格点到四边的距离；指定 reg_max 时截断到分箱范围。"""
    x1y1, x2y2 = bbox.chunk(2, -1)
    dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)  # 左、上、右、下距离
    return dist


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """计算两组 xyxy 框的两两 IoU，输入 [N,4]、[M,4]，输出 [N,M]。

    eps 用于避免除零；计算前转为 float32。参考实现：
        https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py
    """
    # 用 float32 计算交集与并集，避免低精度坐标运算带来的额外误差。
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = 交集面积 / (面积1 + 面积2 - 交集面积)。
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """计算可广播形状的边界框 IoU 或其 GIoU、DIoU、CIoU 变体。

    两个输入最后一维均为 4，其余维度须能广播。xywh=True 时输入为中心点、
    宽、高，否则为左上角、右下角。变体标记同时启用时优先 CIoU，再 DIoU，
    最后 GIoU；均关闭则返回普通 IoU。输出保留长度为 1 的末维。
    当前目标分配与框损失使用 xywh=False、CIoU=True。
    """
    # 统一得到边界框的角点坐标。
    if xywh:  # 从中心点、宽、高转换为角点
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # 输入已经是角点格式。
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # 交集面积。
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # 并集面积。
    union = w1 * h1 + w2 * h2 - inter + eps

    # 普通 IoU，以及可选的几何惩罚项。
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # 最小外接框宽度
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # 最小外接框高度
        if CIoU or DIoU:  # 距离或完整 IoU，参考 https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # 外接框对角线长度平方
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # 两框中心距离的平方
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # 最小外接框面积
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou


def torch_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """按分数降序执行贪心 IoU 抑制，返回保留框在原输入中的索引。

    boxes 为 [N,4] 的 xyxy 框，scores 为 [N] 分数；与当前保留框 IoU 大于
    iou_threshold 的剩余框被移除。不在此区分类别，也不承诺并列分数的
    排序在不同设备上完全一致。
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # 提取角点并预先计算每个框的面积。
    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1) * (y2 - y1)

    # 按分数降序排列。
    order = scores.argsort(0, descending=True)

    # 按候选总数预分配保留索引空间。
    keep = torch.zeros(order.numel(), dtype=torch.int64, device=boxes.device)
    keep_idx = 0
    while order.numel() > 0:
        i = order[0]
        keep[keep_idx] = i
        keep_idx += 1

        if order.numel() == 1:
            break
        # 向量化计算当前框与剩余框的相交区域。
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])

        # 计算交集面积。
        w = (xx2 - xx1).clamp_(min=0)
        h = (yy2 - yy1).clamp_(min=0)
        inter = w * h
        # 当前框与其余框完全不相交时，跳过这一轮的 IoU 计算。
        if inter.sum() == 0:
            # 剩余框仍须在后续迭代中相互比较，而非直接全部作为最终结果。
            order = rest
            continue
        iou = inter / (areas[i] + areas[rest] - inter)
        # 仅保留与当前框 IoU 不超过阈值的候选。
        order = rest[iou <= iou_threshold]

    return keep[:keep_idx]


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.7,
    nc=0,
    multi_label=False,
    agnostic=False,
    max_det=300,
    max_nms=30000,
    max_wh=7680,
):
    """对已解码的轴对齐检测结果做置信度过滤与 NMS，不设置批次耗时截断。

    输入为 [batch,4+nc,anchors]，前四通道为 xywh，后续为类别概率；也接受
    以该张量为首项的元组或列表。返回逐图像 [检测数,6] 张量，列为
    x1、y1、x2、y2、置信度、类别 ID；无检测时保留 [0,6] 空张量。
    multi_label=False 时每个候选仅取最高分类分数，否则可保留多个类别。
    默认按类别偏移框坐标后执行 NMS；agnostic=True 时不区分类别。
    max_nms 限制进入 NMS 的候选数，max_det 限制每张图像的最终框数。
    """
    if not 0 <= conf_thres <= 1 or not 0 <= iou_thres <= 1:
        raise ValueError("Confidence and IoU thresholds must lie within [0, 1].")
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[0]
    nc = nc or prediction.shape[1] - 4
    if prediction.ndim != 3 or prediction.shape[1] != 4 + nc:
        raise ValueError("Expected decoded prediction of shape (batch, 4 + nc, anchors).")
    candidates = prediction[:, 4:4 + nc].amax(1) > conf_thres
    # 复制后再转换坐标，避免修改调用方的已解码张量。
    prediction = prediction.transpose(-1, -2).clone()
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = []
    multi_label = multi_label and nc > 1
    for index, sample in enumerate(prediction):
        sample = sample[candidates[index]]
        if not len(sample):
            output.append(prediction.new_zeros((0, 6)))
            continue
        boxes, classes = sample.split((4, nc), 1)
        if multi_label:
            row, cls = torch.where(classes > conf_thres)
            sample = torch.cat((boxes[row], sample[row, 4 + cls, None], cls[:, None].float()), 1)
        else:
            confidence, cls = classes.max(1, keepdim=True)
            sample = torch.cat((boxes, confidence, cls.float()), 1)[confidence.view(-1) > conf_thres]
        if not len(sample):
            output.append(prediction.new_zeros((0, 6)))
            continue
        if len(sample) > max_nms:
            sample = sample[sample[:, 4].argsort(descending=True)[:max_nms]]
        offsets = sample[:, 5:6] * (0 if agnostic else max_wh)
        keep = torch_nms(sample[:, :4] + offsets, sample[:, 4], iou_thres)[:max_det]
        output.append(sample[keep])
    return output
