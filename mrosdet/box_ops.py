# Derived from Ultralytics (AGPL-3.0), https://github.com/ultralytics/ultralytics.
# Copyright (c) Ultralytics. See LICENSE and THIRD_PARTY_NOTICES.md.
"""Axis-aligned box geometry and suppression used by both detection heads."""
from __future__ import annotations

import math
import numpy as np
import torch


def empty_like(x):
    """Create empty torch.Tensor or np.ndarray with same shape and dtype as input."""
    return torch.empty_like(x, dtype=x.dtype) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=x.dtype)


def xywh2xyxy(x):
    """Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is
    the top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in (x, y, width, height) format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


def xyxy2xywh(x):
    """Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is
    the top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = (x1 + x2) / 2  # x center
    y[..., 1] = (y1 + y2) / 2  # y center
    y[..., 2] = x2 - x1  # width
    y[..., 3] = y2 - y1  # height
    return y


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i in range(len(feats)):  # use len(feats) to avoid TracerWarning from iterating over strides tensor
        stride = strides[i]
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: int | None = None) -> torch.Tensor:
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)  # dist (lt, rb)
    return dist


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Calculate intersection-over-union (IoU) of boxes.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes in (x1, y1, x2, y2) format.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes in (x1, y1, x2, y2) format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.

    References:
        https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py
    """
    # NOTE: Need .float() to get accurate iou values
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
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
    """Calculate the Intersection over Union (IoU) between bounding boxes.

    This function supports various shapes for `box1` and `box2` as long as the last dimension is 4. For instance, you
    may pass tensors shaped like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4). Internally, the code will split the last
    dimension into (x, y, w, h) if `xywh=True`, or (x1, y1, x2, y2) if `xywh=False`.

    Args:
        box1 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        box2 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in (x1, y1,
            x2, y2) format.
        GIoU (bool, optional): If True, calculate Generalized IoU.
        DIoU (bool, optional): If True, calculate Distance IoU.
        CIoU (bool, optional): If True, calculate Complete IoU.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou


def torch_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Optimized NMS with early termination that matches torchvision behavior exactly.

    Args:
        boxes (torch.Tensor): Bounding boxes with shape (N, 4) in xyxy format.
        scores (torch.Tensor): Confidence scores with shape (N,).
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        (torch.Tensor): Indices of boxes to keep after NMS.

    Examples:
        Apply NMS to a set of boxes
        >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
        >>> scores = torch.tensor([0.9, 0.8])
        >>> keep = TorchNMS.nms(boxes, scores, 0.5)
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # Pre-allocate and extract coordinates once
    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1) * (y2 - y1)

    # Sort by scores descending
    order = scores.argsort(0, descending=True)

    # Pre-allocate keep list with maximum possible size
    keep = torch.zeros(order.numel(), dtype=torch.int64, device=boxes.device)
    keep_idx = 0
    while order.numel() > 0:
        i = order[0]
        keep[keep_idx] = i
        keep_idx += 1

        if order.numel() == 1:
            break
        # Vectorized IoU calculation for remaining boxes
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])

        # Fast intersection and IoU
        w = (xx2 - xx1).clamp_(min=0)
        h = (yy2 - yy1).clamp_(min=0)
        inter = w * h
        # Early exit: skip IoU calculation if no intersection
        if inter.sum() == 0:
            # No overlaps with current box, keep all remaining boxes
            order = rest
            continue
        iou = inter / (areas[i] + areas[rest] - inter)
        # Keep boxes with IoU <= threshold
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
    """Source detector's axis-aligned NMS path, without a batch time cutoff.

    Input: decoded ``(batch, 4 + nc, anchors)`` xywh/class probabilities.
    Output: one ``(detections, 6)`` xyxy/confidence/class tensor per image.
    Every image is processed, including on slower CPU-only installations.
    """
    if not 0 <= conf_thres <= 1 or not 0 <= iou_thres <= 1:
        raise ValueError("Confidence and IoU thresholds must lie within [0, 1].")
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[0]
    nc = nc or prediction.shape[1] - 4
    if prediction.ndim != 3 or prediction.shape[1] != 4 + nc:
        raise ValueError("Expected decoded prediction of shape (batch, 4 + nc, anchors).")
    candidates = prediction[:, 4:4 + nc].amax(1) > conf_thres
    # Keep the caller's decoded tensor intact; the original implementation used
    # a view here and replaced its xywh coordinates in place.
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
