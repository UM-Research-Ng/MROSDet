# Derived from Ultralytics (AGPL-3.0), https://github.com/ultralytics/ultralytics.
# Copyright (c) Ultralytics. See LICENSE and THIRD_PARTY_NOTICES.md.
"""Dual-modality detection evaluation using the original AP and matching rules."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from .box_ops import box_iou, non_max_suppression, xywh2xyxy


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def compute_ap(recall: list[float], precision: list[float]) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list[float]): The recall curve.
        precision (list[float]): The precision curve.

    Returns:
        ap (float): Average precision.
        mpre (np.ndarray): Precision envelope curve.
        mrec (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        func = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # np.trapz deprecated
        ap = func(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    plot: bool = False,
    on_plot=None,
    save_dir: Path = Path(),
    names: dict[int, str] = {},
    eps: float = 1e-16,
    prefix: str = "",
) -> tuple:
    """Compute the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the targets.
        plot (bool, optional): Whether to plot PR curves or not.
        on_plot (callable, optional): A callback to pass plots path and data when they are rendered.
        save_dir (Path, optional): Directory to save the PR curves.
        names (dict[int, str], optional): Dictionary of class names to plot PR curves.
        eps (float, optional): A small value to avoid division by zero.
        prefix (str, optional): A prefix string for saving the plot files.

    Returns:
        tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.
        fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class.
        p (np.ndarray): Precision values at threshold given by max F1 metric for each class.
        r (np.ndarray): Recall values at threshold given by max F1 metric for each class.
        f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class.
        ap (np.ndarray): Average precision for each class at different IoU thresholds.
        unique_classes (np.ndarray): An array of unique classes that have data.
        p_curve (np.ndarray): Precision curves for each class.
        r_curve (np.ndarray): Recall curves for each class.
        f1_curve (np.ndarray): F1-score curves for each class.
        x (np.ndarray): X-axis values for the curves.
        prec_values (np.ndarray): Precision values at mAP@0.5 for each class.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    x, prec_values = np.linspace(0, 1, 1000), []

    # Average precision, precision and recall curves
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))  # (nc, 1000)

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = {i: names[k] for i, k in enumerate(unique_classes) if k in names}  # dict: only classes that have data
    if plot:
        raise ValueError("Plotting is not part of the compact evaluation API.")

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values


def postprocess(preds, nc, conf=0.25, iou=0.7, max_det=300, multi_label=False):
    """Decode each branch's evaluation output into xyxy/confidence/class boxes.

    Prediction keeps the best class per anchor. Evaluation explicitly enables
    multi-label NMS, matching the two distinct paths in the source project.
    """
    return {
        branch: non_max_suppression(
            preds[branch], conf_thres=conf, iou_thres=iou, nc=nc,
            multi_label=multi_label, max_det=max_det,
        )
        for branch in ("rgb", "sonar")
    }


def match_predictions(pred_classes, true_classes, iou, thresholds=None):
    """Original greedy class-aware one-to-one matching at ten IoU thresholds."""
    if thresholds is None:
        thresholds = torch.linspace(0.5, 0.95, 10, device=pred_classes.device)
    correct = np.zeros((pred_classes.shape[0], thresholds.shape[0])).astype(bool)
    correct_class = true_classes[:, None] == pred_classes
    iou = (iou * correct_class).cpu().numpy()
    for index, threshold in enumerate(thresholds.cpu().tolist()):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), index] = True
    return correct


def summarize_stats(stats, names, images):
    """Aggregate per-image arrays without changing the source AP averaging."""
    merged = {key: np.concatenate(values, axis=0) for key, values in stats.items()}
    target = merged["target_cls"]
    if not len(target):
        return {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0, "images": images, "instances": 0, "per_class": []}
    results = ap_per_class(merged["tp"], merged["conf"], merged["pred_cls"], target)
    p, r, ap, classes = results[2], results[3], results[5], results[6]
    instances = np.bincount(target.astype(int), minlength=len(names))
    return {
        "precision": float(p.mean()), "recall": float(r.mean()),
        "map50": float(ap[:, 0].mean()), "map50_95": float(ap.mean()),
        "images": images, "instances": int(len(target)),
        "per_class": [
            {"class_id": int(cls), "name": names[int(cls)], "instances": int(instances[cls]),
             "precision": float(p[index]), "recall": float(r[index]),
             "map50": float(ap[index, 0]), "map50_95": float(ap[index].mean())}
            for index, cls in enumerate(classes)
        ],
    }


@torch.inference_mode()
def evaluate(model, dataloader, device, conf=0.001, iou=0.7, max_det=300):
    """Evaluate independent RGB/Sonar labels in the square-resized image space.

    Dataset images must be uint8 CHW batches. The mean of both branches'
    mAP50-95 values is the checkpoint-selection fitness, as in the source.
    """
    device = torch.device(device)
    previous_training = model.training
    model.eval()
    names = model.names
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    dtype = next(model.parameters()).dtype
    stats = {branch: {key: [] for key in ("tp", "conf", "pred_cls", "target_cls")} for branch in ("rgb", "sonar")}
    images = 0
    thresholds = torch.linspace(0.5, 0.95, 10, device=device)
    try:
        for batch in dataloader:
            inputs = {key: batch[key].to(device=device, dtype=dtype) / 255.0 for key in ("img_rgb", "img_sonar")}
            predictions = postprocess(model(inputs), nc=len(names), conf=conf, iou=iou, max_det=max_det, multi_label=True)
            images += inputs["img_rgb"].shape[0]
            for branch in ("rgb", "sonar"):
                for index, detection in enumerate(predictions[branch]):
                    mask = batch[f"batch_idx_{branch}"] == index
                    classes = batch[f"cls_{branch}"][mask].squeeze(-1).to(device)
                    boxes = batch[f"bboxes_{branch}"][mask].to(device)
                    if len(classes):
                        height, width = inputs[f"img_{branch}"].shape[2:]
                        boxes = xywh2xyxy(boxes) * torch.tensor([width, height, width, height], device=device)
                    correct = np.zeros((len(detection), 10), dtype=bool)
                    if len(classes) and len(detection):
                        correct = match_predictions(detection[:, 5], classes, box_iou(boxes, detection[:, :4]), thresholds)
                    stats[branch]["tp"].append(correct)
                    stats[branch]["conf"].append(detection[:, 4].cpu().numpy())
                    stats[branch]["pred_cls"].append(detection[:, 5].cpu().numpy())
                    stats[branch]["target_cls"].append(classes.cpu().numpy())
        if images == 0:
            raise ValueError("Cannot evaluate an empty dataloader.")
        result = {branch: summarize_stats(stats[branch], names, images) for branch in ("rgb", "sonar")}
        result["fitness"] = (result["rgb"]["map50_95"] + result["sonar"]["map50_95"]) / 2
        return result
    finally:
        model.train(previous_training)
