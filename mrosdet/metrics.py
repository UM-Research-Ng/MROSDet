# Derived from Ultralytics (AGPL-3.0), https://github.com/ultralytics/ultralytics.
# Copyright (c) Ultralytics. See LICENSE and THIRD_PARTY_NOTICES.md.
"""按来源实现的匹配与 AP 规则，分别评估 RGB 和 Sonar 检测结果。"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from .box_ops import box_iou, non_max_suppression, xywh2xyxy


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """按序列长度比例 f 构造奇数宽度均值滤波器，边界用端点值填充。"""
    nf = round(len(y) * f * 2) // 2 + 1  # 滤波窗口宽度须为奇数
    p = np.ones(nf // 2)  # 两端填充所需的单位向量
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # 复制端点值填充
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def compute_ap(recall: list[float], precision: list[float]) -> tuple[float, np.ndarray, np.ndarray]:
    """由召回率和精确率曲线计算 AP，并返回精确率包络及补齐后的召回率。

    保留来源实现的端点补齐规则，在 0 到 1 的 101 个点上插值并做梯形积分。
    """
    # 补齐端点，并在末端保留最后一个实际召回率位置。
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))

    # 从右向左取累计最大值，构造单调精确率包络。
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # 当前使用插值积分；保留来源中的连续积分分支。
    method = "interp"  # 当前固定为 interp，另一分支为 continuous
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101 点插值的 AP 计算约定
        func = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # 兼容不同 NumPy 版本
        ap = func(np.interp(x, mrec, mpre), x)  # 梯形积分
    else:  # 连续积分分支
        i = np.where(mrec[1:] != mrec[:-1])[0]  # 召回率变化的位置
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # 曲线下面积

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
    """计算存在真实目标的各类别 AP，以及统一置信度阈值处的 P/R/F1。

    tp 为 [检测数, IoU 阈值数] 匹配布尔数组；conf、pred_cls 为检测的置信度
    和类别，target_cls 为全部真实目标的类别。P/R/F1 使用第一个 IoU 阈值，
    在平滑后的类别平均 F1 最大位置取统一置信度阈值。

    返回顺序：tp、fp、p、r、f1、ap、unique_classes、p_curve、r_curve、
    f1_curve、x、prec_values。前五项为选定阈值处的逐类别结果，ap 为
    [真实类别数, IoU 阈值数]；曲线横轴 x 是 1000 个置信度采样点。
    prec_values 则是在相同 0 到 1 网格上的第一 IoU 阈值精确率包络，
    仅包含同时有真实目标和预测的类别；无可用曲线时返回一行零值。

    此精简接口不提供绘图，plot 必须为 False。on_plot、save_dir、names、
    prefix 仅保留兼容签名，不会生成文件或调用绘图回调。
    """
    # 按最终检测置信度降序排列；此处没有单独的 objectness 分数。
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 只统计真实标签中出现的类别。
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # 有真实目标的类别数

    # 构造逐类别精确率、召回率曲线及 AP。
    x, prec_values = np.linspace(0, 1, 1000), []

    # AP 按各 IoU 阈值保存，P/R 曲线各含 1000 个置信度采样点。
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # 当前类别的真实目标数
        n_p = i.sum()  # 当前类别的预测数
        if n_p == 0 or n_l == 0:
            continue

        # 按置信度排序累计假阳性和真阳性。
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # 第一 IoU 阈值对应的召回率曲线。
        recall = tpc / (n_l + eps)  # 各 IoU 阈值的累计召回率
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # 取负使置信度自变量递增

        # 第一 IoU 阈值对应的精确率曲线。
        precision = tpc / (tpc + fpc)  # 各 IoU 阈值的累计精确率
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # 各置信度采样点的精确率

        # 对每个 IoU 阈值的召回率—精确率曲线分别积分。
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # 第一 IoU 阈值的精确率包络

    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))  # 每条有效包络含 1000 个点

    # 计算精确率与召回率的调和平均 F1。
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = {i: names[k] for i, k in enumerate(unique_classes) if k in names}  # 仅保留存在真实目标的类别名
    if plot:
        raise ValueError("Plotting is not part of the compact evaluation API.")

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # 平滑后的类别平均 F1 最大位置
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # 同一置信度阈值下的逐类别结果
    tp = (r * nt).round()  # 由召回率换算真阳性数
    fp = (tp / (p + eps) - tp).round()  # 由精确率换算假阳性数
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values


def postprocess(preds, nc, conf=0.25, iou=0.7, max_det=300, multi_label=False):
    """对两路已解码预测分别执行 NMS，返回逐图像的 xyxy/置信度/类别张量。

    预测默认仅保留每个候选点最高分的类别；evaluate 显式启用 multi_label，
    使同一候选点中所有超过置信度阈值的类别均可进入按类别执行的 NMS。
    """
    return {
        branch: non_max_suppression(
            preds[branch], conf_thres=conf, iou_thres=iou, nc=nc,
            multi_label=multi_label, max_det=max_det,
        )
        for branch in ("rgb", "sonar")
    }


def match_predictions(pred_classes, true_classes, iou, thresholds=None):
    """按类别和 IoU 建立一对一匹配，保留来源实现的排序、去重次序。

    默认分别使用 0.50 到 0.95 的十个 IoU 阈值；返回 [检测数, 阈值数]
    布尔数组。同一阈值下，每个预测和真实目标最多参与一次匹配。
    """
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
    """合并逐图像统计，只对有真实目标的类别计算 P/R/AP 均值。

    没有任何真实目标时返回零指标及空类别列表；有目标但无预测的类别记零。
    """
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
    """在方形缩放后的坐标空间中，分别用 RGB/Sonar 自己的标签评估。

    数据加载器提供 uint8 BCHW 图像批次；本函数转为模型 dtype 并除以 255。
    归一化标签框换算到各路当前输入尺寸，使用多标签 NMS 和十个 IoU 阈值。
    两路 mAP50_95 的算术平均作为选择权重的 fitness；结束时恢复模型原有的
    训练/验证模式。此函数只返回统计，不绘图或写入评估文件。
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
