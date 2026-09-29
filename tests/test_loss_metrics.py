"""Numerical edge cases for the compact detection loss and evaluation path."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mrosdet.box_ops import box_iou, dist2bbox, make_anchors, non_max_suppression
from mrosdet.loss import DualModalLoss
from mrosdet.metrics import compute_ap, evaluate, match_predictions


def test_anchor_layout_and_decode():
    features = [torch.zeros(1, 4, 2, 3), torch.zeros(1, 4, 1, 1)]
    anchors, strides = make_anchors(features, torch.tensor([8.0, 16.0]))
    assert anchors.shape == (7, 2)
    assert anchors[:3].tolist() == [[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]]
    assert strides[:, 0].tolist() == [8] * 6 + [16]
    boxes = dist2bbox(torch.ones(7, 4), anchors, xywh=False)
    assert boxes[0].tolist() == [-0.5, -0.5, 1.5, 1.5]


def test_nms_is_class_aware_and_does_not_change_input():
    prediction = torch.tensor([[
        [10, 10, 10], [10, 10, 10], [8, 8, 8], [8, 8, 8],
        [0.9, 0.8, 0.1], [0.1, 0.1, 0.7],
    ]], dtype=torch.float32)
    original = prediction.clone()
    boxes = non_max_suppression(prediction, nc=2)[0]
    assert torch.equal(prediction, original)
    assert len(boxes) == 2
    assert boxes[:, 5].tolist() == [0, 1]
    assert boxes[0, :4].tolist() == [6, 6, 14, 14]


def test_matching_disallows_duplicate_or_wrong_class_true_positives():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    detections = boxes.repeat(3, 1)
    correct = match_predictions(torch.tensor([0, 0, 1]), torch.tensor([0]), box_iou(boxes, detections))
    assert correct.sum(axis=0).tolist() == [1] * 10
    assert not correct[2].any()


def test_ap_preserves_source_interpolation_sentinels():
    ap, _, _ = compute_ap(np.array([1.0]), np.array([1.0]))
    assert ap == pytest.approx(0.995)


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nc = 2
        self.reg_max = 1
        self.register_buffer("stride", torch.tensor([8.0, 16.0, 32.0]))


class _LossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reference = torch.nn.Parameter(torch.zeros(1))
        self.model = torch.nn.ModuleList([_Head()])
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, dual_sem=0.0, dual_rel=0.02, dual_unc=0.002)


@pytest.mark.parametrize("empty", [False, True])
def test_dual_loss_is_finite_and_backpropagates(empty):
    torch.manual_seed(7)
    model = _LossModel()
    predictions = {}
    batch = {"condition": torch.tensor([0, 2]), "modality_valid": torch.tensor([[1.0, 1.0], [0.0, 1.0]])}
    for branch in ("rgb", "sonar"):
        predictions[branch] = {
            "boxes": (torch.rand(2, 4, 21) + 1).requires_grad_(),
            "scores": torch.randn(2, 2, 21, requires_grad=True),
            "feats": [torch.zeros(2, 4, 4, 4), torch.zeros(2, 4, 2, 2), torch.zeros(2, 4, 1, 1)],
        }
        batch[f"batch_idx_{branch}"] = torch.empty(0) if empty else torch.tensor([0.0, 1.0])
        batch[f"cls_{branch}"] = torch.empty(0, 1) if empty else torch.tensor([[0.0], [1.0]])
        batch[f"bboxes_{branch}"] = torch.empty(0, 4) if empty else torch.tensor([[0.5, 0.5, 0.5, 0.5]] * 2)
    reliability_logits = torch.randn(2, 3, 2, requires_grad=True)
    weights = reliability_logits.softmax(-1)
    condition_logits = torch.randn(2, 4, requires_grad=True)
    predictions["reliability"] = {"weights": weights, "condition_logits": condition_logits, "uncertainty": 1 - weights.amax(-1)}
    loss, components = DualModalLoss(model)(predictions, batch)
    assert loss.shape == components.shape == (9,)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    for branch in ("rgb", "sonar"):
        assert torch.isfinite(predictions[branch]["scores"].grad).all()
        if not empty:
            assert torch.isfinite(predictions[branch]["boxes"].grad).all()
    assert torch.isfinite(condition_logits.grad).all()
    assert torch.isfinite(reliability_logits.grad).all()


class _PerfectModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reference = torch.nn.Parameter(torch.zeros(1))
        self.names = {0: "object"}

    def forward(self, images):
        batch = images["img_rgb"].shape[0]
        one = torch.tensor([16.0, 16.0, 16.0, 16.0, 0.9], device=self.reference.device).view(1, 5, 1).repeat(batch, 1, 1)
        return {"rgb": one.clone(), "sonar": one.clone(), "reliability": None}


def test_evaluation_keeps_modal_labels_separate_and_restores_mode():
    model = _PerfectModel().train()
    batch = {"img_rgb": torch.zeros(1, 3, 32, 32, dtype=torch.uint8), "img_sonar": torch.zeros(1, 3, 32, 32, dtype=torch.uint8)}
    batch.update({"batch_idx_rgb": torch.tensor([0]), "cls_rgb": torch.tensor([[0.0]]), "bboxes_rgb": torch.tensor([[0.5, 0.5, 0.5, 0.5]])})
    batch.update({"batch_idx_sonar": torch.empty(0), "cls_sonar": torch.empty(0, 1), "bboxes_sonar": torch.empty(0, 4)})
    result = evaluate(model, [batch], torch.device("cpu"))
    assert model.training
    assert result["rgb"]["map50_95"] == pytest.approx(0.995)
    assert result["sonar"]["map50_95"] == 0
    assert result["fitness"] == pytest.approx(0.995 / 2)
    assert result["rgb"]["instances"] == 1
    assert result["sonar"]["instances"] == 0
