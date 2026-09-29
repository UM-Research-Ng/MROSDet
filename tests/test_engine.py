"""Focused checks for optimizer semantics and paired prediction coordinates."""

from pathlib import Path

import pytest
import torch

from mrosdet.engine import build_optimizer, linear_lr_factor, pair_image_files, scale_detections


def test_optimizer_bias_and_batchnorm_are_not_decayed():
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4))
    optimizer = build_optimizer(model)
    settings = {id(parameter): group for group in optimizer.param_groups for parameter in group["params"]}
    assert settings[id(model[0].weight)]["weight_decay"] == 0.0005
    assert settings[id(model[0].bias)]["weight_decay"] == 0
    assert settings[id(model[1].weight)]["weight_decay"] == 0
    assert settings[id(model[0].weight)]["betas"] == (0.9, 0.999)


def test_lr_matches_original_last_epoch():
    assert 0.000769 * linear_lr_factor(19, 20) == pytest.approx(4.57555e-5)
    assert linear_lr_factor(19, 50) > linear_lr_factor(19, 20)


def test_boxes_scale_to_each_modality_without_fixed_640():
    boxes = torch.tensor([[32.0, 64.0, 160.0, 256.0, 0.8, 2.0]])
    rgb = scale_detections(boxes, (480, 800, 3), imgsz=320)
    sonar = scale_detections(boxes, (200, 300, 3), imgsz=320)
    assert rgb[0, :4].tolist() == [80.0, 96.0, 400.0, 384.0]
    assert sonar[0, :4].tolist() == [30.0, 40.0, 150.0, 160.0]
    assert torch.equal(boxes[0, 4:], rgb[0, 4:])


def test_pairing_preserves_relative_paths_and_rejects_missing(tmp_path):
    rgb, sonar = tmp_path / "rgb", tmp_path / "sonar"
    for root in (rgb, sonar):
        for folder in ("one", "two"):
            (root / folder).mkdir(parents=True)
            (root / folder / "same.jpg").touch()
    pairs = pair_image_files(rgb, sonar)
    assert [key for key, _, _ in pairs] == [Path("one/same.jpg"), Path("two/same.jpg")]
    (sonar / "two/same.jpg").unlink()
    with pytest.raises(ValueError, match="not paired"):
        pair_image_files(rgb, sonar)


def test_prediction_pairs_same_stem_different_formats(tmp_path):
    rgb, sonar = tmp_path / "rgb", tmp_path / "sonar"
    rgb.mkdir()
    sonar.mkdir()
    (rgb / "view.jpg").touch()
    (sonar / "view.png").touch()
    pairs = pair_image_files(rgb, sonar)
    assert pairs == [(Path("view.jpg"), rgb / "view.jpg", sonar / "view.png")]
    (rgb / "view.png").touch()
    with pytest.raises(ValueError, match="Duplicate image key"):
        pair_image_files(rgb, sonar)
