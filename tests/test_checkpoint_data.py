"""检查权重读写和成对数据加载，所有测试文件均写入临时目录。"""

from copy import deepcopy
from pathlib import Path
import pickle

import cv2
import numpy as np
import pytest
import torch
import yaml

from mrosdet.checkpoint import (
    FORMAT_VERSION,
    extra_buffers,
    load_model,
    restore_buffers,
    save_checkpoint,
)
from mrosdet.data import PairedDataset, read_labels
from mrosdet.nn.model import MROSDet


@pytest.fixture(scope="module")
def checkpoint_case(tmp_path_factory):
    """复用两个小模型，检查权重保存后能否正确恢复。"""
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/mrosdet.yaml").read_text(encoding="utf-8"))
    config["nc"] = 2
    # ARC2PSA 在默认扩展率 0.5 下，输入至少 128 通道才能保留一个注意力头。
    config["scales"]["n"] = [0.25, 0.125, 1024]
    config["head"][10][3][1] = [16, 32, 64]
    torch.manual_seed(5)
    model = MROSDet(config).eval()
    with torch.no_grad():
        for index in (26, 29, 32):
            model.model[index].scale_gain.copy_(torch.tensor([0.125, 0.25, 0.5]))
        model.model[35].post_gain.copy_(torch.tensor([0.05, 0.1, 0.2]))
    config_to_save = deepcopy(model.config)
    config_to_save["yaml_file"] = "/private/source-machine/model.yaml"
    config_to_save["training_machine"] = "must-not-be-published"
    path = tmp_path_factory.mktemp("checkpoint") / "best.pt"
    names = {0: "first", 1: "second"}
    save_checkpoint(model, path, config_to_save, names, source_sha256="a" * 64)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    restored = load_model(path, device="cpu")
    return model, restored, payload, names


def _assert_plain_weights_payload(value):
    if isinstance(value, torch.Tensor):
        assert value.device.type == "cpu"
    elif isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, (str, int))
            _assert_plain_weights_payload(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_plain_weights_payload(item)
    else:
        assert value is None or type(value) in (bool, int, float, str)


def test_checkpoint_is_plain_weights_only_and_strips_machine_paths(checkpoint_case):
    _, restored, payload, names = checkpoint_case
    _assert_plain_weights_payload(payload)
    assert payload["format_version"] == FORMAT_VERSION
    assert payload["architecture"] == "MROSDet"
    assert payload["source_sha256"] == "a" * 64
    assert "model" not in payload and "optimizer" not in payload
    assert "yaml_file" not in payload["config"]
    assert "training_machine" not in payload["config"]
    assert payload["preprocess"] == {"resize": "stretch", "color": "RGB", "scale": 255.0}
    assert restored.names == names
    assert not restored.training


def test_checkpoint_roundtrip_preserves_weights_outputs_and_fusion_buffers(checkpoint_case):
    source, restored, payload, _ = checkpoint_case
    expected_buffers = {
        "model.26.scale_gain", "model.29.scale_gain", "model.32.scale_gain", "model.35.post_gain",
    }
    assert set(payload["extra_buffers"]) == expected_buffers
    assert expected_buffers.isdisjoint(payload["state_dict"])
    assert source.state_dict().keys() == restored.state_dict().keys()
    for key, tensor in source.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[key], rtol=0, atol=0)
    for key, tensor in extra_buffers(source).items():
        torch.testing.assert_close(tensor, extra_buffers(restored)[key], rtol=0, atol=0)
    generator = torch.Generator().manual_seed(11)
    images = {
        "img_rgb": torch.rand(1, 3, 64, 64, generator=generator),
        "img_sonar": torch.rand(1, 3, 64, 64, generator=generator),
    }
    with torch.no_grad():
        before, after = source(images), restored(images)
    for branch in ("rgb", "sonar"):
        torch.testing.assert_close(before[branch][0], after[branch][0], rtol=1e-6, atol=1e-6)
    for key in ("weights", "condition_logits", "uncertainty"):
        torch.testing.assert_close(before["reliability"][key], after["reliability"][key], rtol=1e-6, atol=1e-6)


def test_missing_checkpoint_fails_instead_of_random_initialization(tmp_path):
    with pytest.raises(FileNotFoundError, match="Weights not found"):
        load_model(tmp_path / "not-downloaded.pt")


class _UnsupportedPayload:
    """用于验证安全加载器不会反序列化自定义 Python 类。"""


def test_loader_rejects_python_objects_outside_weights_only_format(tmp_path):
    path = tmp_path / "python-object.pt"
    torch.save({"model": _UnsupportedPayload()}, path)
    with pytest.raises(pickle.UnpicklingError):
        load_model(path)


@pytest.mark.parametrize("payload", [
    {"format_version": 999, "architecture": "MROSDet"},
    {"format_version": FORMAT_VERSION, "architecture": "different-network"},
    {"state_dict": {}},
    ["not a checkpoint mapping"],
])
def test_unsupported_checkpoint_format_is_rejected(tmp_path, payload):
    path = tmp_path / "unsupported.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Unsupported checkpoint"):
        load_model(path)


def test_buffer_restore_rejects_missing_extra_and_wrong_shape():
    model = torch.nn.Module()
    model.register_buffer("scale_gain", torch.tensor([0.0, 0.35, 0.65]), persistent=False)
    with pytest.raises(ValueError, match="buffers differ"):
        restore_buffers(model, {})
    with pytest.raises(ValueError, match="buffers differ"):
        restore_buffers(model, {"scale_gain": torch.ones(3), "unknown": torch.ones(1)})
    with pytest.raises(ValueError, match="shape mismatch"):
        restore_buffers(model, {"scale_gain": torch.ones(2)})


def _write_pair(root, key="scene/pair", rgb_label="0 0.5 0.5 0.4 0.2\n", sonar_label="1 0.4 0.4 0.2 0.4\n"):
    for modality, shape, color, label in (
        ("RGB", (11, 17, 3), (20, 40, 60), rgb_label),
        ("Sonar", (9, 19, 3), (100, 130, 160), sonar_label),
    ):
        image_path = root / modality / "train/images" / f"{key}.png"
        label_path = root / modality / "train/labels" / f"{key}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.empty(shape, dtype=np.uint8)
        image[:] = color
        assert cv2.imwrite(str(image_path), image)
        label_path.write_text(label, encoding="utf-8")


def _data_config(root):
    return {
        "path": str(root), "nc": 2, "names": {0: "first", 1: "second"},
        "modalities": {"rgb": {"train": "RGB/train/images"}, "sonar": {"train": "Sonar/train/images"}},
    }


def test_dataset_reads_paired_colors_shapes_and_independent_labels(tmp_path):
    _write_pair(tmp_path)
    dataset = PairedDataset(_data_config(tmp_path), imgsz=32, augment=False)
    sample = dataset[0]
    assert len(dataset) == 1 and sample["pair_key"] == "scene/pair"
    assert sample["img_rgb"].shape == sample["img_sonar"].shape == (3, 32, 32)
    assert sample["img_rgb"].dtype == torch.uint8
    assert sample["img_rgb"][:, 0, 0].tolist() == [60, 40, 20]
    assert sample["img_sonar"][:, 0, 0].tolist() == [160, 130, 100]
    assert sample["ori_shape_rgb"] == (11, 17)
    assert sample["ori_shape_sonar"] == (9, 19)
    assert sample["cls_rgb"].tolist() == [[0.0]]
    assert sample["cls_sonar"].tolist() == [[1.0]]
    assert not torch.equal(sample["bboxes_rgb"], sample["bboxes_sonar"])
    # 修改返回批次中的标签，不应影响数据集缓存的原始标签。
    sample["bboxes_rgb"].zero_()
    assert dataset[0]["bboxes_rgb"].count_nonzero() == 4


def test_empty_labels_are_valid_and_collation_offsets_each_stream(tmp_path):
    _write_pair(tmp_path, "a", sonar_label="")
    _write_pair(tmp_path, "b", rgb_label="", sonar_label="1 0.5 0.5 0.2 0.2\n")
    dataset = PairedDataset(_data_config(tmp_path), imgsz=32)
    assert dataset[0]["cls_sonar"].shape == (0, 1)
    assert dataset[0]["bboxes_sonar"].shape == (0, 4)
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    assert batch["img_rgb"].shape == (2, 3, 32, 32)
    assert batch["batch_idx_rgb"].tolist() == [0.0]
    assert batch["batch_idx_sonar"].tolist() == [1.0]
    assert batch["cls_rgb"].tolist() == [[0.0]]
    assert batch["cls_sonar"].tolist() == [[1.0]]


@pytest.mark.parametrize("label", [
    "0 0.5 0.5 0.2",                 # 缺少一列
    "0 0.5 0.5 0.2 0.2 0",           # 多出一列
    "0 0.5 nan 0.2 0.2",             # 坐标不是有限数值
    "0 inf 0.5 0.2 0.2",             # 坐标不是有限数值
    "0.5 0.5 0.5 0.2 0.2",           # 类别编号不是整数
    "2 0.5 0.5 0.2 0.2",             # 类别编号超出范围
    "-1 0.5 0.5 0.2 0.2",            # 类别编号为负
    "0 1.2 0.5 0.2 0.2",             # 坐标超出归一化范围
    "0 0.5 0.5 0 0.2",               # 框宽度为零
    "0 0.05 0.5 0.2 0.2",            # 框超出图像边界
    "not a detection label",          # 非法文本
])
def test_invalid_labels_are_rejected(tmp_path, label):
    path = tmp_path / "invalid.txt"
    path.write_text(label, encoding="utf-8")
    with pytest.raises(ValueError):
        read_labels(path, nc=2)


def test_missing_label_is_not_silently_treated_as_background(tmp_path):
    _write_pair(tmp_path)
    (tmp_path / "Sonar/train/labels/scene/pair.txt").unlink()
    with pytest.raises(FileNotFoundError, match="Missing label"):
        PairedDataset(_data_config(tmp_path), imgsz=32)


def test_missing_pair_is_rejected_even_when_both_folders_have_images(tmp_path):
    _write_pair(tmp_path, "a")
    _write_pair(tmp_path, "b")
    (tmp_path / "Sonar/train/images/b.png").unlink()
    with pytest.raises(ValueError, match="Unpaired"):
        PairedDataset(_data_config(tmp_path), imgsz=32)


def test_duplicate_image_stems_are_rejected(tmp_path):
    _write_pair(tmp_path)
    source = tmp_path / "RGB/train/images/scene/pair.png"
    duplicate = source.with_suffix(".jpg")
    assert cv2.imwrite(str(duplicate), cv2.imread(str(source)))
    with pytest.raises(ValueError, match="Duplicate image key"):
        PairedDataset(_data_config(tmp_path), imgsz=32)


def test_data_yaml_relative_root_is_independent_of_working_directory(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    _write_pair(dataset_root)
    config = _data_config(dataset_root)
    config["path"] = "../dataset"
    config_path = tmp_path / "configs/sample.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert len(PairedDataset(config_path, imgsz=32)) == 1
