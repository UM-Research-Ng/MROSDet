# MROSDet

**MROSDet: A Modality-Robust Optical-Sonar Fusion Paradigm for Underwater Perception under Sensor Degradation**

Mingxin Liu, Yujie Wu, Ruixin Li, Ziliang Ji, and Cong Lin (corresponding author).

Paper link: **to be added after acceptance**. [中文说明](README.zh-CN.md)

MROSDet addresses underwater object detection from paired optical and sonar images under sensor degradation. It estimates the reliability of each modality to guide feature fusion, then produces detections in the coordinate system of each sensor.

This repository contains the model, training and evaluation scripts, a trained checkpoint, and 100 optical-sonar image pairs for running the examples.

## Method

Two matched backbones extract optical and sonar features at multiple scales. The subsequent modules estimate reliability, exchange complementary information, and retain a detection path for each modality:

1. **Modality Reliability Estimator (MRE)** predicts relative modality weights at each scale and condition-classification logits. An uncertainty score is computed from the modality weights.
2. **Reliability-Guided Cross-modal Fusion (RGCF)** uses these estimates to control bidirectional residual feature fusion.
3. **Synergistic Dual-Pyramid Neck (SDPN)** processes the fused features into **two groups of three-scale detection features**.
4. **Bimodal Collaborative Detection Head (BCDHead)** uses the two feature groups to produce optical and sonar detections with separate labels and image coordinates.

The implementation resizes each input image to a square and restores predicted boxes to the corresponding original image size.

## Installation

Python 3.10 is the reference environment. Use a separate virtual environment:

```bash
git clone https://github.com/UM-Research-Ng/MROSDet.git
cd MROSDet
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For the reference CUDA 13.0 environment, install PyTorch first:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install -e .
```

For CPU use:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
```

The CUDA build requires a compatible NVIDIA driver. CPU and single-GPU execution are supported. `requirements.txt` pins the reference non-PyTorch dependencies; `pyproject.toml` declares the package's dependency ranges.

## Model checkpoint

Download the trained checkpoint from [release v0.1.0](https://github.com/UM-Research-Ng/MROSDet/releases/tag/v0.1.0):

```bash
mkdir -p weights
curl -fL https://github.com/UM-Research-Ng/MROSDet/releases/download/v0.1.0/best.pt -o weights/best.pt
curl -fL https://github.com/UM-Research-Ng/MROSDet/releases/download/v0.1.0/SHA256SUMS -o weights/SHA256SUMS
(cd weights && sha256sum -c SHA256SUMS)
```

On macOS, use `shasum -a 256 -c SHA256SUMS` in place of `sha256sum -c SHA256SUMS`.

The checkpoint supports evaluation, prediction, and fine-tuning for the nine UMOD categories. It stores parameters and model configuration in a weights-only format. Optimizer state is not included, so fine-tuning starts a new run.

## Example data

The repository includes 100 optical-sonar pairs from UMOD: **70 train, 20 validation, and 10 test**. Each pair retains its original split. The sample covers the nine categories in both modalities and includes separate optical and sonar labels. See the [dataset README](data/umod_sample/README.md), manifest, and class-count summary for provenance and checksums.

```text
data/umod_sample/
  RGB/
    train/{images,labels}/
    val/{images,labels}/
    test/{images,labels}/
  Sonar/
    train/{images,labels}/
    val/{images,labels}/
    test/{images,labels}/
  manifest.json
  summary.json
```

Labels contain five fields: `class_id center_x center_y width height`. Class IDs start at zero, and box coordinates are normalized to the corresponding modality's original image. The two modalities have separate annotations. An empty label file denotes a background image; a missing label is an error. Images are paired by their relative path without the extension, so `scene/a.jpg` can pair with `scene/a.png`.

The subset is provided for running the examples. The complete RUMOD benchmark, full paper test set, original initialization resources, and degradation-generation tools are outside this release; training on these 100 pairs does not reproduce the paper experiments.

## Train

The default is training from scratch:

```bash
python train.py --data configs/umod_sample.yaml --model configs/mrosdet.yaml \
  --epochs 20 --batch 32 --imgsz 640 --device 0 --workers 0 \
  --output outputs/train
```

To initialize a new run from the released model:

```bash
python train.py --data configs/umod_sample.yaml --model configs/mrosdet.yaml \
  --weights weights/best.pt --epochs 20 --batch 32 --imgsz 640 --device 0 \
  --output outputs/finetune
```

With `--weights`, the architecture and class vocabulary come from the checkpoint, and the dataset must have matching class IDs and names. `--model` selects the architecture for training from scratch.

A one-epoch example uses:

```bash
python train.py --data configs/umod_sample.yaml --model configs/mrosdet.yaml \
  --epochs 1 --batch 2 --imgsz 320 --device 0 --workers 0 \
  --output outputs/smoke
```

Use `--device cpu` for CPU execution and reduce `--batch` when GPU memory is limited.

The defaults are 20 epochs, batch 32, image size 640, zero data-loader workers, and seed 0. Training uses AdamW with initial learning rate 0.000769, weight decay 0.0005, a linear learning-rate schedule, and three warmup epochs. CUDA training enables AMP; gradient accumulation and EMA are also used.

**Configuration note:** the current training entry point uses AdamW with an initial learning rate of 0.000769, whereas the manuscript describes SGD with an initial learning rate of 0.01. These are different optimizer settings. The commands above use the current code defaults.

## Evaluate

```bash
python val.py --weights weights/best.pt --data configs/umod_sample.yaml \
  --split test --imgsz 640 --device 0 --output outputs/val
```

Evaluation reports precision, recall, mAP50, and mAP50-95 separately for RGB and sonar on the selected split.

## Predict

```bash
python predict.py --weights weights/best.pt \
  --rgb data/umod_sample/RGB/test/images \
  --sonar data/umod_sample/Sonar/test/images \
  --imgsz 640 --device 0 --output outputs/predict
```

Prediction writes annotated optical/sonar images and machine-readable JSON detections. Coordinates are restored independently to each modality's original image size, and each annotated image retains its input format. Use `--conf` for the confidence threshold (default 0.25), `--iou` for the NMS IoU threshold (default 0.7), `--max-det` for the maximum detections per image (default 300), and `--limit` for the maximum number of input pairs (default 0, meaning all pairs).

All three entry points provide `--help`. Relative command-line paths resolve from the repository root; a dataset YAML's `path` resolves from that YAML's directory, and modality paths resolve from the dataset root. Absolute paths are also supported. For example, `python /path/to/MROSDet/val.py --weights weights/best.pt` works from another shell directory.

| Entry point | Files under the selected output directory |
| --- | --- |
| `train.py` | `weights/best.pt`, `weights/last.pt`, `results.csv`, `args.yaml`, and `summary.json` |
| `val.py` | `metrics.json`, with aggregate and per-class RGB/Sonar metrics |
| `predict.py` | `rgb/`, `sonar/`, and `predictions.json`; each detection contains pixel `xyxy` coordinates, confidence, class ID, and class name |

If an output directory already exists, a numbered sibling such as `outputs/train2` is created. Training selects `best.pt` by the mean RGB/Sonar mAP50-95.

## Use your own paired dataset

Arrange the two modalities and their labels as above, copy `configs/umod_sample.yaml`, and update its dataset root and category names. Use consistent class IDs across modalities and modality-specific boxes. Keep related captures in the same split to prevent leakage. For a different class set, update the model configuration and train from scratch by omitting `--weights`.

The [dataset README](data/umod_sample/README.md) describes the label checks and example-subset selection procedure.

## Tests

Run the public unit tests from the repository root:

```bash
python -m pip install -e '.[test]' -c requirements.txt
python -m pytest -q
```

The tests cover checkpoint loading, paired data, losses, detection metrics, and the command-line utilities using synthetic inputs and temporary directories.

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite this software. The paper link will be added after acceptance.

Code and model checkpoint: [GNU AGPL-3.0](LICENSE). Example images and annotations: [CC BY 4.0](data/umod_sample/LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for component attribution and adaptation details.
