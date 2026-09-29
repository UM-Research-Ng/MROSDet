# MROSDet

**MROSDet: A Modality-Robust Optical-Sonar Fusion Paradigm for Underwater Perception under Sensor Degradation**

Mingxin Liu, Yujie Wu, Ruixin Li, Ziliang Ji, and Cong Lin (corresponding author).

Paper link: **to be added after acceptance**. [中文说明](README.zh-CN.md)

MROSDet learns from paired optical (RGB) and sonar images and produces a separate set of object detections for each modality. It estimates modality reliability before exchanging features, so the fusion process can respond to sensor degradation. The optical and sonar views retain their own bounding-box annotations and image coordinates.

This repository provides the independent `mrosdet` package, training/evaluation/prediction entry points, one converted model checkpoint, and a 100-pair UMOD demonstration subset. It does **not** include the complete research dataset, all original initialization resources, historical experiment logs, or sensor-degradation generation tools. The small example dataset is intended to exercise the code; training on it is not a reproduction of the paper's reported experiments.

## Method

The model follows the manuscript's four main components after matched dual backbones:

1. **MRE** estimates multi-scale modality reliability, degradation logits, and uncertainty from the two streams.
2. **RGCF** uses reliability guidance for bidirectional residual feature fusion.
3. **SDPN** forms **two groups of three-scale detection features**, preserving modality-specific outputs after fusion.
4. **BCDHead** predicts optical and sonar detections through two detection branches.

The release preserves the research model's computation, modality-specific labels, stretch-resize preprocessing, and dual-modality loss behavior. Its importable package has no `ultralytics` runtime dependency. Components adapted from Ultralytics retain their required attribution and are described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

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

The CUDA build requires a compatible NVIDIA driver. The first release targets CPU and one CUDA GPU; multi-GPU training is not validated. `requirements.txt` pins the reference non-PyTorch dependencies; `pyproject.toml` declares the package's compatible dependency ranges. No cloud account, telemetry service, or automatic model download is required during training or inference.

## Model checkpoint

Download the converted checkpoint from [release v0.1.0](https://github.com/UM-Research-Ng/MROSDet/releases/tag/v0.1.0):

```bash
mkdir -p weights
curl -fL https://github.com/UM-Research-Ng/MROSDet/releases/download/v0.1.0/best.pt -o weights/best.pt
curl -fL https://github.com/UM-Research-Ng/MROSDet/releases/download/v0.1.0/SHA256SUMS -o weights/SHA256SUMS
(cd weights && sha256sum -c SHA256SUMS)
```

On macOS, use `shasum -a 256 -c SHA256SUMS` in place of `sha256sum -c SHA256SUMS`.

The checkpoint contains model parameters, configuration, class names, necessary non-parameter state, a format version, and source-checkpoint provenance. It is loaded with `torch.load(..., weights_only=True)`. It does not contain an executable model object, the original server path, optimizer state, or historical training logs. It initializes a new fine-tuning run and is not an optimizer-resume checkpoint. Other research or third-party pretrained weights are not distributed.

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

Each text label uses normalized YOLO detection rows: `class_id center_x center_y width height`. IDs are zero-based; coordinates refer to that modality's own original image. An existing empty label denotes a background image. Missing labels and missing image pairs are errors. RGB and sonar files are matched by their relative path without the extension, so `scene/a.jpg` can pair with `scene/a.png`; multiple files with the same stem in one modality are rejected. Copying an RGB label onto its sonar counterpart is not appropriate when their object locations differ.

This is a **UMOD example subset**, not the complete RUMOD benchmark or the complete paper test set. Image and annotation use is governed by [CC BY 4.0](data/umod_sample/LICENSE), separately from the source-code license.

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

When `--weights` is supplied, the architecture and class vocabulary come from the checkpoint. The dataset must have exactly matching class IDs and names. `--model` selects the architecture for training from scratch; it does not replace the architecture of a loaded checkpoint.

A small end-to-end execution check uses:

```bash
python train.py --data configs/umod_sample.yaml --model configs/mrosdet.yaml \
  --epochs 1 --batch 2 --imgsz 320 --device 0 --workers 0 \
  --output outputs/smoke
```

Use `--device cpu` for CPU execution. CPU training is substantially slower. Choose a smaller batch explicitly if your GPU cannot accommodate the default; the code does not silently choose a different experiment configuration.

Reference defaults are 20 epochs, batch 32, image size 640, zero data-loader workers, seed 0, AdamW with initial learning rate 0.000769 and weight decay 0.0005, linear learning-rate scheduling, and three warmup epochs. CUDA training uses AMP. Gradient accumulation, EMA, and the two modality losses follow the adapted research implementation. Training writes checkpoints and metrics under `--output`; the public model is never implicitly loaded when `--weights` is omitted.

## Evaluate

```bash
python val.py --weights weights/best.pt --data configs/umod_sample.yaml \
  --split test --imgsz 640 --device 0 --output outputs/val
```

Evaluation reports precision, recall, mAP50, and mAP50-95 **separately for RGB and sonar**. These describe performance on the selected dataset split. Scores on ten demonstration test pairs should not be presented as full-benchmark paper results.

## Predict

```bash
python predict.py --weights weights/best.pt \
  --rgb data/umod_sample/RGB/test/images \
  --sonar data/umod_sample/Sonar/test/images \
  --imgsz 640 --device 0 --output outputs/predict
```

Prediction writes annotated optical/sonar images and machine-readable JSON detections. Coordinates are restored independently to each modality's original image size, and each annotated image retains its input format. Use `--conf` for the confidence threshold (default 0.25), `--iou` for the NMS IoU threshold (default 0.7), `--max-det` for the maximum detections per image (default 300), and `--limit` for the maximum number of input pairs (default 0, meaning all pairs).

All three entry points provide `--help`. Relative command-line paths resolve from the repository root; a dataset YAML's `path` resolves from that YAML's directory, and its modality paths resolve from the dataset root. Absolute input/output paths are also supported. For example, `python /path/to/MROSDet/val.py --weights weights/best.pt` works from another shell directory. Invalid or missing weights, pairs, labels, and configurations produce errors instead of silently substituting a random model or another dataset.

| Entry point | Files under the selected output directory |
| --- | --- |
| `train.py` | `weights/best.pt`, `weights/last.pt`, `results.csv`, `args.yaml`, and `summary.json` |
| `val.py` | `metrics.json`, with aggregate and per-class RGB/Sonar metrics |
| `predict.py` | `rgb/`, `sonar/`, and `predictions.json`; each detection contains pixel `xyxy` coordinates, confidence, class ID, and class name |

If an output directory already exists, a numbered sibling such as `outputs/train2` is created to preserve the previous run. Training's `best.pt` is selected by the mean RGB/Sonar mAP50-95, not independently for each branch.

## Use your own paired dataset

Arrange the two modalities and their labels as above, copy `configs/umod_sample.yaml`, and update its dataset root and category names. Keep the same class-ID meaning in both modalities, use modality-specific boxes, and keep related captures in the same split to prevent leakage. When changing the number of classes, update the model configuration consistently and train from scratch by omitting `--weights`. The bundled checkpoint is for the original nine categories; the entry points reject a different class vocabulary instead of partially loading an incompatible detection head.

For maintainers with the complete authorized UMOD source, `tools/prepare_sample.py --source <UMOD-root> --destination <new-directory>` deterministically prepares the 70/20/10 demonstration split with seed 0. It requires a new destination, audits all source labels and image content, and does not overwrite the source. Missing, malformed, undecodable, or identical-cross-modality pairs are excluded and recorded in the audit summary; source files are never repaired. The selected pairs must still meet the counts and nine-class coverage requirements. Its output manifest contains relative paths and SHA-256 checksums. This helper does not download the full dataset.

## Verification

The release verification procedure checks standalone imports without the original research tree, strict checkpoint conversion and FP32 numerical agreement, all sample pairs and labels, one real training epoch, evaluation and prediction with both public and freshly trained weights, and failure handling for malformed inputs. See `tests/` for the executable checks. The new implementation is compared against the trusted research implementation with `atol=1e-5` and `rtol=1e-4`; test success is a code-portability check, not an assertion that the full paper experiment has been reproduced.

Run the public unit tests from the repository root:

```bash
python -m pip install -e '.[test]' -c requirements.txt
python -m pytest -q
```

The public tests use synthetic inputs and temporary directories and do not need the private research source tree. The old/new numerical parity check is a separate release-maintainer check against that source tree.

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite this software. The manuscript is titled *MROSDet: A Modality-Robust Optical-Sonar Fusion Paradigm for Underwater Perception under Sensor Degradation*. Its accepted-paper link and bibliographic details will be added after acceptance; no journal, DOI, or acceptance status is implied by this repository.

Source code and the released model checkpoint are provided under [GNU AGPL-3.0](LICENSE). The UMOD example images and annotations are provided under [CC BY 4.0](data/umod_sample/LICENSE). Copyright and attribution for adapted components remain in their source headers and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Removing a runtime dependency does not remove those components' provenance or license obligations.
