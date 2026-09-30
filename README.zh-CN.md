# MROSDet 中文说明

论文：*MROSDet: A Modality-Robust Optical-Sonar Fusion Paradigm for Underwater Perception under Sensor Degradation*

作者：Mingxin Liu、Yujie Wu、Ruixin Li、Ziliang Ji、Cong Lin（通讯作者）。论文链接：**接收后更新**。[English README](README.md)

MROSDet 面向传感器退化条件下的水下光学—声纳目标检测，通过估计两种模态的可靠性来引导特征融合，并在各传感器的图像坐标系中分别输出检测结果。

本仓库提供模型、训练与评估脚本、一份训练权重，以及 100 对用于运行示例的光学—声纳图像。

## 方法

匹配的双路骨干首先提取光学与声纳的多尺度特征，随后通过四个模块估计可靠性、交互互补信息，并保留各自的检测分支：

1. **Modality Reliability Estimator（MRE，模态可靠性估计器）**：预测各尺度的相对模态权重和条件分类输出，并由模态权重计算不确定性分数。
2. **Reliability-Guided Cross-modal Fusion（RGCF，可靠性引导的跨模态融合）**：根据可靠性估计进行双向残差特征融合。
3. **Synergistic Dual-Pyramid Neck（SDPN，协同双金字塔颈部网络）**：处理融合特征，构建**两组三尺度检测特征**。
4. **Bimodal Collaborative Detection Head（BCDHead，双模态协同检测头）**：利用两组特征分别完成光学与声纳检测，使用各自的标注和图像坐标。

输入图像分别拉伸为正方形，预测框再按对应原图尺寸还原。

## 安装

参考环境为 Python 3.10、PyTorch 2.11.0、torchvision 0.26.0。建议使用单独虚拟环境：

```bash
git clone https://github.com/UM-Research-Ng/MROSDet.git
cd MROSDet
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install -e .
```

CUDA 13.0 构建需要兼容的 NVIDIA 驱动。只使用 CPU 时，将 PyTorch 安装命令中的 `cu130` 改成 `cpu`，运行脚本时使用 `--device cpu`。当前支持 CPU 和单张 CUDA GPU。`requirements.txt` 固定参考环境的非 PyTorch 依赖版本，`pyproject.toml` 声明依赖范围。

## 下载权重

训练权重可从 [v0.1.0 Release](https://github.com/UM-Research-Ng/MROSDet/releases/tag/v0.1.0) 下载：

```bash
mkdir -p weights
curl -fL https://github.com/UM-Research-Ng/MROSDet/releases/download/v0.1.0/best.pt -o weights/best.pt
curl -fL https://github.com/UM-Research-Ng/MROSDet/releases/download/v0.1.0/SHA256SUMS -o weights/SHA256SUMS
(cd weights && sha256sum -c SHA256SUMS)
```

macOS 校验命令可用 `shasum -a 256 -c SHA256SUMS`。

该权重对应 UMOD 的 9 个类别，可用于验证、预测和微调。文件以纯权重格式保存模型参数与配置，不包含优化器状态，因此微调会开始一轮新的训练。

## 示例数据

100 对样本沿用原始划分：**训练 70 对、验证 20 对、测试 10 对**，两种模态均覆盖 9 类。RGB 与 Sonar 根据不含扩展名的相对路径配对，例如 `scene/a.jpg` 可以与 `scene/a.png` 配对；同一模态中的同名多格式文件会报错。标签分别保留在对应模态目录下：

```text
data/umod_sample/
  RGB/{train,val,test}/{images,labels}/
  Sonar/{train,val,test}/{images,labels}/
  manifest.json
  summary.json
```

标签每行包含五列：`class_id center_x center_y width height`。类别从 0 开始，坐标相对于对应模态的原图归一化，两种模态各自标注。空标签文件表示背景图，缺失标签会报错。格式与校验信息见[数据说明](data/umod_sample/README.md)。

该子集用于运行示例。本次发布不包含完整 RUMOD 基准、完整论文测试集、原始初始化资源和退化生成工具；仅在这 100 对样本上训练不能复现论文实验。

## 训练

默认从头训练：

```bash
python train.py --data configs/umod_sample.yaml --model configs/mrosdet.yaml \
  --epochs 20 --batch 32 --imgsz 640 --device 0 --workers 0 \
  --output outputs/train
```

使用公开权重初始化新一轮微调：

```bash
python train.py --data configs/umod_sample.yaml --model configs/mrosdet.yaml \
  --weights weights/best.pt --epochs 20 --batch 32 --imgsz 640 --device 0 \
  --output outputs/finetune
```

提供 `--weights` 后，模型结构和类别来自该权重，数据集的类别 ID 与名称必须一致。`--model` 用于从头训练时选择结构。

运行一轮训练示例：

```bash
python train.py --data configs/umod_sample.yaml --model configs/mrosdet.yaml \
  --epochs 1 --batch 2 --imgsz 320 --device 0 --workers 0 \
  --output outputs/smoke
```

默认配置为 20 轮、batch 32、640 输入尺寸、workers 0 和 seed 0。训练采用 AdamW，初始学习率为 0.000769，weight decay 为 0.0005，使用线性学习率调度和 3 轮 warmup。CUDA 下启用 AMP，并使用梯度累积和 EMA。显存有限时可减小 `--batch`。

**配置说明：**当前训练入口默认使用 AdamW，初始学习率为 0.000769；稿件描述的是 SGD，初始学习率为 0.01。两者的优化器设置不同，上述命令使用当前代码的默认设置。

## 验证与预测

验证分别报告 RGB/Sonar 的 Precision、Recall、mAP50 和 mAP50-95：

```bash
python val.py --weights weights/best.pt --data configs/umod_sample.yaml \
  --split test --imgsz 640 --device 0 --output outputs/val
```

配对预测输出两路标注图像及 JSON 检测结果，框坐标分别恢复到对应原图尺寸：

```bash
python predict.py --weights weights/best.pt \
  --rgb data/umod_sample/RGB/test/images \
  --sonar data/umod_sample/Sonar/test/images \
  --imgsz 640 --device 0 --output outputs/predict
```

预测支持 `--conf`（置信度，默认 0.25）、`--iou`（NMS IoU，默认 0.7）、`--max-det`（每图最多检测数，默认 300）和 `--limit`（最多输入对数，默认 0 即全部）。输出图保留各自的输入格式。

三个入口均支持 `--help` 和绝对路径。命令行相对路径以仓库根目录为基准，数据 YAML 内的 `path` 以 YAML 所在目录为基准，模态路径以数据根目录为基准。在其他目录执行 `python /path/to/MROSDet/val.py --weights weights/best.pt` 也能定位仓库内的权重。

| 入口 | 输出目录中的文件 |
| --- | --- |
| `train.py` | `weights/best.pt`、`weights/last.pt`、`results.csv`、`args.yaml`、`summary.json` |
| `val.py` | `metrics.json`，包括 RGB/Sonar 总体与逐类指标 |
| `predict.py` | `rgb/`、`sonar/`、`predictions.json`；每个检测包含原图像素 `xyxy` 坐标、置信度、类别 ID 和名称 |

输出目录已存在时会新建带编号的同级目录，例如 `outputs/train2`。`best.pt` 按 RGB/Sonar 的 mAP50-95 均值选择。

## 接入自己的数据

参考示例组织目录，复制 `configs/umod_sample.yaml` 并修改根目录与类别名称。两路使用一致的类别 ID 和各自的标注框，相关视频帧或同次采集应放在同一划分，避免信息泄漏。使用新类别时，同步调整模型配置，并省略 `--weights` 从头训练。

标签检查和示例抽样方法见[数据说明](data/umod_sample/README.md)。

## 测试

在仓库根目录运行公开单元测试：

```bash
python -m pip install -e '.[test]' -c requirements.txt
python -m pytest -q
```

测试使用合成输入和临时目录，覆盖权重加载、成对数据、损失、检测指标和命令行工具。

## 引用与许可

软件引用见 [CITATION.cff](CITATION.cff)，论文链接将在接收后补充。

代码与模型权重使用 [AGPL-3.0](LICENSE)，示例图像和标注使用 [CC BY 4.0](data/umod_sample/LICENSE)。组件来源、版权和修改说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
