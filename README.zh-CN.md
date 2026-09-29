# MROSDet 中文说明

论文：*MROSDet: A Modality-Robust Optical-Sonar Fusion Paradigm for Underwater Perception under Sensor Degradation*

作者：Mingxin Liu、Yujie Wu、Ruixin Li、Ziliang Ji、Cong Lin（通讯作者）。论文链接：**接收后更新**。[English README](README.md)

MROSDet 面向配对的光学与声纳图像，在估计模态可靠性后进行特征交互，分别输出 RGB 和 Sonar 检测结果。两种模态使用各自的标注框和原图坐标，不能将光学标注直接当作声纳标注。

本次发布包括独立 `mrosdet` 包、`train.py`、`val.py`、`predict.py`、一份转换后的模型权重，以及 100 对 UMOD 示例数据。不包含完整研究数据、全部原始初始化资源、历史实验日志和传感器退化生成工具。示例数据用于检查代码流程，在这个小子集上训练不等于复现论文实验。

## 方法与代码来源

匹配的双路骨干提取特征后，**MRE** 估计多尺度模态可靠性、退化 logits 和不确定性；**RGCF** 完成可靠性引导的双向残差融合；**SDPN** 构建**两组三尺度检测特征**；**BCDHead** 分别进行光学和声纳检测。

独立版本保留研究模型的计算逻辑、双路独立标签、拉伸缩放预处理和损失行为，无需安装 `ultralytics`。其中源自 Ultralytics 的网络层、检测与训练组件继续保留版权和许可说明，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。运行依赖的独立不意味着代码没有上游来源。

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

CUDA 13.0 构建需要兼容的 NVIDIA 驱动。只使用 CPU 时，将 PyTorch 安装命令中的 `cu130` 改成 `cpu`，运行脚本时使用 `--device cpu`。首版支持 CPU 和单张 CUDA GPU；未验证多 GPU 训练。`requirements.txt` 固定参考环境的非 PyTorch 依赖版本，`pyproject.toml` 声明兼容范围；运行时不要求云账号，也不会自动下载模型。

## 下载权重

权重作为 [v0.1.0 Release](https://github.com/UM-Research-Ng/MROSDet/releases/tag/v0.1.0) 附件提供，不存入源码 Git 历史：

```bash
mkdir -p weights
curl -fL https://github.com/UM-Research-Ng/MROSDet/releases/download/v0.1.0/best.pt -o weights/best.pt
curl -fL https://github.com/UM-Research-Ng/MROSDet/releases/download/v0.1.0/SHA256SUMS -o weights/SHA256SUMS
(cd weights && sha256sum -c SHA256SUMS)
```

macOS 校验命令可用 `shasum -a 256 -c SHA256SUMS`。

转换后的权重仅保留参数、模型配置、类别、必要的非参数状态、格式版本和来源校验值，使用 `weights_only=True` 加载。文件不包含旧 Python 模型对象、服务器路径、优化器状态和历史训练记录。它适合预测、评估或新一轮微调，不用于恢复旧训练的优化器进度。其他研究权重和第三方预训练权重不在本次发布范围内。

## 示例数据

100 对样本沿用原始划分：**训练 70 对、验证 20 对、测试 10 对**，两种模态均覆盖 9 类。RGB 与 Sonar 根据不含扩展名的相对路径配对，例如 `scene/a.jpg` 可以与 `scene/a.png` 配对；同一模态中的同名多格式文件会报错。标签分别保留在对应模态目录下：

```text
data/umod_sample/
  RGB/{train,val,test}/{images,labels}/
  Sonar/{train,val,test}/{images,labels}/
  manifest.json
  summary.json
```

标签每行格式为 `class_id center_x center_y width height`，类别从 0 开始，坐标相对于该模态的原图归一化。存在但为空的标签文件表示无目标背景图；缺失标签与空标签不是一回事。文件配对、类别范围、图像解码和 SHA-256 校验信息见[数据说明](data/umod_sample/README.md)。

该数据是 **UMOD 示例子集**，不应称为完整 RUMOD 或完整论文测试集。图像和标注使用 [CC BY 4.0](data/umod_sample/LICENSE) 许可。

## 训练

默认从头训练，不隐式读取其他权重：

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

提供 `--weights` 后，模型结构和类别来自该权重，数据集的类别 ID 与名称必须完全一致。`--model` 用于从头训练时选择结构，不会覆盖已加载权重的模型结构。

执行一轮小规模流程检查：

```bash
python train.py --data configs/umod_sample.yaml --model configs/mrosdet.yaml \
  --epochs 1 --batch 2 --imgsz 320 --device 0 --workers 0 \
  --output outputs/smoke
```

默认配置为 20 轮、batch 32、640 输入尺寸、workers 0、seed 0、AdamW、初始学习率 0.000769、weight decay 0.0005、线性学习率调度及 3 轮 warmup；CUDA 下启用 AMP，保留梯度累积、EMA 与双模态损失逻辑。训练权重和指标写入 `--output`。显存有限时需显式减小 batch；CPU 训练较慢。

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

三个入口均支持 `--help` 和绝对路径。命令行相对路径以仓库根目录为基准，数据 YAML 内的 `path` 以 YAML 所在目录为基准，模态路径以数据根目录为基准。因此，在其他目录执行 `python /path/to/MROSDet/val.py --weights weights/best.pt` 也能定位仓库内的权重。缺失权重、图像对或配置会明确报错。10 对示例测试图的结果不代表完整论文基准性能。

| 入口 | 输出目录中的文件 |
| --- | --- |
| `train.py` | `weights/best.pt`、`weights/last.pt`、`results.csv`、`args.yaml`、`summary.json` |
| `val.py` | `metrics.json`，包括 RGB/Sonar 总体与逐类指标 |
| `predict.py` | `rgb/`、`sonar/`、`predictions.json`；每个检测包含原图像素 `xyxy` 坐标、置信度、类别 ID 和名称 |

输出目录已存在时会新建带编号的同级目录，例如 `outputs/train2`，保留上次结果。`best.pt` 按 RGB/Sonar 的 mAP50-95 均值选择，不是分别保存两路各自最优模型。

## 接入自己的数据

参考示例组织目录，复制 `configs/umod_sample.yaml` 并修改根目录与类别名称。两路相同 ID 应对应相同语义类别，但标注框应独立标注；相关视频帧或同次采集应放在同一划分，避免信息泄漏。改变类别数时需同步调整模型配置，并省略 `--weights` 从头训练。公开权重针对原始 9 类；入口会拒绝不同类别定义，不会自动跳过不兼容检测头进行部分加载。

维护者可通过 `tools/prepare_sample.py --source <UMOD根目录> --destination <不存在的新目录>` 使用固定 seed 0 重新制作 70/20/10 示例。脚本检查全部源数据标签、图像解码与内容哈希；缺失、非法标签、无法解码或两模态内容完全相同的图像对会排除并记录，不会修复原数据。选出的样本仍必须满足数量与双模态 9 类覆盖要求。清单仅包含相对路径和校验值；脚本不会下载完整数据。

## 验证、引用与许可

发布前验收包括：脱离原研究目录的独立导入、新旧模型 FP32 数值对齐（`atol=1e-5、rtol=1e-4`）、100 对样本检查、真实单轮训练、公开/新训练权重的验证与预测，以及异常输入检查。测试代码位于 `tests/`。这些检查用于确认移植和调用成功，不等于完成全部论文复现实验。

在仓库根目录运行公开单元测试：

```bash
python -m pip install -e '.[test]' -c requirements.txt
python -m pytest -q
```

公开测试使用合成输入与临时目录，不需要私有研究源码。新旧数值对齐由发布维护者使用原研究源码单独完成。

软件引用见 [CITATION.cff](CITATION.cff)。论文链接和正式书目信息将在接收后补充，当前不声明期刊、DOI 或接收状态。

代码与公开模型权重使用 [AGPL-3.0](LICENSE)；示例图像和标注使用 [CC BY 4.0](data/umod_sample/LICENSE)。上游版权、许可证和修改说明见源文件及 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
