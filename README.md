# 基于深度学习的滑坡识别
# Landslide Detection Based on Deep Learning

## 项目简介
## Project Overview

本项目为深度学习入门学习项目。以公开基准数据集 **Landslide4Sense（L4S）** 为对象，用**手写实现的 U-Net** 完成"从遥感影像中自动识别滑坡"的语义分割任务，构建可复现的"数据—模型—训练—评估—预测"全流程。  
- 2026年8月17日-2026年8月21日

This is a beginner-level deep learning project (for a GIS / remote-sensing background, starting from scratch). Using the public **Landslide4Sense (L4S)** benchmark dataset, a **hand-written U-Net** performs landslide semantic segmentation from remote-sensing imagery, building a reproducible data → model → train → evaluate → predict pipeline.
- 2026-08-17 to 2026-08-21

基本信息：  
- 数据集 / Dataset：Landslide4Sense（Zenodo 记录号 10463239）
- 输入 / Input：14 通道（12 个 Sentinel-2 光谱波段 + 坡度 slope + 高程 elevation）的 128×128 图像块
- 标签 / Label：逐像素二值掩膜（1 = 滑坡，0 = 非滑坡）
- 基线结果 / Baseline：验证集滑坡类 IoU ≈ 0.569、F1 ≈ 0.725（最优模型第 18 epoch，早停于第 26 epoch）
- 实验报告 / Report：`基于深度学习的滑坡识别.docx`（本报告配套文档）

---

## 项目结构
## Project Structure

```
LandslideDetection/
│
├── README.md                          # 项目说明（本文件）/ Project documentation
├── requirements.txt                   # 依赖清单 / Dependency list (torch +cu121 需官方源)
├── configs/
│   └── default.yaml                   # 所有超参数集中配置 / Central hyper-parameter config
│                                     #   （模型名、batch、lr、epochs、mean/std、早停 patience…）
│
├── data/                              # 数据目录（大文件，运行时生成）/ Data (large, generated at runtime)
│   ├── raw/
│   │   ├── train/  img/ mask/         # 训练集 3554（带标签）/ Train (labeled)
│   │   ├── val/    img/ mask/         # 验证集 245（带标签，从 TrainData 切出）/ Val (labeled)
│   │   ├── test/   img/              # 测试集 800（无标签，供 predict.py 预测）/ Test (unlabeled)
│   │   └── _downloads/                # 下载 zip 临时目录 / Temp download dir
│   └── sample/                        # 2 个样例 h5（冒烟测试用）/ 2 sample h5 for smoke test
│
├── dataset/                           # 数据读取模块 / Data loading module
│   ├── __init__.py
│   └── landslide_dataset.py           # LandslideDataset：读 h5、(H,W,C)→(C,H,W)、Z-score 标准化
│
├── models/                            # 模型包 / Model package
│   ├── __init__.py                    # build_model(cfg) 统一入口（现仅 unet）/ Unified model entry
│   └── unet/                          # U-Net 子包（参考 milesial/Pytorch-UNet，全部手写）
│       ├── __init__.py
│       ├── unet_model.py              # UNet 组装（14 → 64 → … → 1024 → … → 2 通道）
│       └── unet_parts.py              # DoubleConv / Down / Up / OutConv 等基础模块
│
├── train.py                           # 训练主循环：前向/反向、验证、早停、存最优权重、自存 loss.png
│                                     #   Training loop: fwd/bwd, val, early-stop, save best, loss.png
├── evaluate.py                        # 加载最优权重在验证集评估 + 存"输入/真值/预测"三栏对比图
│                                     #   Eval on val set + save 3-column comparison figures
├── predict.py                         # 对 test 集前 N 张盲预测，存掩膜 + 概览 PNG
│                                     #   Blind prediction on test set, save masks + overview
├── test_unet.py                       # 冒烟测试：确认模型前向输出形状 (1, 2, 128, 128)
│                                     #   Smoke test: assert forward output shape
│
├── scripts/                           # 工具脚本 / Utility scripts
│   ├── download_data.py               # 下载 + 切分数据到 data/raw / Download + split into data/raw
│   ├── losses.py                      # build_loss(cfg)：现仅 cross_entropy / Loss builder
│   ├── metrics.py                     # compute_metrics()：IoU / F1 / Precision / Recall / MeanIoU
│   └── test_h5.py                     # 打印查看 h5 内部 keys / shape / Inspect h5 keys & shapes
│
├── runs/                              # 实验输出（运行时自动生成）/ Run outputs (auto-generated)
│   └── exp_01_unet/
│       ├── checkpoints/best_model.pt # 验证集 landslide_IoU 最优权重 / Best checkpoint
│       ├── logs/train_log.csv         # 每轮 train_loss/val_loss/IoU/F1 记录 / Per-epoch metrics
│       ├── logs/loss.png              # 损失曲线 / Loss curve
│       ├── preds/                     # evaluate.py 出的三栏对比图 / 3-column comparison figs
│       └── predictions/               # predict.py 出的掩膜 + 概览图 / Masks + overview
│
└── docs/                              # 文档 / Docs
    ├── 00_project_structure.md
    ├── 01_research_framework.md       # 研究框架与报告章节结构 / Framework & report structure
    ├── 02_theory_knowledge.md         # CNN / U-Net 等原理笔记 / Theory notes
    └── 03_directory.md                # 目录与文件说明 / Directory notes
```

---

## 环境配置
## Environment Setup

### 依赖安装
### Dependency Installation

```bash
# 1) 创建并激活 conda 环境（推荐）/ Create & activate conda env (recommended)
conda create -n torch_gpu python=3.10
conda activate torch_gpu

# 2) 安装 PyTorch 2.5.1 + CUDA 12.1（必须走官方源）/ PyTorch must use the official index
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 3) 安装其余依赖 / Install remaining dependencies
pip install -r requirements.txt
```

> 注：`h5py`、`tqdm` 等若 pip 安装慢，可单独用 pip 安装。
> Note: install `h5py` / `tqdm` via pip if needed.

### 核心依赖
### Core Dependencies

- **Python 3.10+**
- **torch 2.5.1+cu121**：深度学习框架（GPU 训练）/ Deep learning framework
- **h5py**：读取 Landslide4Sense 的 `.h5` 数据 / Read `.h5` dataset files
- **numpy**：数值运算 / Numerical ops
- **PyYAML**：读取 `configs/default.yaml` / Parse config
- **matplotlib**：损失曲线与预测对比图可视化 / Plot loss curves & predictions

---

## 实验设计
## Experimental Design

### 数据集与划分
### Dataset & Split

| 子集 / Split | 样本数 / Count | 说明 / Description |
|------|------|------|
| 训练集 train | 3554 | 带标签，用于训练 / Labeled, for training |
| 验证集 val | 245 | 带标签，从 TrainData 切出，用于早停与评估 / Labeled, for early-stop & eval |
| 测试集 test | 800 | 来自 TestData，无标签，仅用于盲预测 / Unlabeled, blind prediction only |

### 模型与训练配置
### Model & Training Config

| 项目 / Item | 取值 / Value | 说明 / Description |
|------|------|------|
| 模型 / Model | 手写 U-Net（`models/unet`） | 编码器—解码器 + 跳跃连接，输出 2 通道 / Encoder-decoder + skip, 2-channel out |
| 损失 / Loss | CrossEntropyLoss | `scripts/losses.py` 的 `build_loss(cfg)` |
| 优化器 / Optimizer | Adam | lr = 1e-3，weight_decay = 5e-4 |
| 批大小 / Batch | 8 | 受 4 GB 显存（RTX 3050 Laptop）约束 / 4 GB VRAM limited |
| 轮数 / Epochs | 30（上限）| 实际由早停提前结束 / Early-stop ends earlier |
| 早停 / Early stop | patience = 8 | 监控验证集 landslide_IoU，无提升则停 / Monitor val landslide_IoU |
| 随机种子 / Seed | 42 | 保证可复现 / Reproducibility |
| 主指标 / Metric | landslide_IoU / landslide_F1 | 滑坡类交并比与 F1；另报 mean IoU |

所有超参数集中在 `configs/default.yaml`，修改配置即可调整实验，**无需改动主流程代码**。

All hyper-parameters live in `configs/default.yaml`; tune experiments by editing the config—no changes to the main pipeline code are required.

---

## 模型运行
## Model Execution

### 1. 准备数据
### 1. Prepare Data

```bash
# 下载 Landslide4Sense 并切分为 train/val/test 到 data/raw
# Download L4S and split into train/val/test under data/raw
python scripts/download_data.py
```

### 2. 训练
### 2. Train

```bash
# 按 configs/default.yaml 训练 U-Net，自动早停并保存最优权重
# Train U-Net per configs/default.yaml; auto early-stop & save best
python train.py
```

### 3. 评估
### 3. Evaluate

```bash
# 在验证集评估最优模型，并生成"输入 / 真值 / 预测"三栏对比图
# Evaluate best model on val set; produce 3-column comparison figures
python evaluate.py
```

### 4. 盲预测
### 4. Predict

```bash
# 对测试集前 N 张无标签影像推理，输出掩膜与概览图
# Blind-predict first N test images; output masks & overview
python predict.py
```

### 5. 冒烟测试（可选）
### 5. Smoke Test (optional)

```bash
python test_unet.py      # 确认模型前向输出形状为 (1, 2, 128, 128)
python scripts/test_h5.py  # 查看某个 h5 的 keys / shape
```

---

## 输出结果
## Output Results

实验产物默认写入 `runs/exp_01_unet/`：

Run artifacts are written under `runs/exp_01_unet/`:

- **`checkpoints/best_model.pt`**：验证集滑坡类 IoU 最高的模型权重 / Best checkpoint by val landslide_IoU
- **`logs/train_log.csv`**：逐轮 `train_loss / val_loss / landslide_IoU / landslide_F1` 记录 / Per-epoch metrics
- **`logs/loss.png`**：训练 / 验证损失曲线 / Train & val loss curve
- **`preds/`**：验证集"输入影像 / 真值掩膜 / 模型预测"三栏对比图 / 3-column comparison
- **`predictions/`**：测试集盲预测掩膜与概览图 / Test masks & overview

最优结果（验证集）：**landslide_IoU ≈ 0.5689，landslide_F1 ≈ 0.7252**（第 18 epoch），并于第 26 epoch 触发早停。

Best val result: **landslide_IoU ≈ 0.5689, landslide_F1 ≈ 0.7252** (epoch 18), early-stopped at epoch 26.

---

## 扩展：添加 / 更换模型与调整参数
## Extension: Add / Replace Models & Tune Parameters

本项目代码已对"更换或添加模型""调整实验参数"做了适配，扩展**无需改动主流程代码**（`train.py` / `evaluate.py` / `predict.py`）。

The code is adapted so that adding/replacing models or tuning parameters requires **no change to the main pipeline** (`train.py` / `evaluate.py` / `predict.py`).

### 如何添加 / 更换模型
### How to add / replace a model

1. 在 `models/` 下新建模型子包，如 `models/deeplabv3/__init__.py`，定义模型类（如 `DeepLabV3`）；
   Create a sub-package under `models/`, e.g. `models/deeplabv3/__init__.py`, defining the model class.
2. 在 `models/__init__.py` 的 `build_model(cfg)` 中增加分支：
   Add a branch in `build_model(cfg)` inside `models/__init__.py`:

   ```python
   if name == "deeplabv3":
       from .deeplabv3 import DeepLabV3
       return DeepLabV3(in_channels=cfg["data"]["in_channels"],
                        out_channels=cfg["model"]["out_channels"])
   ```
3. 将 `configs/default.yaml` 的 `model.name` 改为新模型名（如 `deeplabv3`），按需补充该模型的专属参数；
   Set `model.name` in `configs/default.yaml` to the new name; add model-specific params if needed.
4. 重新运行 `python train.py` 即可。Attention U-Net、SegFormer 等对比模型同理。
   Re-run `python train.py`. Same pattern for Attention U-Net, SegFormer, etc.

### 如何更改实验参数
### How to tune parameters

所有超参数集中在 `configs/default.yaml`，常见调整对照：

All hyper-parameters are in `configs/default.yaml`; common adjustments:

| 想调整的内容 / What to change | 修改位置 / Where | 示例 / Example |
|------|------|------|
| 批大小 / 轮数 / 学习率 / 权重衰减 / 早停耐心 | `train` 块 | `batch_size: 8`、`lr: 0.001`、`patience: 8` |
| 优化器 / 损失函数 | `train` 块 | `optimizer: adam`、`loss: cross_entropy` |
| 输入通道数 / 图像尺寸 / 标准化均值方差 | `data` 块 | `in_channels: 14` |
| 模型名称与输出通道 | `model` 块 | `name: unet`、`out_channels: 2` |

如需更换损失（如 Dice / Focal），在 `scripts/losses.py` 的 `build_loss(cfg)` 中增加对应分支并返回损失对象，再把 `train.loss` 改为新名称即可。

To switch loss (e.g. Dice / Focal), add a branch in `build_loss(cfg)` (`scripts/losses.py`) and set `train.loss` to the new name.

---

## 注意事项
## Notes

1. **数据准备**：运行训练前需先执行 `scripts/download_data.py`，确保 `data/raw/{train,val,test}` 已就绪。
   **Data Prep**: run `scripts/download_data.py` before training so `data/raw/{train,val,test}` exist.
2. **显存约束**：本项目在 4 GB 显存（RTX 3050 Laptop）下以 `batch_size=8` 跑通；显存不足时请调小 batch 或图像尺寸。
   **VRAM**: validated at `batch_size=8` on 4 GB VRAM; lower batch / image size if OOM.
3. **标准化参数**：14 通道的 `mean / std` 为官方在训练集统计的固定值，已在 `configs/default.yaml` 中给出，请勿随意改动以免破坏可复现性。
   **Normalization**: per-channel `mean/std` are fixed official stats in `configs/default.yaml`; keep them for reproducibility.
4. **类别不平衡**：滑坡像素占比极低，主指标用 IoU / F1 而非 Accuracy；如需缓解可在 `CrossEntropyLoss(weight=...)` 或换用 Dice / Focal 损失。
   **Class imbalance**: landslide pixels are rare—use IoU/F1, not Accuracy; mitigate via class weights or Dice/Focal loss.

---

## 许可证
## License

本项目仅供学习使用。
This project is for academic and learning purposes only.
