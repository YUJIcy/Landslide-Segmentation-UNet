# 项目结构与分工说明（承接文档 / Handoff）

> 用途：本文件记录项目的目录结构、代码模块划分、已确认的环境与数据事实。
> 新建对话时，先读本文件即可无缝接手，无需重新向用户询问。
> ⚠️ 本文件已根据当前代码（2026-08-25）校对。早期版本描述的 SMP 库 / HuggingFace 源 / `pylibs/` / `visualize.py` / `run.py` / BCE+Dice 损失均**与实际不符**，请以本版为准。

---

## 0. 项目一句话

基于 Landslide4Sense 公开数据集，用 **PyTorch 从零手写 U-Net** 做遥感影像滑坡语义分割（逐像素二分类：滑坡 / 非滑坡）。baseline = 手写的 milesial 风格 U-Net；对比模型（DeepLabV3+ / Attention U-Net / SegFormer）为**规划中、尚未实现**。评估用 IoU / F1，产出训练代码 + 实验报告。**不依赖 segmentation-models-pytorch（SMP）**。

---

## 1. 已确认的环境事实（不要再问用户）

| 项 | 值 |
|---|---|
| 工作目录 | `E:\PreviewStudy\LandslideDetection` |
| Python 环境 | conda env `torch_gpu`，解释器 `D:\anaconda3\envs\torch_gpu\python.exe` |
| 环境内已装 | torch 2.5.1+cu121、numpy 2.4.6、matplotlib 3.11.1、PyYAML 6.0.3、**h5py 3.16.0**、**tqdm 4.70.0** |
| 第三方库方式 | h5py / tqdm 已用 `pip` 直接装进 env；早期的 `pylibs/` 离线副本已删除，代码里的 `sys.path.insert(pylibs)` 已移除 |
| GPU | NVIDIA RTX 3050 Laptop，**显存仅 4GB**（batch_size 固定为 8） |
| CUDA | 12.1，`torch.cuda.is_available()` = True |
| 用户分工 | **用户自己写全部代码**，AI 只负责逐模块讲解怎么写 |

---

## 2. 已确认的数据集事实（Landslide4Sense，Zenodo 版）

- 来源：Zenodo record **10463239**（`https://zenodo.org/records/10463239`）
  - 直链格式：`https://zenodo.org/api/records/10463239/files/TrainData.zip/content`（注意：必须是 `api/records/10463239/files/<名>/content`；版本记录号是 `10463239`，不是概念记录 `10463238`；路径里要带 `api/` 和 `/content`）
  - **关键：只有 `TrainData.zip` 带标签（mask）；`TestData.zip` 仅含 img、无 mask（竞赛盲评集）**。`ValidData.zip` 本项目未使用（也无标签）。
- 影像：`image_N.h5`，内部 key=`img`，shape `(128,128,14)`，dtype float64，14 通道 = Sentinel-2 的 B1~B12 + B13 坡度 + B14 高程
- 标签：`mask_N.h5`，内部 key=`mask`，shape `(128,128)`，dtype uint8，二值 0/1（1=滑坡）
- 下载与划分：由 `scripts/download_data.py` 完成
  - 下载 `TrainData.zip`（3799 个，带标签）+ `TestData.zip`（800 个，无标签）
  - 从 TrainData 切出验证集（固定 `random.seed(42)` 可复现）：
    - **train = 3554**（带标签）
    - **val = 245**（带标签，从 TrainData 切出）
    - **test = 800**（无标签，来自 TestData，供 `predict.py` 盲预测）
- 本地样例（试跑用，已存在）：`data/sample/image_1.h5` + `data/sample/mask_1.h5`

---

## 3. 目录结构（实际）

```
LandslideDetection/
├── README.md                  # 项目说明（目前近乎空白，待补）
├── requirements.txt           # 依赖清单（torch 的 +cu121 需官方源装）
├── configs/
│   └── default.yaml           # 所有超参数集中配置（模型名、batch、lr、epochs、mean/std…）
│
├── data/
│   ├── raw/                   # 真实数据（脚本下载生成，大文件）
│   │   ├── train/{img,mask}/  # 3554 个带标签 patch
│   │   ├── val/{img,mask}/    # 245 个带标签 patch（从 TrainData 切出）
│   │   ├── test/img/          # 800 个无标签 patch（供 predict.py 盲预测，无 mask 目录）
│   │   └── _downloads/        # 下载的 zip 临时目录（解压切分后可直接删）
│   └── sample/                # 2 个样例 h5（写代码试跑用）
│
├── dataset/                   # 数据读取模块（与官方仓库 dataset/ 对齐思路）
│   ├── __init__.py
│   └── landslide_dataset.py   # ① 数据：LandslideDataset 读 h5、transpose(HWC→CHW)、标准化
│
├── models/                    # 模型模块包
│   ├── __init__.py            # build_model() 统一入口（现仅支持 unet）
│   └── unet/                  # U-Net 子包（参考 milesial/Pytorch-UNet，全部手写）
│       ├── __init__.py
│       ├── unet_model.py      # UNet 组装（14→64→…→2 通道）
│       └── unet_parts.py      # DoubleConv / Down / Up / OutConv
│
├── train.py                   # ② 训练主循环（含验证、早停、存最优，自存 loss.png）
├── evaluate.py                # ③ 加载最优权重在验证集评估 + 存 3 栏对比图
├── predict.py                 # ④ 对 test 集前 N 张盲预测、存掩膜 + 概览 PNG
├── test_unet.py               # 冒烟测试：确认模型前向输出 (1,2,128,128)
│
├── scripts/                   # 工具脚本
│   ├── download_data.py       # 下载 + 切分数据到 data/raw
│   ├── losses.py              # build_loss()（现仅 cross_entropy）
│   ├── metrics.py             # compute_metrics()（IoU / F1 / Precision / Recall）
│   └── test_h5.py             # 打印 h5 内部 keys / shape
│
├── runs/                      # 实验输出（运行时自动生成）
│   └── exp_01_unet/
│       ├── checkpoints/best_model.pt   # 验证集 landslide_IoU 最优的权重
│       ├── logs/train_log.csv + loss.png
│       ├── preds/             # evaluate.py 出的 3 栏对比图（Input/GT/Prediction）
│       └── predictions/       # predict.py 出的掩膜 + 概览图
│
└── docs/                      # 文档
    ├── 00_project_structure.md
    ├── 01_research_framework.md
    ├── 02_theory_knowledge.md
    └── 03_directory.md
```

---

## 4. 代码分几个部分（每个文件职责）

| 文件 | 职责 | 输入 → 输出 |
|---|---|---|
| `dataset/landslide_dataset.py` | `LandslideDataset` 类：读 h5、HWC→CHW、逐通道标准化；`mask_dir=None` 时只返回影像（盲预测用） | `(img_dir, mask_dir)` → `(影像(C,H,W), 标签(H,W))` 或仅影像 |
| `models/unet/*.py` | 手写 U-Net（DoubleConv/Down/Up/OutConv），bilinear 上采样 | `(B,14,H,W)` → `(B,2,H,W)` |
| `models/__init__.py` | `build_model(cfg)` → `UNet(n_channels=14, n_classes=2, bilinear=True)` | cfg → model |
| `scripts/losses.py` | `build_loss(cfg)` → `nn.CrossEntropyLoss()` | — |
| `scripts/metrics.py` | `compute_metrics(pred, target, num_classes=2)` → 返回 `landslide_iou` / `landslide_f1` / `mean_iou` 等 | (预测 0/1, 标签 0/1) → 字典 |
| `train.py` | 训练主循环 + 验证 + 早停 + 存最优；自己存 `loss.png` | data + model + loss → 权重 / 日志 |
| `evaluate.py` | 加载 `best_model.pt` 在 val 算指标、存 3 栏对比图 | checkpoint + val → 指标表 + png |
| `predict.py` | 读 `data/raw/test/img` 前 `NUM_PREDICT` 张（无标签），存掩膜 + 概览 PNG | checkpoint + test → png |
| `scripts/download_data.py` | Zenodo 下载 + 切分成 train/val/test | 无 → 数据文件 |
| `test_unet.py` | 模型前向尺寸冒烟测试 | — |
| `scripts/test_h5.py` | 查看 h5 内部结构 | — |

### 数据流

```
scripts/download_data.py → data/raw/{train,val,test}/*.h5
        ↓
dataset/landslide_dataset.py 读 h5 → (影像, 标签) 张量
        ↓
models/UNet 前向 → (B,2,H,W) logits
        ↓
scripts/losses.py  CrossEntropyLoss(logits, 标签) → loss
        ↓
train.py  反向更新 → best_model.pt（按 val landslide_IoU 早停）
        ↓
evaluate.py / predict.py 加载 best_model.pt → 预测
        ↓
scripts/metrics.py 算 IoU/F1 → runs/exp_01_unet/ 出图与 csv
```

### 建议编写顺序（历史，已大体完成）

1. `configs/default.yaml`
2. `dataset/landslide_dataset.py`
3. `models/unet/unet_parts.py` + `unet_model.py` + `models/__init__.py`（`build_model`）
4. `scripts/losses.py` → `scripts/metrics.py`
5. `train.py`
6. `evaluate.py` → `predict.py`
7. `scripts/download_data.py`（最后准备真实数据）

---

## 5. 关键实现约定（与早期文档不同，重点！）

- **模型输出 2 通道 + CrossEntropyLoss**：`out_channels=2`，标签是 0/1 **类别索引**（非 one-hot）；推理用 `torch.max(Pred, dim=1)` 取类别，**不用 Sigmoid + 0.5 阈值**。
- **无 SMP、无预训练 backbone**：U-Net 全部手写（milesial/Pytorch-UNet 风格），从随机权重训练；`build_model()` 现只支持 `unet`，加模型时在此扩展。
- **损失 / 指标模块在 `scripts/` 下**：`train.py` / `evaluate.py` 用 `from scripts.losses import build_loss` / `from scripts.metrics import compute_metrics` 导入（**不在根目录**）。
- **`losses.py` 目前只实现 `cross_entropy`**（`build_loss`）；Dice / Focal / 组合损失尚未实现，留作后续消融实验。
- **`predict.py` 不是生成竞赛 h5**：它读 `data/raw/test/img` 前 `NUM_PREDICT` 张（无标签），逐张存预测图（PNG，非 h5）。⚠️ 当前代码第 65 行把文件名写成了 `pred_mask_name_{k:03d}.png`（多了字面量 `name`，应改为 `pred_mask_{k:03d}.png`），属小 bug，跑之前建议顺手修。
- **没有 `visualize.py`**：训练曲线由 `train.py` 自己 `fig.savefig(.../loss.png)` 生成；没有 `scripts/run.py` 总入口（分别跑 `train.py` / `evaluate.py` / `predict.py`）。
- **没有 `pylibs/` / `setup_deps.py`**：h5py、tqdm 已装进 conda env，import 直接用。

---

## 6. 关键约束与提醒（与代码一致）

- GPU 仅 4GB 显存：`batch_size=8`（yaml 里固定），别调大。
- 主指标用 **IoU 和 F1**，不用 Accuracy（滑坡像素占比低，全预测 0 也有高准确率）。
- 训练前先跑 `python scripts/download_data.py` 准备数据；`train.py` 读 `data.train.train_dir` / `val_dir`。
- `evaluate.py` 默认评 `val_dir`（有标签，能算 IoU）；`test` 集无标签、无真值，只能跑 `predict.py` 出图，无法算指标。
- 数据集已切成 128×128，无需自己切片；波段顺序固定 `B1..B12, 坡度, 高程`，真彩色预览取索引 `[3,2,1]`（B4/B3/B2）。
