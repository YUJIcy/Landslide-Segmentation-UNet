# 基于深度学习的遥感影像滑坡识别 — 研究框架

> 数据集：Landslide4Sense（Sentinel-2 多光谱滑坡分割基准）
> 任务形式：语义分割（逐像素二分类：滑坡 / 非滑坡）
> 主框架：PyTorch（**不使用** segmentation-models-pytorch；U-Net 手写实现）
> ⚠️ 本文件已根据当前代码（2026-08-25）校对：早期版里的 SMP / HuggingFace / Sigmoid+阈值 / AdamW 等描述已修正。

---

## 0. 一句话概括本研究

以 Landslide4Sense 公开基准数据集为对象，用**从零手写的 U-Net** 做滑坡语义分割 baseline，系统评估其在遥感影像滑坡识别上的精度（IoU / F1），形成一份可复现、可答辩的研究报告。后续规划引入 DeepLabV3+ / Attention U-Net / SegFormer 做对比（**当前代码仅完成 U-Net**）。

---

## 1. 总体技术路线

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  ① 问题定义  │ → │  ② 数据层    │ → │  ③ 方法层    │ → │  ④ 实验层    │
│ 任务与目标  │   │ 数据+预处理  │   │ 模型+损失    │   │ 训练+评估    │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
                                                                  │
                    ┌─────────────────────────────────────────────┘
                    ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  ⑤ 结果层    │ → │  ⑥ 讨论结论  │ → │  ⑦ 报告撰写  │
│ 可视化+误差  │   │ 局限+展望    │   │ 章节结构     │
└──────────────┘   └──────────────┘   └──────────────┘
```

---

## 2. 六大模块详细说明

### 模块 ①｜问题定义与目标

- **研究问题**：如何用深度学习从遥感影像中自动、准确地识别滑坡范围？
- **任务形式**：语义分割（semantic segmentation）——对影像每个像素做二分类（1=滑坡，0=非滑坡）
- **为什么是分割而不是分类/检测**：滑坡范围不规则、边界模糊，只有像素级 mask 才能刻画真实范围
- **研究目标**：
  1. 复现并跑通 U-Net baseline（✅ 已完成）；
  2. 引入 2~3 个主流模型做对比（⏳ 规划中：DeepLabV3+ / Attention U-Net / SegFormer）；
  3. 通过消融实验分析关键因素影响（⏳ 规划中）；
  4. 得出精度/效率权衡的结论。

---

### 模块 ②｜数据层（研究区 + 数据集 + 预处理）

- **数据集**：Landslide4Sense（L4S）
  - 数据源：Sentinel-2 多光谱卫星影像，约 10 m 分辨率
  - 通道：12 个光谱波段（竞赛版另加坡度 + 高程 = 14 通道）
  - 样本：统一 128×128 像素 patch
  - 标签：逐像素二值 mask（0/1）
- **数据来源与划分（实际）**：
  - 来源：**Zenodo** record `10463239`（`scripts/download_data.py` 下载），**不是** HuggingFace。
  - 只有 `TrainData.zip` 带标签；`TestData.zip` 仅含 img、无标签。
  - `scripts/download_data.py` 把 TrainData(3799) 按 `seed=42` 切成：
    - **train = 3554**（带标签）
    - **val = 245**（带标签，从 TrainData 切出）
    - **test = 800**（无标签，来自 TestData，供盲预测）
  - 覆盖：4 个研究区（日本 Iburi、印度 Kodagu、尼泊尔 Gorkha、中国台湾）
- **预处理流水线**（代码里已做）：
  1. 数据下载与解析（`scripts/download_data.py`，读 h5）
  2. 切片：数据集已切好 128×128，**不用自己切**
  3. **归一化**：逐通道 Z-score 标准化 `(x - mean) / std`（mean/std 固定在 `configs/default.yaml`，用官方在训练集算好的值）
  4. 数据增强：**当前未实现**（`LandslideDataset` 无 augment 参数，训练集直接原样用）
  5. 划分：train / val（从 TrainData 切）/ test（TestData 盲评）
- **关键考虑**：滑坡像素占比极低 → 类别不平衡问题（当前用 CrossEntropyLoss，后续可上 Dice/Focal 缓解）

---

### 模块 ③｜方法层（模型设计 + 训练策略）

- **Baseline 模型：U-Net（手写实现）**
  - 结构：编码器-解码器 + 跳跃连接（skip connection），参考 `milesial/Pytorch-UNet`
  - **无预训练 backbone**：编码器从随机权重训练（`DoubleConv/Down` 是普通卷积，不是 resnet/efficientnet）
  - 上采样：`bilinear=True`（双线性插值 + 卷积，简单可靠）
  - 输出：**2 个通道**（背景 / 滑坡各一个 logit），不是 1 通道
  - 参数量链条：`14 → 64 → 128 → 256 → 512 → 1024 → … → 2`
- **对比模型（规划中，未实现）**：
  - DeepLabV3+（ASPP 多尺度空洞卷积）
  - Attention U-Net（注意力门控）
  - SegFormer / Swin-UNet（视觉 Transformer）
  - 加入方式：在 `models/` 下新建子包 + 在 `models/__init__.py` 的 `build_model()` 加分支
- **损失函数（当前）**：
  - `nn.CrossEntropyLoss()`（2 通道输出，标签为 0/1 类别索引）
  - 由 `scripts/losses.py: build_loss(cfg)` 返回；现仅实现 `cross_entropy`
  - （规划）Dice / Focal / 组合损失，作为消融实验
- **训练策略（当前）**：
  - 优化器：**Adam**（lr=1e-3，weight_decay=5e-4），**不是** AdamW
  - 学习率调度：当前**无**（固定 lr），早停充当收尾
  - 早停：`patience=8`，按验证集 `landslide_IoU` 监控，保存最优 `best_model.pt`
  - batch_size=8（4GB 显存约束）
  - 类别不平衡处理：当前未做加权（后续可在 `CrossEntropyLoss(weight=...)` 加）

---

### 模块 ④｜实验层（评估体系 + 实验设计）

- **评估指标**（`scripts/metrics.py: compute_metrics`，二元混淆矩阵逐类统计）：
  - IoU（交并比）/ mean IoU
  - F1、Precision（查准率）、Recall（召回率）
  - 滑坡类主指标：`landslide_iou`、`landslide_f1`（第 1 类）
  - ⚠️ 不用 Accuracy 作为主指标（滑坡像素占比低，全预测"非滑坡"也有高准确率）
- **三类实验**：
  1. **对比实验**（规划）：不同模型在相同数据/设置下对比 IoU/F1
  2. **消融实验**（规划）：损失函数 / 数据增强 / 波段数 / 是否加坡度高程 的影响
  3. **泛化实验（加分项）**：跨数据集验证

---

### 模块 ⑤｜结果层（可视化 + 误差分析）

- **定性分析**：`evaluate.py` 出 3 栏对比图（Input / Ground Truth / Prediction）；`predict.py` 出 输入概览 + 预测掩膜
- **定量分析**：`train.py` 自己存 `loss.png`（每 step 一个 train loss 点）；`train_log.csv` 记录每 epoch 的 train_loss / val_loss / val_landslide_iou / val_landslide_f1
- **误差分析**（报告内容，代码未自动出）：
  - 漏检（FN）：小滑坡、低对比度、阴影遮挡
  - 误检（FP）：道路、裸露岩体、云影、河流被误判为滑坡

---

### 模块 ⑥｜讨论与结论

- **模型优劣总结**：精度 vs 参数量 vs 推理速度的权衡
- **研究局限**：单一数据源、仅单时相、未融合地形/地质因子、样本仍偏少、U-Net 无预训练
- **未来方向**：多时相变化检测、多模态融合、Transformer 轻量化、对比模型扩展

---

## 3. 报告章节结构

1. **摘要** — 问题、方法、数据集、主要结果（300~500 字）
2. **引言** — 背景、滑坡危害、遥感自动识别的意义、本文贡献
3. **相关工作** — 滑坡检测综述 + 深度学习语义分割综述
4. **数据与方法** — 研究区、数据、预处理、模型、损失、指标（对应模块 ②③④）
5. **实验与结果** — 实验设置、对比、消融、可视化、误差分析（对应模块 ④⑤）
6. **讨论** — 对应模块 ⑥
7. **结论与展望**
8. **参考文献**

---

## 4. 当前进度（2026-08-25）

| 模块 | 状态 |
|---|---|
| 数据下载与划分 | ✅ `scripts/download_data.py` 已跑，data/raw 有 3554/245/800 |
| U-Net baseline | ✅ 手写完成，train/evaluate/predict 跑通，已训出 best_model.pt |
| 损失 / 指标 | ✅ CrossEntropyLoss + IoU/F1 |
| 对比模型 | ⏳ 规划，未实现 |
| 消融实验 | ⏳ 规划 |
| 报告撰写 | ⏳ 进行中（用户另有一份 Word 报告在根目录） |

---

## 5. 环境与工具栈（约定）

- 语言：Python 3.10+（torch_gpu 环境）
- 框架：PyTorch 2.5.1 + CUDA 12.1
- 分割实现：**手写 U-Net**，不依赖 SMP
- 数据：Zenodo 下载（非 HuggingFace）
- 可视化：matplotlib（训练曲线 / 预测对比图）
- 依赖管理：conda env + `requirements.txt`（torch 的 +cu121 需 `--index-url https://download.pytorch.org/whl/cu121`）
