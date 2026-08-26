LandslideDetection/
│
├── README.md                  # 项目说明（目前近乎空白，待补）
├── requirements.txt           # 依赖清单（torch 的 +cu121 需官方源装）
├── configs/
│   └── default.yaml           # 所有超参数集中配置（模型名、batch、lr、epochs、mean/std…）
│
├── data/                      # 数据目录（大文件）
│   ├── raw/                   # 真实数据（scripts/download_data.py 生成）
│   │   ├── train/
│   │   │   ├── img/           #   image_1.h5 ... image_3554.h5（带标签）
│   │   │   └── mask/          #   mask_1.h5 ... mask_3554.h5
│   │   ├── val/
│   │   │   ├── img/           #   image_*.h5（245，带标签，从 TrainData 切出）
│   │   │   └── mask/          #   mask_*.h5（245）
│   │   ├── test/
│   │   │   └── img/           #   image_1.h5 ... image_800.h5（无标签，供 predict.py 盲预测）
│   │   └── _downloads/        #   下载的 zip 临时目录（解压切分后可删）
│   └── sample/                # （已有）2 个样例 h5（试跑用）
│       ├── image_1.h5
│       └── mask_1.h5
│
├── dataset/                   # 数据读取模块（与官方 dataset/ 对齐思路）
│   ├── __init__.py
│   └── landslide_dataset.py   # LandslideDataset：读 h5、transpose(HWC→CHW)、标准化
│
├── models/                    # 模型模块包
│   ├── __init__.py            # build_model() 统一入口（现仅 unet）
│   └── unet/                  # U-Net 子包（参考 milesial/Pytorch-UNet，全部手写）
│       ├── __init__.py
│       ├── unet_model.py      # UNet 组装（14→…→2 通道）
│       └── unet_parts.py      # DoubleConv / Down / Up / OutConv
│
├── train.py                   # 训练主循环（含验证、早停、存最优，自存 loss.png）
├── evaluate.py                # 加载最优权重在验证集评估 + 存 3 栏对比图
├── predict.py                 # 对 test 集前 N 张盲预测、存掩膜 + 概览 PNG
├── test_unet.py               # 冒烟测试：确认模型前向输出 (1,2,128,128)
│
├── scripts/                   # 工具脚本
│   ├── download_data.py       # 下载 + 切分数据到 data/raw
│   ├── losses.py              # build_loss()（现仅 cross_entropy）
│   ├── metrics.py             # compute_metrics()（IoU / F1 / Precision / Recall）
│   └── test_h5.py             # 打印 h5 内部 keys / shape
│
├── runs/                      # 实验输出（运行时自动生成，别手建）
│   └── exp_01_unet/
│       ├── checkpoints/        # best_model.pt（验证集 landslide_IoU 最优权重）
│       ├── logs/               # train_log.csv + loss.png
│       ├── preds/              # evaluate.py 出的 3 栏对比图
│       └── predictions/        # predict.py 出的掩膜 + 概览图
│
└── docs/                      # 文档
    ├── 00_project_structure.md
    ├── 01_research_framework.md
    ├── 02_theory_knowledge.md
    └── 03_directory.md
