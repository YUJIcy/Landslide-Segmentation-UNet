"""
测试unet，读取default配置，建立模型，建立数据集，测试模型
"""

import yaml
import torch
from pathlib import Path

from dataset import LandslideDataset
from models import build_model

PROJECT_ROOT = Path(__file__).resolve().parent

cfg_path = PROJECT_ROOT / "configs" / "default.yaml"
with open(cfg_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 使用sample测试
ds = LandslideDataset(
    image_dir=PROJECT_ROOT / "data" / "sample",
    mask_dir=PROJECT_ROOT / "data" / "sample",
    mean=cfg["data"]["mean"],
    std=cfg["data"]["std"]
)
img, _ = ds[0]

# 按配置构建模型
model = build_model(cfg)
model.eval()
with torch.no_grad():
    out = model(img.unsqueeze(0))

print("input :", img.unsqueeze(0).shape)
print("output:", out.shape)
assert out.shape == (1, 2, 128, 128), "输出尺寸不对！"
print("U-Net 链路已打通（2 通道输出）")
