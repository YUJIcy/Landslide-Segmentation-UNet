"""
在测试集上评估最优模型，保存预测可视化结果
"""

from pathlib import Path
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import LandslideDataset
from models import build_model
from scripts.metrics import compute_metrics


PROJECT_ROOT = Path(__file__).resolve().parent

device = "cuda:0"
cfg_path = PROJECT_ROOT / "configs" / "default.yaml"
with open(cfg_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 训练中保存的结果目录
exp_dir = PROJECT_ROOT / "runs" / cfg["exp_name"]
ckpt_path = exp_dir / "checkpoints" / "best_model.pt"       # 存储的最优权重
pred_dir = exp_dir / "preds"                                # 预测可视化输出目录
pred_dir.mkdir(parents=True, exist_ok=True)


def main():
    val_dir = PROJECT_ROOT / cfg["data"]["val_dir"]      # 正式训练时改为验证集目录
    mean, std = cfg["data"]["mean"], cfg["data"]["std"]
    test_data = LandslideDataset(val_dir / "img", val_dir / "mask", mean, std)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=cfg["train"]["num_workers"])

    # 重建模型，加载权重
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()                                             # 评估模式

    preds, targets = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device).long()
            Pred = model(x)
            _, predicted = torch.max(Pred.data, dim=1)
            preds.append(predicted.cpu().numpy())
            targets.append(y.cpu().numpy())

            # 只可视化前10张
            if i < 10:
                save_prediction(x[0].cpu().numpy(), y[0].cpu().numpy(),
                                predicted[0].cpu().numpy(), pred_dir / f"pred_{i:03d}.png")

    m = compute_metrics(np.concatenate(preds), np.concatenate(targets), num_classes=2)
    print("评估完成。测试集指标：")
    print(f"  landslide_IoU = {m['landslide_iou']:.4f}")
    print(f"  landslide_F1  = {m['landslide_f1']:.4f}")
    print(f"  mean_IoU      = {m['mean_iou']:.4f}")
    print(f"预测图已保存到: {pred_dir}")


def save_prediction(img, mask, pred, out_path):
    # img: (C,H,W) → 取前 3、2、1通道当 RGB 显示（Sentinel-2 的 0/1/2 波段 ≈ 红绿蓝）
    rgb = img[[3, 2, 1]].transpose(1, 2, 0)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)  # 简单归一化到 0~1 便于显示
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax[0].imshow(rgb)
    ax[0].set_title("Input (RGB)")
    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Ground Truth")
    ax[2].imshow(pred, cmap="gray")
    ax[2].set_title("Prediction")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)     # 用 close 代替 show，规避 PyCharm 后端崩溃


if __name__ == "__main__":
    main()
