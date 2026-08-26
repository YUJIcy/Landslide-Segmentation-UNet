"""
指定n张无标签影像，对该影像进行滑坡识别
"""

from pathlib import Path
import numpy as np
import yaml
import torch
import h5py
import matplotlib.pyplot as plt
from models import build_model
PROJECT_ROOT = Path(__file__).resolve().parent

device = "cuda:0"
# 待预测的影像路径：
TEST_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "test" / "img"
NUM_PREDICT = 10
cfg_path = PROJECT_ROOT / "configs" / "default.yaml"
with open(cfg_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
exp_dir = PROJECT_ROOT / "runs" / cfg["exp_name"]
ckpt_path = exp_dir / "checkpoints" / "best_model.pt"
out_dir = exp_dir / "predictions"
out_dir.mkdir(parents=True, exist_ok=True)
# 结果名称
# pred_mask_name = "pred_mask"            # 结果掩膜图
# pred_overview_name = "pred_overview"    # 展示图


def load_image_for_predict(img_path, mean, std):
    with h5py.File(img_path, "r") as f:
        image = f["img"][:]        # 注意：源格式为(H,W,14)
        image = np.array(image, dtype=np.float32)
        image = image.transpose(2, 0, 1)
        mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
        std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)
        image = (image - mean) / std
        image = torch.from_numpy(image.copy())
        image = image.unsqueeze(0)     # 增加batch维，(1,14,H,W)
        return image


def main():
    mean, std = cfg["data"]["mean"], cfg["data"]["std"]

    # 读取模型和训练好的参数
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 列出所有image
    img_files = sorted(TEST_IMAGE_DIR.glob("image_*.h5"))
    img_files = img_files[:NUM_PREDICT]
    print(f"待预测影像 {len(img_files)} 张（来自 {TEST_IMAGE_DIR}）")

    for k,img_path in enumerate(img_files):
        # 加载影像
        x = load_image_for_predict(img_path, mean, std).to(device)
        with torch.no_grad():
            Pred = model(x)
            _, predicted = torch.max(Pred.data, dim=1)  # (1,H,W)
        pred = predicted[0].cpu().numpy()  # (H,W)

        # 保存栅格
        plt.imsave(out_dir / f"pred_mask_name_{k:03d}.png", pred, cmap="gray")

        # 绘图
        rgb = x[0, [3, 2, 1]].cpu().numpy().transpose(1, 2, 0)  # (H,W,3)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(rgb)
        ax[0].set_title("Input Image")
        ax[1].imshow(pred, cmap="gray")
        ax[1].set_title("Prediction(landslide)")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"pred_overview_{k:03d}.png", dpi=300)
        plt.close(fig)

        print(f"预测完成：{TEST_IMAGE_DIR.name}")
        print(f"  滑坡像元数 = {int(pred.sum())} / 总像元 = {pred.size} "
            f"(占比 {pred.mean() * 100:.2f}%)")

    print(f"全部预测完成，结果已保存：{out_dir}")


if __name__ == "__main__":
    main()
