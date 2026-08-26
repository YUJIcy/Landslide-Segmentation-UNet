"""
主训练文件
制作数据集 -> 搭建神经网络 -> 损失函数、优化算法 -> 训练网络
"""

from pathlib import Path
import random
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import LandslideDataset
from models import build_model
from scripts.losses import build_loss
from scripts.metrics import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parent


# --------全局常量-------- #

# 基本配置信息
device = "cuda:0"
cfg_path = PROJECT_ROOT / "configs" / "default.yaml"
with open(cfg_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 过程结果目录
exp_dir = PROJECT_ROOT / "runs" / cfg["exp_name"]    # 单次实验目录
ckpt_dir = exp_dir / "checkpoints"                   # 模型权重
log_dir = exp_dir / "logs"                           # 训练日志
ckpt_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)


# ------模型构建与训练------ #

# 随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# 主函数
def main():
    # 基本参数信息
    mean = cfg["data"]["mean"]
    std = cfg["data"]["std"]
    batch_size = cfg["train"]["batch_size"]      # 批次大小
    num_workers = cfg["train"]["num_workers"]    # dataloader多线程
    set_seed(cfg["seed"])

    # 数据与数据集构建
    train_dir = PROJECT_ROOT / cfg["data"]["train_dir"]
    val_dir = PROJECT_ROOT / cfg["data"]["val_dir"]
    train_data = LandslideDataset(train_dir / "img", train_dir / "mask", mean, std)
    test_data = LandslideDataset(val_dir / "img", val_dir / "mask", mean, std)

    # 批次加载器
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 神经网络模型
    model = build_model(cfg).to(device)

    # 损失函数、优化器
    loss_fn = build_loss(cfg)
    learning_rate = cfg["train"]["lr"]
    # Adam优化器，带权重衰减
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=cfg["train"]["weight_decay"])

    # 网络训练参数
    epochs = cfg["train"]["epochs"]         # 总训练轮数
    patience = cfg["train"]["patience"]     # 早停耐心值，多少轮指标不提升就停止训练
    losses = []
    best_iou = -1.0                         # 最优IoU初始值
    no_improve = 0                          # 记录连续多少轮验证集没有提升
    log_path = log_dir / "train_log.csv"    # 日志路径
    log_path.write_text("epoch,train_loss,val_loss,val_landslide_iou,val_landslide_f1\n", encoding="utf-8")

    for epoch in range(epochs):
        # 训练
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).long()
            Pred = model(x)                 # 前向传播
            loss = loss_fn(Pred, y)         # 计算损失
            losses.append(loss.item())      # 存储每个step的损失值
            optimizer.zero_grad()           # 梯度清零
            loss.backward()                 # 反向传播
            optimizer.step()                # 更新模型参数

        # 验证
        model.eval()
        val_loss = 0.0                      # 验证集总损失
        preds, targets = [], []             # 预测结果、真实标签
        with torch.no_grad():               # 评估阶段不计算梯度
            for x, y in test_loader:
                x, y = x.to(device), y.to(device).long()
                Pred = model(x)
                val_loss += loss_fn(Pred, y).item() * x.size(0)    # 累加损失，乘以样本数做加权
                _, predicted = torch.max(Pred.data, dim=1)         # 取通道维度上最大值作为预测类别
                preds.append(predicted.cpu().numpy())              # 预测结果移回cpu，转numpy存入列表
                targets.append(y.cpu().numpy())                    # 真实标签移回cpu，转numpy存入列表
        val_loss /= len(test_data)                                 # 计算平均验证损失，总损失除以验证样本总数
        # 拼接所有batch预测与标签，计算分割指标，二分类(背景+滑坡)
        m = compute_metrics(np.concatenate(preds), np.concatenate(targets), num_classes=2)

        # 打印本轮训练信息：轮次、训练损失、验证损失、IoU、F1
        print(f"epoch {epoch:02d} | train_loss {losses[-1]:.4f} | val_loss {val_loss:.4f} | "
              f"landslide_IoU {m['landslide_iou']:.4f} | landslide_F1 {m['landslide_f1']:.4f}")
        with log_path.open("a", encoding="utf-8") as logf:
            logf.write(f"{epoch},{losses[-1]:.4f},{val_loss:.4f},{m['landslide_iou']:.4f},{m['landslide_f1']:.4f}\n")

        # 早停+保存最优
        if m['landslide_iou'] > best_iou:
            best_iou = m['landslide_iou']     # 更新最优IoU
            no_improve = 0                    # 重置无提升计数
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")    # 保存最优模型权重
            print(f"  → 新最优模型已保存 (landslide_IoU={best_iou:.4f})")
        else:
            no_improve += 1                   # 没有提升，计数+1
            if no_improve >= patience:        # 达到耐心阈值，触发早停
                print(f"验证集 {patience} 轮无提升，提前停止。")
                break

    # 绘制损失曲线
    fig, ax = plt.subplots()
    ax.plot(range(len(losses)), losses)
    ax.set_title("Loss Curve")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(log_dir / "loss.png", dpi=300)
    # plt.show()

    print("训练结束。最优模型:", ckpt_dir / "best_model.pt")
    print("训练日志:", log_path)


if __name__ == "__main__":
    main()
