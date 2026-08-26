# 损失函数模块：根据配置返回对应的损失函数
import torch.nn as nn


def build_loss(cfg):
    """
    读取配置中loss信息，返回对应损失函数
    """
    name = cfg["train"]["loss"].lower()    # 取出名称并统一转小写

    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"位置损失函数:{name}")


# 测试
if __name__ == "__main__":
    import torch
    logits = torch.randn(2, 2, 128, 128)    # 模型原始输出
    target = torch.randint(0, 2, (2, 128, 128)).long()    # 类别索引
    loss_fn = build_loss({"train": {"loss": "cross_entropy"}})
    loss = loss_fn(logits, target)
    print("loss:", loss)
