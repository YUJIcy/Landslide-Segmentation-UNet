# 从unet子包中导入UNet类，unet/__init__.py
from .unet import UNet

__all__ = ['UNet', 'build_model']


# 定义统一入口函数，传入配置字典cfg，返回模型
def build_model(cfg):
    # 从配置里取出模型名字，转为小写，防止UNet/unet不匹配
    name = cfg['model']['name'].lower()

    # 如果配置名是unet，就创建一个UNet实例
    if name == 'unet':
        return UNet(
            n_channels=cfg['data']['in_channels'],      # 输入通道数，14
            n_classes=cfg['model']['out_channels'],     # 输出通道数，2
            bilinear=True,                              # 采用双线性插值进行采样
        )

    # 若配置名不对，报错提醒
    raise ValueError(f'Unexpected model name: {name}')