# parts of the U-Net model
# U-Net基础积木模块：DoubleConv,Down,Up,OutConv

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    定义双卷积块
    连续做两次：3*3卷积->批归一化->ReLU激活
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:    # 中间通道数
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),    # 批归一化：把这一批数据拉到均值0方差1附近，让训练更稳
            nn.ReLU(inplace=True),     # ReLU 激活：负数变 0，引入非线性；inplace=True 省内存
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    定义下采样块，缩小尺寸，增加通道
    先2*2最大池化把尺寸减半，再做双卷积
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),                         # 2*2最大池化，每2*2区域获取最大值，宽高减半
            DoubleConv(in_channels, out_channels)    # 接双卷积块
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv
    定义上采样块，放大尺寸，融合跳跃连接
    先把特征图放大2倍，再和编码器对应层拼接，最后双卷积
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 如果使用双线性插值，则上采样
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)    # 宽高各放大2倍
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)           # 拼接后通道减半处理
        # 否则用转置卷积上采样
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):    # x1为解码器传上来的特征图,x2为编码器同尺度的跳跃连接
        x1 = self.up(x1)          # 先把x1放大2倍
        # input is CHW
        # 计算x2和放大后x1的尺寸差
        diffY = x2.size()[2] - x1.size()[2]    # 高度差
        diffX = x2.size()[3] - x1.size()[3]    # 宽度差
        # 给x1补边,让它和x2尺寸完全一致(左/右/上/下分别补多少)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)    # 沿通道维(dim=1)拼接:跳跃连接的核心
        return self.conv(x)


class OutConv(nn.Module):
    """
    输出层:1*1卷积,把64通道压成n_classes通道
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)    # 1*1卷积只改通道数,不改尺寸

    def forward(self, x):
        return self.conv(x)
