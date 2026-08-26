# Full assembly of the parts to form the complete network
# 使用unet_parts的模块组装成完整的U-Net

import torch.nn as nn
from .unet_parts import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    """完整的U-Net网络"""
    def __init__(self, n_channels, n_classes, bilinear=False):    # n_channels为输入通道(14),n_classes为输出通道(2)
        super(UNet, self).__init__()
        self.n_channels = n_channels    # 记录输入通道数
        self.n_classes = n_classes      # 记录输出通道数
        self.bilinear = bilinear        # 记录用哪种上采样方式

        self.inc = (DoubleConv(n_channels, 64))    # 输入层:14通道->64通道,尺寸不变
        self.down1 = (Down(64, 128))     # 第1次下采样:64->128通道,128*128->64*64
        self.down2 = (Down(128, 256))    # 第2次下采样:128->256通道,64*64->32*32
        self.down3 = (Down(256, 512))    # 第3次下采样:256->512通道,32*32->16*16
        factor = 2 if bilinear else 1                         # 双线性模式下桥接层通道数减半,省显存
        self.down4 = (Down(512, 1024 // factor))    # 第4次下采样(桥接层):16*16->8*8
        self.up1 = (Up(1024, 512 // factor, bilinear))    # 第1次上采样:8*8->16*16,拼接down3
        self.up2 = (Up(512, 256 // factor, bilinear))     # 第2次上采样:16*16->32*32,拼接down2
        self.up3 = (Up(256, 128 // factor, bilinear))     # 第3次上采样:32*32->64*64,拼接down1
        self.up4 = (Up(128,64, bilinear))      # 第4次上采样:64*64->128*128,拼接inc
        self.outc = (OutConv(64, n_classes))              # 输出层:64通道->2通道

    def forward(self, x):
        x1 = self.inc(x)       # 编码第0层(64, 128×128)，留给跳跃连接
        x2 = self.down1(x1)    # 编码第1层 (128, 64×64)
        x3 = self.down2(x2)    # 编码第2层 (256, 32×32)
        x4 = self.down3(x3)    # 编码第3层 (512, 16×16)
        x5 = self.down4(x4)    # 桥接层 (512, 8×8)
        x = self.up1(x5, x4)   # 解码：上采样 x5，拼接编码第3层 x4
        x = self.up2(x, x3)    # 拼接编码第2层 x3
        x = self.up3(x, x2)    # 拼接编码第1层 x2
        x = self.up4(x, x1)    # 拼接编码第0层 x1
        logits = self.outc(x)  # 输出 (1, 128×128)，每个像素是"滑坡得分"
        return logits          # 返回得分图（训练时用 CrossEntropyLoss 内部自带 sigmoid）
