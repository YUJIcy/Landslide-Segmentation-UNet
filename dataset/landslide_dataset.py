# 读取Landslide4Sense数据集
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LandslideDataset(Dataset):
    # image_dir: 存放image_N.h5的目录
    # mask_dir: 存放mask_N.h5的目录；测试集没标签时传None
    # mean/std: 14通道标准化参数（用官方在训练集上算好的值）
    # augment: 是否做数据增强（仅训练集开）
    def __init__(self, image_dir, mask_dir=None, mean=None, std=None):
        self.mask_dir = mask_dir
        self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)    # 转为三维矩阵，便于与影响进行广播运算
        self.std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)
        # 获取image_开头的h5文件
        # 按数字排序，保证位于不同路径下的image顺序与mask一致
        # image_files即为排好序的image文件
        self.image_files = sorted(
            f for f in os.listdir(image_dir)
            if f.startswith('image') and f.endswith('.h5')
        )
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_files)    # 指定len()读取image_files的长度

    def __getitem__(self, index):
        img_name = self.image_files[index]

        with h5py.File(os.path.join(self.image_dir, img_name), 'r') as f:
            image = f['img'][:]    # 读取.h5文件中的img数据集
        image = np.asarray(image, dtype=np.float32)

        mask = None
        if self.mask_dir is not None:
            mask_name = img_name.replace('image', 'mask')    # 将image替换为mask即可得到mask_dir下的对应标签数据
            with h5py.File(os.path.join(self.mask_dir, mask_name), 'r') as f:
                mask = f['mask'][:]    # 打开mask的.h5文件，读取mask数据集
            mask = np.asarray(mask, dtype=np.float32)

        image = image.transpose(2, 0, 1)    # 将提取到的image数据集转为CHW形式（mask为二维(H,W)，无需进行此处理）
        image = (image - self.mean) / self.std

        image = torch.from_numpy(image.copy())
        if mask is None:
            return image
        return image, torch.from_numpy(mask.copy())

if __name__ == '__main__':
    import yaml
    with open(os.path.join(PROJECT_ROOT, 'configs', 'default.yaml'), encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    ds = LandslideDataset(
        image_dir=os.path.join(PROJECT_ROOT, 'data', 'sample'),
        mask_dir=os.path.join(PROJECT_ROOT, 'data', 'sample'),
        mean=cfg["data"]["mean"],
        std=cfg["data"]["std"]
    )
    print("样本数：", len(ds))
    img, msk = ds[0]
    print("img:", img.shape, img.dtype, "值域 %.2f ~ %.2f" % (img.min(), img.max()))
    print("mask:", msk.shape, msk.dtype, "取值:", msk.unique())
    print("滑坡像素占比: %.2f%%" % (msk.mean().item() * 100))





