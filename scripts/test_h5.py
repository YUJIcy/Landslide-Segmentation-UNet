"""
用于测试数据集的格式
学习时所遇到的问题：

## 1. mask 不用转换为 CHW 的形式吗？

结论：二分类分割任务中，mask 不需要强制转 CHW 格式，这是分割任务的标准写法，完全符合 PyTorch 的规范。
mask 是标签数据，不是输入特征，它不需要经过卷积层，维度只需要和损失函数的要求匹配即可。
对于你的二分类滑坡分割场景（只有滑坡/非滑坡两类）：
- 代码中 mask 是二维的 `(H, W)`，每个像素的值是 0 或 1，代表该像素的类别索引。
- PyTorch 最常用的两个分割损失函数都支持这种格式：
  - `nn.CrossEntropyLoss`：要求标签形状为 `(N, H, W)`，每个位置是类别序号，**不需要通道维度**。DataLoader 打包 batch 后 mask 会变成 `(batch_size, H, W)`，刚好符合要求。
  - `nn.BCEWithLogitsLoss`：标签支持 `(N, 1, H, W)` 和 `(N, H, W)` 两种格式，无需手动加通道维。
只有当你的损失函数强制要求标签和网络输出的 shape 完全一致时（比如网络输出是 `(N, 1, H, W)`），才需要在返回 mask 时加一句 `mask.unsqueeze(0)` 变成 `(1, H, W)`。但这不是必须操作，绝大多数开源分割代码里，二分类 mask 都直接使用二维格式。

---

## 2. 怎么知道 .h5 里的数据集名称是 img 和 mask？

结论：这是 Landslide4Sense 官方数据集的标准格式，是数据集发布时就规定好的固定命名。可参考github源代码
官方规范约定：
Landslide4Sense 是 IEEE GRSS 官方发布的滑坡识别竞赛基准数据集，官方在数据说明文档、下载页面和基线代码中都明确规定了存储格式：
- 影像文件 `image_*.h5` 内部，存储影像的数据集键名固定为 `img`，形状为 `(128, 128, 14)`（HWC 格式，14 个光谱波段）。
- 标签文件 `mask_*.h5` 内部，存储标签的数据集键名固定为 `mask`，形状为 `(128, 128)`，二值（0=背景，1=滑坡）。
所有使用该数据集的论文、开源代码都遵循这个统一命名约定。
通用调试方法（陌生 h5 文件也能用）：
如果拿到一个未知结构的 h5 文件，你可以用两行代码快速查看内部所有数据集名称和形状：
```python
import h5py
with h5py.File('image_0.h5', 'r') as f:
    print(f.keys())       # 输出所有数据集名称，比如 ['img']
    print(f['img'].shape) # 查看数据集的维度形状
```

---

## 3. __len__ 和 __getitem__ 是怎么被 DataLoader 调用的？

- `Dataset`：只负责「单个样本按索引怎么取」，相当于一个带序号的数据清单。
- `DataLoader`：负责「批量组织数据」，它会自动调用 Dataset 的两个方法，帮你生成 batch、打乱顺序、多进程加速。
① `__len__` 的调用时机
`__len__` 返回数据集的总样本数，它有三个核心作用：
1. 手动调用 `len(ds)` 时直接执行，就是测试代码里的 `print("样本数：", len(ds))`。
2. DataLoader 生成索引时必须使用：当设置 `shuffle=True` 时，DataLoader 会先生成 `0 到 总样本数-1` 的整数序列再打乱，这个「总样本数」就来自 `__len__`。
3. 计算一个 epoch 的 batch 数量：`总迭代轮数 = 总样本数 ÷ batch_size`，这个总数也来自 `__len__`。
② `__getitem__` 的完整调用流程
DataLoader 加载一个 batch 的完整过程：
1. 生成当前批次的索引列表：比如 `batch_size=4`，就取出 4 个索引，例如 `[5, 17, 3, 9]`。
2. 逐个索引调用 `dataset.__getitem__(index)`：
   - 索引 5 → 调用 `ds[5]` → 得到第 5 张影像 `(14,128,128)` 和 mask `(128,128)`
   - 索引 17 → 调用 `ds[17]` → 得到第 17 张影像和 mask
   - 以此类推，拿到 4 组独立样本。
3. 自动拼接成批次张量：
   DataLoader 内部默认的 `collate_fn` 会把这 4 个样本堆叠成 batch 格式：
   - 影像：4 个 `(14,128,128)` → 拼成 `(4, 14, 128, 128)`（标准 NCHW 批次格式）
   - 标签：4 个 `(128,128)` → 拼成 `(4, 128, 128)`
4. 把拼接好的 batch 返回给训练循环，也就是你 `for imgs, masks in dataloader` 里拿到的变量。
通俗类比：
可以把 Dataset 想象成一本带页码的相册：
- `__len__` 就是相册总页数，告诉别人这本相册有多少张照片。
- `__getitem__(页码)` 就是翻到第 x 页，把那张照片拿出来。
而 DataLoader 就是助理，每次按你的要求（batch_size）从相册里抽若干张照片，整理成一叠交给你，它不需要知道照片怎么来的，只需要调用相册的“翻页”功能就行。
补充：如果设置 `num_workers > 0` 开启多进程加载，每个子进程会分到一部分索引，在子进程里各自调用 `__getitem__` 加载数据，最后把结果传回主进程拼接 batch，以此提升读取速度。
"""

from pathlib import Path
import h5py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
with h5py.File( PROJECT_ROOT / "data" / "sample" / "image_1.h5") as f:
    print(f.keys())
    print(f['img'].shape)

with h5py.File( PROJECT_ROOT / "data" / "sample" / "mask_1.h5") as f:
    print(f.keys())
    print(f['mask'].shape)

# 打印结果：
# <KeysViewHDF5 ['img']>
# (128, 128, 14)
# <KeysViewHDF5 ['mask']>
# (128, 128)
