"""
下载Landslide4Sense官方数据集，划分为训练集/验证集/测试集
注意：只有TrainData带标签，故验证集要从TrainData中切一部分出来
"""

from pathlib import Path
import sys
import urllib.request
import zipfile
import shutil
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw"
DOWNLOAD_DIR = RAW / "_downloads"

URLS = {
    "TrainData.zip": "https://zenodo.org/api/records/10463239/files/TrainData.zip/content",
    "TestData.zip":  "https://zenodo.org/api/records/10463239/files/TestData.zip/content",
}

VAL_COUNT = 245

SEED = 42


def download(url, dest):
    if dest.exists():
        print(f"已存在，跳过: {dest.name}")
        return
    print(f"下载中: {dest.name} ...")

    def hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = block_num * block_size * 100 // total_size
            sys.stdout.write(f"\r  {percent:3d}%  ({block_num * block_size // 1024} KB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, hook)
    print()


def unzip(zip_path, extract_to):
    print(f"解压：: {zip_path.name}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_to)


def split_train_val():
    """
    TrainData/img 与 TrainData/mask 一一对应：image_N.h5 ↔ mask_N.h5
    """
    src_img = DOWNLOAD_DIR / "TrainData" / "img"
    src_mask = DOWNLOAD_DIR / "TrainData" / "mask"
    img_files = sorted(src_img.glob("image_*.h5"))
    print(f"TrainData共: {len(img_files)}个patch")

    rng = random.Random(SEED)
    idx = list(range(len(img_files)))
    rng.shuffle(idx)
    val_idx = idx[:VAL_COUNT]
    train_idx = idx[VAL_COUNT:]

    def move(group_idx, out_img, out_mask):
        out_img.mkdir(parents=True, exist_ok=True)
        out_mask.mkdir(parents=True, exist_ok=True)
        for i in group_idx:
            name = img_files[i].name
            mask_name = name.replace("image", "mask")
            shutil.move(str(img_files[i]), str(out_img / name))
            shutil.move(str(src_mask / mask_name), str(out_mask / mask_name))

    move(train_idx, RAW / "train" / "img", RAW / "train" / "mask")
    move(val_idx, RAW / "val" / "img", RAW / "val" / "mask")
    print(f"划分完成：train={len(train_idx)}  val={len(val_idx)}")


def move_test():
    """
    TestData 仅 img、无 mask → 放到 data/raw/test/img，供 predict.py 盲预测
    """
    src = DOWNLOAD_DIR / "TestData" / "img"
    dst = RAW / "test" / "img"
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("image_*.h5")):
        shutil.move(str(f), str(dst / f.name))
    print(f"TestData 已放到 {dst}（{len(sorted(dst.glob('image_*.h5')))} 张，无标签）")


def main():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        download(url, DOWNLOAD_DIR / name)
    for name in URLS:
        unzip(DOWNLOAD_DIR / name, DOWNLOAD_DIR)
    split_train_val()
    move_test()
    print("数据准备完成。结构：")
    print(f"  {RAW}/train/img + train/mask  (训练，带标签)")
    print(f"  {RAW}/val/img   + val/mask    (验证，带标签，从 TrainData 切出)")
    print(f"  {RAW}/test/img                (测试，无标签，供 predict.py)")


if __name__ == "__main__":
    main()
