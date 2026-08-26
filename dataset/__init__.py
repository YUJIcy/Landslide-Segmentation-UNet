from .landslide_dataset import LandslideDataset

# __all__声明当其余文件from dataset import * 时，只导出LandslideDataset这一个类
__all__ = ['LandslideDataset']