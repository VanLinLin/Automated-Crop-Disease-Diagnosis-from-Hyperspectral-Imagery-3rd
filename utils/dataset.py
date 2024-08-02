import torch
from torch.utils.data import Dataset
import numpy as np
import warnings
from pathlib import Path


class EnsembleDataset(Dataset):
    def __init__(self, data_dir, transform_convnext_swin):
        self.data_info = self.get_img_info(Path(data_dir))
        self.transform_convnext_swin = transform_convnext_swin

    def __getitem__(self, index):
        path_img = self.data_info[index]
        img = torch.load(path_img) # 加載 .pt 文件，這裡 img 已經是 torch.Tensor
        img_32 = self.normHSI(img)

        img_64 = self.transform_convnext_swin(img)

        return img_32, img_64

    def __len__(self):
        return len(self.data_info)
    
    @staticmethod
    def get_img_info(data_dir):
        return sorted(list(data_dir.glob('*.pt')))
    
    @staticmethod
    def normHSI(R):
        if isinstance(R, torch.Tensor):
            rmax, rmin = torch.max(R), torch.min(R)
            R = (R - rmin)/(rmax - rmin)
        elif isinstance(R, np.ndarray):
            rmax, rmin = np.max(R), np.min(R)
            R = (R - rmin)/(rmax - rmin)
        else:
            warnings.warn("Unsupport data type of input HSI")
            return
        return R