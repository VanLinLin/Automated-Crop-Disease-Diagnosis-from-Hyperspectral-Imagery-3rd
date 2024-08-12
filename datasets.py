import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import warnings
import numpy as np

def normHSI(R, eps=1e-6, stat=False):
    if isinstance(R, torch.Tensor):
        rmax, rmin = torch.max(R), torch.min(R)
        R = (R - rmin)/(rmax - rmin + eps)
    elif isinstance(R, np.ndarray):
        rmax, rmin = np.max(R), np.min(R)
        R = (R - rmin)/(rmax - rmin + eps)
    else:
        warnings.warn("Unsupport data type of input HSI")
        return
    if stat:
        return R, rmax, rmin
    return R

def addGaussianNoise(hsi, snr=1000):
    xpower = torch.sum(hsi**2)/hsi.numel()
    npower = torch.sqrt(xpower/snr)
    hsi = hsi + torch.randn(hsi.shape)*npower
    return hsi

class FHB_Dataset(Dataset):
    def __init__(self, data_dir, transform=None, noisy=False, rotate=False):
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform
        self.nosiy = noisy
        self.rotate = rotate

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = torch.load(path_img) # 加載 .pt 文件，這裡 img 已經是 torch.Tensor
        # print(f"Loaded {path_img} with shape {img.shape}")  # 添加這一行來檢查形狀
        if self.transform:
            img = normHSI(img)
            img = self.transform(img)
            if self.nosiy:
                if np.random.rand() > 0.5:
                    img = addGaussianNoise(img)
            if self.rotate:
                r = np.random.rand()
                if  r > 0.66:
                    img = transforms.functional.rotate(img, 180)
                elif r > 0.33:
                    img = transforms.functional.rotate(img, 270)
                else:
                    img = transforms.functional.rotate(img, 90)
        return img, label

    def __len__(self):
        return len(self.data_info)
    
    @staticmethod
    def get_img_info(data_dir):
        data_info = []
        label_name = {"1": 0, "2": 1}
        for root, dirs, files in os.walk(data_dir):
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.pt'), img_names))
                for img_name_ in img_names:
                    img_name = img_name_
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = label_name.get(sub_dir)
                    if label is not None:
                        data_info.append((path_img, int(label)))
        return data_info
    
import imageio
import skimage.io as skio
class HOR_Dataset(Dataset):
    def __init__(self, root, transform=None, noisy=False, rotate=False, norm=True):
        self.root = root
        self.transform = transform
        self.nosiy = noisy
        self.rotate = rotate
        self.norm = norm

        type_list = os.listdir(self.root)
        self.data_path = []
        for t in type_list:
            for f in os.listdir(os.path.join(root, t)):
                self.data_path.append(os.path.join(root, t, f))
        self.label_dict = {'Health': 0, 'Other': 1, 'Rust': 2}

    def __getitem__(self, index):
        img_path = self.data_path[index]
        # img = imageio.imread(img_path).astype(np.float32)[:,:,:125]
        img = skio.imread(img_path)
        img = img.astype('float')[:,:,:125]
        label = self.label_dict[img_path.split('/')[-2]]
        # print(type(img), img.shape) # 64, 64, 125
        # print(img.shape[-1])
        # print(img)
        img = torch.tensor(img).permute(2,0,1)
        if self.norm:
            img = normHSI(img)
        if self.transform:
            img = self.transform(img)
            if self.nosiy:
                if np.random.rand() > 0.5:
                    img = addGaussianNoise(img)
            if self.rotate:
                r = np.random.rand()
                if  r > 0.66:
                    img = transforms.functional.rotate(img, 180)
                elif r > 0.33:
                    img = transforms.functional.rotate(img, 270)
                else:
                    img = transforms.functional.rotate(img, 90)
        return img, label

    def __len__(self):
        return len(self.data_path)
    