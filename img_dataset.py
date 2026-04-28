from dataclasses import dataclass
from pathlib import Path
#--------------------------------------------------------------------
import torch
from torch.utils.data import Dataset

import torchvision.transforms.v2 as transforms
from torchvision.io import decode_image, write_jpeg

#--------------------------------------------------------------------
torch.set_default_device('cpu') # 'cuda:0  [IMPORTANT!!!]
DEVICE = torch.device('cpu') # 'cuda:0'
torch.set_default_dtype(torch.float32)

#--------------------------------------------------------------------
IMG_FLODER = Path(r"E:\CodeHub\Mydata\AnimeFace") # [IMPORTANT!!!]


@dataclass
class AnimeFaceDataset(Dataset):
    floder: Path = IMG_FLODER
    size: int = 80

    mean = (0.6882, 0.5888, 0.5723)
    std = (0.2813, 0.2822, 0.2626)
    # mean = (0.7428, 0.6926, 0.7218)  # Quan_AnimeFace
    # std =  (0.2652, 0.2792, 0.2578)  # Quan_AnimeFace
    inv_std = tuple(1/std_i for std_i in std)
    inv_mean = tuple(-istdi*meani for istdi,meani in zip(inv_std,mean))

    orignal_transform = transforms.Compose([
        transforms.RandomResizedCrop(size, (0.9,1.0), (6/7,7/6)),
        # transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(0.05,0.05,0.05,0.02),
        transforms.RandomHorizontalFlip(),
        transforms.ToDtype(torch.float32,scale=True),
        # transforms.GaussianNoise(),
        transforms.Normalize(mean, std),
    ])  # the default transform for training data
    transform: transforms.Compose = orignal_transform

    orignal_inv_trans = transforms.Compose([
        transforms.Normalize(inv_mean,inv_std),
        transforms.Lambda(lambda x:torch.clamp(x,min=0.0,max=1.0)),
        transforms.ToDtype(torch.uint8,scale=True)
    ])  # the default transform for converting tensor to image (for preview)
    inv_trans: transforms.Compose = orignal_inv_trans

    def __post_init__(self):
        self.path: tuple[Path,...] = tuple(self.floder.iterdir())

    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, index):
        img_path = self.path[index]
        img_t = decode_image(img_path, "RGB").to(device=DEVICE)
        return self.transform(img_t)
    
    def reset(self):
        """reset the transform to the default one (for training data and preview)"""
        self.transform = self.orignal_transform
        self.inv_trans = self.orignal_inv_trans