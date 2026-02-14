## Code to load dataset
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

class ECPC_Dataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # List files and ensure they are sorted so images and masks match
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        # Load image and mask
        # ECPC images can be RGB or Grayscale; masks are usually 'L' (8-bit pixels)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") 

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=np.array(image), mask=np.array(mask))
            image = augmented['image']
            mask = augmented['mask']
            
        # Ensure mask is binary (0 and 1) instead of 0 and 255
        mask = torch.where(mask > 0, 1.0, 0.0)

        return image, mask
