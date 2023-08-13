from pathlib import Path
from typing import Any

import torch
import torch.nn
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize
import torch.nn.functional as F

class GTSRBDataset(Dataset):
    def __init__(self, dataset_root:Path, mode="Train", num_classes=100) -> None:
        super().__init__()
        self._dataset_root = dataset_root
        self._num_classes = num_classes
        self._df = pd.read_csv(self._dataset_root/ f"{mode}.csv")
        self._transform = Compose([Resize((32,32)),ToTensor()])
    
    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, index:int) -> Any:
        img_path = self._dataset_root / self._df.Path[index]
        label = torch.tensor(self._df.ClassId[index],dtype=torch.int64)
        img = Image.open(img_path)
        img_tensor = self._transform(img)
        label_onehot = F.one_hot(label, num_classes=self._num_classes).float()
        return img_tensor, label_onehot