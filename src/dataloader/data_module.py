from .gtsrb_dataset import GTSRBDataset
from pathlib import Path
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
import torch

class GTSRBDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, num_classes:int = 100):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self._num_classes = num_classes

    def setup(self, stage: str):
        self.gtsrb_test = GTSRBDataset(self.data_dir, mode="Test", num_classes=self._num_classes)
        mnist_full = GTSRBDataset(self.data_dir, mode="Train", num_classes=self._num_classes)
        generator = torch.Generator().manual_seed(42)
        self.gtsrb_train, self.gtsrb_val = random_split(mnist_full, [0.8, 0.2], generator=generator)

    def train_dataloader(self):
        return DataLoader(self.gtsrb_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.gtsrb_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.gtsrb_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.gtsrb_test, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass