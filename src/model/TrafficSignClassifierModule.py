from typing import Any, Optional
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from src.model.simple_cnn import SimpleCNN
import torch
import torch.nn.functional as F

class TrafficSignClassifier(pl.LightningModule):
    def __init__(self, n_classes: int=100) -> None:
        super().__init__()
        self._model = SimpleCNN(n_classes=n_classes)
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        img, label = batch
        pred = self._model(img)
        loss = F.mse_loss(pred, label)        
        self.log("train_loss",loss.item(), prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        img, label = batch
        pred = self._model(img)
        loss = F.mse_loss(pred, label)
        self.log("val_loss",loss.item(), prog_bar=True)
        return loss
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        img, label = batch
        pred = self._model(img)
        label = torch.argmax(pred)
        return {'img':img, "label":label}
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer