from pathlib import Path
import torch
import lightning.pytorch as pl
from typing import Dict
from lightning.pytorch.loggers import TensorBoardLogger
import sys

sys.path.insert(0, "/workspaces/deep_learning/Traffic_sign_classifier")
from src.dataloader.data_module import GTSRBDataModule
from src.model.TrafficSignClassifierModule import TrafficSignClassifier
import argparse
import yaml
from dataclasses import dataclass


@dataclass
class TrainConfig:
    dataset_root: Path
    max_epochs: int
    batch_size: int
    optimizer: Dict


torch.set_float32_matmul_precision("high")


def train(cfg: TrainConfig) -> None:
    logger = TensorBoardLogger("tb_logs", name="traffic_sign_classifier")
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        log_every_n_steps=5,
        logger=logger,
        accelerator="gpu",
        devices=1,
        strategy="auto",
        enable_model_summary=True,
    )
    model = TrafficSignClassifier(n_classes=100)
    data_module = GTSRBDataModule(Path("/data/GTSRB"), batch_size=cfg.batch_size)
    trainer.fit(model, datamodule=data_module)
    print(cfg.dataset_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        help="path to training config",
        default="scripts/config.yaml",
    )
    args = parser.parse_args()

    with args.config.open("r") as cfg_fp:
        cfg = yaml.safe_load(cfg_fp)
        train_cfg = TrainConfig(**cfg)

    train(train_cfg)
