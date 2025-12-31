import logging
from pathlib import Path
from typing import Any, Dict
import os

import hydra
import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import DictConfig, OmegaConf

from option_diffusion.data_utils import (
    S3Config,
    load_multiline_panel,
    prepare_tensors,
    split_train_val,
)
from option_diffusion.lightning_module import DiffusionLightningModule

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    logger.info("=" * 70)
    logger.info("Starting Tiny Conditional DDPM Training with PyTorch Lightning")
    logger.info("=" * 70)

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)
    pl.seed_everything(cfg.experiment.seed)

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    logger.info(f"Using device: {device}")

    logger.info("Pulling data with DVC...")
    os.system("dvc pull")

    logger.info("Loading data...")
    if hasattr(cfg.data, 'url') and cfg.data.url:
        logger.info(f"Using public URL: {cfg.data.url}")
        df = load_multiline_panel(url=cfg.data.url)
    else:
        logger.info("Using S3 configuration...")
        s3_config = S3Config(
            endpoint=cfg.s3.endpoint,
            access_key=cfg.s3.access_key,
            secret_key=cfg.s3.secret_key,
            region=cfg.s3.region,
            bucket=cfg.s3.bucket,
        )
        df = load_multiline_panel(s3_config=s3_config, key=cfg.data.s3_key)

    logger.info("Preparing tensors...")
    X, y, stats = prepare_tensors(df, cfg.features, cfg.target)

    logger.info("Splitting data...")
    X_train, y_train, X_val, y_val, _ = split_train_val(df, X, y, val_frac=cfg.split.val_frac)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.training.batch_size,
        num_workers=0
    )

    logger.info("Creating Lightning module...")
    lightning_model = DiffusionLightningModule(
        x_dim=len(cfg.features),
        hidden_dim=cfg.model.architecture.hidden_dim,
        t_emb_dim=cfg.model.architecture.t_emb_dim,
        timesteps=cfg.model.diffusion.timesteps,
        beta_start=cfg.model.diffusion.beta_start,
        beta_end=cfg.model.diffusion.beta_end,
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
    )

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_name=cfg.mlflow.run_name,
    )
    
    if cfg.mlflow.log_params:
        mlflow_logger.log_hyperparams({
            "model_name": cfg.model.name,
            "hidden_dim": cfg.model.architecture.hidden_dim,
            "t_emb_dim": cfg.model.architecture.t_emb_dim,
            "timesteps": cfg.model.diffusion.timesteps,
            "beta_start": cfg.model.diffusion.beta_start,
            "beta_end": cfg.model.diffusion.beta_end,
            "optimizer": cfg.training.optimizer.name,
            "learning_rate": cfg.training.optimizer.lr,
            "weight_decay": cfg.training.optimizer.weight_decay,
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "n_features": len(cfg.features),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "val_frac": cfg.split.val_frac,
            "target": cfg.target,
            "seed": cfg.experiment.seed,
            "device": cfg.device,
        })

    checkpoint_dir = Path(cfg.checkpoint.dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        check_val_every_n_epoch=cfg.training.val_every,
        enable_progress_bar=True,
    )

    logger.info(f"Training for {cfg.training.epochs} epochs...")
    trainer.fit(lightning_model, train_loader, val_loader)

    final_path = checkpoint_dir / "final_model.pt"
    final_checkpoint = {
        "epoch": cfg.training.epochs,
        "model_state_dict": lightning_model.model.state_dict(),
        "config": {
            "t_emb_dim": cfg.model.architecture.t_emb_dim,
            "hidden": cfg.model.architecture.hidden_dim,
            "T": cfg.model.diffusion.timesteps,
            "beta_start": cfg.model.diffusion.beta_start,
            "beta_end": cfg.model.diffusion.beta_end,
            "device": device,
        },
        "stats": stats,
        "feature_columns": cfg.features,
        "target_column": cfg.target,
    }
    torch.save(final_checkpoint, final_path)
    logger.info(f"Saved final model: {final_path}")

    if cfg.mlflow.log_artifacts:
        mlflow_logger.experiment.log_artifact(
            mlflow_logger.run_id,
            str(final_path)
        )

    logger.info("=" * 70)
    logger.info("Training completed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
