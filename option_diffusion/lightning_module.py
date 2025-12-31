import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Tuple

from .model import TinyCondEpsNet, TinyCondDDPM
from .constants import ZERO_CONSTANT, ONE_CONSTANT


class DiffusionLightningModule(pl.LightningModule):
    def __init__(
        self,
        x_dim: int,
        hidden_dim: int = 64,
        t_emb_dim: int = 32,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = TinyCondEpsNet(
            x_dim=x_dim,
            t_emb_dim=t_emb_dim,
            hidden=hidden_dim
        )
        
        self.ddpm = TinyCondDDPM(
            model=self.model,
            T=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=self.device
        )
        
        self.lr = lr
        self.weight_decay = weight_decay
    
    def forward(self, y_t: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.model(y_t, t, x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        X_batch, y_batch = batch
        
        B = y_batch.shape[ZERO_CONSTANT]
        t = torch.randint(ZERO_CONSTANT, self.ddpm.T, (B,), device=self.device)
        
        noise = torch.randn_like(y_batch)
        y_noisy = self.ddpm.q_sample(y_batch, t, noise)
        
        noise_pred = self.model(y_noisy, t, X_batch)
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        X_batch, y_batch = batch
        
        B = y_batch.shape[ZERO_CONSTANT]
        t = torch.randint(ZERO_CONSTANT, self.ddpm.T, (B,), device=self.device)
        
        noise = torch.randn_like(y_batch)
        y_noisy = self.ddpm.q_sample(y_batch, t, noise)
        
        noise_pred = self.model(y_noisy, t, X_batch)
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
    
    def on_train_start(self):
        self.ddpm.device = self.device
        self.ddpm.betas = self.ddpm.betas.to(self.device)
        self.ddpm.alphas = self.ddpm.alphas.to(self.device)
        self.ddpm.alpha_bar = self.ddpm.alpha_bar.to(self.device)
