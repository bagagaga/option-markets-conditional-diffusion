import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, Tuple
from dataclasses import dataclass

from .constants import (
    SINUSOIDAL_EMBEDDING_BASE,
    WEIGHT_INIT_SCALE,
    DEFAULT_N_DDPM_SAMPLES,
    ALPHA_CONSTANT,
    ZERO_CONSTANT,
    ONE_CONSTANT,
)

logger = logging.getLogger(__name__)


def make_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2, device: str = "cpu"):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = ALPHA_CONSTANT - betas
    alpha_bar = torch.cumprod(alphas, dim=ZERO_CONSTANT)
    return betas, alphas, alpha_bar


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(SINUSOIDAL_EMBEDDING_BASE, device=t.device)) 
        * torch.arange(half, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=ONE_CONSTANT)
    if dim % 2 == ONE_CONSTANT:
        emb = torch.cat([emb, torch.zeros((t.shape[ZERO_CONSTANT], ONE_CONSTANT), device=t.device)], dim=ONE_CONSTANT)
    return emb


class TinyCondEpsNet(nn.Module):
    def __init__(self, x_dim: int, t_emb_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.x_dim = x_dim
        self.hidden = hidden
        
        input_dim = ONE_CONSTANT + x_dim + t_emb_dim
        self.fc1_weight = nn.Parameter(torch.randn(hidden, input_dim) * WEIGHT_INIT_SCALE)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden))
        self.fc2_weight = nn.Parameter(torch.randn(ONE_CONSTANT, hidden) * WEIGHT_INIT_SCALE)
        self.fc2_bias = nn.Parameter(torch.zeros(ONE_CONSTANT))
    
    def forward(self, y_t: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        te = timestep_embedding(t, self.t_emb_dim)
        inp = torch.cat([y_t, x, te], dim=ONE_CONSTANT)
        
        h = torch.zeros(inp.shape[ZERO_CONSTANT], self.hidden, device=inp.device)
        for i in range(self.hidden):
            h[:, i] = torch.sum(inp * self.fc1_weight[i], dim=ONE_CONSTANT) + self.fc1_bias[i]
        h = torch.relu(h)
        
        out = torch.zeros(h.shape[ZERO_CONSTANT], ONE_CONSTANT, device=h.device)
        out[:, ZERO_CONSTANT] = torch.sum(h * self.fc2_weight[ZERO_CONSTANT], dim=ONE_CONSTANT) + self.fc2_bias[ZERO_CONSTANT]
        
        return out


class TinyCondDDPM:
    def __init__(
        self,
        model: TinyCondEpsNet,
        T: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str = "cpu"
    ):
        self.model = model
        self.T = T
        self.device = device
        
        self.betas, self.alphas, self.alpha_bar = make_beta_schedule(
            T, beta_start, beta_end, device
        )
    
    def q_sample(self, y0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a_bar_t = self.alpha_bar[t].unsqueeze(ONE_CONSTANT)
        return torch.sqrt(a_bar_t) * y0 + torch.sqrt(ALPHA_CONSTANT - a_bar_t) * noise
    
    @torch.no_grad()
    def p_sample(self, y: torch.Tensor, t: int, x: torch.Tensor) -> torch.Tensor:
        B = y.shape[ZERO_CONSTANT]
        tt = torch.full((B,), t, device=self.device, dtype=torch.long)
        
        eps_hat = self.model(y, tt, x)
        
        a_t = self.alphas[t]
        ab_t = self.alpha_bar[t]
        beta_t = self.betas[t]
        
        mean = (ALPHA_CONSTANT / torch.sqrt(a_t)) * (y - (beta_t / torch.sqrt(ALPHA_CONSTANT - ab_t)) * eps_hat)
        
        if t > ZERO_CONSTANT:
            z = torch.randn_like(y)
            return mean + torch.sqrt(beta_t) * z
        else:
            return mean
    
    @torch.no_grad()
    def sample(self, x: torch.Tensor, n_samples: int = DEFAULT_N_DDPM_SAMPLES) -> torch.Tensor:
        self.model.eval()
        B = x.shape[ZERO_CONSTANT]
        ys = []
        
        for _ in range(n_samples):
            y = torch.randn((B, ONE_CONSTANT), device=self.device)
            
            for t in reversed(range(self.T)):
                y = self.p_sample(y, t, x)
            
            ys.append(y)
        
        y0 = torch.stack(ys, dim=ZERO_CONSTANT).mean(dim=ZERO_CONSTANT)
        return y0
    
    def train_step(
        self,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> float:
        self.model.train()
        
        B = y_batch.shape[ZERO_CONSTANT]
        t = torch.randint(ZERO_CONSTANT, self.T, (B,), device=self.device)
        
        noise = torch.randn_like(y_batch)
        y_noisy = self.q_sample(y_batch, t, noise)
        
        noise_pred = self.model(y_noisy, t, X_batch)
        
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def validate(self, X_val: torch.Tensor, y_val: torch.Tensor) -> float:
        self.model.eval()
        
        B = y_val.shape[ZERO_CONSTANT]
        t = torch.randint(ZERO_CONSTANT, self.T, (B,), device=self.device)
        
        noise = torch.randn_like(y_val)
        y_noisy = self.q_sample(y_val, t, noise)
        
        noise_pred = self.model(y_noisy, t, X_val)
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        return loss.item()


def create_model(x_dim: int, config: Dict[str, Any]) -> Tuple[TinyCondEpsNet, TinyCondDDPM]:
    model = TinyCondEpsNet(
        x_dim=x_dim,
        t_emb_dim=config.get("t_emb_dim", 32),
        hidden=config.get("hidden", 64)
    )
    
    ddpm = TinyCondDDPM(
        model=model,
        T=config.get("T", 50),
        beta_start=config.get("beta_start", 1e-4),
        beta_end=config.get("beta_end", 2e-2),
        device=config.get("device", "cpu")
    )
    
    logger.info(f"Created TinyCondEpsNet: x_dim={x_dim}, hidden={config.get('hidden', 64)}, T={config.get('T', 50)}")
    
    return model, ddpm
