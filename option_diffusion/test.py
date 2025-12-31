import hydra
import torch
import logging
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from pathlib import Path

from option_diffusion.data_utils import S3Config, S3DataLoader, load_multiline_panel
from option_diffusion.model import TinyCondEpsNet, TinyCondDDPM

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    logger.info("="*50)
    logger.info("Starting Tiny Conditional DDPM Inference")
    logger.info("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    checkpoint_path = Path(cfg.inference.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model_config = checkpoint["config"]
    stats = checkpoint["stats"]
    
    if "feature_columns" in checkpoint:
        feature_cols = checkpoint["feature_columns"]
        target_col = checkpoint["target_column"]
    else:
        feature_cols = stats["feature_columns"]
        target_col = stats["target_column"]
    
    logger.info(f"Model config: {model_config}")
    logger.info(f"Features: {feature_cols}")
    logger.info(f"Target: {target_col}")
    
    x_dim = len(feature_cols)
    model = TinyCondEpsNet(
        x_dim=x_dim,
        t_emb_dim=model_config.get("t_emb_dim", 32),
        hidden=model_config.get("hidden", 64)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    ddpm = TinyCondDDPM(
        model=model,
        T=model_config["T"],
        beta_start=model_config["beta_start"],
        beta_end=model_config["beta_end"],
        device=device
    )
    
    logger.info("Loading multiline panel data...")
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
            bucket=cfg.s3.bucket
        )
        df = load_multiline_panel(s3_config=s3_config, key=cfg.data.s3_key)
    
    X = df[feature_cols].values.astype("float32")
    X_standardized = (X - stats["X_mu"]) / stats["X_sd"]
    X_tensor = torch.tensor(X_standardized, dtype=torch.float32, device=device)
    
    y_actual = df[target_col].values
    
    n_samples = cfg.inference.n_samples
    logger.info(f"Generating predictions with {n_samples} DDPM samples...")
    
    with torch.no_grad():
        y_pred_standardized = ddpm.sample(X_tensor, n_samples=n_samples)
    
    y_pred = y_pred_standardized.cpu().numpy() * stats["y_sd"] + stats["y_mu"]
    y_pred = y_pred.reshape(-1)
    
    mse = np.mean((y_pred - y_actual)**2)
    mae = np.mean(np.abs(y_pred - y_actual))
    
    logger.info(f"Test MSE: {mse:.4f}")
    logger.info(f"Test MAE: {mae:.4f}")
    
    results_df = df.copy()
    results_df["y_actual"] = y_actual
    results_df["y_pred"] = y_pred
    results_df["error"] = y_pred - y_actual
    results_df["abs_error"] = np.abs(y_pred - y_actual)
    
    logger.info("\nSample predictions (last 5 rows):")
    logger.info(results_df[["date", "y_actual", "y_pred", "error"]].tail())
    
    results_path = Path("results") / "predictions.csv"
    results_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nSaved predictions to: {results_path}")
    
    if cfg.inference.output.save_to_s3 and hasattr(cfg, 's3') and cfg.s3.bucket:
        s3_config = S3Config(
            endpoint=cfg.s3.endpoint,
            access_key=cfg.s3.access_key,
            secret_key=cfg.s3.secret_key,
            region=cfg.s3.region,
            bucket=cfg.s3.bucket
        )
        s3_key = f"results/predictions_{checkpoint_path.stem}.csv"
        loader = S3DataLoader(s3_config)
        loader.save_csv_to_s3(results_df, s3_key)
        logger.info(f"Uploaded results to s3://{s3_config.bucket}/{s3_key}")
    
    logger.info("\nInference completed!")


if __name__ == "__main__":
    main()
