import os
import io
import logging
import pandas as pd
import numpy as np
import torch
import boto3
import ssl
import urllib.request
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class S3Config:
    endpoint: str = os.getenv("YC_S3_ENDPOINT", "https://storage.yandexcloud.net")
    access_key: Optional[str] = os.getenv("YC_S3_ACCESS_KEY")
    secret_key: Optional[str] = os.getenv("YC_S3_SECRET_KEY")
    region: str = os.getenv("YC_S3_REGION", "ru-central1")
    bucket: Optional[str] = os.getenv("YC_S3_BUCKET")


class S3DataLoader:
    def __init__(self, s3_config: S3Config):
        self.config = s3_config
        self.s3_client = boto3.client(
            's3',
            endpoint_url=s3_config.endpoint,
            aws_access_key_id=s3_config.access_key,
            verify=False,
            aws_secret_access_key=s3_config.secret_key,
            region_name=s3_config.region
        )
    
    def load_csv_from_s3(self, s3_key: str) -> pd.DataFrame:
        try:
            logger.info(f"Loading from s3://{self.config.bucket}/{s3_key}")
            obj = self.s3_client.get_object(Bucket=self.config.bucket, Key=s3_key)
            body = obj["Body"].read().decode("utf-8")
            df = pd.read_csv(io.StringIO(body))
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading from S3: {e}")
            raise
    
    def save_csv_to_s3(self, df: pd.DataFrame, s3_key: str):
        try:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            
            self.s3_client.put_object(
                Bucket=self.config.bucket,
                Key=s3_key,
                Body=csv_buffer.getvalue()
            )
            logger.info(f"Saved to s3://{self.config.bucket}/{s3_key}")
        except Exception as e:
            logger.error(f"Error saving to S3: {e}")
            raise


def load_multiline_panel(url: Optional[str] = None, s3_config: Optional[S3Config] = None, key: str = "baselines/spx_multiline_panel.csv") -> pd.DataFrame:
    if url:
        logger.info(f"Loading from public URL: {url}")
        import requests
        response = requests.get(url, verify=False)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        logger.info(f"Loaded multiline panel: {len(df)} rows")
        return df
    
    if s3_config is None:
        raise ValueError("Either 'url' or 's3_config' must be provided")
    
    loader = S3DataLoader(s3_config)
    df = loader.load_csv_from_s3(key)
    logger.info(f"Loaded multiline panel: {len(df)} rows")
    return df


def prepare_tensors(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = "dC_next"
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    X = df[feature_columns].values.astype("float32")
    y = df[target_column].values.astype("float32").reshape(-1, 1)
    
    X_mu, X_sd = X.mean(0), X.std(0) + 1e-8
    y_mu, y_sd = y.mean(0), y.std(0) + 1e-8
    
    X_standardized = (X - X_mu) / X_sd
    y_standardized = (y - y_mu) / y_sd
    
    X_tensor = torch.tensor(X_standardized, dtype=torch.float32)
    y_tensor = torch.tensor(y_standardized, dtype=torch.float32)
    
    stats = {
        "feature_columns": feature_columns,
        "target_column": target_column,
        "X_mu": X_mu,
        "X_sd": X_sd,
        "y_mu": float(y_mu[0]),
        "y_sd": float(y_sd[0])
    }
    
    logger.info(f"Standardized features: X shape={X.shape}")
    logger.info(f"Feature means: {X_mu[:3]} ... (showing first 3)")
    logger.info(f"Target: y_mu={y_mu[0]:.4f}, y_sd={y_sd[0]:.4f}")
    
    return X_tensor, y_tensor, stats


def split_train_val(
    df: pd.DataFrame,
    X: torch.Tensor,
    y: torch.Tensor,
    val_frac: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    dates = np.sort(df_copy["date"].unique())
    
    cut = int(len(dates) * (1 - val_frac))
    train_dates = set(dates[:cut])
    is_train = df_copy["date"].isin(train_dates).values
    
    X_train, y_train = X[is_train], y[is_train]
    X_val, y_val = X[~is_train], y[~is_train]
    
    logger.info(f"Train: {len(X_train)} samples from {len(train_dates)} dates")
    logger.info(f"Val: {len(X_val)} samples from {len(dates) - len(train_dates)} dates")
    logger.info(f"Date range: {dates[0]} to {dates[-1]}")
    
    return X_train, y_train, X_val, y_val, is_train
