import numpy as np
import pandas as pd
import torch
from typing import Tuple, Dict


def ols_slope(y: np.ndarray, x: np.ndarray, ridge: float = 0.0) -> float:
    y = np.asarray(y)
    x = np.asarray(x)
    
    num = np.nanmean((y - np.nanmean(y)) * (x - np.nanmean(x)))
    den = np.nanvar(x) + ridge
    
    return 0.0 if den <= 0 else num / den


def hedge_and_pnl_multi(
    meta_train: pd.DataFrame,
    Y_train: torch.Tensor,
    meta_test: pd.DataFrame,
    Y_test: torch.Tensor,
    samples: torch.Tensor,
    roll_window: int = 60,
    ridge: float = 0.0
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    Y_train_np = Y_train.numpy() if isinstance(Y_train, torch.Tensor) else Y_train
    Y_test_np = Y_test.numpy() if isinstance(Y_test, torch.Tensor) else Y_test
    samples_np = samples.numpy() if isinstance(samples, torch.Tensor) else samples
    
    dC_train = Y_train_np[:, 0]
    dF_train = Y_train_np[:, 1]
    dC_test = Y_test_np[:, 0]
    dF_test = Y_test_np[:, 1]
    
    dates_test = pd.to_datetime(meta_test["date"].values) if "date" in meta_test.columns else pd.RangeIndex(len(dC_test))
    
    h_hist = ols_slope(dC_train, dF_train, ridge=ridge)
    
    dC_scenarios = samples_np[:, :, 0]
    dF_scenarios = samples_np[:, :, 1]
    num = np.sum(dC_scenarios * dF_scenarios, axis=1)
    den = np.sum(dF_scenarios * dF_scenarios, axis=1) + ridge
    h_scenario = np.where(den > 0, num / den, 0.0)
    
    meta_train_copy = meta_train.copy()
    if "date" in meta_train.columns:
        meta_train_copy["date"] = pd.to_datetime(meta_train_copy["date"])
    
    df_hist = pd.DataFrame({
        "date": pd.concat([
            meta_train_copy["date"] if "date" in meta_train_copy.columns else pd.Series(range(len(dC_train))),
            pd.Series(dates_test) if isinstance(dates_test, pd.DatetimeIndex) else pd.Series(range(len(dC_train), len(dC_train) + len(dC_test)))
        ], ignore_index=True),
        "dC": np.concatenate([dC_train, dC_test]),
        "dF": np.concatenate([dF_train, dF_test]),
    }).reset_index(drop=True)
    
    date_to_idx = {d: i for i, d in enumerate(df_hist["date"])}
    h_rolling = []
    
    for d in dates_test:
        i = date_to_idx.get(d, len(dC_train) + len(h_rolling))
        lo = max(0, i - roll_window)
        up = i
        
        if up - lo < 10:
            h_rolling.append(h_hist)
        else:
            h_rolling.append(ols_slope(
                df_hist["dC"].iloc[lo:up].values,
                df_hist["dF"].iloc[lo:up].values,
                ridge=ridge
            ))
    
    h_rolling = np.array(h_rolling)
    
    h_zero = np.zeros_like(h_scenario)
    h_hist_arr = np.full_like(h_scenario, h_hist)
    
    pnl_unhedged = dC_test
    pnl_zero = dC_test - h_zero * dF_test
    pnl_hist = dC_test - h_hist_arr * dF_test
    pnl_rolling = dC_test - h_rolling * dF_test
    pnl_scenario = dC_test - h_scenario * dF_test
    
    per_day = pd.DataFrame({
        "date": dates_test,
        "dC": dC_test,
        "dF": dF_test,
        "h_hist": h_hist_arr,
        "h_rolling": h_rolling,
        "h_scenario": h_scenario,
        "h_zero": h_zero,
        "pnl_unhedged": pnl_unhedged,
        "pnl_hist": pnl_hist,
        "pnl_rolling": pnl_rolling,
        "pnl_scenario": pnl_scenario,
        "pnl_zero": pnl_zero
    }).sort_values("date").reset_index(drop=True)
    
    def compute_stats(pnl: np.ndarray) -> Tuple:
        mu = np.nanmean(pnl)
        sd = np.nanstd(pnl)
        sharpe = (mu / (sd + 1e-12)) * np.sqrt(252.0)
        hit_rate = np.mean(pnl > 0)
        p5, p95 = np.nanpercentile(pnl, [5, 95])
        return mu, sd, sharpe, hit_rate, p5, p95
    
    summary = pd.DataFrame(
        [
            ("UNHEDGED",) + compute_stats(per_day["pnl_unhedged"]),
            ("ZERO",) + compute_stats(per_day["pnl_zero"]),
            ("HIST_OLS",) + compute_stats(per_day["pnl_hist"]),
            ("ROLL_OLS",) + compute_stats(per_day["pnl_rolling"]),
            ("SCENARIO_DDPM",) + compute_stats(per_day["pnl_scenario"])
        ],
        columns=["strategy", "mean", "stdev", "sharpe_ann", "hit_rate", "p5", "p95"]
    ).sort_values("sharpe_ann", ascending=False).reset_index(drop=True)
    
    baselines = {"h_hist_global": h_hist}
    
    return per_day, summary, baselines
