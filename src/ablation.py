"""Ablation runner for the four ResNet1D variants (A1-A4)."""

from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import log, set_global_seed, RANDOM_SEED
from .models import ResNet1D
from .train import train_dl, predict_dl
from .evaluate import compute_metrics, per_class_f1, save_table


ABLATION_CONFIGS: List[Dict] = [
    {"name": "Full",           "kwargs": dict()},
    {"name": "A1_NoResidual",  "kwargs": dict(use_residual=False)},
    {"name": "A2_Shallow",     "kwargs": dict(shallow=True)},
    {"name": "A3_NoBatchNorm", "kwargs": dict(use_bn=False)},
    {"name": "A4_NoDropout",   "kwargs": dict(dropout=0.0)},
]


def run_ablation(X_tr, y_tr, X_te, y_te, epochs: int = 50) -> pd.DataFrame:
    rows = []
    for cfg in ABLATION_CONFIGS:
        tag  = f"ablation_{cfg['name']}"
        log.info(f"[ablation] {cfg['name']} -> kwargs={cfg['kwargs']}")
        set_global_seed(RANDOM_SEED)
        model = ResNet1D(**cfg["kwargs"])
        train_dl(model, X_tr, y_tr, X_te, y_te, epochs=epochs, tag=tag)

        y_pred, y_prob = predict_dl(model, X_te)
        m  = compute_metrics(y_te, y_pred, y_prob)
        pc = per_class_f1(y_te, y_pred)
        rows.append({"variant": cfg["name"], **m, **pc})

    df = pd.DataFrame(rows)
    save_table(df, "ablation")
    return df
