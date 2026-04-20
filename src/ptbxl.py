"""PTB-XL 12-lead cross-dataset evaluation (optional, deep models only).

Downloads PTB-XL (~1.7 GB) and evaluates ResNet1D/CNN1D adapted to 12-channel
1000-sample inputs. Returns a metrics DataFrame.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import ast

import numpy as np
import pandas as pd

from .config import DATA_RAW, log, set_global_seed, RANDOM_SEED, TABLES
from .evaluate import compute_metrics, save_table

PTBXL_DIR = DATA_RAW / "ptbxl"
PTBXL_CLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]


def download_ptbxl() -> Path:
    import wfdb
    PTBXL_DIR.mkdir(parents=True, exist_ok=True)
    marker = PTBXL_DIR / "ptbxl_database.csv"
    if marker.exists():
        log.info(f"PTB-XL already present at {PTBXL_DIR}")
        return PTBXL_DIR
    log.info(f"Downloading PTB-XL to {PTBXL_DIR} (this can take several minutes) ...")
    wfdb.dl_database("ptb-xl/1.0.3", str(PTBXL_DIR))
    return PTBXL_DIR


def _aggregate_superclass(scp_codes: dict, agg_df: pd.DataFrame):
    labels = set()
    for code in scp_codes:
        if code in agg_df.index:
            sc = agg_df.loc[code].diagnostic_class
            if isinstance(sc, str):
                labels.add(sc)
    return list(labels)


def load_ptbxl(sampling_rate: int = 100) -> Tuple[np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray]:
    import wfdb
    base = download_ptbxl()
    meta = pd.read_csv(base / "ptbxl_database.csv", index_col="ecg_id")
    meta.scp_codes = meta.scp_codes.apply(lambda x: ast.literal_eval(x))
    agg = pd.read_csv(base / "scp_statements.csv", index_col=0)
    agg = agg[agg.diagnostic == 1]
    meta["superclass"] = meta.scp_codes.apply(lambda sc: _aggregate_superclass(sc, agg))
    meta = meta[meta.superclass.map(len) == 1].copy()
    meta["label"] = meta.superclass.apply(lambda s: PTBXL_CLASSES.index(s[0]))

    files = meta.filename_lr if sampling_rate == 100 else meta.filename_hr
    X = np.stack([wfdb.rdsamp(str(base / f))[0].astype(np.float32) for f in files], axis=0)
    # Per-lead z-score
    mean = X.mean(axis=1, keepdims=True); std = X.std(axis=1, keepdims=True) + 1e-6
    X = (X - mean) / std
    X = X.transpose(0, 2, 1)             # (N, 12, T)

    tr = meta.strat_fold.isin(range(1, 9)).values
    te = (meta.strat_fold == 10).values
    return X[tr], meta.label.values[tr], X[te], meta.label.values[te]


# ------------------------------------------------------------
# PTB-XL-adapted deep models (12 input channels, longer sequences)
# ------------------------------------------------------------
def build_ptbxl_resnet1d():
    import torch.nn as nn
    from .models import ResNet1D
    m = ResNet1D(in_channels=12, num_classes=len(PTBXL_CLASSES))
    return m


def build_ptbxl_cnn1d():
    from .models import CNN1D
    return CNN1D(in_channels=12, num_classes=len(PTBXL_CLASSES))


# ------------------------------------------------------------
# Train + evaluate wrapper
# ------------------------------------------------------------
def run_ptbxl(epochs: int = 20) -> pd.DataFrame:
    from .train import train_dl, predict_dl

    set_global_seed(RANDOM_SEED)
    X_tr, y_tr, X_te, y_te = load_ptbxl(sampling_rate=100)
    log.info(f"PTB-XL shapes: train {X_tr.shape} | test {X_te.shape}")

    rows = []
    for name, builder in [("ResNet1D", build_ptbxl_resnet1d),
                          ("CNN1D",    build_ptbxl_cnn1d)]:
        log.info(f"[ptbxl] training {name}")
        model = builder()
        train_dl(model, X_tr, y_tr, X_te, y_te, epochs=epochs, tag=f"ptbxl_{name}")
        y_pred, y_prob = predict_dl(model, X_te)

        # compute_metrics expects NUM_CLASSES-sized probs; ptbxl has 5 -> same size
        from .config import NUM_CLASSES
        assert y_prob.shape[1] == NUM_CLASSES == len(PTBXL_CLASSES)
        m = compute_metrics(y_te, y_pred, y_prob)
        rows.append({"model": name, **m})

    df = pd.DataFrame(rows)
    save_table(df, "ptbxl")
    return df
