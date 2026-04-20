"""Metrics, confusion matrices, ROC curves, and significance tests."""

from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

from .config import AAMI_CLASSES, NUM_CLASSES, FIGURES, TABLES, log


# ============================================================
# Core metrics
# ============================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
    )
    f_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Macro AUC: one-vs-rest, guard against classes absent from y_true
    try:
        present = np.unique(y_true)
        if len(present) < 2:
            auc_m = float("nan")
        else:
            y_onehot = np.eye(NUM_CLASSES)[y_true]
            auc_m = roc_auc_score(
                y_onehot[:, present], y_prob[:, present],
                average="macro", multi_class="ovr",
            )
    except Exception as e:
        log.warning(f"AUC computation failed: {e}")
        auc_m = float("nan")

    return {
        "accuracy":    float(acc),
        "precision_M": float(p_m),
        "recall_M":    float(r_m),
        "f1_M":        float(f_m),
        "f1_W":        float(f_w),
        "auc_M":       float(auc_m),
    }


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)),
        average=None, zero_division=0,
    )
    return {f"f1_{c}": float(v) for c, v in zip(AAMI_CLASSES, f1)}


# ============================================================
# Visualisations
# ============================================================
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          tag: str, normalize: bool = True):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    if normalize:
        with np.errstate(all="ignore"):
            cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_n = np.nan_to_num(cm_n)
    else:
        cm_n = cm

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(AAMI_CLASSES)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(AAMI_CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {tag}")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_n[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = FIGURES / f"cm_{tag}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    log.info(f"  -> {out}")


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, tag: str):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for c, name in enumerate(AAMI_CLASSES):
        yt = (y_true == c).astype(int)
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        fpr, tpr, _ = roc_curve(yt, y_prob[:, c])
        ax.plot(fpr, tpr, label=name, linewidth=1.5)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC (one-vs-rest) — {tag}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = FIGURES / f"roc_{tag}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    log.info(f"  -> {out}")


def plot_model_comparison(results_df: pd.DataFrame):
    metrics = ["accuracy", "f1_M", "f1_W", "auc_M"]
    labels  = ["Acc", "Macro F1", "Weighted F1", "Macro AUC"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(results_df))
    w = 0.2
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        ax.bar(x + (i - 1.5) * w, results_df[m], width=w, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(results_df["model"], rotation=20)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Score")
    ax.set_title("Model comparison — MIT-BIH DS2 (inter-patient)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGURES / "model_comparison.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    log.info(f"  -> {out}")


def plot_per_class_f1(results_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(AAMI_CLASSES))
    n = len(results_df); w = 0.8 / max(n, 1)
    for i, row in results_df.reset_index(drop=True).iterrows():
        vals = [row.get(f"f1_{c}", 0.0) for c in AAMI_CLASSES]
        ax.bar(x + (i - (n - 1) / 2) * w, vals, width=w, label=row["model"])
    ax.set_xticks(x); ax.set_xticklabels(AAMI_CLASSES)
    ax.set_ylabel("F1"); ax.set_title("Per-class F1 — MIT-BIH DS2")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGURES / "per_class_f1.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    log.info(f"  -> {out}")


# ============================================================
# Significance tests
# ============================================================
def wilcoxon_correctness(pred_a: np.ndarray, pred_b: np.ndarray,
                         y_true: np.ndarray) -> float:
    a = (pred_a == y_true).astype(int)
    b = (pred_b == y_true).astype(int)
    if np.array_equal(a, b):
        return 1.0
    try:
        _, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(p)
    except Exception:
        return float("nan")


def mcnemar_test(pred_a: np.ndarray, pred_b: np.ndarray,
                 y_true: np.ndarray) -> Tuple[float, int, int]:
    a = (pred_a == y_true); b = (pred_b == y_true)
    b01 = int(((a) & (~b)).sum())   # A correct, B wrong
    b10 = int(((~a) & (b)).sum())   # A wrong, B correct
    table = [[0, b01], [b10, 0]]
    try:
        res = mcnemar(table, exact=(b01 + b10) < 25, correction=True)
        return float(res.pvalue), b01, b10
    except Exception:
        return float("nan"), b01, b10


def save_table(df: pd.DataFrame, name: str):
    path = TABLES / f"{name}.csv"
    df.to_csv(path, index=False)
    log.info(f"  -> {path}")
