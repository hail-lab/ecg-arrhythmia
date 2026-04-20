"""Training loops: class-weighted CE for deep models, balanced-CW for classical ML."""

from __future__ import annotations
import time
import joblib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

from .config import (
    log, MODELS_DIR, NUM_CLASSES, RANDOM_SEED,
    BATCH_SIZE, LEARNING_RATE, EPOCHS,
    EARLY_STOP_PATIENCE, PLATEAU_PATIENCE, PLATEAU_FACTOR,
    set_global_seed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def class_weights(y: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    classes = np.arange(num_classes)
    present = np.unique(y)
    w = np.ones(num_classes, dtype=np.float32)
    cw = compute_class_weight("balanced", classes=present, y=y)
    for c, weight in zip(present, cw):
        w[c] = float(weight)
    return w


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
                 shuffle: bool) -> DataLoader:
    X_t = torch.from_numpy(X).float().unsqueeze(1)   # (N, 1, T)
    y_t = torch.from_numpy(y).long()
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


# ---------------------------------------------------------------------------
# Deep model training
# ---------------------------------------------------------------------------
def train_dl(model: nn.Module, X_tr: np.ndarray, y_tr: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = EPOCHS, lr: float = LEARNING_RATE,
             batch_size: int = BATCH_SIZE,
             tag: str = "model") -> Dict:
    set_global_seed(RANDOM_SEED)
    device = get_device()
    model = model.to(device)

    tr_loader  = _make_loader(X_tr,  y_tr,  batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, batch_size, shuffle=False)

    cw = torch.from_numpy(class_weights(y_tr)).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE,
    )

    best_val, best_state, bad = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tr_loss += float(loss.item()) * xb.size(0)
        tr_loss /= len(tr_loader.dataset)

        # --- val ---
        model.eval()
        v_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                v_loss += float(criterion(logits, yb).item()) * xb.size(0)
                correct += int((logits.argmax(1) == yb).sum().item())
                total   += xb.size(0)
        v_loss /= total
        v_acc = correct / total

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        scheduler.step(v_loss)
        log.info(f"[{tag}] epoch {epoch:3d} | train {tr_loss:.4f} | "
                 f"val {v_loss:.4f} | val_acc {v_acc:.4f}")

        if v_loss < best_val - 1e-4:
            best_val   = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= EARLY_STOP_PATIENCE:
                log.info(f"[{tag}] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODELS_DIR / f"{tag}.pt")
    log.info(f"[{tag}] done in {time.time()-t0:.1f}s -> {MODELS_DIR/f'{tag}.pt'}")
    return {"history": history, "best_val_loss": best_val}


# ---------------------------------------------------------------------------
# Classical ML training
# ---------------------------------------------------------------------------
def train_ml(model, X_tr: np.ndarray, y_tr: np.ndarray, tag: str = "model"):
    set_global_seed(RANDOM_SEED)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    joblib.dump(model, MODELS_DIR / f"{tag}.joblib")
    log.info(f"[{tag}] fit in {time.time()-t0:.1f}s -> {MODELS_DIR/f'{tag}.joblib'}")
    return model


# ---------------------------------------------------------------------------
# Prediction helpers (unified shape: returns logits or probabilities)
# ---------------------------------------------------------------------------
def predict_dl(model: nn.Module, X: np.ndarray,
               batch_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    device = get_device()
    model = model.to(device).eval()
    loader = _make_loader(X, np.zeros(len(X), dtype=np.int64), batch_size, shuffle=False)
    all_prob, all_pred = [], []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            all_prob.append(prob)
            all_pred.append(prob.argmax(axis=1))
    return np.concatenate(all_pred), np.concatenate(all_prob)


def predict_ml(model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    prob = model.predict_proba(X)
    # XGBoost may emit num_class < NUM_CLASSES if labels absent — pad defensively
    if prob.shape[1] != NUM_CLASSES:
        padded = np.zeros((prob.shape[0], NUM_CLASSES), dtype=prob.dtype)
        padded[:, :prob.shape[1]] = prob
        prob = padded
    return prob.argmax(axis=1), prob
