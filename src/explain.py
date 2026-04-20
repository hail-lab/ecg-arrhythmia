"""Unified SHAP + Grad-CAM pipeline.

Deep models:   SHAP GradientExplainer -> per-class mean |phi| over 260 samples
Tree models:   SHAP TreeExplainer -> beeswarm over the same 260-sample space
ResNet1D:      Grad-CAM on final residual block
"""

from __future__ import annotations
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .config import AAMI_CLASSES, NUM_CLASSES, WINDOW_SIZE, FIGURES, log
from .train import get_device


# ============================================================
# SHAP — deep models (GradientExplainer)
# ============================================================
def shap_temporal_deep(model, X_background: np.ndarray, X_explain: np.ndarray,
                       y_explain: np.ndarray, tag: str,
                       n_background: int = 128,
                       per_class_samples: int = 200) -> Dict[str, np.ndarray]:
    """Return {class: mean(|phi|) over samples of that class} as 260-d arrays,
    and write one SHAP temporal figure per class."""
    import shap

    device = get_device()
    model = model.to(device).eval()

    rng = np.random.default_rng(0)
    bg_idx = rng.choice(len(X_background), size=min(n_background, len(X_background)), replace=False)
    bg = torch.from_numpy(X_background[bg_idx]).float().unsqueeze(1).to(device)

    explainer = shap.GradientExplainer(model, bg)

    # cuDNN's LSTM backward requires training mode; SHAP puts model in eval.
    # Disabling cuDNN forces the pure-CUDA path which supports eval-mode backward.
    _has_rnn = any(isinstance(m, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN))
                   for m in model.modules())

    profiles: Dict[str, np.ndarray] = {}
    for c, cname in enumerate(AAMI_CLASSES):
        idx = np.where(y_explain == c)[0]
        if len(idx) == 0:
            continue
        take = rng.choice(idx, size=min(per_class_samples, len(idx)), replace=False)
        Xc = torch.from_numpy(X_explain[take]).float().unsqueeze(1).to(device)

        if _has_rnn:
            with torch.backends.cudnn.flags(enabled=False):
                sv = explainer.shap_values(Xc)
        else:
            sv = explainer.shap_values(Xc)
        # shap returns list[num_classes] of arrays (N, 1, T) OR a single (N, 1, T, C)
        if isinstance(sv, list):
            phi_c = sv[c]                               # (N, 1, T)
        else:
            phi_c = sv[..., c]                          # (N, 1, T)
        phi_c = np.asarray(phi_c).squeeze(1)            # (N, T)
        profile = np.mean(np.abs(phi_c), axis=0)        # (T,)
        profiles[cname] = profile

        # Plot: mean beat + |SHAP| profile overlay
        mean_beat = X_explain[take].mean(axis=0)
        fig, ax1 = plt.subplots(figsize=(7.5, 3.8))
        ax1.plot(mean_beat, color="#444", linewidth=1.2, label="mean beat")
        ax1.set_xlabel("Sample index")
        ax1.set_ylabel("Normalised amplitude", color="#444")
        ax1.tick_params(axis="y", labelcolor="#444")

        ax2 = ax1.twinx()
        ax2.fill_between(np.arange(len(profile)), 0, profile, alpha=0.4,
                         color="#d62728", label="mean |SHAP|")
        ax2.set_ylabel("mean |SHAP|", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax1.set_title(f"SHAP temporal attribution — {tag} class {cname}")
        fig.tight_layout()
        out = FIGURES / f"shap_temporal_{tag}_class_{cname}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        log.info(f"  -> {out}")

    return profiles


# ============================================================
# SHAP — tree / classical models
# ============================================================
def shap_beeswarm_tree(model, X_background: np.ndarray, X_explain: np.ndarray,
                       tag: str, n_background: int = 200,
                       n_explain: int = 2000) -> np.ndarray:
    """Tree SHAP -> beeswarm plot + returns mean |phi| over samples (shape (T,))."""
    import shap

    rng = np.random.default_rng(0)
    bg_idx = rng.choice(len(X_background), size=min(n_background, len(X_background)), replace=False)
    ex_idx = rng.choice(len(X_explain),    size=min(n_explain,    len(X_explain)),    replace=False)
    bg = X_background[bg_idx]
    ex = X_explain[ex_idx]

    try:
        explainer = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
        sv = explainer.shap_values(ex)
    except Exception:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(ex)

    # Multi-class -> list of (N, T). Aggregate to mean(|phi|) across classes & samples.
    if isinstance(sv, list):
        phi = np.stack([np.abs(s) for s in sv], axis=0).mean(axis=0)    # (N, T)
    else:
        phi = np.abs(sv)
        if phi.ndim == 3:
            phi = phi.mean(axis=-1)
    mean_abs = phi.mean(axis=0)

    # Beeswarm-like plot using dot strip over top-K features (indices by mean |phi|)
    top_k = 20
    top_idx = np.argsort(mean_abs)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for rank, fidx in enumerate(top_idx):
        x_vals = ex[:, fidx]
        s_vals = phi[:, fidx] if phi.ndim == 2 else np.full(len(ex), mean_abs[fidx])
        ax.scatter(s_vals * np.sign(x_vals - x_vals.mean()),
                   np.full_like(s_vals, rank, dtype=float),
                   c=x_vals, cmap="coolwarm", s=6, alpha=0.5)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"sample {int(i)}" for i in top_idx])
    ax.invert_yaxis()
    ax.set_xlabel("Signed |SHAP| (colour = feature value)")
    ax.set_title(f"SHAP beeswarm — {tag}  (top {top_k} sample indices)")
    fig.tight_layout()
    out = FIGURES / f"shap_beeswarm_{tag}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    log.info(f"  -> {out}")
    return mean_abs


# ============================================================
# Grad-CAM for ResNet1D
# ============================================================
def grad_cam_resnet1d(model, X: np.ndarray, y: np.ndarray,
                     tag: str = "resnet1d", per_class_samples: int = 1):
    device = get_device()
    model = model.to(device).eval()

    # Register hooks on the last residual block
    feats, grads = {}, {}

    def fwd_hook(mod, inp, out):
        feats["val"] = out.detach()

    def bwd_hook(mod, grad_in, grad_out):
        grads["val"] = grad_out[0].detach()

    target_layer = model.blocks[-1]
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    fig, axes = plt.subplots(NUM_CLASSES, 1, figsize=(7.5, 2.1 * NUM_CLASSES), sharex=True)
    if NUM_CLASSES == 1:
        axes = [axes]
    rng = np.random.default_rng(0)

    try:
        for c, cname in enumerate(AAMI_CLASSES):
            idx = np.where(y == c)[0]
            if len(idx) == 0:
                axes[c].set_title(f"{cname}: no samples"); continue
            i = int(rng.choice(idx))
            x = torch.from_numpy(X[i:i + 1]).float().unsqueeze(1).to(device)
            x.requires_grad_(True)

            model.zero_grad()
            logits = model(x)
            score = logits[0, c]
            score.backward()

            fmap = feats["val"][0]           # (C, T')
            grad = grads["val"][0]           # (C, T')
            weights = grad.mean(dim=1)       # (C,)
            cam = F.relu((fmap * weights.unsqueeze(1)).sum(dim=0))   # (T',)
            cam = cam / (cam.max() + 1e-8)
            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                size=WINDOW_SIZE, mode="linear",
                                align_corners=False).squeeze().cpu().numpy()

            beat = X[i]
            axes[c].plot(beat, color="#333", linewidth=1.0)
            axes[c].imshow(cam[None, :], aspect="auto",
                           extent=(0, WINDOW_SIZE, beat.min(), beat.max()),
                           cmap="jet", alpha=0.35)
            axes[c].set_title(f"Grad-CAM — class {cname}")
    finally:
        h1.remove(); h2.remove()

    axes[-1].set_xlabel("Sample index")
    fig.tight_layout()
    out = FIGURES / f"gradcam_{tag}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    log.info(f"  -> {out}")
