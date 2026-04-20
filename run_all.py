"""End-to-end local runner for the ECG arrhythmia benchmark.

Mirrors the colab/run_all.ipynb notebook cells. Run from the ecg-arrhythmia/ folder:
    python run_all.py
    python run_all.py --bilstm-epochs 10  # fewer BiLSTM epochs if time-constrained
    python run_all.py --run-ptbxl         # include PTB-XL (~1.7 GB download)

Produces ./outputs.zip alongside the unpacked outputs/ tree.
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import shutil
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _setup_paths():
    """Pin ECG_V2_ROOT to this script's directory before importing src.*"""
    here = Path(__file__).resolve().parent
    os.environ["ECG_ROOT"] = str(here)
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    return here


ROOT = _setup_paths()

# These imports depend on ECG_V2_ROOT being set above.
from src.config   import (log, set_global_seed, RANDOM_SEED,
                          AAMI_CLASSES, NUM_CLASSES,
                          TABLES, FIGURES, MODELS_DIR, OUTPUTS)
from src.data     import download_mitbih, preprocess_mitbih
from src.models   import (DL_MODELS, ML_MODELS, count_parameters)
from src.train    import (train_dl, train_ml, predict_dl, predict_ml, get_device)
from src.evaluate import (compute_metrics, per_class_f1,
                          plot_confusion_matrix, plot_roc,
                          plot_model_comparison, plot_per_class_f1,
                          wilcoxon_correctness, mcnemar_test, save_table)
from src.explain  import (shap_temporal_deep, shap_beeswarm_tree,
                          grad_cam_resnet1d)
from src.ablation import run_ablation


# ============================================================
# Helpers
# ============================================================
def _section(title: str):
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}", flush=True)


def _banner_env():
    dev = get_device()
    print(f"torch  = {torch.__version__}", flush=True)
    print(f"device = {dev}", flush=True)
    if dev.type == "cuda":
        print(f"gpu    = {torch.cuda.get_device_name(0)}", flush=True)
        print(f"vram   = {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB", flush=True)


# ============================================================
# Pipeline steps
# ============================================================
def step_data():
    _section("Step 1/8 | Download + preprocess MIT-BIH")
    download_mitbih()
    X_tr, y_tr, X_te, y_te = preprocess_mitbih(force=False)
    print(f"DS1 train: X={X_tr.shape}, y dist={np.bincount(y_tr, minlength=NUM_CLASSES)}", flush=True)
    print(f"DS2 test : X={X_te.shape}, y dist={np.bincount(y_te, minlength=NUM_CLASSES)}", flush=True)
    return X_tr, y_tr, X_te, y_te


def step_train(X_tr, y_tr, X_te, y_te, bilstm_epochs: int):
    _section("Step 2/8 | Train models (resume from disk where possible)")
    trained: dict = {}

    for name, builder in DL_MODELS.items():
        pt = MODELS_DIR / f"{name}.pt"
        m = builder()
        if pt.exists():
            print(f"[{name}] loading cached weights ({pt.stat().st_size/1e6:.2f} MB)", flush=True)
            m.load_state_dict(torch.load(pt, map_location="cpu"))
        else:
            ep = bilstm_epochs if name == "BiLSTM" else 50
            print(f"=== training {name} (epochs={ep}) ===", flush=True)
            t0 = time.time()
            train_dl(m, X_tr, y_tr, X_te, y_te, epochs=ep, tag=name)
            print(f"[{name}] done in {(time.time()-t0)/60:.1f} min", flush=True)
        trained[name] = m
        print(f"[{name}] params = {count_parameters(m):,}", flush=True)

    for name, builder in ML_MODELS.items():
        jb = MODELS_DIR / f"{name}.joblib"
        if jb.exists():
            print(f"[{name}] loading cached model", flush=True)
            trained[name] = joblib.load(jb)
        else:
            print(f"=== fitting {name} ===", flush=True)
            t0 = time.time()
            trained[name] = train_ml(builder(), X_tr, y_tr, tag=name)
            print(f"[{name}] done in {(time.time()-t0)/60:.1f} min", flush=True)

    print(f"\nTrained: {list(trained)}", flush=True)
    return trained


def step_predict(trained, X_te, y_te):
    _section("Step 3/8 | Predict + overall metrics")
    preds: dict = {}
    rows: list = []
    for name, model in trained.items():
        if name in DL_MODELS:
            y_pred, y_prob = predict_dl(model, X_te)
        else:
            y_pred, y_prob = predict_ml(model, X_te)
        preds[name] = (y_pred, y_prob)
        m  = compute_metrics(y_te, y_pred, y_prob)
        pc = per_class_f1(y_te, y_pred)
        rows.append({
            "model": name, **m, **pc,
            "params": count_parameters(model) if name in DL_MODELS else np.nan,
        })
        print(f"[{name}] acc={m['accuracy']:.4f}  f1_M={m['f1_M']:.4f}  auc_M={m['auc_M']:.4f}", flush=True)

    overall_df = (pd.DataFrame(rows)
                    .sort_values("f1_M", ascending=False)
                    .reset_index(drop=True))
    save_table(overall_df, "overall")
    print("\n" + overall_df.to_string(index=False), flush=True)
    return preds, overall_df


def step_plots(preds, overall_df, y_te):
    _section("Step 4/8 | Confusion matrices, ROC, bar charts")
    for name, (y_pred, y_prob) in preds.items():
        plot_confusion_matrix(y_te, y_pred, tag=name, normalize=True)
        plot_roc(y_te, y_prob, tag=name)
    plot_model_comparison(overall_df)
    plot_per_class_f1(overall_df)


def step_significance(preds, y_te, ref: str = "ResNet1D"):
    _section("Step 5/8 | Statistical significance (Wilcoxon + McNemar)")
    if ref not in preds:
        # fall back to first available model if ref was skipped
        ref = next(iter(preds))
    ref_pred, _ = preds[ref]

    rows = []
    for name, (y_pred, _) in preds.items():
        if name == ref:
            continue
        p_wil = wilcoxon_correctness(ref_pred, y_pred, y_te)
        p_mc, b01, b10 = mcnemar_test(ref_pred, y_pred, y_te)
        rows.append({
            "comparison":       f"{ref} vs {name}",
            "p_wilcoxon":       p_wil,
            "p_mcnemar":        p_mc,
            f"{ref}_correct_other_wrong": b01,
            f"{ref}_wrong_other_correct": b10,
            "net_advantage":    b01 - b10,
        })
    sig_df = pd.DataFrame(rows)
    save_table(sig_df, "significance")
    if not sig_df.empty:
        print("\n" + sig_df.to_string(index=False), flush=True)


def step_shap(trained, X_tr, X_te, y_te):
    _section("Step 6/8 | Unified SHAP (deep + tree) + Grad-CAM")
    for name in ("ResNet1D", "CNN1D", "BiLSTM"):
        if name not in trained:
            continue
        print(f"[SHAP deep] {name}", flush=True)
        shap_temporal_deep(trained[name], X_tr, X_te, y_te, tag=name,
                           n_background=128, per_class_samples=200)

    for name in ("RandomForest", "XGBoost"):
        if name not in trained:
            continue
        print(f"[SHAP tree] {name}", flush=True)
        shap_beeswarm_tree(trained[name], X_tr, X_te, tag=name,
                          n_background=200, n_explain=2000)

    if "ResNet1D" in trained:
        print("[Grad-CAM] ResNet1D", flush=True)
        grad_cam_resnet1d(trained["ResNet1D"], X_te, y_te, tag="resnet1d")


def step_ablation(X_tr, y_tr, X_te, y_te, ablation_epochs: int):
    _section(f"Step 7/8 | Ablation study A1-A4 ({ablation_epochs} epochs each)")
    abl_df = run_ablation(X_tr, y_tr, X_te, y_te, epochs=ablation_epochs)
    print("\n" + abl_df.to_string(index=False), flush=True)


def step_ptbxl(ptbxl_epochs: int):
    _section(f"Step 7b/8 | PTB-XL 12-lead cross-dataset ({ptbxl_epochs} epochs each)")
    from src.ptbxl import run_ptbxl
    df = run_ptbxl(epochs=ptbxl_epochs)
    print("\n" + df.to_string(index=False), flush=True)


def step_package():
    _section("Step 8/8 | Zip outputs/ -> outputs.zip")
    archive = shutil.make_archive(str(OUTPUTS.parent / "outputs"), "zip", OUTPUTS)
    print(f"Wrote {archive}  ({Path(archive).stat().st_size/1e6:.2f} MB)", flush=True)
    print(f"Tables:  {sorted(p.name for p in TABLES.glob('*.csv'))}", flush=True)
    print(f"Figures: {sorted(p.name for p in FIGURES.glob('*.png'))}", flush=True)
    print(f"Models:  {sorted(p.name for p in MODELS_DIR.glob('*'))}", flush=True)


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bilstm-epochs",  type=int, default=20, help="BiLSTM epoch count (default 20)")
    p.add_argument("--ablation-epochs",type=int, default=30, help="ablation variants epoch count (default 30)")
    p.add_argument("--run-ptbxl",      action="store_true", default=False, help="include PTB-XL cross-dataset eval")
    p.add_argument("--ptbxl-epochs",   type=int, default=15, help="PTB-XL epoch count (default 15)")
    p.add_argument("--stop-after",     choices=["data","train","predict","plots","sig","shap","ablation","ptbxl"],
                                       help="short-circuit pipeline for debugging")
    args = p.parse_args()

    set_global_seed(RANDOM_SEED)
    _section("ECG Arrhythmia Benchmark — local run")
    print(f"root   = {ROOT}", flush=True)
    _banner_env()

    # 1-2. Data + training
    X_tr, y_tr, X_te, y_te = step_data()
    if args.stop_after == "data": return

    trained = step_train(X_tr, y_tr, X_te, y_te,
                         bilstm_epochs=args.bilstm_epochs)
    if args.stop_after == "train": return

    # 3-5. Evaluation
    preds, overall_df = step_predict(trained, X_te, y_te)
    if args.stop_after == "predict": return

    step_plots(preds, overall_df, y_te)
    if args.stop_after == "plots": return

    step_significance(preds, y_te)
    if args.stop_after == "sig": return

    # 6-7. Explanations + ablation
    step_shap(trained, X_tr, X_te, y_te)
    if args.stop_after == "shap": return

    step_ablation(X_tr, y_tr, X_te, y_te, ablation_epochs=args.ablation_epochs)
    if args.stop_after == "ablation": return

    if args.run_ptbxl:
        step_ptbxl(ptbxl_epochs=args.ptbxl_epochs)
        if args.stop_after == "ptbxl": return

    # 8. Package
    step_package()
    _section("DONE")


if __name__ == "__main__":
    main()
