"""MIT-BIH download and preprocessing.

Steps per record:
    1. Load raw record via wfdb
    2. Bandpass (0.5-45 Hz, Butterworth, order 4)
    3. Segment 260-sample window centred on each annotation's R-peak sample
    4. Map annotation symbols to AAMI 5-class labels (drop unmapped)
    5. Per-beat Z-score normalisation (drop flat beats)

Inter-patient DS1/DS2 split per De Chazal (2004).
"""

from __future__ import annotations
from collections import Counter
from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt

from .config import (
    DATA_RAW, DATA_CLEAN, log,
    SAMPLING_RATE, WINDOW_SIZE,
    BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER,
    SYMBOL_TO_AAMI, AAMI_CLASSES,
    DS1_RECORDS, DS2_RECORDS, RANDOM_SEED,
)

MITBIH_DIR = DATA_RAW / "mitdb"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_mitbih() -> None:
    """Download MIT-BIH Arrhythmia Database (48 records) via wfdb."""
    import wfdb
    MITBIH_DIR.mkdir(parents=True, exist_ok=True)
    if any(MITBIH_DIR.glob("*.dat")):
        log.info(f"MIT-BIH already present at {MITBIH_DIR} - skipping download.")
        return
    log.info(f"Downloading MIT-BIH Arrhythmia DB to {MITBIH_DIR} ...")
    wfdb.dl_database("mitdb", str(MITBIH_DIR))
    log.info("MIT-BIH download complete.")


# ---------------------------------------------------------------------------
# Signal utilities
# ---------------------------------------------------------------------------
def bandpass(signal: np.ndarray,
             low: float = BANDPASS_LOW,
             high: float = BANDPASS_HIGH,
             fs: int = SAMPLING_RATE,
             order: int = BANDPASS_ORDER) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=0)


def _segment(signal: np.ndarray, peaks: np.ndarray, symbols,
             window: int = WINDOW_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    half = window // 2
    beats, labels = [], []
    for peak, sym in zip(peaks, symbols):
        if sym not in SYMBOL_TO_AAMI:
            continue
        start = peak - half
        end   = start + window
        if start < 0 or end > len(signal):
            continue
        beat = signal[start:end].astype(np.float32).copy()
        std = beat.std()
        if std < 1e-6:
            continue
        beat = (beat - beat.mean()) / std
        beats.append(beat)
        labels.append(AAMI_CLASSES.index(SYMBOL_TO_AAMI[sym]))
    if not beats:
        return np.empty((0, window), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(beats, axis=0), np.asarray(labels, dtype=np.int64)


def _process_record(rec_num: int) -> Tuple[np.ndarray, np.ndarray]:
    import wfdb
    rec_path = str(MITBIH_DIR / str(rec_num))
    record = wfdb.rdrecord(rec_path)
    ann    = wfdb.rdann(rec_path, "atr")
    signal = bandpass(record.p_signal[:, 0])
    return _segment(signal, ann.sample, ann.symbol)


def _process_split(records, name: str):
    all_X, all_y = [], []
    for rec in records:
        X, y = _process_record(rec)
        all_X.append(X); all_y.append(y)
        log.info(f"  record {rec}: {len(X)} beats")
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    np.save(DATA_CLEAN / f"X_{name}.npy", X)
    np.save(DATA_CLEAN / f"y_{name}.npy", y)
    dist = {AAMI_CLASSES[k]: int(v) for k, v in sorted(Counter(y).items())}
    log.info(f"{name}: {len(X)} beats | dist={dist}")
    return X, y


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def preprocess_mitbih(force: bool = False):
    """Preprocess MIT-BIH into DS1/DS2 NumPy arrays. Returns (X_tr, y_tr, X_te, y_te)."""
    paths = {k: DATA_CLEAN / f"{k}.npy" for k in ("X_train", "y_train", "X_test", "y_test")}
    if not force and all(p.exists() for p in paths.values()):
        log.info("Using cached preprocessed arrays.")
        return (np.load(paths["X_train"]), np.load(paths["y_train"]),
                np.load(paths["X_test"]),  np.load(paths["y_test"]))
    log.info("Preprocessing DS1 (train) ...")
    X_tr, y_tr = _process_split(DS1_RECORDS, "train")
    log.info("Preprocessing DS2 (test) ...")
    X_te, y_te = _process_split(DS2_RECORDS, "test")
    return X_tr, y_tr, X_te, y_te


def load_mitbih():
    """Load preprocessed MIT-BIH arrays. Runs preprocessing if missing."""
    return preprocess_mitbih(force=False)
