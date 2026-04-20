"""Unified project configuration: paths, seeds, constants, AAMI map, DS1/DS2 splits."""

from __future__ import annotations
import os
import random
import logging
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths (auto-detect Colab vs. local)
# ---------------------------------------------------------------------------
def _detect_root() -> Path:
    # 1) explicit env var wins
    env = os.environ.get("ECG_ROOT")
    if env:
        return Path(env).resolve()
    # 2) colab convention
    if Path("/content").exists():
        return Path("/content/ecg-arrhythmia").resolve()
    # 3) local: repo root two levels up from this file
    return Path(__file__).resolve().parent.parent


ROOT       = _detect_root()
DATA_RAW   = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"
OUTPUTS    = ROOT / "outputs"
FIGURES    = OUTPUTS / "figures"
TABLES     = OUTPUTS / "tables"
MODELS_DIR = OUTPUTS / "models"

for p in (DATA_RAW, DATA_CLEAN, FIGURES, TABLES, MODELS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Seed python, numpy, and torch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ---------------------------------------------------------------------------
# ECG signal and AAMI 5-class configuration
# ---------------------------------------------------------------------------
SAMPLING_RATE = 360            # Hz (MIT-BIH native)
WINDOW_SIZE   = 260            # samples centred on R-peak (~0.72 s @ 360 Hz)
BANDPASS_LOW  = 0.5            # Hz
BANDPASS_HIGH = 45.0           # Hz
BANDPASS_ORDER = 4

AAMI_MAP = {
    "N": ["N", "L", "R", "e", "j"],
    "S": ["A", "a", "J", "S"],
    "V": ["V", "E"],
    "F": ["F"],
    "Q": ["/", "f", "Q"],
}
SYMBOL_TO_AAMI = {s: cls for cls, syms in AAMI_MAP.items() for s in syms}
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
NUM_CLASSES  = len(AAMI_CLASSES)

# De Chazal (2004) inter-patient split
DS1_RECORDS = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
               122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2_RECORDS = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
               210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# ---------------------------------------------------------------------------
# Training hyperparameters (deep models)
# ---------------------------------------------------------------------------
BATCH_SIZE    = 256
LEARNING_RATE = 1e-3
EPOCHS        = 50
EARLY_STOP_PATIENCE = 8
PLATEAU_PATIENCE    = 5
PLATEAU_FACTOR      = 0.5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    force=True,          # override any pre-existing handler (e.g. Colab's)
)
log = logging.getLogger("ecg")
log.setLevel(logging.INFO)
log.propagate = True
