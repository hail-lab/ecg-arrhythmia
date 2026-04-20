"""Model zoo: ResNet1D (+ablation variants), CNN1D, BiLSTM, RandomForest, XGBoost."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .config import NUM_CLASSES, RANDOM_SEED, WINDOW_SIZE


# ============================================================
# 1D Residual Block
# ============================================================
class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1,
                 use_bn: bool = True, dropout: float = 0.2,
                 use_residual: bool = True):
        super().__init__()
        pad = kernel_size // 2
        self.use_bn = use_bn
        self.use_residual = use_residual

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                               padding=pad, bias=not use_bn)
        self.bn1   = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1,
                               padding=pad, bias=not use_bn)
        self.bn2   = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if use_residual and (stride != 1 or in_ch != out_ch):
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=not use_bn),
                nn.BatchNorm1d(out_ch) if use_bn else nn.Identity(),
            )
        else:
            self.shortcut = nn.Identity() if use_residual else None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        if self.use_residual:
            out = out + self.shortcut(x)
        return F.relu(out)


# ============================================================
# ResNet1D (+ ablation-aware constructor)
# ============================================================
class ResNet1D(nn.Module):
    """1D-ResNet. Flags control ablation variants.

    shallow=True       → use only 2 residual blocks (A2)
    use_residual=False → skip connections removed (A1)
    use_bn=False       → batch norm removed (A3)
    dropout=0.0        → dropout removed (A4)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = NUM_CLASSES,
                 shallow: bool = False, use_residual: bool = True,
                 use_bn: bool = True, dropout: float = 0.2):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, padding=7, bias=not use_bn),
            nn.BatchNorm1d(32) if use_bn else nn.Identity(),
            nn.ReLU(),
        )
        blk = lambda i, o, k, s: ResidualBlock1D(
            i, o, kernel_size=k, stride=s,
            use_bn=use_bn, dropout=dropout, use_residual=use_residual,
        )
        if shallow:
            self.blocks = nn.Sequential(
                blk(32, 32, 7, 1),
                blk(32, 64, 7, 2),
            )
            final_ch = 64
        else:
            self.blocks = nn.Sequential(
                blk(32, 32, 7, 1),
                blk(32, 64, 7, 2),
                blk(64, 128, 5, 2),
                blk(128, 256, 5, 2),
            )
            final_ch = 256
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(final_ch, num_classes)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

    def feature_maps(self, x):
        """Final conv feature maps before GAP — used by Grad-CAM."""
        x = self.conv_in(x)
        return self.blocks(x)


# ============================================================
# CNN1D baseline
# ============================================================
class CNN1D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.fc(self.features(x).squeeze(-1))


# ============================================================
# BiLSTM baseline
# ============================================================
class BiLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 128,
                 num_layers: int = 2, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)           # (B, T, C)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ============================================================
# Classical ML builders
# ============================================================
def build_random_forest():
    return RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=5,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
    )

def build_xgboost():
    return XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=NUM_CLASSES,
        eval_metric="mlogloss", random_state=RANDOM_SEED, n_jobs=-1,
        tree_method="hist",
    )


# ============================================================
# Registry
# ============================================================
DL_MODELS = {
    "ResNet1D": lambda: ResNet1D(),
    "CNN1D":    lambda: CNN1D(),
    "BiLSTM":   lambda: BiLSTM(),
}
ML_MODELS = {
    "RandomForest": build_random_forest,
    "XGBoost":      build_xgboost,
}
ALL_MODELS = list(DL_MODELS) + list(ML_MODELS)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
