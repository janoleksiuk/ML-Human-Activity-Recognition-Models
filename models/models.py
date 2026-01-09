"""
models.py

A single module that collects all implemented models as classes:
- PyTorch (nn.Module): GRU, LSTM, CNN, MLP, TabTransformer (continuous-only)
- scikit-learn wrappers: KNN, SVM
- SOM (Kohonen): SOM + SOMClassifier
- XGBoost: XGBoostClassifierWrapper

Design goals:
- Keep PyTorch models usable with torch training loops (forward returns logits).
- Keep classical models usable with fit/predict/predict_proba style.
- Provide save/load helpers where practical.

Notes:
- KNN/SVM/XGBoost/SOM are NOT nn.Module; they are sklearn-style estimators / wrappers.
- Optional dependencies: scikit-learn, xgboost, joblib. Imported defensively.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

# ----------------------------
# Optional deps (defensive imports)
# ----------------------------
try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

try:
    from sklearn.pipeline import Pipeline  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.neighbors import KNeighborsClassifier  # type: ignore
    from sklearn.svm import SVC  # type: ignore
except Exception:  # pragma: no cover
    Pipeline = None  # type: ignore
    StandardScaler = None  # type: ignore
    KNeighborsClassifier = None  # type: ignore
    SVC = None  # type: ignore

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None  # type: ignore

# PyTorch models
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Utility helpers
# =============================================================================

def _ensure_2d_features(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Ensure input is (B, F). If it's (B, T, F) it will be flattened to (B, T*F).
    """
    if isinstance(x, np.ndarray):
        if x.ndim == 3:
            return x.reshape(x.shape[0], -1)
        return x
    else:
        if x.dim() == 3:
            return x.reshape(x.size(0), -1)
        return x


def _ensure_rnn_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure input is (B, T, F) for RNNs. If x is (B, F), treat as one-timestep (B, 1, F).
    """
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected tensor with dim 2 or 3, got shape {tuple(x.shape)}")


# =============================================================================
# PyTorch models
# =============================================================================

class TwoStreamPoseGRU(nn.Module):
    """
    Two-stream GRU classifier:
      - GRU on joint positions
      - GRU on joint velocities
      - Late fusion of both
      - Dynamic joint dropout (only during training)
    Expected input: (B, T, F) or (B, F) (treated as T=1)
    Output: logits (B, num_classes)
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        velocity_dropout: float = 0.05,
    ):
        super().__init__()

        self.velocity_dropout = velocity_dropout
        out_dim = hidden_dim * (2 if bidirectional else 1)

        self.gru_pos = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.gru_vel = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(out_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_rnn_3d(x).float()  # (B, T, F)

        if self.training and self.velocity_dropout > 0:
            mask = (torch.rand_like(x) > self.velocity_dropout).float()
            x = x * mask

        # Compute velocity
        x_vel = x[:, 1:, :] - x[:, :-1, :]             # (B, T-1, F)
        x_vel = F.pad(x_vel, (0, 0, 1, 0))              # pad first frame to keep length

        out_pos, _ = self.gru_pos(x)
        out_vel, _ = self.gru_vel(x_vel)

        last_pos = out_pos[:, -1, :]                    # (B, H*)
        last_vel = out_vel[:, -1, :]                    # (B, H*)

        combined = torch.cat([last_pos, last_vel], dim=1)
        return self.head(combined)


class PoseGRU(nn.Module):
    """
    GRU classifier.
    Expected input:
      - (B, T, F) or (B, F) (treated as T=1)
    Output: logits (B, num_classes)
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_rnn_3d(x).float()
        out, _ = self.gru(x)           # (B, T, H*)
        last = out[:, -1, :]           # (B, H*)
        return self.head(last)


class PoseLSTM(nn.Module):
    """
    LSTM classifier.
    Expected input:
      - (B, T, F) or (B, F) (treated as T=1)
    Output: logits (B, num_classes)
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_rnn_3d(x).float()
        out, _ = self.lstm(x)          # (B, T, H*)
        last = out[:, -1, :]           # (B, H*)
        return self.head(last)


class PoseCNN1D(nn.Module):
    """
    1D-CNN for 57-feature pose vectors (or any length).
    Accepts:
      - (B, L) or (B, 1, L) or (B, T, F) (flatten to (B, 1, T*F))
    Output: logits (B, num_classes)
    """
    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:          # (B, L)
            x = x.unsqueeze(1)    # (B, 1, L)
        elif x.dim() == 3:
            # (B,T,F) -> (B,1,T*F)
            x = x.reshape(x.size(0), 1, -1)
        else:
            raise ValueError(f"Expected dim 2 or 3, got shape {tuple(x.shape)}")

        x = x.float()
        x = self.features(x)
        return self.classifier(x)


class PoseMLP(nn.Module):
    """
    MLP for 57 features (or any flat feature size).
    Accepts:
      - (B, F) or (B, T, F) (flatten to (B, T*F))
    Output: logits (B, num_classes)
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_2d_features(x).float()
        return self.net(x)


class TabTransformerContinuous(nn.Module):
    """
    TabTransformer-style model for continuous-only tabular data.
    Each scalar feature becomes a token with a learned per-feature affine embedding.

    Input: (B, n_features)
    Output: logits (B, num_classes)
    """
    def __init__(
        self,
        n_features: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        self.feature_weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.feature_index_emb = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected (B,n_features), got {tuple(x.shape)}")
        x = x.float()

        tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        tokens = tokens + self.feature_index_emb.unsqueeze(0)

        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+n_features, d_model)

        out = self.encoder(tokens)
        cls_out = self.norm(out[:, 0])
        return self.head(cls_out)


# =============================================================================
# Classical ML models (sklearn-like wrappers)
# =============================================================================

class KNNClassifier:
    """
    KNN classifier wrapper with StandardScaler in a Pipeline.
    """
    def __init__(
        self,
        n_neighbors: int = 11,
        weights: str = "distance",
        metric: str = "minkowski",
        p: int = 2,
    ):
        if Pipeline is None or StandardScaler is None or KNeighborsClassifier is None:
            raise ImportError("scikit-learn is required for KNNClassifier.")
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric,
                p=p,
            ))
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        X = _ensure_2d_features(np.asarray(X))
        y = np.asarray(y).astype(np.int64)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_2d_features(np.asarray(X))
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_2d_features(np.asarray(X))
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError("Underlying KNN pipeline does not support predict_proba.")

    def save(self, path: str) -> None:
        if joblib is None:
            raise ImportError("joblib is required to save sklearn models.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self.model, path)

    @staticmethod
    def load(path: str) -> "KNNClassifier":
        if joblib is None:
            raise ImportError("joblib is required to load sklearn models.")
        obj = KNNClassifier.__new__(KNNClassifier)
        obj.model = joblib.load(path)
        return obj


class SVMClassifier:
    """
    SVM classifier wrapper with StandardScaler in a Pipeline.
    Default is RBF kernel.
    """
    def __init__(
        self,
        C: float = 10.0,
        gamma: Union[str, float] = "scale",
        kernel: str = "rbf",
        class_weight: Optional[Union[str, Dict[int, float]]] = "balanced",
        probability: bool = True,
    ):
        if Pipeline is None or StandardScaler is None or SVC is None:
            raise ImportError("scikit-learn is required for SVMClassifier.")
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                class_weight=class_weight,
                probability=probability,
            ))
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        X = _ensure_2d_features(np.asarray(X))
        y = np.asarray(y).astype(np.int64)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_2d_features(np.asarray(X))
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_2d_features(np.asarray(X))
        # Only if probability=True
        if hasattr(self.model[-1], "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError("SVM was created with probability=False; predict_proba unavailable.")

    def save(self, path: str) -> None:
        if joblib is None:
            raise ImportError("joblib is required to save sklearn models.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self.model, path)

    @staticmethod
    def load(path: str) -> "SVMClassifier":
        if joblib is None:
            raise ImportError("joblib is required to load sklearn models.")
        obj = SVMClassifier.__new__(SVMClassifier)
        obj.model = joblib.load(path)
        return obj


class XGBoostClassifierWrapper:
    """
    XGBoost multiclass wrapper. Uses xgboost.XGBClassifier internally.

    Note:
      - XGBoost versions differ re: early_stopping_rounds in fit().
        We'll support both by setting early_stopping_rounds in constructor, and not passing to fit().
    """
    def __init__(
        self,
        num_classes: int,
        device: str = "cpu",
        n_estimators: int = 2000,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        early_stopping_rounds: int = 50,
        random_state: int = 0,
    ):
        if xgb is None:
            raise ImportError("xgboost is required for XGBoostClassifierWrapper.")

        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            device=device,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
        )
        self.evals_result_: Optional[Dict[str, Dict[str, List[float]]]] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            verbose: Union[bool, int] = 50) -> "XGBoostClassifierWrapper":
        X_train = _ensure_2d_features(np.asarray(X_train))
        y_train = np.asarray(y_train).astype(np.int64)

        if X_val is not None and y_val is not None:
            X_val = _ensure_2d_features(np.asarray(X_val))
            y_val = np.asarray(y_val).astype(np.int64)
            self.model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=verbose)
        else:
            self.model.fit(X_train, y_train, verbose=verbose)

        try:
            self.evals_result_ = self.model.evals_result()
        except Exception:
            self.evals_result_ = None
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_2d_features(np.asarray(X))
        return self.model.predict(X).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_2d_features(np.asarray(X))
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Recommend JSON model saving if the path ends with .json; otherwise pickle the wrapper.
        if path.lower().endswith(".json"):
            self.model.save_model(path)
        else:
            with open(path, "wb") as f:
                pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "XGBoostClassifierWrapper":
        if path.lower().endswith(".json"):
            if xgb is None:
                raise ImportError("xgboost is required to load XGBoost JSON model.")
            # Create a minimal wrapper and load into it
            obj = XGBoostClassifierWrapper.__new__(XGBoostClassifierWrapper)
            obj.evals_result_ = None
            obj.model = xgb.XGBClassifier()
            obj.model.load_model(path)
            return obj
        else:
            with open(path, "rb") as f:
                return pickle.load(f)


# =============================================================================
# SOM (Kohonen Network) + classifier wrapper
# =============================================================================

class SOM:
    """
    Basic 2D Self-Organizing Map (Kohonen Network).
    Weights: (m, n, dim)
    """
    def __init__(self, m: int, n: int, dim: int, sigma: float = 1.5, lr: float = 0.5, seed: int = 0):
        self.m = int(m)
        self.n = int(n)
        self.dim = int(dim)
        self.sigma0 = float(sigma)
        self.lr0 = float(lr)
        self.rng = np.random.default_rng(seed)
        self.weights = self.rng.normal(0.0, 1.0, size=(self.m, self.n, self.dim)).astype(np.float32)

        xs, ys = np.meshgrid(np.arange(self.m), np.arange(self.n), indexing="ij")
        self.coords = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32)

    def _bmu(self, x: np.ndarray) -> Tuple[int, int]:
        diff = self.weights - x[None, None, :]
        dist2 = np.sum(diff * diff, axis=-1)
        idx = np.unravel_index(np.argmin(dist2), dist2.shape)
        return int(idx[0]), int(idx[1])

    def _decay(self, t: int, T: int) -> Tuple[float, float]:
        lr = self.lr0 * np.exp(-t / max(T, 1))
        sigma = self.sigma0 * np.exp(-t / max(T, 1))
        return float(lr), float(max(sigma, 1e-3))

    def fit(self, X: np.ndarray, num_iters: int = 5000, shuffle: bool = True) -> "SOM":
        X = _ensure_2d_features(np.asarray(X)).astype(np.float32)
        N = X.shape[0]
        idxs = np.arange(N)

        for t in range(num_iters):
            if shuffle and (t % N == 0):
                self.rng.shuffle(idxs)

            x = X[idxs[t % N]]
            lr, sigma = self._decay(t, num_iters)

            bi, bj = self._bmu(x)
            bmu_coord = np.array([bi, bj], dtype=np.float32)

            d2 = np.sum((self.coords - bmu_coord[None, :]) ** 2, axis=1)  # (m*n,)
            h = np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)     # (m*n,)
            h2 = h.reshape(self.m, self.n, 1)

            self.weights += lr * h2 * (x[None, None, :] - self.weights)

        return self

    def bmu_indices(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_2d_features(np.asarray(X)).astype(np.float32)
        bmus = [self._bmu(x) for x in X]
        return np.asarray(bmus, dtype=np.int64)  # (N,2)


class SOMClassifier:
    """
    SOM-based classifier:
      - optional StandardScaler
      - train SOM unsupervised
      - label neurons by majority vote using training labels
      - predict via BMU -> neuron label, with nearest-labeled fallback

    Recommended for your pose vectors (57 dims):
      - scale=True
      - optionally preprocess your coordinates elsewhere (centroid/RMS normalization)
    """
    def __init__(
        self,
        m: int = 10,
        n: int = 10,
        sigma: float = 1.5,
        lr: float = 0.5,
        num_iters: int = 8000,
        scale: bool = True,
        seed: int = 0,
    ):
        if scale and StandardScaler is None:
            raise ImportError("scikit-learn is required for scaling in SOMClassifier.")
        self.m = m
        self.n = n
        self.sigma = sigma
        self.lr = lr
        self.num_iters = num_iters
        self.scale = scale
        self.seed = seed

        self.scaler = StandardScaler() if scale else None
        self.som: Optional[SOM] = None
        self.neuron_label: Optional[np.ndarray] = None
        self.num_classes: Optional[int] = None

    def _label_neurons(self, X: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
        assert self.som is not None
        bmus = self.som.bmu_indices(X)
        counts = np.zeros((self.som.m, self.som.n, num_classes), dtype=np.int64)
        for (i, j), cls in zip(bmus, y):
            counts[i, j, int(cls)] += 1

        labels = np.full((self.som.m, self.som.n), -1, dtype=np.int64)
        for i in range(self.som.m):
            for j in range(self.som.n):
                if counts[i, j].sum() > 0:
                    labels[i, j] = int(np.argmax(counts[i, j]))
        return labels

    def fit(self, X: np.ndarray, y: np.ndarray, num_classes: int) -> "SOMClassifier":
        X = _ensure_2d_features(np.asarray(X)).astype(np.float32)
        y = np.asarray(y).astype(np.int64)
        self.num_classes = int(num_classes)

        if self.scaler is not None:
            Xs = self.scaler.fit_transform(X)
        else:
            Xs = X

        self.som = SOM(m=self.m, n=self.n, dim=Xs.shape[1], sigma=self.sigma, lr=self.lr, seed=self.seed)
        self.som.fit(Xs, num_iters=self.num_iters, shuffle=True)
        self.neuron_label = self._label_neurons(Xs, y, self.num_classes)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.som is None or self.neuron_label is None:
            raise RuntimeError("SOMClassifier not fitted.")
        X = _ensure_2d_features(np.asarray(X)).astype(np.float32)
        Xs = self.scaler.transform(X) if self.scaler is not None else X

        bmus = self.som.bmu_indices(Xs)
        preds = np.empty((len(Xs),), dtype=np.int64)

        labeled_mask = self.neuron_label >= 0
        labeled_coords = np.argwhere(labeled_mask)  # (K,2)
        labeled_labels = self.neuron_label[labeled_mask]  # (K,)

        for k, (i, j) in enumerate(bmus):
            lab = self.neuron_label[i, j]
            if lab >= 0:
                preds[k] = lab
            else:
                # nearest labeled neuron in grid
                d2 = np.sum((labeled_coords - np.array([i, j])) ** 2, axis=1)
                preds[k] = labeled_labels[int(np.argmin(d2))]

        return preds

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "SOMClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)


# =============================================================================
# Convenience: model registry (optional)
# =============================================================================

PYTORCH_MODELS = {
    "GRU": PoseGRU,
    "LSTM": PoseLSTM,
    "CNN": PoseCNN1D,
    "MLP": PoseMLP,
    "TabTransformer": TabTransformerContinuous,
}

CLASSICAL_MODELS = {
    "KNN": KNNClassifier,
    "SVM": SVMClassifier,
    "SOM": SOMClassifier,
    "XGBoost": XGBoostClassifierWrapper,
}
