import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Tuple, Union, Any, List


# -----------------------------
# Robust NPZ loading
# -----------------------------
def _load_npz_xy(
    npz_path: str,
    x_keys: Optional[List[str]] = None,
    y_keys: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads X and y from an NPZ file.
    - Tries common key names.
    - Falls back to heuristic if keys not found.
    """
    x_keys = x_keys or ["X", "x", "data", "features"]
    y_keys = y_keys or ["y", "Y", "labels", "label", "target", "targets"]

    data = np.load(npz_path, allow_pickle=True)
    X = None
    y = None

    for k in x_keys:
        if k in data.files:
            X = data[k]
            break
    for k in y_keys:
        if k in data.files:
            y = data[k]
            break

    # Fallback inference (if file doesn't follow expected naming)
    if X is None or y is None:
        arrays = [data[k] for k in data.files]
        # y: prefer a 1D array with smallest size
        one_d = [a for a in arrays if np.asarray(a).ndim == 1]
        if len(one_d) > 0:
            y = min(one_d, key=lambda a: np.asarray(a).size)
        # X: largest array
        X = max(arrays, key=lambda a: np.asarray(a).size)

    return np.asarray(X), np.asarray(y)


# -----------------------------
# Optional pose normalization (for 57 = 19 * xyz)
# -----------------------------
def normalize_pose_np(X57: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalizes pose-like features assuming 57 dims = 19*(x,y,z).
    - subtract centroid (translation invariance)
    - divide by RMS scale (scale invariance)
    """
    X = X57.reshape(-1, 19, 3).copy()
    centroid = X.mean(axis=1, keepdims=True)
    X -= centroid
    scale = np.sqrt((X ** 2).sum(axis=(1, 2), keepdims=True) / (19 * 3))
    X /= (scale + eps)
    return X.reshape(-1, 57).astype(np.float32)


# -----------------------------
# Dataset
# -----------------------------
class PoseDataset(Dataset):
    """
    General dataset for pose/tabular data stored in NPZ.

    mode:
      - "rnn":   returns (T,F) per sample; guarantees 3D batches (B,T,F) by expanding if needed
      - "flat":  returns (F,) per sample; flattens (T,F)->(T*F)
      - "cnn1d": returns (1,F) per sample; channel-first for Conv1d
      - "raw":   returns X as stored (torch tensor)
    """
    def __init__(
        self,
        npz_path: str,
        mode: str = "rnn",
        class_to_idx: Optional[Dict[Any, int]] = None,
        normalize_pose: bool = False,
        x_keys: Optional[List[str]] = None,
        y_keys: Optional[List[str]] = None,
    ):
        X_raw, y_raw = _load_npz_xy(npz_path, x_keys=x_keys, y_keys=y_keys)

        # Convert X to float32 (handle object/strings defensively)
        if np.issubdtype(X_raw.dtype, np.str_) or X_raw.dtype == object:
            try:
                X = np.array(X_raw, dtype=np.float32)
            except ValueError as e:
                raise ValueError(f"Cannot convert X data to float: {e}")
        else:
            X = np.array(X_raw, dtype=np.float32)

        # Optional pose normalization (expects 57 features after flatten if needed)
        # Apply after any flattening in __getitem__ would be awkward; so we apply here if possible.
        # If X is (N,57) already, normalize directly. If it's (N,T,F) and T*F==57, flatten then normalize.
        if normalize_pose:
            if X.ndim == 2 and X.shape[1] == 57:
                X = normalize_pose_np(X)
            elif X.ndim == 3 and (X.shape[1] * X.shape[2] == 57):
                X = X.reshape(X.shape[0], -1)
                X = normalize_pose_np(X)
            # else: silently skip normalization (shape not compatible)

        # Handle y - numeric or string labels
        if np.issubdtype(y_raw.dtype, np.str_) or y_raw.dtype == object:
            if class_to_idx is None:
                classes = np.unique(y_raw)
                self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
            else:
                self.class_to_idx = class_to_idx
            y_numeric = np.array([self.class_to_idx[label] for label in y_raw], dtype=np.int64)
        else:
            y_numeric = np.array(y_raw, dtype=np.int64)
            self.class_to_idx = class_to_idx  # may be None

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_numeric, dtype=torch.long)

        self.mode = mode.lower().strip()
        if self.mode not in {"rnn", "flat", "cnn1d", "raw"}:
            raise ValueError(f"Unsupported mode='{mode}'. Use: rnn, flat, cnn1d, raw")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.mode == "raw":
            return x, y

        if self.mode == "rnn":
            # Ensure per-sample is (T,F). If (F,), convert to (1,F).
            if x.dim() == 1:
                x = x.unsqueeze(0)
            elif x.dim() != 2:
                # If stored oddly, flatten to (1, F)
                x = x.reshape(1, -1)
            return x, y

        if self.mode == "flat":
            # Ensure per-sample is (F,)
            if x.dim() == 2:
                x = x.reshape(-1)
            elif x.dim() != 1:
                x = x.reshape(-1)
            return x, y

        if self.mode == "cnn1d":
            # Ensure per-sample is (1, F) channel-first
            if x.dim() == 2:
                x = x.reshape(-1)  # (F,)
            elif x.dim() != 1:
                x = x.reshape(-1)
            x = x.unsqueeze(0)      # (1, F)
            return x, y

        raise RuntimeError("Unreachable")

    def get_class_mapping(self):
        return self.class_to_idx


# -----------------------------
# Dataloader helpers
# -----------------------------
def get_dataloader(
    npz_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    mode: str = "rnn",
    class_to_idx: Optional[Dict[Any, int]] = None,
    normalize_pose: bool = False,
) -> DataLoader:
    """
    Backward compatible: default mode="rnn" matches your original GRU/LSTM usage. :contentReference[oaicite:1]{index=1}
    """
    dataset = PoseDataset(
        npz_path=npz_path,
        mode=mode,
        class_to_idx=class_to_idx,
        normalize_pose=normalize_pose,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_rnn_dataloader(
    npz_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    class_to_idx: Optional[Dict[Any, int]] = None,
    normalize_pose: bool = False,
) -> DataLoader:
    return get_dataloader(
        npz_path, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        mode="rnn", class_to_idx=class_to_idx, normalize_pose=normalize_pose
    )


def get_flat_dataloader(
    npz_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    class_to_idx: Optional[Dict[Any, int]] = None,
    normalize_pose: bool = False,
) -> DataLoader:
    """
    For MLP / TabTransformer (continuous-only) where input should be (B,F).
    """
    return get_dataloader(
        npz_path, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        mode="flat", class_to_idx=class_to_idx, normalize_pose=normalize_pose
    )


def get_cnn1d_dataloader(
    npz_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    class_to_idx: Optional[Dict[Any, int]] = None,
    normalize_pose: bool = False,
) -> DataLoader:
    """
    For 1D CNN where input should be (B,1,F).
    """
    return get_dataloader(
        npz_path, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        mode="cnn1d", class_to_idx=class_to_idx, normalize_pose=normalize_pose
    )


# -----------------------------
# Numpy helpers (for sklearn/XGBoost/SOM pipelines)
# -----------------------------
def load_npz_numpy(
    npz_path: str,
    mode: str = "flat",
    normalize_pose: bool = False,
    class_to_idx: Optional[Dict[Any, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[Any, int]]]:
    """
    Returns (X, y, class_to_idx) as numpy arrays.
    - mode="flat": ensures X is (N,F) by flattening (N,T,F)->(N,T*F)
    - mode="rnn": returns X as (N,T,F) if present, else (N,1,F)
    - mode="raw": returns as stored
    """
    X_raw, y_raw = _load_npz_xy(npz_path)

    # X to float32
    if np.issubdtype(X_raw.dtype, np.str_) or X_raw.dtype == object:
        X = np.array(X_raw, dtype=np.float32)
    else:
        X = np.array(X_raw, dtype=np.float32)

    # y to int64
    if np.issubdtype(y_raw.dtype, np.str_) or y_raw.dtype == object:
        if class_to_idx is None:
            classes = np.unique(y_raw)
            class_to_idx = {cls: i for i, cls in enumerate(classes)}
        y = np.array([class_to_idx[label] for label in y_raw], dtype=np.int64)
    else:
        y = np.array(y_raw, dtype=np.int64)

    mode = mode.lower().strip()
    if mode == "flat":
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        if normalize_pose and X.ndim == 2 and X.shape[1] == 57:
            X = normalize_pose_np(X)
        return X.astype(np.float32), y, class_to_idx

    if mode == "rnn":
        if X.ndim == 2:
            X = X[:, None, :]  # (N,1,F)
        elif X.ndim != 3:
            X = X.reshape(X.shape[0], 1, -1)
        # normalize_pose for rnn mode is not applied here (since it assumes flat 57)
        return X.astype(np.float32), y, class_to_idx

    if mode == "raw":
        return X, y, class_to_idx

    raise ValueError(f"Unsupported mode='{mode}'. Use: flat, rnn, raw")
