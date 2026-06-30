import random
import warnings
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

# Scikit-learn & pmdarima
from sklearn.impute import KNNImputer
try:
    from pmdarima import auto_arima
except ImportError:
    pass

# PyTorch (Optional dependency check)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# 0. BASELINE IMPUTATION FUNCTIONS (1D)
# =============================================================================

# ---- 0a. LOCF (1D) ---------------------------------------------------------
def locf_imputation_1D(array: np.ndarray, **kwargs) -> np.ndarray:
    """Forward-fill then backward-fill on a 1D array."""
    s = pd.Series(array)
    return s.ffill().bfill().values


# ---- 0b. Local linear regression (1D) --------------------------------------
def multi_point_linear_imputation_1D(array: np.ndarray, window_size: int = 7, **kwargs) -> np.ndarray:
    """
    For each NaN, fit a degree-1 polynomial to observed values within
    ±window_size and use it to impute. Residual NaNs filled by ffill/bfill.
    """
    s = pd.Series(array)
    
    # 1. Connect bounded intervals with linear segments (Exact PLI definition)
    interpolated = s.interpolate(method='linear')
    
    # 2. Handle edge cases (e.g., if the very first or last hour of the 
    # entire patient record is missing, it has no bounding point to connect to)
    final_array = interpolated.ffill().bfill().values
    
    return final_array


# ---- 0c. autoARIMA (1D) ----------------------------------------------------
def autoarima_imputation_1D(array: np.ndarray, **kwargs) -> np.ndarray:
    """
    auto_arima fit on a ffill/bfill-initialised series; in-sample
    predictions replace only the originally-missing entries.
    """
    s = pd.Series(array)
    initial = s.ffill().bfill().values
    
    try:
        model = auto_arima(initial, error_action='ignore', suppress_warnings=True, stepwise=True)
        predicted = model.predict_in_sample()
    except Exception as e:
        print(f"[warn] auto_arima failed ({e}); returning ffill/bfill series.")
        return initial
        
    imputed = np.array(array, dtype=float)
    nan_mask = np.isnan(imputed)
    imputed[nan_mask] = predicted[nan_mask]
    return imputed


# =============================================================================
# 1. GRID-BASED IMPUTATION FUNCTIONS (2D)
# =============================================================================

# ---- 1a. KNN imputation (2D) -----------------------------------------------
def knn_imputation_2D(arr: np.ndarray, n_neighbors: int = 48, method: str = "uniform", **kwargs) -> np.ndarray:
    """
    Column-wise + row-wise KNN imputation, then averaged.
    Treats 0 as a sentinel inside the merge step.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=method,
                         keep_empty_features=True)
    a = imputer.fit_transform(arr)
    b = imputer.fit_transform(arr.T).T
    a[np.where(a == 0)] = b[np.where(a == 0)]
    b[np.where(b == 0)] = a[np.where(b == 0)]
    mean_arr = (a + b) / 2.0
    mean_arr[mean_arr == 0] = np.nan
    return mean_arr

# =============================================================================
# 2. DEEP LEARNING (CNN 2D)
# =============================================================================

if TORCH_AVAILABLE:

    class CNNImputer2D(nn.Module):
        """
        3-layer CNN that takes a (2, H, W) input — channel 0 is values
        with NaNs zeroed, channel 1 is the observed-mask — and outputs a
        (1, H, W) prediction.
        """
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(2, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=3, padding=1),
            )

        def forward(self, x):
            return self.net(x)


    def make_torch_dataset(pivot_d: Dict, binary_d: Dict,
                           good_k: List, binary_masks: List,
                           shape_2d: Tuple[int, int] = (28, 24),
                           n_per_key: int = 10,
                           seed: int = 2025) -> TensorDataset:
        """Build a TensorDataset of (X, Y) where X = (vals, mask) channels and Y = original."""
        rng = random.Random(seed)
        np.random.seed(seed)
        
        N = len(good_k) * n_per_key
        pairs = list(zip(n_per_key * list(good_k), rng.choices(binary_masks, k=N)))
        
        X_list, Y_list = [], []
        for orig_key, mask_key in pairs:
            flat = pivot_d[orig_key].drop(["week_start"], axis=1).values.flatten()
            orig_2d = flat.reshape(shape_2d)
            
            miss_flat = flat.copy().astype(float)
            miss_flat[binary_d[mask_key].flatten().astype(bool)] = np.nan
            
            miss_2d = miss_flat.reshape(shape_2d)
            mask_2d = (~np.isnan(miss_2d)).astype(float)
            vals_2d = np.nan_to_num(miss_2d, nan=0.0)
            
            X_list.append(np.stack([vals_2d, mask_2d], axis=0))
            Y_list.append(orig_2d[np.newaxis, ...])
            
        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        Y = torch.tensor(np.array(Y_list), dtype=torch.float32)
        return TensorDataset(X, Y)


    def train_cnn_imputer(pivot_d: Dict, binary_d: Dict,
                          good_k: List, binary_masks: List,
                          shape_2d: Tuple[int, int] = (28, 24),
                          n_per_key: int = 10,
                          dataset_seed: int = 2025,
                          batch_size: int = 32,
                          n_epochs: int = 50,
                          lr: float = 1e-3,
                          verbose: bool = True) -> CNNImputer2D:
        """Train CNNImputer2D and return the trained model."""
        dataset = make_torch_dataset(
            pivot_d, binary_d, good_k, binary_masks,
            shape_2d=shape_2d, n_per_key=n_per_key, seed=dataset_seed,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = CNNImputer2D()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        epoch_iter = tqdm(range(n_epochs), desc="Training CNN", disable=not verbose)
        for epoch in epoch_iter:
            epoch_loss = 0.0
            for xb, yb in loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * xb.size(0)
            epoch_iter.set_postfix(loss=f"{epoch_loss/len(dataset):.4f}")
            
        return model


    def make_cnn_imputation_fn(model: CNNImputer2D) -> Callable:
        """
        Wrap a trained model into a function with signature:
        arr_2d (with NaNs) -> arr_2d (NaNs replaced by predictions)
        """
        def impute(arr_2d: np.ndarray, **kwargs) -> np.ndarray:
            model.eval()
            mask_2d = ~np.isnan(arr_2d)
            vals_2d = np.nan_to_num(arr_2d, nan=0.0)
            
            inp = torch.tensor(
                np.stack([vals_2d, mask_2d.astype(float)], axis=0),
                dtype=torch.float32,
            ).unsqueeze(0)
            
            with torch.no_grad():
                pred = model(inp)[0, 0].cpu().numpy()
                
            result = arr_2d.copy()
            result[~mask_2d] = pred[~mask_2d]
            return result
            
        return impute