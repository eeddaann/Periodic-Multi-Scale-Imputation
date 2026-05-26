import random
import numpy as np
import pandas as pd
import warnings
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

class PMSIImputer:
    """
    Periodic Multi-Scale Imputation (PMSI) Framework.
    """
    def __init__(self, x_size: int = 7, y_size: int = 7, n_evals: int = 300, seed: int = 134, pre_fit_params: dict = None):
        self.x_size = x_size
        self.y_size = y_size
        self.pad_width = max(x_size, y_size) // 2
        self.n_evals = n_evals
        self.seed = seed
        
        self.best_params_ = pre_fit_params 
        self.trials_ = None
        self._kernels_cache = {}

    def _get_kernel(self, x_stddev: float, y_stddev: float):
        # if y_stddev is None or missing, make it isotropic (equal to x_stddev)
        if y_stddev is None:
            y_stddev = x_stddev
            
        k_key = (x_stddev, y_stddev, self.x_size, self.y_size)
        if k_key not in self._kernels_cache:
            self._kernels_cache[k_key] = Gaussian2DKernel(
                x_stddev=x_stddev, 
                y_stddev=y_stddev, 
                x_size=self.x_size, 
                y_size=self.y_size
            )
        return self._kernels_cache[k_key]

    def _convolve(self, img_2d: np.ndarray, x_stddev: float, y_stddev: float) -> np.ndarray:
        kernel = self._get_kernel(x_stddev, y_stddev)
        padded = np.pad(img_2d.copy(), mode='edge', pad_width=self.pad_width)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interpolated = interpolate_replace_nans(padded, kernel)
            
        return interpolated[self.pad_width:-self.pad_width, self.pad_width:-self.pad_width]

    def fit(self, pivot_d: dict, binary_d: dict, good_k: list, binary_masks: list, n_pairs_per_key: int = 10):
        random.seed(self.seed)
        np.random.seed(self.seed)

        N = n_pairs_per_key * len(good_k)
        pairs = list(zip(n_pairs_per_key * list(good_k), random.choices(binary_masks, k=N)))

        def objective(weights):
            try:
                tmp_aic_list, tmp_rmse_list = [], []
                
                for orig_key, mask_key in pairs:
                    orig_df = pivot_d[orig_key]
                    if "week_start" in orig_df.columns:
                        orig_flat = orig_df.drop(["week_start"], axis=1).values.flatten()
                    else:
                        orig_flat = orig_df.values.flatten()
                    
                    try:
                        orig_2d = orig_flat.reshape((28, 24))
                    except ValueError:
                        continue
                        
                    missing_flat = orig_flat.copy().astype(float)
                    mask_bool = binary_d[mask_key].flatten().astype(bool)
                    missing_flat[mask_bool] = np.nan
                    missing_2d = missing_flat.reshape((28, 24))
                    
                    missing_indices = np.isnan(missing_2d)
                    n_miss = np.sum(missing_indices)
                    
                    if n_miss == 0:
                        continue
                        
                    # Fits using separate weights as requested by hyperopt space
                    imputed_2d = self._convolve(missing_2d, weights['x'], weights['y'])
                    
                    imputed_values = imputed_2d[missing_indices]
                    actual_values = orig_2d[missing_indices]
                    
                    valid_mask = ~np.isnan(imputed_values)
                    valid_n = np.sum(valid_mask)
                    
                    if valid_n == 0:
                        continue
                        
                    residuals = imputed_values[valid_mask] - actual_values[valid_mask]
                    rss = max(np.sum(residuals**2), 1e-12)
                    
                    log_likelihood = -0.5 * valid_n * np.log(rss / valid_n)
                    aic = 4 - 2 * log_likelihood
                    rmse = np.sqrt(np.mean(residuals**2))
                    
                    tmp_aic_list.append(aic)
                    tmp_rmse_list.append(rmse)

                if not tmp_aic_list:
                    return {'loss': 1e10, 'status': STATUS_OK}
                    
                avg_aic = np.mean(tmp_aic_list)
                if np.isnan(avg_aic):
                    return {'loss': 1e10, 'status': STATUS_OK}
                    
                return {'loss': float(avg_aic), 'aic': float(avg_aic), 'rmse': float(np.mean(tmp_rmse_list)), 'status': STATUS_OK}
                
            except Exception as e:
                print(f"Objective Error: {e}")
                return {'loss': 1e10, 'status': STATUS_OK}

        hp_space = {
            'x': hp.uniform('x', 0.1, 5.0),
            'y': hp.uniform('y', 0.1, 5.0)
        }
        
        self.trials_ = Trials()
        print(f"Fitting PMSI: Running {self.n_evals} evals via Hyperopt TPE...")
        
        self.best_params_ = fmin(
            fn=objective, space=hp_space, algo=tpe.suggest,
            trials=self.trials_, max_evals=self.n_evals,
            rstate=np.random.default_rng(self.seed)
        )
        
        best_trial = min(self.trials_.results, key=lambda x: x['loss'])
        print(f"Fit Complete. Optimal Params: {self.best_params_} | AIC: {best_trial['aic']:.2f}")
        
        return self

    def impute(self, miss_2d: np.ndarray, **kwargs) -> np.ndarray:
        # Prioritize key matching standard naming conventions, fallback to fitted parameters
        x_w = kwargs.get('x_stddev', kwargs.get('x', self.best_params_['x'] if self.best_params_ else None))
        y_w = kwargs.get('y_stddev', kwargs.get('y', self.best_params_['y'] if self.best_params_ else x_w)) # If y missing, mirror x

        if x_w is None:
            raise ValueError("PMSIImputer requires an x_stddev parameter via fit, pre_fit_params, or kwargs.")
            
        # Returns the raw math output from astropy
        return self._convolve(miss_2d, x_w, y_w)