import random
import numpy as np
import pandas as pd
import warnings
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy.signal import find_peaks, stft


class PMSIImputer:
    """
    Periodic Multi-Scale Imputation (PMSI) Framework.
    """
    def __init__(self, x_size: int = 7, y_size: int = 7, sizes: tuple = None, n_evals: int = 300, seed: int = 134, pre_fit_params: dict = None):
        self.x_size = x_size
        self.y_size = y_size
        if sizes is not None:
            self.sizes = tuple(sizes)
        else:
            self.sizes = (y_size, x_size)
        self.ndim = len(self.sizes)
        
        self.pad_width = max(self.sizes) // 2
        self.n_evals = n_evals
        self.seed = seed
        
        self.best_params_ = pre_fit_params 
        self.trials_ = None
        self._kernels_cache = {}

    def _get_kernel(self, *args):
        # Determine stddevs from arguments
        if len(args) == 1 and isinstance(args[0], (list, tuple, np.ndarray)):
            stddevs = list(args[0])
        else:
            # Backward compatibility for 2D args (x_stddev, y_stddev)
            if len(args) == 2:
                x_std = args[0]
                y_std = args[1] if args[1] is not None else x_std
                stddevs = [y_std, x_std]
            elif len(args) == 1:
                stddevs = [args[0], args[0]]
            else:
                raise ValueError("Invalid standard deviations passed to _get_kernel.")

        k_key = (tuple(stddevs), self.sizes)
        if k_key not in self._kernels_cache:
            grids = []
            for i in range(self.ndim):
                size = self.sizes[i]
                stddev = stddevs[i]
                if stddev <= 0:
                    stddev = 1e-5
                center = size // 2
                x = np.arange(size) - center
                g = np.exp(-0.5 * (x / stddev)**2)
                
                shape = [1] * self.ndim
                shape[i] = size
                grids.append(g.reshape(shape))
                
            kernel_array = grids[0]
            for g in grids[1:]:
                kernel_array = kernel_array * g
                
            k_sum = np.sum(kernel_array)
            if k_sum > 0:
                kernel_array = kernel_array / k_sum
                
            self._kernels_cache[k_key] = kernel_array
            
        # Return a kernel object containing the .array property for compatibility with visualization code
        class CompatibleKernel:
            def __init__(self, arr):
                self.array = arr
        return CompatibleKernel(self._kernels_cache[k_key])

    def _convolve(self, img_nd: np.ndarray, stddevs: list, mode: str = 'edge', standard_convolve: bool = False) -> np.ndarray:
        if self.ndim == 2 and not standard_convolve:
            y_std, x_std = stddevs[0], stddevs[1]
            kernel = Gaussian2DKernel(x_stddev=x_std, y_stddev=y_std, x_size=self.sizes[1], y_size=self.sizes[0])
            pad_width = max(self.sizes) // 2
            padded = np.pad(img_nd.copy(), mode=mode, pad_width=pad_width)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                interpolated = interpolate_replace_nans(padded, kernel)
            result = interpolated[pad_width:-pad_width, pad_width:-pad_width]
            
            # Fallback: fill any remaining NaNs with the mean of observed values in img_nd
            if np.any(np.isnan(result)):
                fill_val = np.nanmean(img_nd)
                if np.isnan(fill_val):
                    fill_val = 0.0
                result = np.where(np.isnan(result), fill_val, result)
            return result

        kernel_obj = self._get_kernel(stddevs)
        # Collapse any singleton dimensions in the kernel so its rank matches
        # the dimensionality of the input image. This is required for cases
        # like the 3‑D ablation where the kernel is defined with extra size‑1
        # axes (e.g., (3, 7, 1, 1, 5)).
        kernel_array = np.squeeze(kernel_obj.array)

        # If the kernel still has more dimensions than the input image, collapse the extra dimensions.
        # Extra dimensions arise from size‑1 axes in the user‑provided `sizes` tuple (e.g., (3,7,1,1,5)).
        # These dimensions do not correspond to actual axes of the data and should not affect convolution.
        # We iteratively sum over leading axes until the kernel rank matches the image rank.
        while kernel_array.ndim > img_nd.ndim:
            # Sum over the first axis (any axis works because singleton dimensions have size 1 after squeeze)
            kernel_array = kernel_array.sum(axis=0)
        # If kernel has fewer dimensions than the image (e.g., 1‑D kernel on 2‑D data), expand it
        while kernel_array.ndim < img_nd.ndim:
            kernel_array = np.expand_dims(kernel_array, axis=0)

        # Normalize kernel after shape adjustments
        kernel_sum = np.sum(kernel_array)
        if kernel_sum > 0:
            kernel_array = kernel_array / kernel_sum

        if standard_convolve:
            import scipy.ndimage
            scipy_mode = 'nearest' if mode == 'edge' else mode
            res_scipy = scipy.ndimage.convolve(img_nd, kernel_array, mode=scipy_mode)
            if np.any(np.isnan(res_scipy)):
                fill_val = np.nanmean(img_nd)
                if np.isnan(fill_val):
                    fill_val = 0.0
                res_scipy = np.where(np.isnan(res_scipy), fill_val, res_scipy)
            return res_scipy
            
        # Adjust pad widths to match the dimensionality of the input image.
        # Some ablation configurations use a 5‑D kernel (e.g., sizes=(3,7,1,1,5))
        # but evaluate on a 3‑D array. NumPy requires the pad_width list length to
        # equal the number of dimensions of the array, so we slice self.sizes to the
        # appropriate length.
        effective_sizes = self.sizes[:img_nd.ndim]
        pad_widths = [(size // 2, size // 2) for size in effective_sizes]
        padded = np.pad(img_nd.copy(), mode=mode, pad_width=pad_widths)

        # Unified convolution handling NaNs via mask for any dimensionality
        import scipy.ndimage
        scipy_mode = 'nearest' if mode == 'edge' else mode
        mask = (~np.isnan(padded)).astype(float)
        padded_zero = np.where(np.isnan(padded), 0.0, padded)
        conv_array = scipy.ndimage.convolve(padded_zero, kernel_array, mode=scipy_mode)
        conv_mask = scipy.ndimage.convolve(mask, kernel_array, mode=scipy_mode)
        with np.errstate(invalid='ignore', divide='ignore'):
            interpolated = np.where(conv_mask > 0, conv_array / conv_mask, np.nan)
        # Preserve original non‑NaN values
        padded_result = padded.copy()
        nan_mask = np.isnan(padded)
        padded_result[nan_mask] = interpolated[nan_mask]

        slices = tuple(slice(pw, -pw if pw > 0 else None) for pw, _ in pad_widths)
        result = padded_result[slices]
        
        # Fallback: fill any remaining NaNs with the mean of observed values in img_nd
        if np.any(np.isnan(result)):
            fill_val = np.nanmean(img_nd)
            if np.isnan(fill_val):
                fill_val = 0.0
            result = np.where(np.isnan(result), fill_val, result)
        return result


    def _get_stddevs(self, weights):
        try:
            return [weights[f'x_{i}'] for i in range(self.ndim)]
        except KeyError:
            if 'x' in weights and 'y' in weights:
                return [weights['y'], weights['x']]
            raise KeyError(f"Could not resolve standard deviations from weights: {list(weights.keys())}")

    def fit(self, pivot_d: dict, binary_d: dict, good_k: list, binary_masks: list, n_pairs_per_key: int = 10, shape: tuple = None):
        rng = random.Random(self.seed)
        np.random.seed(self.seed)

        # Determine target shape once, before the objective function
        if shape is not None:
            target_shape = shape
        elif self.ndim == 2:
            target_shape = (28, 24)
        else:
            # Infer shape from first pivot entry
            first_key = good_k[0]
            first_val = pivot_d[first_key]
            if isinstance(first_val, pd.DataFrame):
                if "week_start" in first_val.columns:
                    target_shape = first_val.drop(["week_start"], axis=1).shape
                else:
                    target_shape = first_val.shape
            else:
                target_shape = np.asarray(first_val).shape
            # Flatten for 1‑D models
            if self.ndim == 1:
                target_shape = (int(np.prod(target_shape)),)
        N = n_pairs_per_key * len(good_k)
        pairs = list(zip(n_pairs_per_key * list(good_k), rng.choices(binary_masks, k=N)))

        def objective(weights):
            try:
                tmp_rmse_list = []

                for orig_key, mask_key in pairs:
                    orig_df = pivot_d[orig_key]
                    if isinstance(orig_df, pd.DataFrame):
                        if "week_start" in orig_df.columns:
                            orig_flat = orig_df.drop(["week_start"], axis=1).values.flatten()
                        else:
                            orig_flat = orig_df.values.flatten()
                    else:
                        orig_flat = np.asarray(orig_df).flatten()
                    
                    try:
                        orig_nd = orig_flat.reshape(target_shape)
                    except ValueError:
                        continue
                        
                    missing_flat = orig_flat.copy().astype(float)
                    
                    mask_df = binary_d[mask_key]
                    if isinstance(mask_df, pd.DataFrame):
                        mask_flat = mask_df.values.flatten()
                    else:
                        mask_flat = np.asarray(mask_df).flatten()
                        
                    mask_bool = mask_flat.astype(bool)
                    missing_flat[mask_bool] = np.nan
                    
                    try:
                        missing_nd = missing_flat.reshape(target_shape)
                    except ValueError:
                        continue
                    
                    missing_indices = np.isnan(missing_nd)
                    n_miss = np.sum(missing_indices)
                    
                    if n_miss == 0:
                        continue
                        
                    stddevs = self._get_stddevs(weights)
                    imputed_nd = self._convolve(missing_nd, stddevs)
                    
                    imputed_values = imputed_nd[missing_indices]
                    actual_values = orig_nd[missing_indices]
                    
                    valid_mask = ~np.isnan(imputed_values)
                    valid_n = np.sum(valid_mask)
                    
                    if valid_n == 0:
                        continue
                        
                    residuals = imputed_values[valid_mask] - actual_values[valid_mask]
                    rmse = np.sqrt(np.mean(residuals**2))
                    
                    tmp_rmse_list.append(rmse)

                if not tmp_rmse_list:
                    return {'loss': 1e10, 'status': STATUS_OK}
                    
                avg_rmse = np.mean(tmp_rmse_list)
                if np.isnan(avg_rmse):
                    return {'loss': 1e10, 'status': STATUS_OK}
                    
                return {'loss': float(avg_rmse), 'status': STATUS_OK}
                
            except Exception as e:
                print(f"Objective Error: {e}")
                return {'loss': 1e10, 'status': STATUS_OK}

        hp_space = {
            f'x_{i}': hp.uniform(f'x_{i}', 0.1, 5.0) for i in range(self.ndim)
        }
        
        self.trials_ = Trials()
        print(f"Fitting PMSI: Running {self.n_evals} evals via Hyperopt TPE...")
        
        self.best_params_ = fmin(
            fn=objective, space=hp_space, algo=tpe.suggest,
            trials=self.trials_, max_evals=self.n_evals,
            rstate=np.random.default_rng(self.seed),
            show_progressbar=False
        )
        
        print(f"Fit Complete. Optimal Params: {self.best_params_}")
        return self

    def impute(self, miss_nd: np.ndarray, **kwargs) -> np.ndarray:
        stddevs = None
        
        if 'stddevs' in kwargs:
            stddevs = list(kwargs['stddevs'])
        elif any(f'x_{i}' in kwargs for i in range(self.ndim)):
            stddevs = [kwargs.get(f'x_{i}', self.best_params_[f'x_{i}'] if self.best_params_ else 1.0) for i in range(self.ndim)]
        elif self.best_params_ and all(f'x_{i}' in self.best_params_ for i in range(self.ndim)):
            stddevs = [self.best_params_[f'x_{i}'] for i in range(self.ndim)]
        elif 'x_stddev' in kwargs or 'x' in kwargs or 'y_stddev' in kwargs or 'y' in kwargs:
            x_w = kwargs.get('x_stddev', kwargs.get('x', self.best_params_.get('x', self.best_params_.get('x_1', None)) if self.best_params_ else None))
            y_w = kwargs.get('y_stddev', kwargs.get('y', self.best_params_.get('y', self.best_params_.get('x_0', None)) if self.best_params_ else x_w))
            if x_w is not None:
                stddevs = [y_w if y_w is not None else x_w, x_w]
        elif self.best_params_ and ('x' in self.best_params_ or 'y' in self.best_params_):
            x_w = self.best_params_.get('x', self.best_params_.get('x_1', 1.0))
            y_w = self.best_params_.get('y', self.best_params_.get('x_0', x_w))
            stddevs = [y_w, x_w]
            
        if stddevs is None:
            if self.ndim == 2:
                stddevs = [1.0, 1.0]
            else:
                raise ValueError("PMSIImputer requires standard deviation parameters via fit, pre_fit_params, or kwargs.")
                
        return self._convolve(miss_nd, stddevs)

    def compute_aic(self, pivot_d: dict, binary_d: dict, good_k: list, binary_masks: list, n_pairs_per_key: int = 10, shape: tuple = None) -> float:
        """
        Computes the standardized AIC score of the fitted imputer on a specified set of pairs.
        Every missing element is accounted for (does not ignore nulls since _convolve has fallback).
        """
        # Determine target shape
        if shape is not None:
            target_shape = shape
        elif self.ndim == 2:
            target_shape = (28, 24)
        else:
            first_key = good_k[0]
            first_val = pivot_d[first_key]
            if isinstance(first_val, pd.DataFrame):
                if "week_start" in first_val.columns:
                    target_shape = first_val.drop(["week_start"], axis=1).shape
                else:
                    target_shape = first_val.shape
            else:
                target_shape = np.asarray(first_val).shape
            if self.ndim == 1:
                target_shape = (int(np.prod(target_shape)),)

        rng = random.Random(self.seed)
        np.random.seed(self.seed)
        N = n_pairs_per_key * len(good_k)
        pairs = list(zip(n_pairs_per_key * list(good_k), rng.choices(binary_masks, k=N)))

        tmp_aic_list = []
        for orig_key, mask_key in pairs:
            orig_df = pivot_d[orig_key]
            if isinstance(orig_df, pd.DataFrame):
                if "week_start" in orig_df.columns:
                    orig_flat = orig_df.drop(["week_start"], axis=1).values.flatten()
                else:
                    orig_flat = orig_df.values.flatten()
            else:
                orig_flat = np.asarray(orig_df).flatten()
            
            try:
                orig_nd = orig_flat.reshape(target_shape)
            except ValueError:
                continue

            missing_flat = orig_flat.copy().astype(float)
            mask_df = binary_d[mask_key]
            mask_flat = mask_df.values.flatten() if isinstance(mask_df, pd.DataFrame) else np.asarray(mask_df).flatten()
            missing_flat[mask_flat.astype(bool)] = np.nan

            try:
                missing_nd = missing_flat.reshape(target_shape)
            except ValueError:
                continue

            missing_indices = np.isnan(missing_nd)
            n_miss = np.sum(missing_indices)
            if n_miss == 0:
                continue

            imputed_nd = self.impute(missing_nd)
            imputed_values = imputed_nd[missing_indices]
            actual_values = orig_nd[missing_indices]

            residuals = imputed_values - actual_values
            rss = max(np.sum(residuals**2), 1e-12)

            log_likelihood = -0.5 * n_miss * np.log(rss / n_miss)
            aic = 2 * self.ndim - 2 * log_likelihood
            tmp_aic_list.append(aic)

        return float(np.mean(tmp_aic_list)) if tmp_aic_list else 1e10


