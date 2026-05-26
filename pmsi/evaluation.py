import numpy as np
import pandas as pd
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import tracemalloc
import psutil
import gc
from tqdm.auto import tqdm
import ast
from statsmodels.tsa.stattools import acf

# --- 1. Helper Classes and Constants ---
METRIC_NAMES = ["RMSE", "MAE", "MAPE", "SMAPE", "Bias", "MedAE"]

class ResourceTracker:
    """Records wall-clock time, tracemalloc peak (Python-level allocations),
    and psutil RSS delta (catches numpy/C allocations but is noisier)."""
    def __init__(self):
        self._proc = psutil.Process()

    def __enter__(self):
        gc.collect()
        self._rss0 = self._proc.memory_info().rss
        tracemalloc.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed_sec = time.perf_counter() - self._t0
        _, peak_py = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_python_mb = peak_py / (1024 ** 2)
        rss1 = self._proc.memory_info().rss
        self.rss_delta_mb = (rss1 - self._rss0) / (1024 ** 2)
        return False


# --- 2. Helper Functions ---
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    mape_eps: float = 1e-8) -> Dict[str, float]:
    """Compute multiple error metrics on a single (y_true, y_pred) pair.

    Returns
    -------
    RMSE   : root mean squared error
    MAE    : mean absolute error
    MAPE   : mean absolute percentage error (skips |y_true| < eps), %
    SMAPE  : symmetric MAPE in [0, 200], well-defined at zero, %
    Bias   : mean signed error (y_pred - y_true)
    MedAE  : median absolute error (robust to outliers)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    abs_err = np.abs(err)

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(abs_err))
    bias = float(np.mean(err))
    medae = float(np.median(abs_err))

    safe = np.abs(y_true) > mape_eps
    mape = float(np.mean(abs_err[safe] / np.abs(y_true[safe])) * 100) \
        if safe.any() else np.nan

    denom = np.abs(y_true) + np.abs(y_pred)
    smape_terms = np.where(denom > 0, 2 * abs_err / denom, 0.0)
    smape = float(np.mean(smape_terms) * 100)

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape,
            "SMAPE": smape, "Bias": bias, "MedAE": medae}

def bootstrap_ci(values: np.ndarray, n_replicates: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    """Percentile-bootstrap CI for the mean of `values`.
    Returns (mean, lower, upper). NaNs dropped first."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_replicates, n))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)

# --- 3. Main Evaluation Methods ---
def evaluate_method(
    name: str,
    impute_fn: Callable,
    seed: int,
    pivot_d: Dict, binary_d: Dict, good_k: List, binary_masks: List,
    is_2d: bool = False,
    shape_2d: Tuple[int, int] = (28, 24),
    n_per_key: int = 100,
    n_bootstrap: int = 10_000,
    impute_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate ONE imputation method end-to-end.

    Steps printed / shown progressively:
      1. Header banner
      2. Pair-construction message
      3. tqdm progress bar over pairs (+ resource tracking)
      4. Resource summary (time, peak Python mem, RSS delta)
      5. One line per metric with mean and 95% bootstrap CI
      6. Per-pair CSV + per-method summary CSV saved (if output_dir given)

    Returns
    -------
    dict with keys: name, seed, df_pairs, df_summary, resources
    Pass to combine_results() or compare_methods().
    """
    impute_kwargs = impute_kwargs or {}

    print(f"\n{'=' * 72}")
    print(f"  METHOD: {name}    (seed = {seed})")
    print('=' * 72)

    # --- Step 1: build pairs with the method-specific seed ------------------
    random.seed(seed)
    np.random.seed(seed)
    N = len(good_k) * n_per_key
    pairs = list(zip(n_per_key * list(good_k),
                     random.choices(binary_masks, k=N)))
    print(f"  Built {len(pairs)} (data, mask) pairs.")

    # --- Step 2: per-pair evaluation with progress bar ----------------------
    rows: List[Dict[str, Any]] = []
    skipped = 0
    with ResourceTracker() as rt:
        bar = tqdm(pairs, desc=f"  Evaluating {name}", leave=False, ncols=80)
        for orig_key, mask_key in bar:
            flat = pivot_d[orig_key].drop(["week_start"], axis=1).values.flatten()
            miss = flat.copy().astype(float)
            miss[binary_d[mask_key].flatten().astype(bool)] = np.nan
            mask = np.isnan(miss)
            if mask.sum() == 0:
                skipped += 1
                continue
            if is_2d:
                try:
                    miss_2d = miss.reshape(shape_2d)
                except ValueError:
                    skipped += 1
                    continue
                imputed_2d = impute_fn(miss_2d, **impute_kwargs)
                imputed_flat = np.asarray(imputed_2d).flatten()
            else:
                imputed_flat = impute_fn(miss, **impute_kwargs)
            metrics = compute_metrics(flat[mask], imputed_flat[mask])
            metrics.update({"orig_key": str(orig_key),
                            "mask_key": str(mask_key),
                            "n_missing": int(mask.sum())})
            rows.append(metrics)

    df_pairs = pd.DataFrame(rows)
    n_eval = len(rows)

    # --- Step 3: resource summary -------------------------------------------
    print(f"  Pairs evaluated:    {n_eval}    (skipped: {skipped})")
    print(f"  Wall-clock runtime: {rt.elapsed_sec:9.3f} s   "
          f"({rt.elapsed_sec / max(n_eval, 1) * 1000:.2f} ms/pair)")
    print(f"  Peak Python memory: {rt.peak_python_mb:9.3f} MB  (tracemalloc)")
    print(f"  RSS delta:          {rt.rss_delta_mb:9.3f} MB  (psutil)")

    # --- Step 4: bootstrap CI per metric, printed as we go ------------------
    print(f"\n  Bootstrap 95% CIs  (n_replicates = {n_bootstrap}):")
    print(f"  {'Metric':<7s}  {'Mean':>10s}    {'95% CI':<28s}")
    print(f"  {'-' * 7}  {'-' * 10}    {'-' * 28}")
    summary_rows = []
    for i, m in enumerate(METRIC_NAMES):
        if m not in df_pairs.columns:
            continue
        vals = df_pairs[m].values
        pt, lo, hi = bootstrap_ci(vals, n_replicates=n_bootstrap, seed=seed + i)
        print(f"  {m:<7s}  {pt:10.4f}    [{lo:10.4f}, {hi:10.4f}]")
        summary_rows.append({"Method": name, "Metric": m,
                             "Mean": pt, "CI_lower": lo, "CI_upper": hi,
                             "N": int(np.sum(~np.isnan(vals)))})
    df_summary = pd.DataFrame(summary_rows)

    # --- Step 5: save -------------------------------------------------------
    if output_dir is not None:
        out_pairs = Path(output_dir) / "per_pair_metrics" / f"{name.lower()}_metrics.csv"
        out_summary = Path(output_dir) / "per_method_summary" / f"{name.lower()}_summary.csv"
        df_pairs.to_csv(out_pairs, index=False)
        df_summary.to_csv(out_summary, index=False)
        print(f"\n  [i/o] Per-pair metrics -> {out_pairs}")
        print(f"  [i/o] Summary          -> {out_summary}")

    resources = {
        "method": name,
        "seed": seed,
        "n_pairs_evaluated": n_eval,
        "total_runtime_sec": rt.elapsed_sec,
        "mean_runtime_per_pair_sec": rt.elapsed_sec / max(n_eval, 1),
        "peak_python_mem_mb": rt.peak_python_mb,
        "rss_delta_mb": rt.rss_delta_mb,
    }
    return {
        "name": name, "seed": seed,
        "df_pairs": df_pairs, "df_summary": df_summary,
        "resources": resources,
    }

def combine_results(results_list: List[Dict[str, Any]],
                    output_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Aggregate per-method results into:
      - long_summary : one row per (method, metric)   with Mean / CI bounds
      - wide_summary : one row per method, columns "<metric>_mean" + "<metric>_CI"
      - resources    : runtime and memory per method

    All tables are saved to output_dir if given.
    """
    long_summary = pd.concat([r["df_summary"] for r in results_list], ignore_index=True)

    wide_rows = []
    for r in results_list:
        row = {"Method": r["name"], "seed": r["seed"]}
        for _, m in r["df_summary"].iterrows():
            row[f"{m['Metric']}_mean"] = round(m["Mean"], 4)
            row[f"{m['Metric']}_CI"] = f"[{m['CI_lower']:.4f}, {m['CI_upper']:.4f}]"
        wide_rows.append(row)
        
    wide_summary = pd.DataFrame(wide_rows)
    resources = pd.DataFrame([r["resources"] for r in results_list])

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        long_summary.to_csv(out_path / "all_methods_long.csv", index=False)
        wide_summary.to_csv(out_path / "all_methods_wide.csv", index=False)
        resources.to_csv(out_path / "all_methods_runtime_memory.csv", index=False)

    print("\n" + "=" * 72)
    print("  COMBINED SUMMARY  (all methods, all metrics)")
    print("=" * 72)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(wide_summary.to_string(index=False))

    print("\n" + "=" * 72)
    print("  RUNTIME & MEMORY")
    print("=" * 72)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(resources.round(3).to_string(index=False))

    return {
        "long_summary": long_summary,
        "wide_summary": wide_summary,
        "resources": resources
    }

def dict2percentile(d: Dict) -> Dict:
    """Helper: Converts a dictionary of scores into 5 equal 20% percentile bins."""
    s = pd.Series(d).dropna()
    if s.empty:
        return {}
        
    bins = pd.qcut(
        s.rank(method='first'), 
        q=5, 
        labels=["20", "40", "60", "80", "100"] 
    )
    return bins.astype(str).to_dict()

def stratify_results_by_periodicity(results_list: List[Dict], 
                                    pivot_d: Dict, 
                                    binary_d: Dict, 
                                    good_k: List) -> List[Dict]:
    """
    Calculates 168-hour ACF for observations and masks, bins them into 5 percentiles,
    and maps the evaluation MAE results to these bins for plotting.
    """
    # 1. Observation ACF
    obs_weekly_autocorr_d = {}
    for k in good_k:
        flat = pivot_d[k].drop(["week_start"], axis=1).values.flatten()
        flat_clean = flat[~np.isnan(flat)] 
        
        if len(flat_clean) > 168:
            scores = acf(flat_clean, nlags=168, fft=True)
            obs_weekly_autocorr_d[k] = scores[168] if len(scores) > 168 else 0.0
        else:
            obs_weekly_autocorr_d[k] = 0.0
            
    obs_weekly_autocorr_pct = dict2percentile(obs_weekly_autocorr_d)

    # 2. Mask ACF
    mask_weekly_autocorr_d = {}
    for k in binary_d:
        flat_mask = binary_d[k].flatten()
        if len(flat_mask) > 168:
            scores = acf(flat_mask, nlags=168, fft=True)
            mask_weekly_autocorr_d[k] = scores[168] if len(scores) > 168 else 0.0
        else:
            mask_weekly_autocorr_d[k] = 0.0
            
    mask_weekly_autocorr_pct = dict2percentile(mask_weekly_autocorr_d)

    # 3. Map Bins to Evaluation Results
    formatted_results = []
    for res in results_list:
        res_name = res["name"]
        df = res["df_pairs"].copy()
        
        try:
            mask_keys = (df["mask_key"].astype(str)
                         .str.replace("np.int16(", "", regex=False)
                         .str.replace("),", ",", regex=False)
                         .apply(ast.literal_eval))
            obs_keys = (df["orig_key"].astype(str)
                        .str.replace("np.int16(", "", regex=False)
                        .str.replace("),", ",", regex=False)
                        .apply(ast.literal_eval))
        except Exception:
            mask_keys = df["mask_key"]
            obs_keys = df["orig_key"]

        df["mask_pct"] = mask_keys.map(mask_weekly_autocorr_pct)
        df["obs_pct"] = obs_keys.map(obs_weekly_autocorr_pct)
        
        obs_dict = df.groupby("obs_pct")["MAE"].mean().to_dict()
        mask_dict = df.groupby("mask_pct")["MAE"].mean().to_dict()
        
        formatted_results.append({
            "name": res_name, 
            "obs": obs_dict, 
            "mask": mask_dict
        })
        
    return formatted_results

    """
    Runs all imputers and stratifies MAE by ACF percentiles.
    """
    n_patients = observed.shape[0]
    
    # 1. Calculate ACF for observation and missingness per patient
    obs_acf = [calculate_acf(pd.Series(observed[i]).dropna(), lag=168) for i in range(n_patients)]
    mask_acf = [calculate_acf(masks[i], lag=168) for i in range(n_patients)]
    
    # 2. Assign patients to 5 equal-population percentile bins (0-20, 20-40, etc.)
    obs_bins = pd.qcut(obs_acf, q=5, labels=["20", "40", "60", "80", "100"])
    mask_bins = pd.qcut(mask_acf, q=5, labels=["20", "40", "60", "80", "100"])
    
    results = []
    
    # 3. Run Models and Calculate MAE
    for model in models:
        print(f"Running {model.name}...")
        imputed_data = model.impute(observed)
        
        # Dictionary to hold MAE for this model
        model_metrics = {'name': model.name, 'obs': {}, 'mask': {}}
        
        # Calculate MAE only on the artificially masked missing values
        missing_indices = (masks == 0)
        
        # Stratify by Observation ACF Bins
        for pct in ["20", "40", "60", "80", "100"]:
            idx = (obs_bins == pct)
            mae = np.nanmean(np.abs(ground_truth[idx][missing_indices[idx]] - imputed_data[idx][missing_indices[idx]]))
            model_metrics['obs'][pct] = mae
            
        # Stratify by Missingness ACF Bins
        for pct in ["20", "40", "60", "80", "100"]:
            idx = (mask_bins == pct)
            mae = np.nanmean(np.abs(ground_truth[idx][missing_indices[idx]] - imputed_data[idx][missing_indices[idx]]))
            model_metrics['mask'][pct] = mae
            
        results.append(model_metrics)
        
    return results