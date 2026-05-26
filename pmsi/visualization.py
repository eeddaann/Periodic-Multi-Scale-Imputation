import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

def plot_periodicity_percentiles(data_list):
    """Plots MAE trends across weekly periodicity percentiles."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)
    
    colors = {'PMSI': '#1d3557', 'CNN': '#457b9d', 'LOCF': '#e76f51', 
              'PLI': '#f4a261', 'ARIMA': '#2a9d8f', 'KNN2D': '#7209b7'}
    
    plot_types = ['obs', 'mask']
    titles = ["A", "B"]
    
    for idx, stream_key in enumerate(plot_types):
        ax = axes[idx]
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        for model_entry in data_list:
            raw_name = model_entry['name']
            lookup_name = raw_name.upper() if raw_name.lower() in ['pmsi', 'cnn', 'locf', 'arima', 'knn'] else raw_name
            color = colors.get(lookup_name, '#64748b')
            
            metrics_dict = model_entry[stream_key]
            # Ensure sorting numerically by percentile strings
            sorted_percentiles = sorted(metrics_dict.keys(), key=int) 
            mae_values = [metrics_dict[pct] for pct in sorted_percentiles]
            
            if lookup_name == 'PMSI':
                lw, ms, alpha, zorder, label_name = 3.8, 8, 1.0, 10, r"$\mathbf{PMSI\ }$"
            else:
                lw, ms, alpha, zorder, label_name = 1.8, 5, 0.35, 2, raw_name

            ax.plot(sorted_percentiles, mae_values, marker='o', label=label_name, 
                    color=color, linewidth=lw, markersize=ms, alpha=alpha, zorder=zorder)
            
        ax.set_title(titles[idx], fontsize=23, fontweight='bold', pad=12, color='#212529', loc='left')
        ax.set_ylabel("MAE (BPM)", fontsize=16, labelpad=8)
        
        if idx == 0:
            ax.set_xlabel("Observation Weekly Autocorrelation Percentile Range", fontsize=18, fontweight='bold', labelpad=10)
        else:
            ax.set_xlabel("Missingness Weekly Autocorrelation Percentile Range", fontsize=18, fontweight='bold', labelpad=10)
            
        tick_labels = []
        prev_edge = "0"
        for pct in sorted_percentiles:
            tick_labels.append(f"{prev_edge}-{pct}%")
            prev_edge = pct
            
        ax.set_xticks(range(len(sorted_percentiles)))
        ax.set_xticklabels(tick_labels, fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.grid(axis='both', alpha=0.15, linestyle='--')
        
        # Sort legend
        handles, labels = ax.get_legend_handles_labels()
        pmsi_idx = next((i for i, l in enumerate(labels) if "PMSI" in l), None)
        if pmsi_idx is not None:
            ordered_handles = [handles.pop(pmsi_idx)] + handles
            ordered_labels = [labels.pop(pmsi_idx)] + labels
        else:
            ordered_handles, ordered_labels = handles, labels

        if idx == 0:
            ax.legend(ordered_handles, ordered_labels, fontsize=15, loc='best', frameon=False)

    # Match Y-Axes
    ymin_0, ymax_0 = axes[0].get_ylim()
    ymin_1, ymax_1 = axes[1].get_ylim()
    global_ymin, global_ymax = min(ymin_0, ymin_1), max(ymax_0, ymax_1)
    
    for ax in axes:
        ax.set_ylim(global_ymin, global_ymax)

    plt.tight_layout()
    plt.savefig("Figure_3_Reproduced.png", dpi=300)
    plt.show()
    plt.close()

def plot_pmsi_kernel(model, title_suffix=""):
    """
    Extracts the optimized parameters from a fitted PMSIImputer instance
    and visualizes the resulting 2D Gaussian kernel.
    """
    if not hasattr(model, 'best_params_') or model.best_params_ is None:
        raise ValueError("The provided PMSIImputer model has not been fitted yet. Please run .fit() first.")
    
    # Extract optimized parameters safely (handling different key naming styles)
    x_std = model.best_params_.get('x', model.best_params_.get('x_stddev'))
    y_std = model.best_params_.get('y', model.best_params_.get('y_stddev', x_std))
    
    # Retrieve the exact kernel array from the model cache
    kernel_obj = model._get_kernel(x_std, y_std)
    kernel_matrix = kernel_obj.array
    
    # Set up the plot layout
    plt.figure(figsize=(7, 6))
    
    # Create the heatmap
    ax = sns.heatmap(
        kernel_matrix, 
        cmap='viridis', 
        annot=True, 
        fmt=".3f", 
        annot_kws={"size": 8},
        cbar_kws={'label': 'Weight Intensity'}
    )
    
    # Configure titles and labels
    title_text = f"Optimized PMSI 2D Gaussian Kernel {title_suffix}\n" \
                 f"Grid Size: {model.x_size}x{model.y_size} | " \
                 f"$\sigma_x$ = {x_std:.4f}, $\sigma_y$ = {y_std:.4f}"
    
    plt.title(title_text, fontsize=11, fontweight='semibold', pad=12)
    plt.xlabel("Temporal Proximity: Hours ($x$-axis)", fontsize=10, labelpad=8)
    plt.ylabel("Periodic Proximity: Days ($y$-axis)", fontsize=10, labelpad=8)
    
    # Center the tick marks
    plt.xticks(np.arange(model.x_size) + 0.5, labels=range(model.x_size))
    plt.yticks(np.arange(model.y_size) + 0.5, labels=range(model.y_size))
    
    plt.tight_layout()
    plt.show()

def plot_data_intuition(pivot_d, binary_d, good_k, binary_masks):
    """
    Plots the structural data pipeline for the first participant-mask pair.
    Unifies BOTH matrices on the weekly cycle level (4 weeks x 168 hours).
    """
    first_orig_key = good_k[0]
    first_mask_key = binary_masks[0]
    
    orig_df = pivot_d[first_orig_key]
    if "week_start" in orig_df.columns:
        flat_data = orig_df.drop(["week_start"], axis=1).values.flatten()
    else:
        flat_data = orig_df.values.flatten()
        
    mask_2d = binary_d[first_mask_key]
    
    # -------------------------------------------------------------------------
    # CRITICAL DIMENSION UNIFICATION (4 Weeks x 168 Hours)
    # -------------------------------------------------------------------------
    try:
        # Reshape the flat 672 data sequence into weekly blocks to match the mask
        matrix_2d = flat_data.reshape((4, 168))
        # Ensure the mask is also explicitly viewed in this same shape
        mask_view_2d = mask_2d.reshape((4, 168))
    except ValueError:
        print(f"Error: Could not reshape data array of size {flat_data.size} into (4, 168)")
        return

    # Initialize the 3-panel plotting grid
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # -------------------------------------------------------------------------
    # Panel 1: 1D Continuous Time Series Line Plot
    # -------------------------------------------------------------------------
    axes[0].plot(flat_data, color='#2c3e50', linewidth=1.5, label='Observed Signal')
    axes[0].set_title(f"1. Continuous 1D Time Series View (Participant: {first_orig_key})", 
                      fontsize=12, fontweight='bold', pad=8)
    axes[0].set_xlabel("Timeline Evolution (Continuous Hours: 0 to 671)", fontsize=10)
    axes[0].set_ylabel("Physiological Value", fontsize=10)
    axes[0].set_xlim(0, 672)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Add vertical lines at every 168-hour mark to visually show the week boundaries
    for week_boundary in [168, 336, 504]:
        axes[0].axvline(x=week_boundary, color='darkorange', linestyle=':', alpha=0.7, linewidth=1.5)
    
    # -------------------------------------------------------------------------
    # Panel 2: 2D Structured Pivot Matrix (Weekly Cycle View)
    # -------------------------------------------------------------------------
    sns.heatmap(matrix_2d, cmap='viridis', ax=axes[1], 
                cbar=True, cbar_kws={'label': 'Value Magnitude', 'shrink': 0.8, 'pad': 0.02})
    axes[1].set_title("2. 2D Reshaped Matrix View (Weekly Periodicity: 4 Weeks x 168 Hours)", 
                      fontsize=12, fontweight='bold', pad=8)
    axes[1].set_xlabel("Hour of the Week (0 - 167)", fontsize=10)
    axes[1].set_ylabel("Week Index (0 - 3)", fontsize=10)
    
    # Mark standard 24-hour day increments along the weekly x-axis to make it easily readable
    axes[1].set_xticks(range(0, 169, 24))
    axes[1].set_xticklabels(range(0, 169, 24))

    # -------------------------------------------------------------------------
    # Panel 3: 2D Binary Missingness Mask Matrix (Weekly Cycle View)
    # -------------------------------------------------------------------------
    sns.heatmap(mask_view_2d.astype(int), cmap=sns.xkcd_palette(['light grey', 'dark red']), 
                ax=axes[2], cbar=False)
    axes[2].set_title(f"3. 2D Binary Missingness Mask (Mask Pattern: {first_mask_key})", 
                      fontsize=12, fontweight='bold', pad=8)
    axes[2].set_xlabel("Hour of the Week (0 - 167)", fontsize=10)
    axes[2].set_ylabel("Week Index (0 - 3)", fontsize=10)
    
    axes[2].set_xticks(range(0, 169, 24))
    axes[2].set_xticklabels(range(0, 169, 24))
    
    # Place custom legend elements inside the right pad area
    legend_elements = [
        Patch(facecolor='lightgrey', edgecolor='gray', label='Observed (0)'),
        Patch(facecolor='darkred', edgecolor='gray', label='Missing Block (1)')
    ]
    axes[2].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), title="Data State")

    plt.tight_layout()
    plt.show()