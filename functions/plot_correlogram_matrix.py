import os
import numpy as np
import matplotlib.pyplot as plt
from functions.correlogram import correlogram
from matplotlib import patches as mpatches
from matplotlib.lines import Line2D
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import sys

"""
Definition of problematic correlograms:
- For autocorrelograms: if either center bin exceeds global_threshold (mean + 2 stds of all correlogram center bins), or if the global peak is immediately outside the center bins.
- For cross-correlograms: if both center bins are the minima for the correlogram.
"""

def compute_correlogram(i, j, prefiltered_spikes, binsize, limit):
    # Use prefiltered spike times.
    t1 = prefiltered_spikes[i]
    t2 = prefiltered_spikes[j]
    is_auto = (i == j)
    counts, bins = correlogram(t1, t2=t2, binsize=binsize, limit=limit, auto=is_auto, density=False)
    # Trim counts if necessary.
    if len(counts) > len(bins) - 1:
        counts = counts[:-1]
    return (i, j, counts, bins)

def plot_correlogram_matrix(neurons_data, binsize, dataset_name, limit=0.02, time_window=None, save_folder=None, store_data=True):
    num_neurons = len(neurons_data)
    num_bins = int(2 * limit / binsize)
    common_bins = np.linspace(-limit, limit, num_bins + 1)
    common_bin_centers = (common_bins[:-1] + common_bins[1:]) / 2

    # Precompute center bin indices (same for every correlogram)
    if num_bins % 2 == 0:
        center_left = num_bins // 2 - 1
        center_right = num_bins // 2
    else:
        center_left = center_right = num_bins // 2

    problematic_neuron_indices = set()

    # Pre-filter spike times once per neuron.
    prefiltered_spikes = []
    for neuron in neurons_data:
        spikes = neuron[:3][2]  # Extract spike times
        if time_window is not None:
            spikes = spikes[(spikes >= time_window[0]) & (spikes <= time_window[1])]
        prefiltered_spikes.append(spikes)

    # List of pairs to process (i >= j)
    tasks = [(i, j) for i in range(num_neurons) for j in range(i+1)]

    # First pass: Compute all correlograms in parallel using joblib
    results = Parallel(n_jobs=-1)(
        delayed(compute_correlogram)(i, j, prefiltered_spikes, binsize, limit)
        for i, j in tqdm(tasks, desc="Computing correlogram", ncols=100)
    )

    # Allocate data structures.
    grid_data = [[None] * num_neurons for _ in range(num_neurons)]
    all_center_vals = []  # To compute global threshold
    correlogram_data = {} if store_data else None

    # Process parallel results.
    for i, j, counts, bins in results:
        # Compute bin centers.
        bin_centers = (bins[:-1] + bins[1:]) / 2
        grid_data[i][j] = {
            "counts": counts,
            "bins": bins,
            "bin_centers": bin_centers,
            "center_left": center_left,
            "center_right": center_right
        }
        # Save data if desired.
        if store_data:
            key = f"Neuron {i+1}" if i == j else f"Neuron {i+1} vs Neuron {j+1}"
            correlogram_data[key] = {"counts": counts, "bins": bins}
        # For autocorrelograms, collect center bin values.
        if i == j:
            all_center_vals.append(counts[center_left])
            all_center_vals.append(counts[center_right])

    # Compute the global threshold: mean of all center bin values plus two standard deviations.
    all_center_vals = np.array(all_center_vals)
    global_threshold = all_center_vals.mean() + 2 * all_center_vals.std()

    # Create subplot grid.
    fig, axes = plt.subplots(num_neurons, num_neurons, figsize=(num_neurons * 3, num_neurons * 3))

    # Second pass: Plot each correlogram.
    for i in tqdm(range(num_neurons), desc="Plotting correlograms", ncols=100):
        for j in range(i+1):
            data = grid_data[i][j]
            counts = data["counts"]
            bins = data["bins"]
            bin_centers = data["bin_centers"]
            center_line = (bin_centers[center_left] + bin_centers[center_right]) / 2

            # Determine if the correlogram is problematic.
            if i == j:
                # Autocorrelogram: problematic if either center bin exceeds threshold or global peak is immediately outside center bins.
                condition_A = (counts[center_left] > global_threshold or counts[center_right] > global_threshold)
                global_peak_index = int(np.argmax(counts))
                condition_B = (global_peak_index == center_left - 1 or global_peak_index == center_right + 1)
                is_problematic = condition_A or condition_B
                if is_problematic:
                    problematic_neuron_indices.add(i)
            else:
                # Cross-correlogram: problematic if a center bin is the minimum.
                is_problematic = (counts[center_left] == counts.min() or counts[center_right] == counts.min())
                if is_problematic:
                    problematic_neuron_indices.add(i)
                    problematic_neuron_indices.add(j)

            # Determine color for plotting.
            if is_problematic:
                color = '#FFFF99'
            else:
                color = '#77DD77' if i == j else '#CDA4DE'

            # Plot in the matrix.
            ax = axes[i, j] if num_neurons > 1 else axes
            ax.bar(bin_centers, counts, width=np.diff(bins), align='center', color=color, alpha=0.7,
                   edgecolor='k', linewidth=0.25)
            ax.set_xlim(-limit, limit)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axvline(center_line, color='black', linestyle='--', linewidth=0.5)

            # Highlight the center bins in pastel pink.
            pink_color = '#FFB6C1'
            ax.bar(bin_centers[center_left:center_left+1],
                   counts[center_left:center_left+1],
                   width=np.diff(bins)[center_left:center_left+1],
                   align='center', color=pink_color, alpha=1, edgecolor='k', linewidth=0.25)
            ax.bar(bin_centers[center_right:center_right+1],
                   counts[center_right:center_right+1],
                   width=np.diff(bins)[center_right:center_right+1],
                   align='center', color=pink_color, alpha=1, edgecolor='k', linewidth=0.25)

            # For the upper triangle, mirror the plot.
            if i != j:
                ax_mirror = axes[j, i] if num_neurons > 1 else axes
                ax_mirror.bar(-bin_centers, counts, width=np.diff(bins), align='center', color=color, alpha=0.7,
                              edgecolor='k', linewidth=0.25)
                ax_mirror.set_xlim(-limit, limit)
                ax_mirror.set_xticks([])
                ax_mirror.set_yticks([])
                ax_mirror.axvline(-center_line, color='black', linestyle='--', linewidth=0.5)
                ax_mirror.bar(-bin_centers[center_left:center_left+1],
                              counts[center_left:center_left+1],
                              width=np.diff(bins)[center_left:center_left+1],
                              align='center', color=pink_color, alpha=1, edgecolor='k', linewidth=0.25)
                ax_mirror.bar(-bin_centers[center_right:center_right+1],
                              counts[center_right:center_right+1],
                              width=np.diff(bins)[center_right:center_right+1],
                              align='center', color=pink_color, alpha=1, edgecolor='k', linewidth=0.25)

            # Label the first row and column.
            if i == 0:
                ax.set_title(f"Neuron {j+1}")
            if j == 0:
                ax.set_ylabel(f"Neuron {i+1}")

    plt.suptitle(f"Cross-correlogram (Bin Size = {binsize:.4f}s)", fontsize=16)
    plt.tight_layout()

    # Create legend patches.
    patch_auto = mpatches.Patch(color='#77DD77', label='Autocorrelogram (non-problematic)')
    patch_cross = mpatches.Patch(color='#CDA4DE', label='Cross-correlogram (non-problematic)')
    patch_prob = mpatches.Patch(color='#FFFF99', label='Problematic')
    patch_center = mpatches.Patch(color='#FFB6C1', label='Center bins')
    patch_thresh = Line2D([], [], color='none', label=f'Global threshold: {global_threshold:.2e}')

    fig.legend(handles=[patch_auto, patch_cross, patch_prob, patch_center, patch_thresh],
               loc='upper right', ncol=1)

    # Save the figure.
    if save_folder is None:
        save_folder = os.path.join(os.getcwd(), "reports", "figures")
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{dataset_name}_correlogram.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # Free memory

    if store_data:
        correlogram_data["global_threshold"] = global_threshold
        correlogram_data["problematic_neuron_indices"] = problematic_neuron_indices
        return correlogram_data

    print(f"Correlogram saved: {save_path}")
