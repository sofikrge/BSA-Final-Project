import os
import numpy as np
import matplotlib.pyplot as plt
from functions.correlogram import correlogram
from matplotlib import patches as mpatches
from tqdm.auto import tqdm
from joblib import Parallel, delayed

"""
This script generates a correlogram matrix for a given dataset of neurons. The correlogram matrix is a grid of
subplots where each subplot is a correlogram between two neurons.
Thus, the autocorrelogram is plotted in the diagonal.

Highlights:
- we wrote it such that we only compute the lower triangle and mirror it then to save time
- used joblib to have parallel computation
"""

# Helper function to compute a single correlogram
def compute_correlogram(i, j, prefiltered_spikes, binsize, limit):
    t1 = prefiltered_spikes[i]
    t2 = prefiltered_spikes[j]
    is_auto = (i == j)
    counts, bins = correlogram(t1, t2=t2, binsize=binsize, limit=limit, auto=is_auto)
    if len(counts) > len(bins) - 1: # no of elements might mismatch, so the code trims off last count to match
        counts = counts[:-1]
    return (i, j, counts, bins)

# Main function to plot the correlogram matrix
def plot_correlogram_matrix(neurons_data, binsize, dataset_name, limit=0.02, time_window=None, save_folder=None, store_data=True):
    num_neurons = len(neurons_data)

    problematic_neuron_indices = set()

    # Pre-filter spike times once per neuron according to time window
    prefiltered_spikes = []
    for neuron in neurons_data:
        spikes = neuron[:3][2]
        if time_window is not None:
            spikes = spikes[(spikes >= time_window[0]) & (spikes <= time_window[1])]
        prefiltered_spikes.append(spikes)

    # List of pairs to process, only compute lower triangle including the diagnoal
    tasks = [(i, j) for i in range(num_neurons) for j in range(i+1)]

    # Compute all correlograms in parallel with joblib
    results = Parallel(n_jobs=-1)(
        delayed(compute_correlogram)(i, j, prefiltered_spikes, binsize, limit)
        for i, j in tqdm(tasks, desc="Computing correlogram", ncols=100)
    )

    # Organise results
    grid_data = [[None] * num_neurons for _ in range(num_neurons)]
    correlogram_data = {} if store_data else None

    # Loop through results and store data
    for i, j, counts, bins in results:
        num_bins_actual = len(bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        grid_data[i][j] = {
            "counts": counts,
            "bins": bins,
            "bin_centers": bin_centers,
            "center": num_bins_actual // 2,
        }
        if store_data:  # save data (optional)
            key = f"Neuron {i+1}" if i == j else f"Neuron {i+1} vs Neuron {j+1}"
            correlogram_data[key] = {"counts": counts, "bins": bins}

    # Create matrix plot grid
    fig, axes = plt.subplots(num_neurons, num_neurons, figsize=(num_neurons * 3, num_neurons * 3))

    # Plot each correlogram
    for i in tqdm(range(num_neurons), desc="Plotting correlograms", ncols=100):
        for j in range(i+1):
            data = grid_data[i][j]
            counts = data["counts"]
            bins = data["bins"]
            bin_centers = data["bin_centers"]
            center_bin = data["center"]
            center_line = bin_centers[center_bin]

            # Determine if the correlogram is problematic.
            if i == j: # Autocorrelogram
                # Compute local stats
                local_mean = np.mean(counts)
                local_std = np.std(counts)
                threshold = local_mean - 2 * local_std
                global_peak_index = int(np.argmax(counts))
                # # Flag as problematic if the center bins count is above the threshold
                # or if the bins immediately adjacent to the center are the global maximum.
                condition_A=(counts[center_bin] > threshold)
                condition_B=(global_peak_index == center_bin-1) or (global_peak_index == center_bin+1)
                is_problematic = condition_A or condition_B
            else: # Crosscorrelogram
                # Cross-correlogram: problematic if a center bin count if below the threshold
                local_mean = np.mean(counts)
                local_std = np.std(counts)
                threshold = local_mean - 2 * local_std
                # Flag as problematic if the center bins count is below the threshold.
                is_problematic = (counts[center_bin] < threshold)

            # Determine color based on problematic status
            if is_problematic:
                color = '#FFFF99'
            else:
                color = '#77DD77' if i == j else '#CDA4DE'

            # Plot in the matrix
            ax = axes[i, j] if num_neurons > 1 else axes
            ax.bar(bin_centers, counts, width=np.diff(bins), align='center', color=color, alpha=0.7,
                   edgecolor='k', linewidth=0.1)
            ax.set_xlim(-limit, limit)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axvline(center_line, color='black', linestyle='--', linewidth=0.5)

            # Highlight the center bin in pastel pink
            pink_color = '#FFB6C1'
            ax.bar(bin_centers[center_bin:center_bin+1],
                   counts[center_bin:center_bin+1],
                   width=np.diff(bins)[center_bin:center_bin+1],
                   align='center', color=pink_color, alpha=1, edgecolor='k', linewidth=0.1)

            # For the upper triangle, mirror the plot
            if i != j:
                ax_mirror = axes[j, i] if num_neurons > 1 else axes
                ax_mirror.bar(-bin_centers, counts, width=np.diff(bins), align='center', color=color, alpha=0.7,
                              edgecolor='k', linewidth=0.1)
                ax_mirror.set_xlim(-limit, limit)
                ax_mirror.set_xticks([])
                ax_mirror.set_yticks([])
                ax_mirror.axvline(-center_line, color='black', linestyle='--', linewidth=0.5)
                ax_mirror.bar(-bin_centers[center_bin:center_bin+1],
                              counts[center_bin:center_bin+1],
                              width=np.diff(bins)[center_bin:center_bin+1],
                              align='center', color=pink_color, alpha=1, edgecolor='k', linewidth=0.1)

            # Label the first row and column
            if i == 0:
                ax.set_title(f"Neuron {j+1}")
            if j == 0:
                ax.set_ylabel(f"Neuron {i+1}")

    plt.suptitle(f"Cross-correlogram (Bin Size = {binsize:.4f}s)", fontsize=16)
    plt.tight_layout()

    # Create legends
    patch_auto = mpatches.Patch(color='#77DD77', label='Autocorrelogram (non-problematic)')
    patch_cross = mpatches.Patch(color='#CDA4DE', label='Cross-correlogram (non-problematic)')
    patch_prob = mpatches.Patch(color='#FFFF99', label='Problematic')
    patch_center = mpatches.Patch(color='#FFB6C1', label='Center bin')
    fig.legend(handles=[patch_auto, patch_cross, patch_prob, patch_center],
               loc='upper right', ncol=1)

    # Save the figure & store data (optional)
    if save_folder is None:
        save_folder = os.path.join(os.getcwd(), "reports", "figures")
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{dataset_name}_correlogram.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close() 
    
    if store_data:
        correlogram_data["problematic_neuron_indices"] = problematic_neuron_indices
        return correlogram_data

    print(f"Correlogram saved: {save_path}")
