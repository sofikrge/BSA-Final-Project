import os
import numpy as np
import matplotlib.pyplot as plt
from functions.correlogram import correlogram
from matplotlib import patches as mpatches
from matplotlib.lines import Line2D
from tqdm.auto import tqdm
import sys

# Definition of problematic correlograms:
# for autocorrelograms: if either center bin exceeds global_threshold (mean + 2 stds of all correlogram center bins), or if the global peak is immediately outside the center bins.
# for cross-correlograms: if both center bins are the minima for the correlogram

def plot_correlogram_matrix(neurons_data, binsize, dataset_name, limit=0.02, time_window=None, save_folder=None, store_data=True):
    # Removed the global_threshold argument; it will be computed from the data.
    
    num_neurons = len(neurons_data)
    num_bins = int(2 * limit / binsize)
    common_bins = np.linspace(-limit, limit, num_bins + 1)
    common_bin_centers = (common_bins[:-1] + common_bins[1:]) / 2
    common_widths = np.diff(common_bins)
    
    problematic_neuron_indices = set()
    
    # Precompute center bin indices (they are the same for every correlogram).
    if num_bins % 2 == 0:
        center_left = num_bins // 2 - 1
        center_right = num_bins // 2
    else:
        center_left = center_right = num_bins // 2
    
    # Prepare to store correlogram data and center bin values.
    if store_data:
        correlogram_data = {}
    all_center_vals = []  # Will collect all center bin counts
    grid_data = [[None] * num_neurons for _ in range(num_neurons)] # grid_data will hold computed data for each neuron pair for later plotting.
    
    # First pass: Compute all correlograms and store center bin values.
    for i, neuron_i in enumerate(neurons_data):
        for j, neuron_j in enumerate(neurons_data[:i+1]):  # Compute only for i >= j
            t1 = neuron_i[:3][2]  # Extract spike times
            t2 = neuron_j[:3][2]  # Extract spike times

            tqdm.write(f"Processing Neuron {i+1} vs Neuron {j+1}...", end="\n", file=sys.stdout)
            
            # Filter spikes to the desired window if provided.
            if time_window is not None:
                t1 = t1[(t1 >= time_window[0]) & (t1 <= time_window[1])]
                t2 = t2[(t2 >= time_window[0]) & (t2 <= time_window[1])]
            
            # Compute correlogram.
            counts, bins = correlogram(t1, t2=t2, binsize=binsize, limit=limit, auto=(i == j), density=False)
            # trim counts if necessary
            if len(counts) > len(bins) - 1:
                counts = counts[:-1]
            
            # Store computed data to minimise loop repetitions.
            if store_data:
                key = f"Neuron {i+1}" if i == j else f"Neuron {i+1} vs Neuron {j+1}"
                correlogram_data[key] = {"counts": counts, "bins": bins}
            
            # Collect the center bin counts for global threshold computation from autocorrelograms only
            if i == j:
                all_center_vals.append(counts[center_left])
                all_center_vals.append(counts[center_right])

            
            # Save necessary data for plotting.
            grid_data[i][j] = {
                "counts": counts,
                "bins": bins,
                "bin_centers": (bins[:-1] + bins[1:]) / 2,
                "center_left": center_left,
                "center_right": center_right
            }

    # Compute the global threshold: mean of all center bin values of the autocorrelograms plus their standard deviation.
    all_center_vals = np.array(all_center_vals)
    global_threshold = all_center_vals.mean() + 2 * all_center_vals.std()
    print(f"Computed global_threshold = {global_threshold}")
    
    # Create the subplot grid.
    fig, axes = plt.subplots(num_neurons, num_neurons, figsize=(num_neurons * 3, num_neurons * 3))
    
    # Second pass: Plot each correlogram using the computed global_threshold.
    for i, neuron_i in enumerate(neurons_data):
        for j, neuron_j in enumerate(neurons_data[:i+1]):
            data = grid_data[i][j]
            counts = data["counts"]
            bins = data["bins"]
            bin_centers = data["bin_centers"]
            center_left = data["center_left"]
            center_right = data["center_right"]
            center_line = (bin_centers[center_left] + bin_centers[center_right]) / 2
            
            # Check if the current correlogram is problematic.
            if i == j:  # autocorrelogram: problematic if either center bin exceeds threshold,
                        # or if the global peak is immediately outside the center bins.
                condition_A = (counts[center_left] > global_threshold or counts[center_right] > global_threshold)
                global_peak_index = int(np.argmax(counts))
                condition_B = (global_peak_index == center_left - 1 or global_peak_index == center_right + 1)
                is_problematic = condition_A or condition_B
                if is_problematic:
                    reasons = []
                    if condition_A:
                        reasons.append("center bin count(s) exceed global_threshold")
                    if condition_B:
                        reasons.append("global peak is immediately outside center bins")
                    print(f"Neuron {i+1} autocorrelogram is problematic because: {', '.join(reasons)}")
                    problematic_neuron_indices.add(i)
            else:       # cross-correlogram: problematic if both center bins are below threshold.
                is_problematic = (counts[center_left] == counts.min() and counts[center_right] == counts.min())
                if is_problematic:
                    print(f"Neuron {i+1} vs Neuron {j+1} cross-correlogram is problematic because both center bins are the minimum.")
                    problematic_neuron_indices.add(i)
                    problematic_neuron_indices.add(j)
                    
            # Set plot color.
            color = '#FFFF99' if is_problematic else ('#77DD77' if i == j else '#CDA4DE')
            
            # Plot in the matrix.
            ax = axes[i, j] if num_neurons > 1 else axes
            ax.bar(bin_centers, counts, width=np.diff(bins), align='center', color=color, alpha=0.7,
                   edgecolor='k', linewidth=0.25)
            ax.set_xlim(-limit, limit)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax.axvline(center_line, color='black', linestyle='--', linewidth=0.5)
            
            # Overlay the central bin(s) in pastel pink.
            pink_color = '#FFB6C1'
            ax.bar(bin_centers[center_left:center_left+1],
                   counts[center_left:center_left+1],
                   width=np.diff(bins)[center_left:center_left+1],
                   align='center', color=pink_color, alpha=1, edgecolor='k', linewidth=0.25)
            
            # Annotation above right center bin:
            ax.bar(bin_centers[center_right:center_right+1],
                   counts[center_right:center_right+1],
                   width=np.diff(bins)[center_right:center_right+1],
                   align='center', color=pink_color, alpha=1, edgecolor='k', linewidth=0.25)
            
            # Mirror for the upper triangle.
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
            
            # Labels for the first row and first column.
            if i == 0:
                ax.set_title(f"Neuron {j+1}")
            if j == 0:
                ax.set_ylabel(f"Neuron {i+1}")

    plt.suptitle(f"Cross-correlogram with (Bin Size = {binsize:.4f}s)", fontsize=16)
    plt.tight_layout()

    # Create legend patches.
    patch_auto = mpatches.Patch(color='#77DD77', label='Autocorrelogram (non-problematic)')
    patch_cross = mpatches.Patch(color='#CDA4DE', label='Cross-correlogram (non-problematic)')
    patch_prob = mpatches.Patch(color='#FFFF99', label='Problematic')
    patch_center = mpatches.Patch(color='#FFB6C1', label='Center bins')
    # Create a dummy entry for the global threshold; using a Line2D with no marker.
    patch_thresh = Line2D([], [], color='none', label=f'Global threshold: {global_threshold:.2e}')

    fig.legend(handles=[patch_auto, patch_cross, patch_prob, patch_center, patch_thresh], loc='upper right', ncol=1)
    
    # Save the figure.
    if save_folder is None:
        save_folder = os.path.join(os.getcwd(), "reports", "figures")
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{dataset_name}_correlogram.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Free memory
    
    if store_data:
        correlogram_data["global_threshold"] = global_threshold
        correlogram_data["problematic_neuron_indices"] = problematic_neuron_indices
        return correlogram_data

    print(f"Correlogram saved: {save_path}")
