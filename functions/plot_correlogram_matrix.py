import os
import numpy as np
import matplotlib.pyplot as plt
from functions.correlogram import correlogram
from matplotlib import patches as mpatches

def plot_correlogram_matrix(neurons_data, binsize, dataset_name, limit=0.02, time_window=None, save_folder=None, store_data=True):
    # Sofia modified this such that we only compute the lower triangle and the upper one is simply mirrored
    
    num_neurons = len(neurons_data)
    fig, axes = plt.subplots(num_neurons, num_neurons, figsize=(num_neurons * 3, num_neurons * 3))
    
    print(f"Starting correlogram matrix for {dataset_name}")
    
    # Create a dictionary to store correlogram data if desired.
    if store_data:
        correlogram_data = {}
    
    for i, neuron_i in enumerate(neurons_data):
        for j, neuron_j in enumerate(neurons_data[:i+1]):  # Compute only for i >= j
            t1 = neuron_i[:3][2]  # Extract spike times
            t2 = neuron_j[:3][2]  # Extract spike times

            print(f"Processing Neuron {i+1} vs Neuron {j+1}...")
            
            # Filter spikes to the desired window if provided
            if time_window is not None:
                t1=t1[(t1>=time_window[0]) & (t1<=time_window[1])]
                t2=t2[(t2>=time_window[0]) & (t2<=time_window[1])]
            
            # Compute correlogram
            counts, bins = correlogram(t1, t2=t2, binsize=binsize, limit=limit, auto=(i == j), density=False)
            
           # Instead of np.arange, we use np.linspace to force the correct number of bins.
            num_bins = int(2 * limit / binsize)  # Expected number of bins
            bins = np.linspace(-limit, limit, num_bins + 1)
            if len(counts) > len(bins) - 1:
                counts = counts[:-1]
            
            # Compute bin centers.
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # Minimal modification: store computed data if requested.
            if store_data:
                key = f"Neuron {i+1}" if i == j else f"Neuron {i+1} vs Neuron {j+1}"
                correlogram_data[key] = {"counts": counts, "bins": bins}
            
            # --- New: Determine the center bins ---
            n_bins = len(counts)
            if n_bins % 2 == 0:
                # Even number of bins: define the center as the two middle bins.
                center_left = n_bins // 2 - 1
                center_right = n_bins // 2
            else:
                # Odd number of bins: use the middle bin for both.
                center_left = center_right = n_bins // 2
            
            # Calculate the vertical line position as the average of the two center bin centers.
            center_line = (bin_centers[center_left] + bin_centers[center_right]) / 2
            
            # Check if the current correlogram is problematic based on the center bins.
            # For autocorrelograms: problematic if either center bin is non-empty.
            # For cross-correlograms: problematic if both center bins are empty.
            if i == j:  # autocorrelogram
                is_problematic = (counts[center_left] > 0 or counts[center_right] > 0)
            else:       # cross-correlogram
                is_problematic = (counts[center_left] == 0 and counts[center_right] == 0)
            
            # Set color: if problematic, use pastel yellow; otherwise, use default.
            if is_problematic:
                color = '#FFFF99'  # Pastel yellow
            else:
                color = '#77DD77' if i == j else '#CDA4DE'
            
            # Plot in the matrix
            ax = axes[i, j] if num_neurons > 1 else axes
            ax.bar(bin_centers, counts, width=np.diff(bins), align='center', color=color, alpha=0.7, edgecolor='k', linewidth=0.25)
            ax.set_xlim(-limit, limit)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Vertical line at center x=0
            if n_bins % 2 == 0:
                center_left = n_bins // 2 - 1
                center_right = n_bins // 2
            else:
                center_left = center_right = n_bins // 2
            center_line = (bin_centers[center_left] + bin_centers[center_right]) / 2
            
            ax.axvline(center_line, color='black', linestyle='--', linewidth=0.5)
            
            # Overlay the central bin(s) in pastel pink (for both auto and cross)
            pink_color = '#FFB6C1'
            # Highlight left center bin
            # Overlay the left center bin in pastel pink
            ax.bar(bin_centers[center_left:center_left+1],
                counts[center_left:center_left+1],
                width=np.diff(bins)[center_left:center_left+1],
                align='center',
                color=pink_color, alpha=1, edgecolor='k', linewidth=0.25)
            # Overlay the right center bin in pastel pink
            ax.bar(bin_centers[center_right:center_right+1],
                counts[center_right:center_right+1],
                width=np.diff(bins)[center_right:center_right+1],
                align='center',
                color=pink_color, alpha=1, edgecolor='k', linewidth=0.25)

            
            # Mirror for upper triangle.
            if i != j:
                ax_mirror = axes[j, i] if num_neurons > 1 else axes
                ax_mirror.bar(-bin_centers, counts, width=np.diff(bins), align='center', color=color, alpha=0.7, edgecolor='k', linewidth=0.25)
                ax_mirror.set_xlim(-limit, limit)
                ax_mirror.set_xticks([])
                ax_mirror.set_yticks([])
                ax_mirror.axvline(-center_line, color='black', linestyle='--', linewidth=0.5)
                ax_mirror.bar(-bin_centers[center_left:center_left+1],
                              counts[center_left:center_left+1],
                              width=np.diff(bins)[center_left:center_left+1],
                              align='center',
                              color=pink_color, alpha=1, edgecolor='k', linewidth=0.25)
                ax_mirror.bar(-bin_centers[center_right:center_right+1],
                              counts[center_right:center_right+1],
                              width=np.diff(bins)[center_right:center_right+1],
                              align='center',
                              color=pink_color, alpha=1, edgecolor='k', linewidth=0.25)

            # Labels for the first row and first column
            if i == 0:
                ax.set_title(f"Neuron {j+1}")
            if j == 0:
                ax.set_ylabel(f"Neuron {i+1}")

    plt.suptitle(f"Cross-correlogram with (Bin Size = {binsize:.4f}s)", fontsize=16)  # Show bin size in title
    plt.tight_layout()

    # Create legend patches
    patch_auto = mpatches.Patch(color='#77DD77', label='Autocorrelogram (non-problematic)')
    patch_cross = mpatches.Patch(color='#CDA4DE', label='Cross-correlogram (non-problematic)')
    patch_prob = mpatches.Patch(color='#FFFF99', label='Problematic')
    patch_center = mpatches.Patch(color='#FFB6C1', label='Center bins')
    # Add a figure-level legend.
    fig.legend(handles=[patch_auto, patch_cross, patch_prob, patch_center], loc='upper right', ncol=1)
    
    # If save_folder is provided, use it; otherwise, use the default relative path
    if save_folder is None:
        save_folder = os.path.join(os.getcwd(), "reports", "figures")
    
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{dataset_name}_correlogram.png")

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Free memory
    
    if store_data:
        return correlogram_data

    print(f"Correlogram saved: {save_path}")  # Confirm save location