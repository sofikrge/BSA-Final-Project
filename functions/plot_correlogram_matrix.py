import os
import numpy as np
import matplotlib.pyplot as plt
from functions.correlogram import correlogram

def plot_correlogram_matrix(neurons_data, binsize, dataset_name, limit=0.02, time_window=None):
    # Sofia modified this such that we only compute the lower triangle and the upper one is simply mirrored
    
    num_neurons = len(neurons_data)
    fig, axes = plt.subplots(num_neurons, num_neurons, figsize=(num_neurons * 3, num_neurons * 3))
    
    print(f"Starting correlogram matrix for {dataset_name}")
    
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

            # Ensure counts and bins align correctly
            if len(counts) > len(bins) - 1:
                counts = counts[:-1]

            bin_centers = (bins[:-1] + bins[1:]) / 2

            # Set color dynamically
            color = '#AAF0D1' if i == j else '#C9A0DC'  # Green for auto-correlation, purple for others
            
            # Plot in the matrix
            ax = axes[i, j] if num_neurons > 1 else axes
            ax.bar(bin_centers, counts, width=np.diff(bins), align='center', color=color, alpha=0.7, edgecolor='k')
            ax.set_xlim(-limit, limit)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Mirror results
            if i != j:
                ax_mirror = axes[j, i] if num_neurons > 1 else axes
                ax_mirror.bar(-bin_centers, counts, width=np.diff(bins), align='center', color=color, alpha=0.7, edgecolor='k')
                ax_mirror.set_xlim(-limit, limit)
                ax_mirror.set_xticks([])
                ax_mirror.set_yticks([])

            # Labels for the first row and first column
            if i == 0:
                ax.set_title(f"Neuron {j+1}")
            if j == 0:
                ax.set_ylabel(f"Neuron {i+1}")

    plt.suptitle(f"Cross-correlogram with (Bin Size = {binsize:.4f}s)", fontsize=16)  # Show bin size in title
    plt.tight_layout()
    
    # Define relative save path
    save_dir = os.path.join(os.getcwd(), "reports", "figures")  # Relative path
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
    save_path = os.path.join(save_dir, f"{dataset_name}_correlogram.png")

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Free memory

    print(f"Correlogram saved: {save_path}")  # Confirm save location