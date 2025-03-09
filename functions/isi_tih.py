import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch

def isi_tih(spike_times, binsize=0.0004, min_interval=1/2500, neuron_id=None, bins=100, dataset_name="unknown", save_folder="reports/figures/TIH", time_window=None):
    """
    Calculate the ISIs from spike_times, plot a histogram, and mark problematic bins in yellow.
    
    Parameters:
      spike_times (np.array): Sorted array of spike times.
      min_interval (float): Minimum allowed inter-spike interval.
      neuron_id (optional): Identifier for the neuron (for labeling).
      bins (int): Number of bins for the histogram.
      
    Returns:
      isis (np.array): The computed ISIs.
      problematic_isis (np.array): The ISIs that are below the min_interval.
    """
    
    # If a time window is specified, filter spike_times.
    if time_window is not None:
        start, end = time_window
        spike_times = spike_times[(spike_times >= start) & (spike_times <= end)]
    
    # Calculate ISIs
    isis = np.diff(spike_times)
    
    # Compute bin edges using the provided bin width.
    bin_edges = np.arange(isis.min(), 0.1 + binsize, binsize)
    
    # Create histogram
    counts, bin_edges, patches = plt.hist(isis, bins=bin_edges, color='green', edgecolor='black', alpha=0.7, linewidth=0.25)
    
    # Compute bin centers to decide which bins correspond to problematic ISIs.
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # For each bin, if its center is below the min_interval, change its color to yellow.
    for patch, center in zip(patches, bin_centers):
        if center < min_interval:
            patch.set_facecolor('yellow')
    
    plt.xlabel("Inter-Spike Interval (s)")
    plt.ylabel("Count")
    title = f"Neuron {neuron_id} ISI Histogram for unstimulated time-window" if neuron_id is not None else "ISI Histogram"
    plt.title(title)
    # Create a custom legend:
    legend_elements = [Patch(facecolor='green', edgecolor='black', label='ISI bins'),
                       Patch(facecolor='yellow', edgecolor='black', label='Problematic first ISI bin')]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    
    # Identify and print problematic ISIs
    problematic_isis = isis[isis < min_interval]
    
    # Build the output directory using the provided save_folder.
    output_dir = os.path.join(save_folder, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{neuron_id}.png" if neuron_id is not None else "unknown.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save the figure instead of showing it.
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return isis, problematic_isis