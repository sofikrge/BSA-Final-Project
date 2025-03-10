import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
from functions.load_dataset import load_dataset
import pickle

def filter_invalid_isis(spike_times, min_isi=0.0004, apply_filter=True):
    """
    Removes spikes that occur within absolute refractory period
    """
    if not apply_filter:
        return spike_times  # Skip filtering if disabled
    
    if len(spike_times) < 2:
        return spike_times  # No filtering needed if there's only one spike

    spike_times = np.array(spike_times)
    isi = np.diff(spike_times)  # Compute ISIs

    # Identify indices where ISI is too short
    valid_indices = np.where(isi >= min_isi)[0] + 1  # Keep valid spikes

    # Always keep the first spike, then add valid ones
    filtered_spike_times = np.insert(spike_times[valid_indices], 0, spike_times[0])
    
    # Print number of removed spikes
    removed_spikes = len(spike_times) - len(filtered_spike_times)
    total_spikes = len(spike_times)
    print(f"Removed {removed_spikes} out of {total_spikes} spikes due to ISI ≤ {min_isi} seconds.")
    
    return filtered_spike_times


def isi_tih(spike_times, binsize=0.0004, min_interval=0.0004, neuron_id=None, bins=100, dataset_name="unknown", save_folder="reports/figures/TIH", time_window=None, apply_filter=True):
    """
    Calculate the ISIs from spike_times, plot a histogram, and mark problematic bins in yellow
    """
    
    # If a time window is specified, filter spike_times.
    if time_window is not None:
        start, end = time_window
        spike_times = spike_times[(spike_times >= start) & (spike_times <= end)]
    
    # Apply ISI filtering only if enabled
    filtered_spike_times = filter_invalid_isis(spike_times, min_isi=min_interval, apply_filter=apply_filter)
    
    # Calculate ISIs (before and after filtering for comparison)
    isis = np.diff(spike_times)
    filtered_isis = np.diff(filtered_spike_times)
    
    # Compute bin edges using the provided bin width.
    bin_edges = np.arange(min(isis.min(), filtered_isis.min()), 0.1 + binsize, binsize)
    
    # Create histograms before and after filtering
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot ISI histogram before filtering
    axes[0].hist(isis, bins=bin_edges, color='green', edgecolor='black', alpha=0.7, linewidth=0.25)
    axes[0].set_title(f"Neuron {neuron_id} ISI Histogram (Pre-Filtering)")
    axes[0].set_xlabel("Inter-Spike Interval (s)")
    axes[0].set_ylabel("Count")
    
    # Plot ISI histogram after filtering (if applied)
    if apply_filter:
        axes[1].hist(filtered_isis, bins=bin_edges, color='blue', edgecolor='black', alpha=0.7, linewidth=0.25)
        axes[1].set_title(f"Neuron {neuron_id} ISI Histogram (Post-Filtering)")
        axes[1].set_xlabel("Inter-Spike Interval (s)")
        axes[1].set_ylabel("Count")
    else:
        axes[1].axis('off')  # Hide second plot if no filtering is applied
    
    plt.tight_layout()
    
    # Build the output directory using the provided save_folder.
    output_dir = os.path.join(save_folder, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{neuron_id}_filtered_comparison.png" if neuron_id is not None else "unknown_filtered_comparison.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save the figure instead of showing it.
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return isis, filtered_isis

def save_filtered_isi_datasets(datasets, processed_dir, raw_dir, apply_filter=True):
    """
    Processes ISI filtering for all datasets and saves the updated data only if filtering is toggled on
    """
    if not apply_filter:
        print("ISI filtering is disabled. No datasets will be saved.")
        return

    for dataset_name, (neurons_data, time_window) in datasets.items():
        print(f"\nProcessing ISI filtering for dataset: {dataset_name}")
        total_neurons = len(neurons_data)
        
        # Apply ISI filtering to each neuron's spike train
        filtered_neurons_data = []
        for neuron in neurons_data:
            filtered_spike_times = filter_invalid_isis(neuron[2], min_isi=0.0004, apply_filter=apply_filter)
            new_neuron = neuron.copy()
            new_neuron[2] = filtered_spike_times
            filtered_neurons_data.append(new_neuron)
        
        # Reload original data to preserve metadata
        original_file = os.path.join(raw_dir, dataset_name + ".pkl")
        data, _, _ = load_dataset(original_file)
        data["neurons"] = filtered_neurons_data
        
        # Save the updated dataset with ISI-filtered spikes
        output_filename = dataset_name + "_ISIfiltered.pkl"
        output_path = os.path.join(processed_dir, output_filename)
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Saved ISI-filtered data for {dataset_name} to {output_path}")

