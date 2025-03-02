import numpy as np
import os
import pickle

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  

# Define directory for pickle files
pkl_dir = os.path.join(base_dir, "data", "raw")  # Correct relative path

# Define file paths
file_1 = os.path.join(pkl_dir, "ctrl rat 1.pkl")
file_2 = os.path.join(pkl_dir, "ctrl rat 2.pkl")
file_3 = os.path.join(pkl_dir, "exp rat 2.pkl")
file_4 = os.path.join(pkl_dir, "exp rat 3.pkl")

# Load data manually
with open(file_1, "rb") as f:
    ctrl_rat_1_neurons_data = pickle.load(f)["neurons"]

with open(file_2, "rb") as f:
    ctrl_rat_2_neurons_data = pickle.load(f)["neurons"]

with open(file_3, "rb") as f:
    exp_rat_2_neurons_data = pickle.load(f)["neurons"]

with open(file_4, "rb") as f:
    exp_rat_3_neurons_data = pickle.load(f)["neurons"]

def compute_c_n_optimized(neuron_spike_times, delta_values, total_time, n_trials):
    """
    Compute C_n(Δ) for a range of bin sizes Δ for a single neuron using the correct formula.
    
    Parameters:
    - neuron_spike_times: List of spike times from one neuron.
    - delta_values: Array of candidate bin sizes.
    - total_time: Total observation period T.
    - n_trials: Number of trials.
    
    Returns:
    - delta_opt: Optimal bin size for this neuron.
    """
    c_n_values = np.zeros(len(delta_values))  # Pre-allocate memory

    for idx, delta in enumerate(delta_values):
        N = int(total_time / delta)  # Number of bins
        bin_edges = np.linspace(0, total_time, N+1)  # Define bin edges

        # Fast binning with np.histogram (optimized)
        k_i, _ = np.histogram(neuron_spike_times, bins=bin_edges)

        # Compute mean spike count per bin
        k_bar = np.mean(k_i)

        # Compute variance correctly as given in the formula
        v = np.mean((k_i - k_bar) ** 2)

        # Compute C_n(Δ) using the formula from the image
        c_n = (2 * k_bar - v) / (n_trials * delta) ** 2
        c_n_values[idx] = c_n

    return delta_values[np.argmin(c_n_values)]  # Return Δ* that minimizes C_n(Δ)

def find_average_optimal_bin_size_optimized(neurons_data, total_time, n_trials, delta_min=0.001, delta_max=0.05, num_deltas=100):
    """
    Compute the average optimal bin size across all neurons efficiently using the correct formula.
    
    Parameters:
    - neurons_data: List of neurons, where each neuron contains spike times.
    - total_time: Total observation period T.
    - n_trials: Number of trials.
    
    Returns:
    - average_delta_opt: The averaged optimal bin size.
    """
    delta_values = np.linspace(delta_min, delta_max, num_deltas)  # Candidate Δ values
    optimal_deltas = np.zeros(len(neurons_data))  # Pre-allocate memory

    for i, neuron in enumerate(neurons_data):
        spike_times = neuron[2]  # Extract spike times
        optimal_deltas[i] = compute_c_n_optimized(spike_times, delta_values, total_time, n_trials)  # Compute optimal bin size

    return np.mean(optimal_deltas)  # Return the averaged optimal bin size

# Loop through datasets and save optimal bin size
datasets = {
    "ctr_rat_1": ctrl_rat_1_neurons_data,
    "ctr_rat_2": ctrl_rat_2_neurons_data,
    "exp_rat_2": exp_rat_2_neurons_data,
    "exp_rat_3": exp_rat_3_neurons_data
}

optimal_bin_sizes = {}

for name, neurons_data in datasets.items():
    print(f"Processing {name}...")  
    total_time = max(np.concatenate([neuron[2] for neuron in neurons_data]))  # Total observation period
    n_trials = 1  

    avg_optimal_bin_size = find_average_optimal_bin_size_optimized(neurons_data, total_time, n_trials)
    optimal_bin_sizes[name] = avg_optimal_bin_size  

    print(f"✅ Done: {name} - Optimal Bin Size = {avg_optimal_bin_size:.4f} seconds")

# Print results for reference
for name, bin_size in optimal_bin_sizes.items():
    print(f"Optimal Bin Size for {name}: {bin_size:.4f} seconds")