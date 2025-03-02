import numpy as np

def compute_optimal_bin_size(neuron_spike_times, total_time, n_trials):
    """
    Compute the optimal bin size Δ* for a single neuron using analytical differentiation.
    
    Parameters:
    - neuron_spike_times: List of spike times from one neuron.
    - total_time: Total observation period T.
    - n_trials: Number of trials.

    Returns:
    - delta_opt: Optimal bin size Δ*.
    """
    N = len(neuron_spike_times)  # Number of spike events
    if N < 2:
        return None  # Avoid division errors if too few spikes
    
    # Divide into N bins and count spikes
    bin_edges = np.linspace(0, total_time, N + 1)
    k_i, _ = np.histogram(neuron_spike_times, bins=bin_edges)
    
    # Compute mean spike count per bin (k̄) and variance (v)
    k_bar = np.mean(k_i)
    v = np.var(k_i, ddof=0)  # Population variance

    # Compute optimal bin size using Δ* = (2(2k̄ - v) / n²)^(1/3)
    numerator = 2 * (2 * k_bar - v)
    denominator = n_trials ** 2
    
    if numerator <= 0:
        return None  # Avoid complex numbers if variance dominates

    delta_opt = (numerator / denominator) ** (1/3)
    return delta_opt
