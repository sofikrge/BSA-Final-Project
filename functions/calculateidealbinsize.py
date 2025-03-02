import numpy as np

def compute_optimal_bin_size(T, spike_counts, n_trials, delta_values):
    """
    Computes the optimal bin size that minimizes C_n(Δ)
    
    Parameters:
    T (float): Total observation period
    spike_counts (list): List of spike counts in each bin
    n_trials (int): Number of trials
    delta_values (list): List of candidate bin widths
    
    Returns:
    float: Optimal bin width Δ*
    """
    N = len(spike_counts)
    k_bar = np.mean(spike_counts)
    v = np.mean((spike_counts - k_bar) ** 2)
    
    optimal_delta = None
    min_Cn = float('inf')
    
    for delta in delta_values:
        Cn = (2 * k_bar - v) / ((n_trials * delta) ** 2)
        if Cn < min_Cn:
            min_Cn = Cn
            optimal_delta = delta
    
    return optimal_delta

# Example Usage
T = 100  # Total observation period
spike_counts = np.random.poisson(lam=5, size=50)  # Simulated spike count data
n_trials = 10  # Number of trials
delta_values = np.linspace(0.1, 5, 50)  # Range of possible bin widths

optimal_delta = compute_optimal_bin_size(T, spike_counts, n_trials, delta_values)
print(f"Optimal bin width Δ*: {optimal_delta}")
