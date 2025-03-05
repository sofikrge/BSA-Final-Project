import numpy as np

def compute_fano_factor(neurons, time_window, bin_width=0.05):
    """
    Compute the Fano Factor for each neuron within a specified time window.
    The Fano Factor is defined as the variance of the spike counts divided by the mean spike count
    across bins of width `bin_width`.

    Parameters:
        neurons (list): List of neurons; each neuron is assumed to have its spike times at index 2.
        time_window (tuple): (start, end) time in seconds.
        bin_width (float): The width (in seconds) of each time bin used for counting spikes.

    Returns:
        np.array: Fano Factor for each neuron.
    """
    start, end = time_window
    # Create bin edges covering the full time window.
    bins = np.arange(start, end + bin_width, bin_width)
    fano_factors = []
    
    for neuron in neurons:
        # Unpack neuron data; assuming spikes are at index 2.
        _, _, spikes = neuron
        # Select spikes that occur within the time window.
        spikes_in_window = np.array([spike for spike in spikes if start <= spike <= end])
        # Count spikes in each bin.
        counts, _ = np.histogram(spikes_in_window, bins=bins)
        mean_count = np.mean(counts)
        # If mean is zero, return NaN to avoid division by zero.
        if mean_count == 0:
            fano = np.nan
        else:
            fano = np.var(counts) / mean_count
        fano_factors.append(fano)
    
    return np.array(fano_factors)
