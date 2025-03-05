import numpy as np

def compute_cv_isi(neurons, time_window=None):
    """
    Compute the coefficient of variation (CV) of inter-spike intervals (ISIs) for each neuron.
    CV is calculated as the standard deviation of the ISIs divided by their mean.
    If a time_window is provided, only spikes within that window are considered.

    Parameters:
        neurons (list): List of neurons; each neuron is assumed to have its spike times at index 2.
        time_window (tuple, optional): (start, end) time in seconds. If provided, only consider spikes within this window.
    
    Returns:
        np.array: CV of ISIs for each neuron.
    """
    cv_isi_values = []
    
    for neuron in neurons:
        _, _, spikes = neuron
        # Optionally filter spikes by time window.
        if time_window:
            start, end = time_window
            spikes = np.array([spike for spike in spikes if start <= spike <= end])
        else:
            spikes = np.array(spikes)
        
        if len(spikes) < 2:
            cv_isi_values.append(np.nan)  # Not enough spikes to compute ISIs
        else:
            isis = np.diff(spikes)
            mean_isi = np.mean(isis)
            if mean_isi == 0:
                cv_isi = np.nan
            else:
                cv_isi = np.std(isis) / mean_isi
            cv_isi_values.append(cv_isi)
    
    return np.array(cv_isi_values)