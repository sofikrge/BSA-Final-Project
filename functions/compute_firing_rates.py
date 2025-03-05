import numpy as np

def compute_firing_rates(spike_times_list, time_window):
    start, end = time_window
    duration = end - start
    if duration <= 0:
        return np.zeros(len(spike_times_list))  # Avoid division by zero
    firing_rates = np.array([
        np.sum((times >= start) & (times <= end)) / duration if len(times) > 0 else 0
        for times in spike_times_list
    ])
    return firing_rates