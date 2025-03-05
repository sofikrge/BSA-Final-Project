import numpy as np

def compute_firing_rate_std(spike_times_list, time_window, bin_width=0.05):
    start, end = time_window
    std_list = []
    bins = np.arange(start, end + bin_width, bin_width)
    for spikes in spike_times_list:
        spikes_in_window = spikes[(spikes >= start) & (spikes <= end)]
        counts, _ = np.histogram(spikes_in_window, bins=bins)
        rates = counts / bin_width
        std_rate = np.std(rates)
        std_list.append(std_rate)
    return np.array(std_list)