import numpy as np
from functions.compute_firing_rate_std import compute_firing_rate_std
from functions.compute_firing_rates import compute_firing_rates
from functions.get_spike_times import get_spike_times

def process_and_plot_dataset(ds, file_name, ax):
    """Compute descriptive metrics and plot boxplots with std dev bars on ax."""
    data = ds["data"]
    sacc_start = data.get("sacc drinking session start time", 0)
    cta_time = data.get("CTA injection time", 0)
    spike_times_list = get_spike_times(data)
    
    # Determine maximum spike time (ignoring empty neurons)
    max_time = max((np.max(times) for times in spike_times_list if len(times) > 0), default=0)
    
    # Define time windows
    non_stimuli_time = ds["non_stimuli_time"]
    pre_CTA_time = (sacc_start, cta_time)
    post_CTA_time = (cta_time + 3 * 3600, max_time)
    
    # Compute metrics
    non_stimuli_rates = compute_firing_rates(spike_times_list, non_stimuli_time)
    pre_CTA_rates = compute_firing_rates(spike_times_list, pre_CTA_time)
    post_CTA_rates = compute_firing_rates(spike_times_list, post_CTA_time)
    
    non_stimuli_std = compute_firing_rate_std(spike_times_list, non_stimuli_time)
    pre_CTA_std = compute_firing_rate_std(spike_times_list, pre_CTA_time)
    post_CTA_std = compute_firing_rate_std(spike_times_list, post_CTA_time)
    
    # Plot boxplot and overlay standard deviation bars
    ax.boxplot(
        [non_stimuli_rates, pre_CTA_rates, post_CTA_rates],
        labels=["Non-Stimuli", "Pre-CTA", "Post-CTA"]
    )
    ax.bar([1, 2, 3],
           [np.mean(non_stimuli_std), np.mean(pre_CTA_std), np.mean(post_CTA_std)],
           width=0.3, color='orange', alpha=0.7, label="Std Dev")
    ax.set_ylabel("Firing Rate (Hz)")
    ax.set_title(f"Firing Rates - {file_name}")
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    
    # Return local y-axis limits for later scaling
    y_min = min(np.min(non_stimuli_rates), np.min(pre_CTA_rates), np.min(post_CTA_rates))
    y_max = max(np.max(non_stimuli_rates), np.max(pre_CTA_rates), np.max(post_CTA_rates))
    return y_min, y_max