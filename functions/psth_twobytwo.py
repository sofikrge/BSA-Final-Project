import os
import numpy as np
import matplotlib.pyplot as plt

def plot_neuron_rasters_2x2(group_name, neurons, water_events, sugar_events, cta_time, save_folder="reports/figures/PSTH_TwoByTwo", summary_folder="reports/figures/PSTH_TwoByTwo/Summary", window=(-1, 2), bin_width=0.05):
    """
    Creates 2x2 raster-style histogram plots per neuron + creates a group-level summary
    """
    
    dataset_folder = save_folder
    os.makedirs(dataset_folder, exist_ok=True) 
    os.makedirs(summary_folder, exist_ok=True)

    # Define pre and post CTA event times
    if cta_time is not None:
        water_pre = water_events[water_events < cta_time]
        sugar_pre = sugar_events[sugar_events < cta_time]
        water_post = water_events[water_events >= (cta_time + 3 * 3600)]
        sugar_post = sugar_events[sugar_events >= (cta_time + 3 * 3600)]
    else:
        water_pre, sugar_pre = water_events, sugar_events
        water_post, sugar_post = np.array([]), np.array([])

    group_summary = None 

    def moving_average(data, window_size=3): # 3, so one before and one after
        kernel = np.ones(window_size) / window_size  # equal weights to adjacent bins
        return np.convolve(data, kernel, mode='same')  # same bc we want the same array size

    def compute_spike_histogram(neuron, events, window=(-1, 2), bin_width=0.05, baseline_window=(-1, 0), smooth=True):
        """Compute a histogram-like spike count across event-aligned windows with baseline subtraction."""
        num_bins = int((window[1] - window[0]) / bin_width)
        bins = np.linspace(window[0], window[1], num_bins + 1, endpoint=True)  # Ensure exact bin alignment
        bin_centers = bins[:-1] + (bin_width / 2)  # Ensure bin centers are correct
        num_baseline_bins = int((baseline_window[1] - baseline_window[0]) / bin_width)  
        baseline_bins = np.linspace(baseline_window[0], baseline_window[1], num_baseline_bins + 1)

        all_spikes = []
        for event in events:
            rel_spikes = np.array(neuron[2]) - event # collect relative spike times
            rel_spikes = rel_spikes[(rel_spikes >= window[0]) & (rel_spikes <= window[1])]
            all_spikes.extend(rel_spikes)

        counts, _ = np.histogram(all_spikes, bins=bins)
        num_events = max(len(events), 1)  # Avoid division by zero
        psth = counts / (num_events * bin_width)

        # Compute baseline firing rate = mean over baseline window, then subtract from PSTH
        baseline_counts, _ = np.histogram(all_spikes, bins=baseline_bins)
        baseline_rate = np.mean(baseline_counts / (num_events * bin_width))
        psth_corrected = (psth - baseline_rate)
        
        # Apply smoothing (optional)
        if smooth:
            psth_corrected = moving_average(psth_corrected, window_size=3)
            
        return bin_centers, psth_corrected

    # Loop through neurons
    for i, neuron in enumerate(neurons):
        bin_centers, hist_water_pre = compute_spike_histogram(neuron, water_pre, baseline_window=(-1, 0))
        _, hist_water_post = compute_spike_histogram(neuron, water_post, baseline_window=(-1, 0))
        _, hist_sugar_pre = compute_spike_histogram(neuron, sugar_pre, baseline_window=(-1, 0))
        _, hist_sugar_post = compute_spike_histogram(neuron, sugar_post, baseline_window=(-1, 0))

        # Initialize group_summary only once, based on first computed PSTH size
        if group_summary is None:
            num_bins = len(hist_water_pre)
            group_summary = {
                "water_pre": np.zeros(num_bins),
                "water_post": np.zeros(num_bins),
                "sugar_pre": np.zeros(num_bins),
                "sugar_post": np.zeros(num_bins)
            }

        # Add to group summary
        group_summary["water_pre"] += hist_water_pre
        group_summary["water_post"] += hist_water_post
        group_summary["sugar_pre"] += hist_sugar_pre
        group_summary["sugar_post"] += hist_sugar_post

        # Create a 2×2 figure for this neuron
        fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
        fig.suptitle(f"Neuron {i+1} - {group_name}", fontsize=14)

        # Row 1 - Water Events
        axs[0, 0].bar(bin_centers, hist_water_pre, width=bin_width,align='center', color="blue", alpha=0.7, edgecolor="black")
        axs[0, 0].axvline(0, color="red", linestyle="--")
        axs[0, 0].set_title("Water Pre-CTA")
        axs[0, 0].set_ylabel("Firing Rate (Hz)")

        axs[0, 1].bar(bin_centers, hist_water_post, width=bin_width, align='center', color="cyan", alpha=0.7, edgecolor="black")
        axs[0, 1].axvline(0, color="red", linestyle="--")
        axs[0, 1].set_title("Water Post-CTA")

        # Row 2 - Sugar Events
        axs[1, 0].bar(bin_centers, hist_sugar_pre, width=bin_width,align='center', color="red", alpha=0.7, edgecolor="black")
        axs[1, 0].axvline(0, color="red", linestyle="--")
        axs[1, 0].set_title("Sugar Pre-CTA")
        axs[1, 0].set_ylabel("Firing Rate (Hz)")
        axs[1, 0].set_xlabel("Time (s)")

        axs[1, 1].bar(bin_centers, hist_sugar_post, width=bin_width,align='center', color="pink", alpha=0.7, edgecolor="black")
        axs[1, 1].axvline(0, color="red", linestyle="--")
        axs[1, 1].set_title("Sugar Post-CTA")
        axs[1, 1].set_xlabel("Time (s)")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(dataset_folder, f"neuron_{i+1}_raster.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved updated 2x2 raster-style plots for {group_name} in {dataset_folder}")

    # Group-Level Summary Plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    fig.suptitle(f"Group Summary - {group_name}", fontsize=14)

    axs[0, 0].bar(bin_centers, group_summary["water_pre"] / len(neurons), width=bin_width,align='center', color="blue", alpha=0.7, edgecolor="black")
    axs[0, 0].axvline(0, color="red", linestyle="--")
    axs[0, 0].set_title("Water Pre-CTA")

    axs[0, 1].bar(bin_centers, group_summary["water_post"] / len(neurons), width=bin_width,align='center', color="cyan", alpha=0.7, edgecolor="black")
    axs[0, 1].axvline(0, color="red", linestyle="--")
    axs[0, 1].set_title("Water Post-CTA")

    axs[1, 0].bar(bin_centers, group_summary["sugar_pre"] / len(neurons), width=bin_width,align='center', color="red", alpha=0.7, edgecolor="black")
    axs[1, 0].axvline(0, color="red", linestyle="--")
    axs[1, 0].set_title("Sugar Pre-CTA")
    axs[1, 0].set_xlabel("Time (s)")

    axs[1, 1].bar(bin_centers, group_summary["sugar_post"] / len(neurons), width=bin_width, align='center',color="pink", alpha=0.7, edgecolor="black")
    axs[1, 1].axvline(0, color="red", linestyle="--")
    axs[1, 1].set_title("Sugar Post-CTA")
    axs[1, 1].set_xlabel("Time (s)")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    summary_path = os.path.join(summary_folder, f"{group_name}_summary.png")
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved group summary plot for {group_name} in {summary_folder}")
