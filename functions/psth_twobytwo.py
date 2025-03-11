import os
import numpy as np
import matplotlib.pyplot as plt

def plot_neuron_rasters_2x2(group_name, neurons, water_events, sugar_events, cta_time, save_folder="reports/figures/PSTH_TwoByTwo", summary_folder="reports/figures/PSTH_TwoByTwo/Summary", window=(-1, 2), bin_width=0.05):
    """
    Generates and saves 2×2 raster-style histogram plots per neuron.
    Also creates a group-level summary raster plot stored in `rasters/summary/`.

    Parameters:
        group_name (str): Group name (e.g., "ctrl_rat_1").
        neurons (list): List of neurons, each with spike times at index 2.
        water_events (array-like): Times of water-related stimulus events.
        sugar_events (array-like): Times of sugar-related stimulus events.
        cta_time (float): Time of CTA injection.
        save_folder (str): Base directory where individual neuron figures will be saved.
        summary_folder (str): Directory where the group-level summary will be saved.
        window (tuple): Time window (start, end) relative to each event.
        bin_width (float): Width of time bins for histogram-like representation.
    """
    dataset_folder = save_folder  # Create dataset-specific folder
    os.makedirs(dataset_folder, exist_ok=True)  # Ensure dataset folder exists
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

    bins = np.arange(window[0], window[1] + bin_width, bin_width)
    group_summary = {
        "water_pre": np.zeros(len(bins) - 1),
        "water_post": np.zeros(len(bins) - 1),
        "sugar_pre": np.zeros(len(bins) - 1),
        "sugar_post": np.zeros(len(bins) - 1)
    }

    def compute_spike_histogram(neuron, events):
        """Compute a histogram-like spike count across event-aligned windows."""
        all_spikes = []
        for event in events:
            rel_spikes = np.array(neuron[2]) - event
            rel_spikes = rel_spikes[(rel_spikes >= window[0]) & (rel_spikes <= window[1])]
            all_spikes.extend(rel_spikes)
        
        counts, _ = np.histogram(all_spikes, bins=bins)
        return counts / (len(events) * bin_width) if len(events) > 0 else np.zeros_like(counts)

    # Loop through neurons and create 2x2 raster-style histograms
    for i, neuron in enumerate(neurons):
        hist_water_pre = compute_spike_histogram(neuron, water_pre)
        hist_water_post = compute_spike_histogram(neuron, water_post)
        hist_sugar_pre = compute_spike_histogram(neuron, sugar_pre)
        hist_sugar_post = compute_spike_histogram(neuron, sugar_post)

        # Add to group summary (efficient in-place update)
        group_summary["water_pre"] += hist_water_pre
        group_summary["water_post"] += hist_water_post
        group_summary["sugar_pre"] += hist_sugar_pre
        group_summary["sugar_post"] += hist_sugar_post

        # Create a 2×2 figure for this neuron
        fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
        fig.suptitle(f"Neuron {i+1} - {group_name}", fontsize=14)

        # Row 1 - Water Events
        axs[0, 0].bar(bins[:-1], hist_water_pre, width=bin_width, color="blue", alpha=0.7, edgecolor="black")
        axs[0, 0].axvline(0, color="red", linestyle="--")
        axs[0, 0].set_title("Water Pre-CTA")
        axs[0, 0].set_ylabel("Firing Rate (Hz)")

        axs[0, 1].bar(bins[:-1], hist_water_post, width=bin_width, color="cyan", alpha=0.7, edgecolor="black")
        axs[0, 1].axvline(0, color="red", linestyle="--")
        axs[0, 1].set_title("Water Post-CTA")

        # Row 2 - Sugar Events
        axs[1, 0].bar(bins[:-1], hist_sugar_pre, width=bin_width, color="red", alpha=0.7, edgecolor="black")
        axs[1, 0].axvline(0, color="red", linestyle="--")
        axs[1, 0].set_title("Sugar Pre-CTA")
        axs[1, 0].set_ylabel("Firing Rate (Hz)")
        axs[1, 0].set_xlabel("Time (s)")

        axs[1, 1].bar(bins[:-1], hist_sugar_post, width=bin_width, color="pink", alpha=0.7, edgecolor="black")
        axs[1, 1].axvline(0, color="red", linestyle="--")
        axs[1, 1].set_title("Sugar Post-CTA")
        axs[1, 1].set_xlabel("Time (s)")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(dataset_folder, f"neuron_{i+1}_raster.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved updated 2x2 raster-style plots for {group_name} in {dataset_folder}")

    # Generate Group-Level Summary Plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    fig.suptitle(f"Group Summary - {group_name}", fontsize=14)

    axs[0, 0].bar(bins[:-1], group_summary["water_pre"] / len(neurons), width=bin_width, color="blue", alpha=0.7, edgecolor="black")
    axs[0, 0].axvline(0, color="red", linestyle="--")
    axs[0, 0].set_title("Water Pre-CTA")

    axs[0, 1].bar(bins[:-1], group_summary["water_post"] / len(neurons), width=bin_width, color="cyan", alpha=0.7, edgecolor="black")
    axs[0, 1].axvline(0, color="red", linestyle="--")
    axs[0, 1].set_title("Water Post-CTA")

    axs[1, 0].bar(bins[:-1], group_summary["sugar_pre"] / len(neurons), width=bin_width, color="red", alpha=0.7, edgecolor="black")
    axs[1, 0].axvline(0, color="red", linestyle="--")
    axs[1, 0].set_title("Sugar Pre-CTA")
    axs[1, 0].set_xlabel("Time (s)")

    axs[1, 1].bar(bins[:-1], group_summary["sugar_post"] / len(neurons), width=bin_width, color="pink", alpha=0.7, edgecolor="black")
    axs[1, 1].axvline(0, color="red", linestyle="--")
    axs[1, 1].set_title("Sugar Post-CTA")
    axs[1, 1].set_xlabel("Time (s)")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    summary_path = os.path.join(summary_folder, f"{group_name}_summary.png")
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved group summary plot for {group_name} in {summary_folder}")
