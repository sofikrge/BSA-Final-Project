import numpy as np
import matplotlib.pyplot as plt

def plot_stacked_raster_and_psth(neurons, water_events, sugar_events, window=(-0.5, 2.0), bin_width=0.05,
                                 figure_title="", save_path=None):
    """
    Creates a figure with three vertically-stacked subplots:
      1. Water Raster (top)
      2. Sugar Raster (middle)
      3. PSTH overlay (bottom), with water in cyan and sugar in magenta.
    """

    def gather_raster_data(events):
        raster_data = []
        for neuron_idx, neuron in enumerate(neurons):
            spikes = np.array(neuron[2])
            for event_time in events:
                rel_spikes = spikes - event_time
                mask = (rel_spikes >= window[0]) & (rel_spikes <= window[1])
                raster_data.append((rel_spikes[mask], neuron_idx))
        return raster_data
    
    raster_water = gather_raster_data(water_events)
    raster_sugar = gather_raster_data(sugar_events)

    def compute_psth(events):
        bins = np.arange(window[0], window[1] + bin_width, bin_width)
        all_spikes = []
        num_events = len(events) if len(events) > 0 else 1
        num_neurons = len(neurons)

        for neuron in neurons:
            spikes = np.array(neuron[2])
            for event_time in events:
                rel_spikes = spikes - event_time
                valid = rel_spikes[(rel_spikes >= window[0]) & (rel_spikes <= window[1])]
                all_spikes.extend(valid)
        
        counts, edges = np.histogram(all_spikes, bins=bins)
        psth = counts / (num_events * num_neurons * bin_width)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        return bin_centers, psth

    bin_centers_water, psth_water = compute_psth(water_events)
    bin_centers_sugar, psth_sugar = compute_psth(sugar_events)

    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.suptitle(figure_title)

    # Top: Water Raster
    ax0 = axs[0]
    for spike_times, neuron_idx in raster_water:
        ax0.plot(spike_times, np.full_like(spike_times, neuron_idx),
                 marker='|', linestyle='None', color='#A2D5F2', markersize=1)
    ax0.axvline(0, color='red', linestyle='--')
    ax0.set_title("Water Raster")
    ax0.set_ylabel("Neuron Index")
    ax0.set_xlim(window[0], window[1])
    ax0.set_ylim(-1, len(neurons))
    ax0.invert_yaxis()

    # Middle: Sugar Raster
    ax1 = axs[1]
    for spike_times, neuron_idx in raster_sugar:
        ax1.plot(spike_times, np.full_like(spike_times, neuron_idx),
                 marker='|', linestyle='None', color='#F6A9A9', markersize=1)
    ax1.axvline(0, color='red', linestyle='--')
    ax1.set_title("Sugar Raster")
    ax1.set_ylabel("Neuron Index")
    ax1.set_xlim(window[0], window[1])
    ax1.set_ylim(-1, len(neurons))
    ax1.invert_yaxis()

    # Bottom: PSTH Overlay
    ax2 = axs[2]
    ax2.plot(bin_centers_water, psth_water, color='#A2D5F2', label='Water')
    ax2.plot(bin_centers_sugar, psth_sugar, color='#F6A9A9', label='Sugar')
    ax2.axvline(0, color='red', linestyle='--')
    ax2.set_title("PSTH Overlay")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Firing Rate (Hz)")
    ax2.set_xlim(window[0], window[1])
    ax2.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
    plt.close(fig)