import os
import numpy as np
import matplotlib.pyplot as plt

def psth_raster(group_name, neurons, water_events, sugar_events, cta_time, save_folder="reports/figures/psth"):
    """
    Generates and saves a single figure showing both pre-CTA and post-CTA rasters and PSTH overlays side by side.
    
    The function splits the events based on CTA time:
      - If cta_time is provided, events before CTA are considered pre-CTA and those after (cta_time + 3*3600)
        are considered post-CTA.
      - If cta_time is None, all events are treated as pre-CTA (and the post-CTA plots will be empty).
      
    The resulting figure has 3 rows and 2 columns:
      Left column: Pre-CTA
        Top: Water Raster
        Middle: Sugar Raster
        Bottom: PSTH Overlay
      Right column: Post-CTA (same structure)
    
    The figure is saved under "reports/figures/psth" with filename "{group_name}_stacked.png".
    
    Parameters:
    -----------
    group_name : str
        Name of the group (used for figure title and filename).
    neurons : list
        List of neurons; each neuron is expected to have spike times at index 2.
    water_events : np.array or list
        Array/list of water event times.
    sugar_events : np.array or list
        Array/list of sugar event times.
    cta_time : float or None
        CTA injection time. If provided, used to split events into pre and post CTA.
    """
    # Define the PSTH figures directory and ensure it exists.
    figures_dir = os.path.join("reports", "figures", "psth")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Split events into pre and post CTA.
    if cta_time is not None:
        water_pre = water_events[water_events < cta_time]
        sugar_pre = sugar_events[sugar_events < cta_time]
        water_post = water_events[water_events >= (cta_time + 3 * 3600)]
        sugar_post = sugar_events[sugar_events >= (cta_time + 3 * 3600)]
    else:
        water_pre = water_events
        sugar_pre = sugar_events
        water_post = []
        sugar_post = []
    
    # Parameters for plotting.
    window = (-0.5, 2.0)
    bin_width = 0.05
    
    # Helper functions.
    def gather_raster_data(events):
        raster_data = []
        for neuron_idx, neuron in enumerate(neurons):
            spikes = np.array(neuron[2])
            for event_time in events:
                rel_spikes = spikes - event_time
                mask = (rel_spikes >= window[0]) & (rel_spikes <= window[1])
                raster_data.append((rel_spikes[mask], neuron_idx))
        return raster_data

    def moving_average(data, window_size=3):
        """Computes a moving average with a given window size using a convolution."""
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='same') # same bc we want same length as input, note edge bins will be incorrect 

    def compute_psth(events, apply_smoothing=True):
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
        
        if apply_smoothing:
            psth = moving_average(psth, window_size=3)
        
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        return bin_centers, psth
    
    # Compute PSTHs and store them
    psth_results = {
        "bin_centers_water_pre": None,
        "psth_water_pre": None,
        "bin_centers_sugar_pre": None,
        "psth_sugar_pre": None,
        "bin_centers_water_post": None,
        "psth_water_post": None,
        "bin_centers_sugar_post": None,
        "psth_sugar_post": None
    }
    
    psth_results["bin_centers_water_pre"], psth_results["psth_water_pre"] = compute_psth(water_pre)
    psth_results["bin_centers_sugar_pre"], psth_results["psth_sugar_pre"] = compute_psth(sugar_pre)
    psth_results["bin_centers_water_post"], psth_results["psth_water_post"] = compute_psth(water_post)
    psth_results["bin_centers_sugar_post"], psth_results["psth_sugar_post"] = compute_psth(sugar_post)
    
    # Create one figure with 3 rows x 2 columns.
    fig, axs = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    fig.suptitle(f"{group_name} Group - PSTH & Raster (Pre vs Post CTA)", fontsize=16)
    
    # --- Pre-CTA (left column) ---
    raster_water_pre = gather_raster_data(water_pre)
    raster_sugar_pre = gather_raster_data(sugar_pre)
    
    # Use precomputed PSTH values
    bin_centers_water_pre = psth_results["bin_centers_water_pre"]
    psth_water_pre = psth_results["psth_water_pre"]
    bin_centers_sugar_pre = psth_results["bin_centers_sugar_pre"]
    psth_sugar_pre = psth_results["psth_sugar_pre"]
    
    # Top: Water Raster.
    ax0 = axs[0, 0]
    for spike_times, neuron_idx in raster_water_pre:
        ax0.plot(spike_times, np.full_like(spike_times, neuron_idx),
                 marker='|', linestyle='None', color='#A2D5F2', markersize=1)
    ax0.axvline(0, color='red', linestyle='--')
    ax0.set_title("Pre-CTA Water Raster")
    ax0.set_ylabel("Neuron Index")
    ax0.set_xlim(window)
    ax0.set_ylim(-1, len(neurons))
    ax0.invert_yaxis()
    
    # Middle: Sugar Raster.
    ax1 = axs[1, 0]
    for spike_times, neuron_idx in raster_sugar_pre:
        ax1.plot(spike_times, np.full_like(spike_times, neuron_idx),
                 marker='|', linestyle='None', color='#F6A9A9', markersize=1)
    ax1.axvline(0, color='red', linestyle='--')
    ax1.set_title("Pre-CTA Sugar Raster")
    ax1.set_ylabel("Neuron Index")
    ax1.set_xlim(window)
    ax1.set_ylim(-1, len(neurons))
    ax1.invert_yaxis()
    
    # Bottom: PSTH Overlay.
    ax2 = axs[2, 0]
    ax2.plot(bin_centers_water_pre, psth_water_pre, color='#A2D5F2', label='Water')
    ax2.plot(bin_centers_sugar_pre, psth_sugar_pre, color='#F6A9A9', label='Sugar')
    ax2.axvline(0, color='red', linestyle='--')
    ax2.set_title("Pre-CTA PSTH Overlay")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Firing Rate (Hz)")
    ax2.set_xlim(window)
    ax2.legend()
    
    # --- Post-CTA (right column) ---
    raster_water_post = gather_raster_data(water_post)
    raster_sugar_post = gather_raster_data(sugar_post)
    
    bin_centers_water_post = psth_results["bin_centers_water_post"]
    psth_water_post = psth_results["psth_water_post"]
    bin_centers_sugar_post = psth_results["bin_centers_sugar_post"]
    psth_sugar_post = psth_results["psth_sugar_post"]
    
    # Top: Water Raster.
    ax3 = axs[0, 1]
    for spike_times, neuron_idx in raster_water_post:
        ax3.plot(spike_times, np.full_like(spike_times, neuron_idx),
                 marker='|', linestyle='None', color='#A2D5F2', markersize=1)
    ax3.axvline(0, color='red', linestyle='--')
    ax3.set_title("Post-CTA Water Raster")
    ax3.set_xlim(window)
    ax3.set_ylim(-1, len(neurons))
    ax3.invert_yaxis()
    
    # Middle: Sugar Raster.
    ax4 = axs[1, 1]
    for spike_times, neuron_idx in raster_sugar_post:
        ax4.plot(spike_times, np.full_like(spike_times, neuron_idx),
                 marker='|', linestyle='None', color='#F6A9A9', markersize=1)
    ax4.axvline(0, color='red', linestyle='--')
    ax4.set_title("Post-CTA Sugar Raster")
    ax4.set_xlim(window)
    ax4.set_ylim(-1, len(neurons))
    ax4.invert_yaxis()
    
    # Bottom: PSTH Overlay.
    ax5 = axs[2, 1]
    ax5.plot(bin_centers_water_post, psth_water_post, color='#A2D5F2', label='Water')
    ax5.plot(bin_centers_sugar_post, psth_sugar_post, color='#F6A9A9', label='Sugar')
    ax5.axvline(0, color='red', linestyle='--')
    ax5.set_title("Post-CTA PSTH Overlay")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Firing Rate (Hz)")
    ax5.set_xlim(window)
    ax5.legend()
    
    # Uniform y-axis scaling for PSTH overlays (bottom row):
    # Compute the maximum firing rate from both pre and post PSTHs.
    pre_max = max(np.nanmax(psth_water_pre) if psth_water_pre.size > 0 else 0,
                  np.nanmax(psth_sugar_pre) if psth_sugar_pre.size > 0 else 0)
    post_max = max(np.nanmax(psth_water_post) if psth_water_post.size > 0 else 0,
                   np.nanmax(psth_sugar_post) if psth_sugar_post.size > 0 else 0)
    global_max = max(pre_max, post_max)
    # Set the y-axis limit uniformly on both PSTH subplots.
    buffer = 0.1 * global_max  # Add 10% extra space above the max
    ax2.set_ylim(0, global_max + buffer)
    ax5.set_ylim(0, global_max + buffer)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(figures_dir, f"{group_name}_stacked.png")
    fig.savefig(os.path.join(save_folder, f"{group_name}_stacked.png"), dpi=150, bbox_inches="tight")
    print(f"Saved figure: {save_path}")
    plt.close(fig)

    return psth_results
