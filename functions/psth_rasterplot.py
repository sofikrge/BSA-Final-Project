import os
import numpy as np
import matplotlib.pyplot as plt

def psth_raster(group_name, neurons, water_events, sugar_events, cta_time, save_folder="reports/figures/psth"):
    """
    Generates and saves a single figure showing both pre-CTA and post-CTA rasters and PSTH overlays side by side.
    (See original docstring for details.)
    """
    # Define the PSTH figures directory and ensure it exists.
    figures_dir = os.path.join("reports", "figures", "Rasterplot_SmoothedPSTH")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Split events into pre and post CTA.
    if cta_time is not None:
        water_pre = water_events[water_events < cta_time]
        sugar_pre = sugar_events[sugar_events < cta_time]
        water_post = water_events[water_events >= (cta_time + 3 * 3600)]
        sugar_post = sugar_events[ sugar_events >= (cta_time + 3 * 3600)]
    else:
        water_pre = water_events
        sugar_pre = sugar_events
        water_post = np.array([])
        sugar_post = np.array([])
    
    # Parameters for plotting.
    window = (-0.5, 2.0)
    bin_width = 0.05

    # Helper functions
    def gather_raster_data(events):
        # Instead of nested loops, using  list comprehension
        return [
            (np.array(neuron[2]) - event, neuron_idx)
            for neuron_idx, neuron in enumerate(neurons)
            for event in events
            if window[0] <= (np.array(neuron[2]) - event).min() <= window[1]  # quick check; adjust if needed
        ]
    
    def moving_average(data, window_size=3):
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='same') # as we want the resulting array to have the same size as the input
    
    def compute_psth(events, apply_smoothing=True):
        bins = np.arange(window[0], window[1] + bin_width, bin_width)
        # Use a list comprehension and np.concatenate to vectorize the collection of relative spike times.
        # For each neuron, subtract all event times in one go.
        all_rel_spikes = []
        for neuron in neurons:
            spikes = np.array(neuron[2])
            # Use broadcasting: subtract all events at once.
            rel = spikes[:, None] - events  # shape: (n_spikes, n_events)
            rel = rel.ravel()  # flatten all differences
            # Use vectorized filtering.
            all_rel_spikes.append(rel[(rel >= window[0]) & (rel <= window[1])])
        if all_rel_spikes:
            all_spikes = np.concatenate(all_rel_spikes)
        else:
            all_spikes = np.array([])
        
        counts, edges = np.histogram(all_spikes, bins=bins)
        num_events = len(events) if len(events) > 0 else 1
        num_neurons = len(neurons)
        psth = counts / (num_events * num_neurons * bin_width)
        if apply_smoothing:
            psth = moving_average(psth, window_size=3)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        return bin_centers, psth

    

    # Compute PSTHs and store results
    psth_results = {}
    psth_results["bin_centers_water_pre"], psth_results["psth_water_pre"] = compute_psth(water_pre)
    psth_results["bin_centers_sugar_pre"], psth_results["psth_sugar_pre"] = compute_psth(sugar_pre)
    psth_results["bin_centers_water_post"], psth_results["psth_water_post"] = compute_psth(water_post)
    psth_results["bin_centers_sugar_post"], psth_results["psth_sugar_post"] = compute_psth(sugar_post)
    
    # --- Plotting ---
    fig, axs = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    fig.suptitle(f"{group_name} Group - PSTH & Raster (Pre vs Post CTA)", fontsize=16)
    
    # Pre-CTA plots.
    raster_water_pre = gather_raster_data(water_pre)
    raster_sugar_pre = gather_raster_data(sugar_pre)
    
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
    ax2.plot(psth_results["bin_centers_water_pre"], psth_results["psth_water_pre"], color='#A2D5F2', label='Water')
    ax2.plot(psth_results["bin_centers_sugar_pre"], psth_results["psth_sugar_pre"], color='#F6A9A9', label='Sugar')
    ax2.axvline(0, color='red', linestyle='--')
    ax2.set_title("Pre-CTA PSTH Overlay")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Firing Rate (Hz)")
    ax2.set_xlim(window)
    ax2.legend()
    
    # Post-CTA plots.
    raster_water_post = gather_raster_data(water_post)
    raster_sugar_post = gather_raster_data(sugar_post)
    
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
    ax5.plot(psth_results["bin_centers_water_post"], psth_results["psth_water_post"], color='#A2D5F2', label='Water')
    ax5.plot(psth_results["bin_centers_sugar_post"], psth_results["psth_sugar_post"], color='#F6A9A9', label='Sugar')
    ax5.axvline(0, color='red', linestyle='--')
    ax5.set_title("Post-CTA PSTH Overlay")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Firing Rate (Hz)")
    ax5.set_xlim(window)
    ax5.legend()
    
    # Uniform y-axis scaling for PSTH overlays.
    pre_max = max(np.nanmax(psth_results["psth_water_pre"]) if psth_results["psth_water_pre"].size > 0 else 0,
                  np.nanmax(psth_results["psth_sugar_pre"]) if psth_results["psth_sugar_pre"].size > 0 else 0)
    post_max = max(np.nanmax(psth_results["psth_water_post"]) if psth_results["psth_water_post"].size > 0 else 0,
                   np.nanmax(psth_results["psth_sugar_post"]) if psth_results["psth_sugar_post"].size > 0 else 0)
    global_max = max(pre_max, post_max)
    buffer = 0.1 * global_max
    ax2.set_ylim(0, global_max + buffer)
    ax5.set_ylim(0, global_max + buffer)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(figures_dir, f"{group_name}_stacked.png")
    fig.savefig(os.path.join(save_folder, f"{group_name}_stacked.png"), dpi=150, bbox_inches="tight")
    print(f"Saved figure: {save_path}")
    plt.close(fig)

    return psth_results
