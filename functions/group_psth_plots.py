import os
import numpy as np
import matplotlib.pyplot as plt
from functions.load_dataset import load_dataset

def group_psth_plots(final_filtered_datasets, final_filtered_files, processed_dir):
    """
    Generates a figure with raster plots (top row) and PSTHs (bottom row) for Control and Experimental groups.
    The layout is as follows:
    
      - Row 1: Raster plots (left: Control, right: Experimental)
      - Row 2: PSTH plots (left: Control, right: Experimental)
    
    """
    window = (-0.5, 2.0)
    bin_width = 0.05
    
    def compute_psth(neurons, events):
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
    
    groups = {"Control": {"pre": [], "post": []},
              "Experimental": {"pre": [], "post": []}}
    raster_data = {"Control": {"pre": [], "post": []},
                   "Experimental": {"pre": [], "post": []}}
    
    for dataset_name, (neurons, non_stimuli_time) in final_filtered_datasets.items():
        data, _, _ = load_dataset(os.path.join(processed_dir, final_filtered_files[dataset_name]))
        water_events = np.array(data["event_times"].get("water", []))
        sugar_events = np.array(data["event_times"].get("sugar", []))
        cta_time = data.get("CTA injection time", None)
        
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
        
        bc_water_pre, psth_water_pre = compute_psth(neurons, water_pre)
        bc_sugar_pre, psth_sugar_pre = compute_psth(neurons, sugar_pre)
        bc_water_post, psth_water_post = compute_psth(neurons, water_post)
        bc_sugar_post, psth_sugar_post = compute_psth(neurons, sugar_post)
        
        spikes_per_trial = lambda spikes, events: [spikes - e for e in events]
        
        spike_trains_pre = [spikes_per_trial(np.array(neuron[2]), water_pre.tolist() + sugar_pre.tolist()) for neuron in neurons]
        spike_trains_post = [spikes_per_trial(np.array(neuron[2]), water_post.tolist() + sugar_post.tolist()) for neuron in neurons]
        
        group = "Control" if "ctrl" in dataset_name.lower() else "Experimental"
        groups[group]["pre"].append((bc_water_pre, psth_water_pre, bc_sugar_pre, psth_sugar_pre))
        groups[group]["post"].append((bc_water_post, psth_water_post, bc_sugar_post, psth_sugar_post))
        raster_data[group]["pre"].extend(spike_trains_pre)
        raster_data[group]["post"].extend(spike_trains_post)
    
    global_max = max(
        max(np.nanmax(psth) if psth.size > 0 else 0 for _, psth, _, _ in grp["pre"] + grp["post"])
        for grp in groups.values()
    )
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    fig.suptitle("Raster and PSTH by Group and CTA Condition", fontsize=16)
    
    for i, (grp_name, grp_data) in enumerate(raster_data.items()):
        for j, cond in enumerate(["pre", "post"]):
            ax = axs[0, j] if i == 0 else axs[1, j]
            trials = sum(len(neuron_spikes) for neuron_spikes in grp_data[cond])
            trial_idx = 0
            for neuron_spikes in grp_data[cond]:
                for spikes in neuron_spikes:
                    ax.vlines(spikes, trial_idx, trial_idx + 1, color='black', alpha=0.7)
                    trial_idx += 1
            ax.set_title(f"{grp_name} {cond.capitalize()}-CTA Raster")
            ax.set_ylabel("Trials")
            ax.set_xlim(window)
    
    for i, (grp_name, grp_data) in enumerate(groups.items()):
        for j, cond in enumerate(["pre", "post"]):
            ax = axs[1, j]
            for bc_water, psth_water, bc_sugar, psth_sugar in grp_data[cond]:
                ax.plot(bc_water, psth_water, color='#A2D5F2', alpha=0.7)
                ax.plot(bc_sugar, psth_sugar, color='#F6A9A9', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--')
            ax.set_title(f"{grp_name} {cond.capitalize()}-CTA PSTH")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Firing Rate (Hz)")
            ax.set_xlim(window)
            ax.set_ylim(0, global_max)
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_dir = os.path.join("reports", "figures", "psth")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "grouped_raster_psth.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved raster and PSTH figure: {save_path}")
    plt.close(fig)
