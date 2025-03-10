import os
import numpy as np
import matplotlib.pyplot as plt
from functions.load_dataset import load_dataset

def group_psth_plots(final_filtered_datasets, final_filtered_files, processed_dir):
    """
    Groups datasets into "Control" and "Experimental" based on the dataset name and 
    generates a grouped PSTH overlay figure. For each dataset, the PSTHs for water and sugar 
    are computed for both pre-CTA and post-CTA conditions. Then, for each group, the PSTH curves 
    are overlaid in a 2x2 figure:
    
      - Row 1: Control group (left: pre-CTA, right: post-CTA)
      - Row 2: Experimental group (left: pre-CTA, right: post-CTA)
    
    All PSTH subplots share the same x-axis (window = (-0.5, 2.0)) and y-axis (set uniformly 
    based on the maximum firing rate across all curves). The figure is saved under:
    
         reports/figures/psth/grouped_psth.png
    """
    # Define parameters.
    window = (-0.5, 2.0)
    bin_width = 0.05

    # Helper function to compute PSTH for a given list of events.
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

    # Initialize a dictionary to group PSTH curves.
    groups = {"Control": {"pre": [], "post": []},
              "Experimental": {"pre": [], "post": []}}

    # Loop over each dataset.
    for dataset_name, (neurons, non_stimuli_time) in final_filtered_datasets.items():
        # Load the associated data.
        data, _, _ = load_dataset(os.path.join(processed_dir, final_filtered_files[dataset_name]))
        
        # Extract water and sugar events.
        water_events = np.array(data["event_times"].get("water", []))
        sugar_events = np.array(data["event_times"].get("sugar", []))
        # Get CTA injection time.
        cta_time = data.get("CTA injection time", None)
        # Split events into pre and post CTA using your logic.
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
        
        # Compute PSTHs for pre-CTA.
        bc_water_pre, psth_water_pre = compute_psth(neurons, water_pre)
        bc_sugar_pre, psth_sugar_pre = compute_psth(neurons, sugar_pre)
        # Compute PSTHs for post-CTA.
        bc_water_post, psth_water_post = compute_psth(neurons, water_post)
        bc_sugar_post, psth_sugar_post = compute_psth(neurons, sugar_post)
        
        # Group based on dataset name.
        if "ctrl" in dataset_name.lower():
            groups["Control"]["pre"].append((bc_water_pre, psth_water_pre, bc_sugar_pre, psth_sugar_pre))
            groups["Control"]["post"].append((bc_water_post, psth_water_post, bc_sugar_post, psth_sugar_post))
        else:
            groups["Experimental"]["pre"].append((bc_water_pre, psth_water_pre, bc_sugar_pre, psth_sugar_pre))
            groups["Experimental"]["post"].append((bc_water_post, psth_water_post, bc_sugar_post, psth_sugar_post))
    
    # Determine global maximum firing rate (y-axis uniform scaling) from all PSTH curves.
    global_max = 0
    for grp in groups.values():
        for cond in ["pre", "post"]:
            for bc_water, psth_water, bc_sugar, psth_sugar in grp[cond]:
                max_val = max(np.nanmax(psth_water) if psth_water.size > 0 else 0,
                              np.nanmax(psth_sugar) if psth_sugar.size > 0 else 0)
                if max_val > global_max:
                    global_max = max_val

    # Create a grouped 2x2 figure.
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle("Grouped PSTH Overlays by Group and CTA Condition", fontsize=16)
    group_order = ["Control", "Experimental"]
    conditions = ["pre", "post"]
    for i, grp_name in enumerate(group_order):
        for j, cond in enumerate(conditions):
            ax = axs[i, j]
            # Plot each dataset's PSTH curves.
            for bc_water, psth_water, bc_sugar, psth_sugar in groups[grp_name][cond]:
                ax.plot(bc_water, psth_water, color='#A2D5F2', alpha=0.7)
                ax.plot(bc_sugar, psth_sugar, color='#F6A9A9', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--')
            ax.set_title(f"{grp_name} {cond.capitalize()}-CTA")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Firing Rate (Hz)")
            ax.set_xlim(window)
            ax.set_ylim(0, global_max)
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # Ensure the PSTH folder exists.
    save_dir = os.path.join("reports", "figures", "psth")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "grouped_psth.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved grouped PSTH figure: {save_path}")
    plt.close(fig)
