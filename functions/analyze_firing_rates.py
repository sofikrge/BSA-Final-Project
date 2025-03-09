import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from functions.load_dataset import load_dataset

def extract_spike_times(neurons_data):
    """Extracts spike times from neuron data."""
    return [np.array(neuron[2]) for neuron in neurons_data]  # Convert to NumPy array for filtering

def compute_firing_rates(spike_times_list, time_window):
    """Computes firing rates for a given time window."""
    start, end = time_window
    duration = end - start
    return [len(spikes[(spikes >= start) & (spikes <= end)]) / duration if duration > 0 else 0 for spikes in spike_times_list]

def analyze_firing_rates(filtered_datasets, filtered_files, processed_dir, save_folder):
    """
    Processes and analyzes firing rates across different datasets, generating figures and summary statistics.
    """
    firingrates_dir = os.path.join(save_folder, "firingrates")
    os.makedirs(firingrates_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    summary_stats = []  # Collect summary stats

    # Loop through each filtered dataset and process it
    for ax, (dataset_name, (neurons_data, non_stimuli_time)) in zip(axes, filtered_datasets.items()):
        try:
            # Extract spike times
            spike_times_list = extract_spike_times(neurons_data)
            data = load_dataset(os.path.join(processed_dir, filtered_files[dataset_name]))[0]
            sacc_start = data.get("sacc drinking session start time", 0)
            cta_time = data.get("CTA injection time", 0)
            max_time = max((np.max(times) for times in spike_times_list if len(times) > 0), default=0)
            
            # Define time windows
            pre_CTA_time = (sacc_start, cta_time)
            post_CTA_time = (cta_time + 3 * 3600, max_time)
            
            # Compute firing rates
            non_stimuli_rates = compute_firing_rates(spike_times_list, non_stimuli_time)
            pre_CTA_rates = compute_firing_rates(spike_times_list, pre_CTA_time)
            post_CTA_rates = compute_firing_rates(spike_times_list, post_CTA_time)
            
            # Compute standard deviations
            non_stimuli_std = np.std(non_stimuli_rates)
            pre_CTA_std = np.std(pre_CTA_rates)
            post_CTA_std = np.std(post_CTA_rates)
            
            # Store summary stats
            group = "Control" if "ctrl" in dataset_name.lower() else "Experimental"
            summary_stats.append({
                "Recording": dataset_name,
                "Group": group,
                "Non-Stimuli Mean": np.mean(non_stimuli_rates),
                "Pre-CTA Mean": np.mean(pre_CTA_rates),
                "Post-CTA Mean": np.mean(post_CTA_rates),
                "Non-Stimuli Std": non_stimuli_std,
                "Pre-CTA Std": pre_CTA_std,
                "Post-CTA Std": post_CTA_std
            })
            
            # Simple bar plot for each dataset with error bars
            ax.bar(["Non-Stimuli", "Pre-CTA", "Post-CTA"], 
                   [np.mean(non_stimuli_rates), np.mean(pre_CTA_rates), np.mean(post_CTA_rates)], 
                   yerr=[non_stimuli_std, pre_CTA_std, post_CTA_std],
                   color='skyblue', edgecolor='k', alpha=0.7, capsize=5)
            ax.set_title(dataset_name)
            ax.set_ylabel("Mean Firing Rate (Hz)")
            ax.set_xlabel("Time Window")
            ax.grid(axis='y', linestyle='--', alpha=0.6)

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
    
    fig.suptitle("Firing Rates Across Time Windows for Each Recording", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.ioff() 
    plt.savefig(os.path.join(firingrates_dir, "firing_rates_combined.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Compute group-level summary and plot
    summary_df = pd.DataFrame(summary_stats)
    print("\nIndividual recording summary:")
    print(summary_df)
    group_summary = summary_df.groupby("Group")[["Non-Stimuli Mean", "Pre-CTA Mean", "Post-CTA Mean"]].mean()
    group_std = summary_df.groupby("Group")[["Non-Stimuli Mean", "Pre-CTA Mean", "Post-CTA Mean"]].std()
    group_std.columns = ["Non-Stimuli Std", "Pre-CTA Std", "Post-CTA Std"]
    print("\nGroup-level summary (Control vs Experimental):")
    print(group_summary)

    # Plot group-level firing rates with standard deviations
    fig2, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (group, row) in zip(axs, group_summary.iterrows()):
        stds = group_std.loc[group, ["Non-Stimuli Std", "Pre-CTA Std", "Post-CTA Std"]].values
        ax.bar(["Non-Stimuli", "Pre-CTA", "Post-CTA"], row, 
               yerr=stds, color='skyblue', edgecolor='k', alpha=0.7, capsize=5)
        ax.set_title(f"{group} Group")
        ax.set_ylabel("Mean Firing Rate (Hz)")
        ax.set_xlabel("Time Window")
        ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.suptitle("Group-Level Mean Firing Rates")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.ioff() 
    plt.savefig(os.path.join(firingrates_dir, "group_level_firing_rates.png"), dpi=300, bbox_inches="tight")
    plt.close()
    