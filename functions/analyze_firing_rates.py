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
    Processes and analyzes firing rates across different datasets. 
    This function creates a composite figure with two panels:
      - The top panel shows three subplots (one for each time window) of the individual dataset firing rates.
      - The bottom panel shows three subplots of the group-level mean firing rates per window.
    """
    firingrates_dir = os.path.join(save_folder, "firingrates")
    os.makedirs(firingrates_dir, exist_ok=True)
    
    # Lists to collect per-recording data
    recording_names = []
    non_stimuli_means = []
    pre_CTA_means = []
    post_CTA_means = []
    non_stimuli_stds = []
    pre_CTA_stds = []
    post_CTA_stds = []
    group_list = []
    
    summary_stats = []  # To store summary stats for each recording

    # Process each dataset
    for dataset_name, (neurons_data, non_stimuli_time) in filtered_datasets.items():
        try:
            # Extract spike times and load additional dataset info
            spike_times_list = extract_spike_times(neurons_data)
            data = load_dataset(os.path.join(processed_dir, filtered_files[dataset_name]))[0]
            sacc_start = data.get("sacc drinking session start time", 0)
            cta_time = data.get("CTA injection time", 0)
            max_time = max((np.max(times) for times in spike_times_list if len(times) > 0), default=0)
            
            # Define time windows
            pre_CTA_time = (sacc_start, cta_time)
            post_CTA_time = (cta_time + 3 * 3600, max_time)
            
            # Compute firing rates for each window
            rates_non = compute_firing_rates(spike_times_list, non_stimuli_time)
            rates_pre = compute_firing_rates(spike_times_list, pre_CTA_time)
            rates_post = compute_firing_rates(spike_times_list, post_CTA_time)
            
            # Calculate means and standard deviations
            mean_non = np.mean(rates_non)
            mean_pre = np.mean(rates_pre)
            mean_post = np.mean(rates_post)
            
            std_non = np.std(rates_non)
            std_pre = np.std(rates_pre)
            std_post = np.std(rates_post)
            
            # Determine group based on dataset name
            group = "Control" if "ctrl" in dataset_name.lower() else "Experimental"
            
            # Append per-recording data
            recording_names.append(dataset_name)
            non_stimuli_means.append(mean_non)
            pre_CTA_means.append(mean_pre)
            post_CTA_means.append(mean_post)
            non_stimuli_stds.append(std_non)
            pre_CTA_stds.append(std_pre)
            post_CTA_stds.append(std_post)
            group_list.append(group)
            
            # Store summary statistics for this recording
            summary_stats.append({
                "Recording": dataset_name,
                "Group": group,
                "Non-Stimuli Mean": mean_non,
                "Pre-CTA Mean": mean_pre,
                "Post-CTA Mean": mean_post,
                "Non-Stimuli Std": std_non,
                "Pre-CTA Std": std_pre,
                "Post-CTA Std": std_post
            })
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
    
    # Convert collected stats to DataFrame for further grouping
    summary_df = pd.DataFrame(summary_stats)
    
    # Group-level summary: calculate mean and std for each group (Control vs Experimental)
    group_summary = summary_df.groupby("Group")[["Non-Stimuli Mean", "Pre-CTA Mean", "Post-CTA Mean"]].mean()
    group_std = summary_df.groupby("Group")[["Non-Stimuli Mean", "Pre-CTA Mean", "Post-CTA Mean"]].std()
    group_std.columns = ["Non-Stimuli Std", "Pre-CTA Std", "Post-CTA Std"]

    # Create a composite figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # -------------------
    # Top row: Individual recordings for each time window
    # -------------------
    # Non-Stimuli window (top left)
    axes[0, 0].bar(recording_names, non_stimuli_means, yerr=non_stimuli_stds,
                   color='skyblue', edgecolor='k', alpha=0.7, capsize=5)
    axes[0, 0].set_title("Non-Stimuli Firing Rates (Individual)")
    axes[0, 0].set_ylabel("Firing Rate (Hz)")
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Pre-CTA window (top middle)
    axes[0, 1].bar(recording_names, pre_CTA_means, yerr=pre_CTA_stds,
                   color='skyblue', edgecolor='k', alpha=0.7, capsize=5)
    axes[0, 1].set_title("Pre-CTA Firing Rates (Individual)")
    axes[0, 1].set_ylabel("Firing Rate (Hz)")
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Post-CTA window (top right)
    axes[0, 2].bar(recording_names, post_CTA_means, yerr=post_CTA_stds,
                   color='skyblue', edgecolor='k', alpha=0.7, capsize=5)
    axes[0, 2].set_title("Post-CTA Firing Rates (Individual)")
    axes[0, 2].set_ylabel("Firing Rate (Hz)")
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(axis='y', linestyle='--', alpha=0.6)
    
    # -------------------
    # Bottom row: Group-level summary for each time window
    # -------------------
    groups = ["Control", "Experimental"]
    
    # Non-Stimuli window group summary (bottom left)
    non_means = [group_summary.loc[g, "Non-Stimuli Mean"] if g in group_summary.index else 0 for g in groups]
    non_err = [group_std.loc[g, "Non-Stimuli Std"] if g in group_std.index else 0 for g in groups]
    axes[1, 0].bar(groups, non_means, yerr=non_err,
                   color='skyblue', edgecolor='k', alpha=0.7, capsize=5)
    axes[1, 0].set_title("Non-Stimuli Firing Rates (Group)")
    axes[1, 0].set_ylabel("Firing Rate (Hz)")
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Pre-CTA window group summary (bottom middle)
    pre_means = [group_summary.loc[g, "Pre-CTA Mean"] if g in group_summary.index else 0 for g in groups]
    pre_err = [group_std.loc[g, "Pre-CTA Std"] if g in group_std.index else 0 for g in groups]
    axes[1, 1].bar(groups, pre_means, yerr=pre_err,
                   color='skyblue', edgecolor='k', alpha=0.7, capsize=5)
    axes[1, 1].set_title("Pre-CTA Firing Rates (Group)")
    axes[1, 1].set_ylabel("Firing Rate (Hz)")
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Post-CTA window group summary (bottom right)
    post_means = [group_summary.loc[g, "Post-CTA Mean"] if g in group_summary.index else 0 for g in groups]
    post_err = [group_std.loc[g, "Post-CTA Std"] if g in group_std.index else 0 for g in groups]
    axes[1, 2].bar(groups, post_means, yerr=post_err,
                   color='skyblue', edgecolor='k', alpha=0.7, capsize=5)
    axes[1, 2].set_title("Post-CTA Firing Rates (Group)")
    axes[1, 2].set_ylabel("Firing Rate (Hz)")
    axes[1, 2].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Compute global maximum for file-level plots
    global_max_individual = max(
        max([x + y for x, y in zip(non_stimuli_means, non_stimuli_stds)]),
        max([x + y for x, y in zip(pre_CTA_means, pre_CTA_stds)]),
        max([x + y for x, y in zip(post_CTA_means, post_CTA_stds)])
    )

    # Compute global maximum for group-level plots
    global_max_group = max(
        max([x + y for x, y in zip(non_means, non_err)]),
        max([x + y for x, y in zip(pre_means, pre_err)]),
        max([x + y for x, y in zip(post_means, post_err)])
    )

    # Determine the overall maximum y-axis value
    global_ymax = max(global_max_individual, global_max_group)

    # Set the same y-axis limit for every subplot
    for ax in axes.flatten():
        ax.set_ylim(0, global_ymax)
    
    # Set overall figure title and adjust layout
    plt.suptitle("Firing Rates: Individual Recordings and Group-Level Summary", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.ioff()
    composite_filename = os.path.join(firingrates_dir, "firing_rates_dataset_and_group_level.png")
    plt.savefig(composite_filename, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Optional: Print the summary stats for debugging
    print("Final summary_df structure:")
    print(summary_df.head())
    print("\nGroup-level summary (Control vs Experimental):")
    print(group_summary)