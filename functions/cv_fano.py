import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from functions.load_dataset import load_dataset

def compute_cv_isi(neurons, time_window):
    """Computes the Coefficient of Variation (CV) of Interspike Intervals (ISI) for neurons. ratio of the standard deviation of the inter-spike intervals (ISIs) to their mean"""
    start, end = time_window
    cv_values = []
    for neuron in neurons:
        spike_times = np.array(neuron[2])
        isi = np.diff(spike_times[(spike_times >= start) & (spike_times <= end)])
        if len(isi) > 1: # Need enough spikes to compute ISIs => with one value, std is 0 so not meaningful data
            cv_values.append(np.std(isi) / np.mean(isi))
        else:
            cv_values.append(np.nan)
    return cv_values

def compute_fano_factor(neurons, time_window, bin_width=0.05):
    """Computes the Fano Factor for neurons over a given time window dividing the variance of the spike counts by the mean spike count"""
    start, end = time_window
    fano_values = []
    for neuron in neurons:
        spike_times = np.array(neuron[2])
        bins = np.arange(start, end, bin_width)
        spike_counts, _ = np.histogram(spike_times, bins=bins)
        if np.mean(spike_counts) > 0:
            fano_values.append(np.var(spike_counts) / np.mean(spike_counts))
        else:
            fano_values.append(np.nan)
    return fano_values

def analyze_variability(filtered_datasets, processed_dir, filtered_files, save_folder):
    """
    Computes and plots the Coefficient of Variation (CV) of ISIs and the Fano Factor across different time windows.
    This version creates composite figures with file-level plots (boxplots) in the top row and group-level plots (bar plots)
    in the bottom row (one column per time window). The y-axis scaling is set to be identical across all subplots.
    """
    
    # Pre-load all datasets to avoid re-loading for every time window
    loaded_data = {}
    for dataset_name in filtered_datasets.keys():
        loaded_data[dataset_name] = load_dataset(os.path.join(processed_dir, filtered_files[dataset_name]))[0]

    
    variability_dir = os.path.join(save_folder, "CV_FF")
    os.makedirs(variability_dir, exist_ok=True)
    
    time_windows = ["Non-Stimuli", "Pre-CTA", "Post-CTA"]
    cv_results, fano_results = {}, {}
    
    # For global y-axis scaling (CV and Fano)
    cv_file_max_list = []
    cv_group_max_list = []
    fano_file_max_list = []
    fano_group_max_list = []
    
    # Create composite figures (2 rows x 3 columns)
    fig_cv, axs_cv = plt.subplots(2, 3, figsize=(18, 10))
    fig_fano, axs_fano = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loop over time windows
    for i, window_name in enumerate(time_windows):
        cv_data, fano_data, labels = [], [], []
        cv_file_means = []  # to compute group-level values
        fano_file_means = []
        dataset_groups = []
        
        for dataset_name, (neurons, non_stimuli_time) in filtered_datasets.items():
            data = loaded_data[dataset_name]
            sacc_start = data.get("sacc drinking session start time", 0)
            cta_time = data.get("CTA injection time", 0)
            
            if window_name == "Non-Stimuli":
                twindow = non_stimuli_time
            elif window_name == "Pre-CTA":
                twindow = (sacc_start, cta_time)
            elif window_name == "Post-CTA":
                max_time = max((np.max(neuron[2]) for neuron in neurons if len(neuron[2]) > 0), default=0)
                twindow = (cta_time + 3 * 3600, max_time)
            
            cv_values = compute_cv_isi(neurons, time_window=twindow)
            fano_values = compute_fano_factor(neurons, time_window=twindow, bin_width=0.05)
            
            cv_data.append(cv_values)
            fano_data.append(fano_values)
            labels.append(dataset_name)
            
            # Compute file-level mean values for group-level plotting
            cv_mean = np.nanmean(cv_values)
            fano_mean = np.nanmean(fano_values)
            cv_file_means.append(cv_mean)
            fano_file_means.append(fano_mean)
            
            # Determine group from dataset name (minimal logic change)
            group = "Control" if "ctrl" in dataset_name.lower() else "Experimental"
            dataset_groups.append(group)
            
            cv_results[dataset_name] = cv_values
            fano_results[dataset_name] = fano_values
        
        # File-level boxplots (top row)
        axs_cv[0, i].boxplot(cv_data, labels=labels, patch_artist=True,
                             boxprops=dict(facecolor='skyblue', color='black'),
                             medianprops=dict(color='black'))
        axs_cv[0, i].set_title(window_name)
        axs_cv[0, i].set_ylabel("CV")
        
        axs_fano[0, i].boxplot(fano_data, labels=labels, patch_artist=True,
                               boxprops=dict(facecolor='skyblue', color='black'),
                               medianprops=dict(color='black'))
        axs_fano[0, i].set_title(window_name)
        axs_fano[0, i].set_ylabel("Fano Factor")
        
        # Record global max for file-level (CV and Fano)
        try:
            file_cv_max = np.nanmax(np.concatenate([np.array(x) for x in cv_data]))
        except ValueError:
            file_cv_max = 0
        cv_file_max_list.append(file_cv_max)
        
        try:
            file_fano_max = np.nanmax(np.concatenate([np.array(x) for x in fano_data]))
        except ValueError:
            file_fano_max = 0
        fano_file_max_list.append(file_fano_max)
        
        # Compute group-level summaries
        group_dict_cv = {"Control": [], "Experimental": []}
        group_dict_fano = {"Control": [], "Experimental": []}
        for j, grp in enumerate(dataset_groups):
            group_dict_cv[grp].append(cv_file_means[j])
            group_dict_fano[grp].append(fano_file_means[j])
        
        group_cv_means = {}
        group_cv_stds = {}
        group_fano_means = {}
        group_fano_stds = {}
        for grp in ["Control", "Experimental"]:
            if group_dict_cv[grp]:
                group_cv_means[grp] = np.nanmean(group_dict_cv[grp])
                group_cv_stds[grp] = np.nanstd(group_dict_cv[grp])
            else:
                group_cv_means[grp] = 0
                group_cv_stds[grp] = 0
            if group_dict_fano[grp]:
                group_fano_means[grp] = np.nanmean(group_dict_fano[grp])
                group_fano_stds[grp] = np.nanstd(group_dict_fano[grp])
            else:
                group_fano_means[grp] = 0
                group_fano_stds[grp] = 0
        
        groups = ["Control", "Experimental"]
        # Group-level bar plots (bottom row)
        cv_group_vals = [group_cv_means[g] for g in groups]
        cv_group_errs = [group_cv_stds[g] for g in groups]
        axs_cv[1, i].bar(groups, cv_group_vals, yerr=cv_group_errs,
                         color='skyblue', edgecolor='k', alpha=0.7, capsize=5)
        axs_cv[1, i].set_title(window_name + " (Group)")
        axs_cv[1, i].set_ylabel("CV")
        group_cv_max = max([cv_group_vals[k] + cv_group_errs[k] for k in range(len(groups))])
        cv_group_max_list.append(group_cv_max)
        
        fano_group_vals = [group_fano_means[g] for g in groups]
        fano_group_errs = [group_fano_stds[g] for g in groups]
        axs_fano[1, i].bar(groups, fano_group_vals, yerr=fano_group_errs,
                           color='skyblue', edgecolor='k', alpha=0.7, capsize=5)
        axs_fano[1, i].set_title(window_name + " (Group)")
        axs_fano[1, i].set_ylabel("Fano Factor")
        group_fano_max = max([fano_group_vals[k] + fano_group_errs[k] for k in range(len(groups))])
        fano_group_max_list.append(group_fano_max)
    
    # Set the y-axis scaling for CV plots to 0-20
    for ax in axs_cv.flatten():
        ax.set_ylim(0, 15)

    
    # Set the same y-axis scaling across all subplots for Fano Factor
    global_fano_ymax = max(max(fano_file_max_list), max(fano_group_max_list))
    for ax in axs_fano.flatten():
        ax.set_ylim(0, global_fano_ymax)
    
    # Finalize and save composite figures
    fig_cv.suptitle("CV of ISIs Across Time Windows and Recordings (File & Group Level)", fontsize=16)
    fig_cv.tight_layout(rect=[0, 0, 1, 0.96])
    cv_filename = os.path.join(variability_dir, "cv_composite.png")
    fig_cv.savefig(cv_filename, dpi=300, bbox_inches="tight")  # Use fig_cv.savefig(...)
    plt.close(fig_cv)

    fig_fano.suptitle("Fano Factor of Spike Counts Across Time Windows and Recordings (File & Group Level)", fontsize=16)
    fig_fano.tight_layout(rect=[0, 0, 1, 0.96])
    fano_filename = os.path.join(variability_dir, "fano_composite.png")
    fig_fano.savefig(fano_filename, dpi=150, bbox_inches="tight")  # Use fig_fano.savefig(...)
    plt.close(fig_fano)

# Explicitly export it
__all__ = ["analyze_variability"]