import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from functions.load_dataset import load_dataset

def compute_cv_isi(neurons, time_window):
    """Computes the Coefficient of Variation (CV) of Interspike Intervals (ISI) for neurons."""
    start, end = time_window
    cv_values = []
    for neuron in neurons:
        spike_times = np.array(neuron[2])
        isi = np.diff(spike_times[(spike_times >= start) & (spike_times <= end)])
        if len(isi) > 1:
            cv_values.append(np.std(isi) / np.mean(isi))
        else:
            cv_values.append(np.nan)
    return cv_values

def compute_fano_factor(neurons, time_window, bin_width=0.05):
    """Computes the Fano Factor for neurons over a given time window."""
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
    """
    variability_dir = os.path.join(save_folder, "variability")
    os.makedirs(variability_dir, exist_ok=True)
    
    time_windows = ["Non-Stimuli", "Pre-CTA", "Post-CTA"]
    cv_results, fano_results = {}, {}
    
    fig_cv, axs_cv = plt.subplots(1, 3, figsize=(15, 5))
    fig_fano, axs_fano = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, window_name in enumerate(time_windows):
        cv_data, fano_data, labels = [], [], []
        
        for dataset_name, (neurons, non_stimuli_time) in filtered_datasets.items():
            data = load_dataset(os.path.join(processed_dir, filtered_files[dataset_name]))[0]
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
            
            cv_results[dataset_name] = cv_values
            fano_results[dataset_name] = fano_values
        
        axs_cv[i].boxplot(cv_data, labels=labels, patch_artist=True,
                          boxprops=dict(facecolor='skyblue', color='black'),
                          medianprops=dict(color='black'))
        axs_cv[i].set_title(window_name)
        axs_cv[i].set_ylabel("CV")
        
        axs_fano[i].boxplot(fano_data, labels=labels, patch_artist=True,
                            boxprops=dict(facecolor='skyblue', color='black'),
                            medianprops=dict(color='black'))
        axs_fano[i].set_title(window_name)
        axs_fano[i].set_ylabel("Fano Factor")
    
    plt.suptitle("CV of ISIs Across Time Windows and Recordings")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(variability_dir, "cv_temporal.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    plt.suptitle("Fano Factor of Spike Counts Across Time Windows and Recordings")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(variability_dir, "fano_temporal.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Variability analysis plots saved in:", variability_dir)

# Explicitly export it
__all__ = ["analyze_variability"]