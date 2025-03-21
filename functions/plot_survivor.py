import numpy as np
import matplotlib.pyplot as plt
import os

def plot_survivor(neuron, non_stimuli_time, sacc_start, cta_time, dataset_max_time,
                                          figure_title="Survivor Function for Neuron",
                                          save_folder=None, subfolder=None, neuron_label="neuron1"):
    """
    Compute and plot the Survivor function for a single neuron across all 3 time windows
    S(t)=P(ISI>t) -> prob that an ISI exceeds t
    """
    
    metrics = {}
    window_names = ["Non-Stimuli", "Pre-CTA", "Post-CTA"]
    n_windows = len(window_names)
    limit_x = 10  # Only consider the first 10 s for plotting
    
    # Create a figure with 1 row& 1 column per time window
    fig, axs = plt.subplots(1, n_windows, figsize=(6 * n_windows, 5))
    fig.suptitle(figure_title)
    
    # Loop over each time window.
    for i, window_name in enumerate(window_names):
        if window_name == "Non-Stimuli":
            twindow = non_stimuli_time
        elif window_name == "Pre-CTA":
            twindow = (sacc_start, cta_time)
        elif window_name == "Post-CTA":
            twindow = (cta_time + 3 * 3600, dataset_max_time)
        else:
            continue
        
        # Filter spikes for the current window
        spikes = np.array(neuron[2])
        spikes = spikes[(spikes >= twindow[0]) & (spikes <= twindow[1])]
        
        if len(spikes) < 2:
            print(f"Not enough spikes for time window '{window_name}' in neuron {neuron_label}.")
            continue
        
        # Compute ISIs & avoid 0s
        isis = np.diff(np.sort(spikes))
        isis = np.maximum(isis, 1e-6)
        sorted_isi = np.sort(isis)
        
        # Restrict to ISI values within the first 10 seconds -> approaches 0 either way
        idx = sorted_isi <= limit_x
        sorted_isi = sorted_isi[idx]
        if len(sorted_isi) == 0:
            print(f"No ISI values under {limit_x} sec for window '{window_name}' in neuron {neuron_label}.")
            continue
        
        # Compute the survivor function
        cumulative_prob = np.arange(1, len(sorted_isi) + 1) / len(sorted_isi)
        survivor_prob = 1 - cumulative_prob
        survivor_prob = np.maximum(survivor_prob, 1e-6)
        
        # Store metrics
        metrics[window_name] = {
            "isis": isis,
            "sorted_isi": sorted_isi,
            "survivor_prob": survivor_prob
        }
        
        # Plot
        if n_windows > 1:
            ax_survivor = axs[i]
        else:
            ax_survivor = axs
        ax_survivor.plot(sorted_isi, survivor_prob, color='blue')
        ax_survivor.set_title(f"Survivor: {window_name}")
        ax_survivor.set_xlabel("ISI (s)")
        ax_survivor.set_xlim(0, limit_x)
        ax_survivor.set_ylabel("Survivor Probability")
        ax_survivor.set_ylim(0, 1)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save fig
    if save_folder:
        full_folder = os.path.join(save_folder, "Survivor_Function")

        if subfolder:
            full_folder = os.path.join(full_folder, subfolder)
        os.makedirs(full_folder, exist_ok=True)
        save_path = os.path.join(full_folder, f"{neuron_label}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        # print(f"Survivor function plot saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()

    
    return metrics

def plot_survivor_dataset_summary(metrics_list, dataset_name, save_folder, limit_x=10):
    """
    Average per dataset
    """
    window_names = ["Non-Stimuli", "Pre-CTA", "Post-CTA"]
    n_windows = len(window_names)
    
    # Define a common x-axis
    x_common = np.linspace(0, limit_x, 200)
    
    # Prep dictionary survivor functions for each window
    avg_survivor = {window: [] for window in window_names}
    
    # Iterate over the precomputed metrics for each neuron
    for metrics in metrics_list:
        for window_name in window_names:
            if window_name in metrics:
                sorted_isi = metrics[window_name]["sorted_isi"]
                survivor_prob = metrics[window_name]["survivor_prob"]
                
                # Create x and y values
                x_vals = np.concatenate(([0], sorted_isi))
                y_vals = np.concatenate(([1], survivor_prob))
                
                # Common x grid
                interp_survivor = np.interp(x_common, x_vals, y_vals, left=1, right=y_vals[-1])
                avg_survivor[window_name].append(interp_survivor)
    
    # Compute mean across neurons
    avg_survivor_mean = {}
    for window_name in window_names:
        if len(avg_survivor[window_name]) > 0:
            avg_survivor_mean[window_name] = np.mean(avg_survivor[window_name], axis=0)
        else:
            avg_survivor_mean[window_name] = None
    
    # Plot
    fig, axs = plt.subplots(1, n_windows, figsize=(6 * n_windows, 5))
    fig.suptitle(f"Averaged Survivor Function for {dataset_name}")
    for i, window_name in enumerate(window_names):
        ax = axs[i]
        if avg_survivor_mean[window_name] is not None:
            ax.plot(x_common, avg_survivor_mean[window_name], color='red', label='Averaged Survivor')
        else:
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                    horizontalalignment='center', verticalalignment='center')
        ax.set_title(window_name)
        ax.set_xlabel("ISI (s)")
        ax.set_xlim(0, limit_x)
        ax.set_ylabel("Survivor Probability")
        ax.set_ylim(0, 1)
        ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save fig
    summary_folder = os.path.join(save_folder, "Survivor_Summary")
    os.makedirs(summary_folder, exist_ok=True)
    summary_filename = os.path.join(summary_folder, f"{dataset_name}Survivor_Summary.png")
    fig.savefig(summary_filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Averaged survivor function plot saved: {summary_filename}")
