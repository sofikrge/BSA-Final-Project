import numpy as np
import matplotlib.pyplot as plt
import os

def plot_survivor_hazard(neuron, non_stimuli_time, sacc_start, cta_time, dataset_max_time,
                                          figure_title="Survivor and Hazard Functions for Neuron",
                                          save_folder=None, subfolder=None, neuron_label="neuron1"):
    """
    Compute and plot the Survivor and Hazard functions for a single neuron across three time windows:
      - Non-Stimuli: use non_stimuli_time
      - Pre-CTA: (sacc_start, cta_time)
      - Post-CTA: (cta_time + 3*3600, dataset_max_time)
    
    For each time window, the survivor function S(t)=P(ISI>t) and the hazard function
    h(t) = - (dS/dt) / S(t) are computed.
    
    Parameters:
    -----------
    neuron : list
        A single neuron's data; expected to have spike times at index 2.
    non_stimuli_time : tuple
        (start, end) time for the Non-Stimuli period.
    sacc_start : float
        The start time of the sacc drinking session.
    cta_time : float
        The CTA injection time.
    dataset_max_time : float
        The maximum spike time (across all neurons in the dataset); used for Post-CTA.
    figure_title : str
        Title for the overall figure.
    save_folder : str or None
        Base folder where the figure should be saved (e.g., "reports/figures").
    subfolder : str or None
        Subfolder name under "Survivor_Hazard" in which to save the figure.
    neuron_label : str
        The label for this neuron (e.g., "neuron1"); also used as the file name.
        
    Returns:
    --------
    dict
        A dictionary with computed metrics for each time window.
        Each key (window name) maps to a dictionary containing:
            "isis", "sorted_isi", "survivor_prob", and "hazard_rate".
    """
    metrics = {}
    window_names = ["Non-Stimuli", "Pre-CTA", "Post-CTA"]
    n_windows = len(window_names)
    limit_x=10 # only consider first 10 s for plotting
    hazard_max_values = []  # To later set uniform y-axis for hazard plots
    
    # Create a figure with 2 rows (survivor on top, hazard below) and one column per time window.
    fig, axs = plt.subplots(2, n_windows, figsize=(6 * n_windows, 10))
    fig.suptitle(figure_title)
    
    # Loop over each time window.
    for i, window_name in enumerate(window_names):
        # Determine the time window based on the window name.
        if window_name == "Non-Stimuli":
            twindow = non_stimuli_time
        elif window_name == "Pre-CTA":
            twindow = (sacc_start, cta_time)
        elif window_name == "Post-CTA":
            twindow = (cta_time + 3 * 3600, dataset_max_time)
        else:
            continue  # Should not occur.
        
        # Filter spikes for the current window.
        spikes = np.array(neuron[2])
        spikes = spikes[(spikes >= twindow[0]) & (spikes <= twindow[1])]
        
        if len(spikes) < 2:
            print(f"Not enough spikes for time window '{window_name}' in neuron {neuron_label}.")
            continue
        
        # Compute ISIs and avoid zeros.
        isis = np.diff(np.sort(spikes))
        isis = np.maximum(isis, 1e-6)
        sorted_isi = np.sort(isis)
        
        # Restrict to ISI values within the first 10 seconds.
        idx = sorted_isi <= limit_x
        sorted_isi = sorted_isi[idx]
        if len(sorted_isi) == 0:
            print(f"No ISI values under {limit_x} sec for window '{window_name}' in neuron {neuron_label}.")
            continue
        
        # Compute the Survivor Function: S(t) = 1 - empirical CDF.
        cumulative_prob = np.arange(1, len(sorted_isi) + 1) / len(sorted_isi)
        survivor_prob = 1 - cumulative_prob
        survivor_prob = np.maximum(survivor_prob, 1e-6)
        
        # Compute derivative of the Survivor Function.
        derivative_survivor = np.gradient(survivor_prob, sorted_isi)
        derivative_survivor = np.nan_to_num(derivative_survivor, nan=0, posinf=0, neginf=0)
        
        # Compute Hazard Function: h(t) = - (dS/dt) / S(t).
        hazard_rate = -derivative_survivor / survivor_prob
        
        # Record the maximum hazard (within 0-10 sec) for uniform scaling.
        hazard_max_values.append(np.nanmax(hazard_rate))
        
        
        # Store computed metrics.
        metrics[window_name] = {
            "isis": isis,
            "sorted_isi": sorted_isi,
            "survivor_prob": survivor_prob,
            "hazard_rate": hazard_rate
        }
        
        # Plot Survivor Function (top row).
        if n_windows > 1:
            ax_survivor = axs[0, i]
        else:
            ax_survivor = axs[0]
        ax_survivor.plot(sorted_isi, survivor_prob, color='blue')
        ax_survivor.set_title(f"Survivor: {window_name}")
        ax_survivor.set_xlabel("ISI (s)")
        ax_survivor.set_xlim(0, limit_x)
        ax_survivor.set_ylabel("Survivor Probability")
        ax_survivor.set_ylim(0, 1)
        
        # Plot Hazard Function (bottom row).
        if n_windows > 1:
            ax_hazard = axs[1, i]
        else:
            ax_hazard = axs[1]
        ax_hazard.plot(sorted_isi, hazard_rate, color='red')
        ax_hazard.set_title(f"Hazard: {window_name}")
        ax_hazard.set_xlabel("ISI (s)")
        ax_hazard.set_ylabel("Hazard Rate")
        ax_hazard.set_xlim(0, limit_x)
    
    # Set uniform y-axis for Hazard functions based on the maximum hazard across windows.
    if hazard_max_values:
        global_hazard_max = max(hazard_max_values)
    else:
        global_hazard_max = 1
    for i in range(n_windows):
        if n_windows > 1:
            axs[1, i].set_ylim(0, global_hazard_max)
        else:
            axs[1].set_ylim(0, global_hazard_max)
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure in the specified folder structure.
    if save_folder:
        full_folder = os.path.join(save_folder, "Survivor_Hazard")
        if subfolder:
            full_folder = os.path.join(full_folder, subfolder)
        os.makedirs(full_folder, exist_ok=True)
        save_path = os.path.join(full_folder, f"{neuron_label}.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Survivor and hazard functions plot saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()
    
    return metrics
