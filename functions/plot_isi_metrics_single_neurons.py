import numpy as np
import matplotlib.pyplot as plt

def plot_isi_metrics_single_neuron(neuron, time_window, bins=50, figure_title="ISI Metrics for Neuron", save_path=None):
    """
    Compute and plot ISI metrics for a single neuron within a given time window.
    
    The function computes:
      - TIH (Time Interval Histogram)
      - Survivor Function (S(t) = P(ISI > t))
      - Hazard Function (instantaneous risk to fire, -dS/dt / S)
    
    Parameters:
    -----------
    neuron : list
        A single neuron's data; expected to have spike times at index 2.
    time_window : tuple
        (start, end) in seconds. Only spikes within this window are used.
    bins : int or sequence
        Binning for the TIH histogram.
    figure_title : str
        Title for the figure.
    save_path : str or None
        If provided, the figure is saved to this path; otherwise, it is displayed.
        
    Returns:
    --------
    dict
        A dictionary with computed metrics.
    """
    spikes = np.array(neuron[2])
    # Filter spikes within the time window
    spikes = spikes[(spikes >= time_window[0]) & (spikes <= time_window[1])]
    
    if len(spikes) < 2:
        print(f"Not enough spikes in the window {time_window} for this neuron.")
        return None
    
    # Compute inter-spike intervals (ISIs)
    isis = np.diff(np.sort(spikes))
    # Avoid zero ISIs (and extremely small ones) for stability
    isis = np.maximum(isis, 1e-6)
    
    # Compute TIH: already given by histogram of isis
    # (We will plot it directly below)
    
    # Compute Survivor Function:
    sorted_isi = np.sort(isis)
    cumulative_prob = np.arange(1, len(sorted_isi)+1) / len(sorted_isi)
    survivor_prob = 1 - cumulative_prob
    survivor_prob = np.maximum(survivor_prob, 1e-6)
    
    # Compute derivative of survivor function manually:
    if len(sorted_isi) < 3:
        # If there are too few points, use forward difference only
        derivative_survivor = np.gradient(survivor_prob, sorted_isi)
    else:
        first_deriv = (survivor_prob[1] - survivor_prob[0]) / (sorted_isi[1] - sorted_isi[0] + 1e-6)
        central_deriv = np.diff(survivor_prob) / (np.diff(sorted_isi) + 1e-6)
        last_deriv = (survivor_prob[-1] - survivor_prob[-2]) / (sorted_isi[-1] - sorted_isi[-2] + 1e-6)
        derivative_survivor = np.concatenate(([first_deriv], central_deriv, [last_deriv]))
        derivative_survivor = np.nan_to_num(derivative_survivor, nan=0, posinf=0, neginf=0)
    
    # Compute Hazard Function: h(t) = - (dS/dt) / S(t)
    hazard_rate = np.zeros_like(survivor_prob)
    for i in range(len(survivor_prob)):
        if survivor_prob[i] > 1e-6:
            hazard_rate[i] = -derivative_survivor[i] / survivor_prob[i]
        else:
            hazard_rate[i] = 0
    
    # Plotting: Create a figure with three subplots (TIH, Survivor, Hazard)
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    fig.suptitle(figure_title)
    
    # Subplot 1: TIH
    axs[0].hist(isis, bins=bins, color='#A2D5F2', edgecolor='black', alpha=0.7)
    axs[0].set_title("Time Interval Histogram (TIH)")
    axs[0].set_xlabel("ISI (s)")
    axs[0].set_ylabel("Frequency")
    
    # Subplot 2: Survivor Function
    axs[1].plot(sorted_isi, survivor_prob, color='blue')
    axs[1].set_title("Survivor Function")
    axs[1].set_xlabel("ISI (s)")
    axs[1].set_ylabel("Survivor Probability")
    
    # Subplot 3: Hazard Function
    axs[2].plot(sorted_isi, hazard_rate, color='red')
    axs[2].set_title("Hazard Function")
    axs[2].set_xlabel("ISI (s)")
    axs[2].set_ylabel("Hazard Rate")
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Per-neuron ISI metrics plot saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()
    
    # Return computed metrics for further analysis if needed
    return {
        "isis": isis,
        "sorted_isi": sorted_isi,
        "survivor_prob": survivor_prob,
        "hazard_rate": hazard_rate
    }
