#%%
import pickle
import re
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys

from functions.load_dataset import load_dataset
from functions.plot_correlogram_matrix import plot_correlogram_matrix
from functions.compute_firing_rates import compute_firing_rates
from functions.get_spike_times import get_spike_times
from functions.process_and_plot_dataset import process_and_plot_dataset
from functions.find_outliers import find_outliers
from functions.compute_fano_factor import compute_fano_factor
from functions.compute_cv_isi import compute_cv_isi
from functions.merge_datasets import merge_datasets
from functions.plot_group_figures import plot_group_figures
from functions.plot_isi_metrics_single_neurons import plot_isi_metrics_single_neuron

#%% 
"""
=================================================================================================================================================================================
BSA Final Assignment - Denise Jaeschke & Sofia Karageorgiou
=================================================================================================================================================================================
"""
#%% All TODO s and ideas
"""
- IMPORTANT SOFIA: Need to change saving directory bc its saving to the wrong folder rn
- I think I am pooling all neurons per file for some metrics which are only meaningful per neuron?
- consider removing outliers?
- comment out data inspection before submission maybe
- Make correlograms nicer looking: titles + legends etc
- need to change the time window we're looking at, way too large for TIH etc.
- Unsure: I intentionally did not plot the PSTHs for each neuron separately bc I don't think that makes sense with the amount of data we're working with 
- we should consider having all plots with the same scale for better comparison
- create calculateidealbinsize (if we have time)
- add std to group level firing rates plot
- maybe for isi metrics we can have all neurons in one plot so we have 4 figures in total? because the current set-up is quite messy with each neuron getting its own plot
- save processed data in its prospective folder
- adjust naming of figures to be more descriptive
- have own folders per general code section for figures to make it easier to follow

To compare at the end:
- subtract baseline non-stimulus firing rate from evoked responses to see how much they change -> 2 plots: control water and sugar + experimental water and sugar
- statistical analysis to see if the changes are significant
- can one correlate metrics over time windows? e.g., correlation of neuron 1 precta with neuron 1 postcta: high correlation would mean firing remains similar
    while low correlation would mean the response changed -> maybe create heatmaps with different colors for different degrees of correlation?
Chats notes on that:
1. Subtract Baseline from Evoked Response: 
   - **Idea:** Compute the difference between the baseline (non-stimulus) firing rate and the evoked firing rate (during taste presentations).  
   - **Visualization:** Create separate plots for control and experimental conditions for water and sugar. This directly shows how much each neuron's firing rate changes from baseline.  
   - **Statistical Analysis:** You can run paired statistical tests (e.g., paired t-test or Wilcoxon signed-rank test) to assess if the change is significant within groups, and then compare between control and experimental groups.
2. Correlation of Metrics Over Time Windows:
   - **Idea:** Compute metrics (e.g., CV, Fano Factor, or even raw firing rate) per neuron for different time windows (e.g., pre-CTA vs. post-CTA). Then, correlate these metrics for each neuron.  
   - **Interpretation:**  
     - A high correlation indicates that the relative firing properties remain similar across conditions (e.g., a neuron that is bursty pre-CTA remains bursty post-CTA).  
     - A low correlation suggests that the firing properties have shifted—perhaps due to the effects of CTA.  
   - **Visualization:** Creating heatmaps of the correlation coefficients (using, say, different colors to indicate degrees of similarity/difference) is a great idea. This can provide a quick visual reference for which neurons change most dramatically.
3. Implementation Considerations:
   - When subtracting baseline from evoked responses, ensure you're aligning the time windows correctly.  
   - For the correlation analysis, you might want to compute the metric per neuron in each time window and then use a scatter plot (or a correlation matrix/heatmap) to compare the values.
   - It's also useful to check the overall distribution of these metrics to see if any neurons are outliers and how that might affect your statistical tests.

Descriptive metrics:
- Mean spiking rate
- Variance
- Coefficient of Variation (CV)
    1 for Poisson, because both mean and variance = lambda (rate parameter of process so average number of events in a given time interval)
    Shows how random vs regular activity
    When >1 then neurons are bursty, variability higher than expected
    When <1 usually dealing with regular neurons 
- Fano Factor
    F >1 indicates overdispersion so larger variance than mean, could be clustering or correlation among events
    F<1 underdispersion, more regular or uniform distribution of events than what a Poisson assumes
    If Fano factor approaches Cv^2 over long time intervals, it means that the next spike depends on the previous one
- ISI
    What is the chance for a spike t seconds after the previous
- TIH
    Histogram of time difference between adjacent spikes (so of ISI)
- Survivor function
    probability neuron stays quiet for time t after previous spike, initial value is 1 and decreases to 0 as t approaches infinity 
- Hazard function
    independent probability to fire at any single point
    mainly used to detect burst activity and refractory period
    focuses on risk of an event happening regardless of history
    basically rate at which survivor function decays
    might get very noisy at the end because there are only few neurons that spike with such long ISI

Gameplan for exclusion
- only look at unstimulated phase
- 0.5ms bin size as that is the absolute refractory period -> one spike is happening so look at 2 bins
- no immediate peak next to absolute refractory period
- 2ms relative refractory period -> look at 4 bins, there should be close to none as we are looking at the unstimulated 
phase and a very strong stimulus would be needed for a new spike
- chose a conservative criterion because our biggest enemy too high is data loss

"""

#%% Step 1 Inspect data
"""
=================================================================================================================================================================================
Step 1: Inspect data
=================================================================================================================================================================================
First we will check whether the data matches the documentation we were provided with.
Thus, we looked at the data types, the keys of the dictionary, and the content of each key.
"""
#%% Inspect the data using the helper function
base_dir = os.path.dirname(os.path.abspath(__file__))
pkl_dir = os.path.join(base_dir, 'data', 'raw')
file_path = os.path.join(pkl_dir, "exp rat 2.pkl")

# Use the helper function to load the dataset
data, neurons, non_stimuli_time = load_dataset(file_path)

print(type(data))  # It should be a dictionary
print(data.keys())  # Check the keys of the dictionary

# Look at content of each key
print("Event Times:", data["event_times"].keys())
print("Event Times:", data["event_times"])
print("Saccharin drinking start time:", data["sacc drinking session start time"])
print("CTA injection time:", data["CTA injection time"])
print("Number of neurons recorded:", len(data["neurons"]))
print("Example neuron data:", data["neurons"][0])  # Checking the first neuron

"""
This matches our expectations, namely:
- the times when saccharin and water were given.
- the start of the saccharin drinking session and the time of LiCl/saline injection.
- count the number of neurons recorded.
- print one neuron's data to understand its format.

Now, we'll extract the spike data
"""

#%% Step 2: Exclude Data
"""
=================================================================================================================================================================================
Step 2: Data exclusion
=================================================================================================================================================================================
Next, our main goal will be to check whether we are actually 
dealing with separate neuronal spiking data or whether some 
spiking activity that is supposed to come from a single 
neuron, is actually a result of multiple neurons being
treated as one.

For that, we will plot the correlogram of each data set and 
for now focus on the auto-correlograms.

""" 
#%% Extract data 
# Extract spiking data from each pkl file and save it as its own variable
# Define file paths
file_1 = os.path.join(pkl_dir, "ctrl rat 1.pkl")
file_2 = os.path.join(pkl_dir, "ctrl rat 2.pkl")
file_3 = os.path.join(pkl_dir, "exp rat 2.pkl")
file_4 = os.path.join(pkl_dir, "exp rat 3.pkl")

# Load data using helper function we defined
data1, ctrl_rat_1_neurons_data, non_stimuli_time_1 = load_dataset(file_1)
data2, ctrl_rat_2_neurons_data, non_stimuli_time_2 = load_dataset(file_2)
data3, exp_rat_2_neurons_data, non_stimuli_time_3 = load_dataset(file_3)
data4, exp_rat_3_neurons_data, non_stimuli_time_4 = load_dataset(file_4)

# Debug print
# print(ctrl_rat_1_neurons_data[:5])  # Print first 5 neurons of ctrl_rat_1

"""
print("sacc_start_1:", sacc_start_1)
print("sacc_start_2:", sacc_start_2)
print("sacc_start_3:", sacc_start_3)
print("sacc_start_4:", sacc_start_4)
"""

# Check if data is loaded properly
# print(ctrl_rat_1_neurons_data[:5])  # Print first 5 neurons of ctr_rat_1

# Print the number of neurons for each dataset
""" 
print("Number of neurons in ctrl_rat_1:", len(ctrl_rat_1_neurons_data))
print("Number of neurons in ctrl_rat_2:", len(ctrl_rat_2_neurons_data))
print("Number of neurons in exp_rat_2:", len(exp_rat_2_neurons_data))
print("Number of neurons in exp_rat_3:", len(exp_rat_3_neurons_data)) 
"""
"""
Now we know we have 27 neurons recorded for ctr rat 1, 4 for ctr rat 2, 13 for exp rat 2 and 25 for exp rat 3
"""

# Before we work with correlograms, we want to check which bin size is the most optimal one per dataset
"""
With some research, we found out about the Cn(Delta) function which quantifies how well a particular bin size captures spike train information.
Our goal is to find a delta that minimises the Cn. 

Why care about bin size?
- if bins are too small, the histograms will be too noisy with too many empty bins
- if bins are too large, we might miss important information
The optimal bin size achieves the best balance between these two extremes.

We computed Cn(Delta) based on the formula from this video: https://youtu.be/VJGtyeR87R4?si=wsTlEeRorVug9kJC

The calculations showed that ___ is the optimal bin size for the dataset ___.
(For details look at functions/calculateidealbinsize.py)

"""
#%% Correlogram calculation + plotting 
# Define optimal bin sizes for each dataset
optimal_bin_sizes = {
    "ctrl_rat_1": 0.0005,  # Replace with the actual optimal bin size
    "ctrl_rat_2": 0.0005,  
    "exp_rat_2": 0.0005,   
    "exp_rat_3": 0.0005   
}
# Plot all 4 correlograms using their respective bin sizes
plot_correlogram_matrix(ctrl_rat_1_neurons_data, binsize=optimal_bin_sizes["ctrl_rat_1"], dataset_name="ctrl_rat_1", time_window=non_stimuli_time_1)
plot_correlogram_matrix(ctrl_rat_2_neurons_data, binsize=optimal_bin_sizes["ctrl_rat_2"], dataset_name="ctrl_rat_2", time_window=non_stimuli_time_2)
plot_correlogram_matrix(exp_rat_2_neurons_data, binsize=optimal_bin_sizes["exp_rat_2"], dataset_name="exp_rat_2", time_window=non_stimuli_time_3)
plot_correlogram_matrix(exp_rat_3_neurons_data, binsize=optimal_bin_sizes["exp_rat_3"], dataset_name="exp_rat_3", time_window=non_stimuli_time_4)

"""
Interpretation of the correlograms:
Features to note
- how flat is it? -> what does it mean if it's flat??
- does it show a pause in the middle? we expect one for auto-correlograms but not for cross-correlograms

What if we do find a pause in cross-correlograms?
- inspect subclusters, exploit fact that they are not symmetric and find out when they fire
- maybe also check adaptation over time as that might explain that
""" 
#%% Step 3: Descriptive metrics
"""
=================================================================================================================================================================================
Step 3: Descriptive metrics
=================================================================================================================================================================================
"""
# Let's check out the data we have with some descriptive measures at crucial time points: the Pre-CTA and Post-CTA window
# Define time windows for analysis & extract spike times for descriptive metrics

# List of already defined file paths
file_paths = [file_1, file_2, file_3, file_4]
file_names = [os.path.basename(path).replace(".pkl", "") for path in file_paths]

# Ensure figure save directory exists
figures_dir = os.path.join("reports", "figures")
os.makedirs(figures_dir, exist_ok=True)

# Create a dictionary to store the datasets
datasets = {}
for file_path, file_name in zip(file_paths, file_names):
    data, neurons, non_stimuli_time = load_dataset(file_path)
    datasets[file_name] = {
        "data": data,
        "neurons": neurons, 
        "non_stimuli_time": non_stimuli_time
    }
    
#%%  Create a figure with 2x2 subplots and shared axes
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()

# Collect y-axis limits from each subplot
y_mins, y_maxes = [], []

# Prep list to collect summary stats for individual recordings
summary_stats = []

# Loop through each dataset and process it
for ax, (file_name, ds) in zip(axes, datasets.items()):
    try:
        local_ymin, local_ymax = process_and_plot_dataset(ds, file_name, ax)
        y_mins.append(local_ymin)
        y_maxes.append(local_ymax)
        
        # Extract data and compute firing rates for individual summary
        data = ds["data"]
        sacc_start = data.get("sacc drinking session start time", 0)
        cta_time = data.get("CTA injection time", 0)
        spike_times_list = get_spike_times(data)
        
        # Define time windows
        non_stimuli_time = ds["non_stimuli_time"]
        pre_CTA_time = (sacc_start, cta_time)
        max_time = max((np.max(times) for times in spike_times_list if len(times) > 0), default=0)
        post_CTA_time = (cta_time + 3 * 3600, max_time)
        
        # Compute firing rates using helper functions
        non_stimuli_rates = compute_firing_rates(spike_times_list, non_stimuli_time)
        pre_CTA_rates = compute_firing_rates(spike_times_list, pre_CTA_time)
        post_CTA_rates = compute_firing_rates(spike_times_list, post_CTA_time)
        
        # Print summary statistics for the individual dataset
        print(f"Summary statistics for {file_name}:")
        print("Mean non-stimuli firing rate:", np.mean(non_stimuli_rates))
        print("Mean pre-CTA firing rate:", np.mean(pre_CTA_rates))
        print("Mean post-CTA firing rate:", np.mean(post_CTA_rates))
        print("-" * 50)
        
        # Identify outliers for each time window
        for window_name, rates in zip(["Non-Stimuli", "Pre-CTA", "Post-CTA"],
                                      [non_stimuli_rates, pre_CTA_rates, post_CTA_rates]):
            outlier_indices = find_outliers(rates)
            if len(outlier_indices) > 0:
                for idx in outlier_indices:
                    print(f"{file_name}, {window_name} outlier: Neuron index {idx}, rate = {rates[idx]}")
        
        # Determine group based on file name
        group = "Control" if "ctrl" in file_name.lower() else "Experimental"
        
        # Store individual means along with the group label
        summary_stats.append({
            "Recording": file_name,
            "Group": group,
            "Non-Stimuli Mean": np.mean(non_stimuli_rates),
            "Pre-CTA Mean": np.mean(pre_CTA_rates),
            "Post-CTA Mean": np.mean(post_CTA_rates)
        })
        
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Set uniform y-axis scaling across all subplots based on collected limits
global_ymin = min(y_mins) - 1
global_ymax = max(y_maxes) + 1
for ax in axes:
    ax.set_ylim(global_ymin, global_ymax)
    ax.set_xlim(-0.5, 3.5)

fig.suptitle("Firing Rates Across Time Windows for Each Recording", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
combined_figure_path = os.path.join(figures_dir, "firing_rates_combined.png")
plt.savefig(combined_figure_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Combined plot saved: {combined_figure_path}")

# Create a DataFrame for individual recordings and compute group-level summary
summary_df = pd.DataFrame(summary_stats)
print("\nIndividual recording summary:")
print(summary_df)

group_summary = summary_df.groupby("Group").mean(numeric_only=True)
print("\nGroup-level summary (Control vs Experimental):")
print(group_summary)

# Create a 1x2 figure for group-level means
fig2, axs = plt.subplots(1, 2, figsize=(12, 5))
# Iterate over groups and plot a bar chart for each
for ax, (group, row) in zip(axs, group_summary.iterrows()):
    time_windows = ["Non-Stimuli", "Pre-CTA", "Post-CTA"]
    means = [row["Non-Stimuli Mean"], row["Pre-CTA Mean"], row["Post-CTA Mean"]]
    
    ax.bar(time_windows, means, color='skyblue', edgecolor='k', alpha=0.7)
    ax.set_title(f"{group} Group")
    ax.set_ylabel("Mean Firing Rate (Hz)")
    ax.set_xlabel("Time Window")
    ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.suptitle("Group-Level Mean Firing Rates")
plt.tight_layout(rect=[0, 0, 1, 0.96])
group_plot_path = os.path.join(figures_dir, "group_level_firing_rates.png")
plt.savefig(group_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Group-level plot saved: {group_plot_path}")

"""
Interpretation of the descriptive metrics:

"""

# %% Compute Fano Factor + CV ISI
# Create dictionaries to store CV and Fano Factor results for each file
cv_results = {}
fano_results = {}

# Loop through each dataset and compute CV and Fano Factor
for file_name, ds in datasets.items():
    neurons = ds["neurons"]
    # For CV, here we use the non-stimuli time window (adjust if desired)
    cv_values = compute_cv_isi(neurons, time_window=ds["non_stimuli_time"])
    # For Fano Factor, also use the non-stimuli time window with a chosen bin width (e.g., 0.05 sec)
    fano_values = compute_fano_factor(neurons, time_window=ds["non_stimuli_time"], bin_width=0.05)
    
    cv_results[file_name] = cv_values
    fano_results[file_name] = fano_values

# Define the three time window names
time_windows = ["Non-Stimuli", "Pre-CTA", "Post-CTA"]

# Create a figure for CV with 1x3 subplots
fig_cv, axs_cv = plt.subplots(1, 3, figsize=(15, 5))
for i, window_name in enumerate(time_windows):
    data_to_plot = []
    labels = []
    for file_name, ds in datasets.items():
        data = ds["data"]
        neurons = ds["neurons"]
        spike_times_list = get_spike_times(data)
        sacc_start = data.get("sacc drinking session start time", 0)
        cta_time = data.get("CTA injection time", 0)
        
        if window_name == "Non-Stimuli":
            twindow = ds["non_stimuli_time"]
        elif window_name == "Pre-CTA":
            twindow = (sacc_start, cta_time)
        elif window_name == "Post-CTA":
            max_time = max((np.max(times) for times in spike_times_list if len(times) > 0), default=0)
            twindow = (cta_time + 3 * 3600, max_time)
        
        # Compute CV for this recording and time window
        cv_values = compute_cv_isi(neurons, time_window=twindow)
        data_to_plot.append(cv_values)
        labels.append(file_name)
    
    # Plot a boxplot for the current time window
    axs_cv[i].boxplot(data_to_plot, labels=labels, patch_artist=True,
                      boxprops=dict(facecolor='skyblue', color='black'),
                      medianprops=dict(color='black'))
    axs_cv[i].set_title(window_name)
    axs_cv[i].set_ylabel("CV")
    
plt.suptitle("CV of ISIs Across Time Windows and Recordings")
plt.tight_layout(rect=[0, 0, 1, 0.96])
cv_plot_path = os.path.join(figures_dir, "cv_temporal.png")
plt.savefig(cv_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print("Temporal CV plot saved:", cv_plot_path)

# Create a figure for Fano Factor with 1x3 subplots
fig_fano, axs_fano = plt.subplots(1, 3, figsize=(15, 5))
for i, window_name in enumerate(time_windows):
    data_to_plot = []
    labels = []
    for file_name, ds in datasets.items():
        data = ds["data"]
        neurons = ds["neurons"]
        spike_times_list = get_spike_times(data)
        sacc_start = data.get("sacc drinking session start time", 0)
        cta_time = data.get("CTA injection time", 0)
        
        if window_name == "Non-Stimuli":
            twindow = ds["non_stimuli_time"]
        elif window_name == "Pre-CTA":
            twindow = (sacc_start, cta_time)
        elif window_name == "Post-CTA":
            max_time = max((np.max(times) for times in spike_times_list if len(times) > 0), default=0)
            twindow = (cta_time + 3 * 3600, max_time)
        
        # Compute Fano Factor for this recording and time window (using a bin width of 0.05 sec)
        fano_values = compute_fano_factor(neurons, time_window=twindow, bin_width=0.05)
        data_to_plot.append(fano_values)
        labels.append(file_name)
    
    # Plot a boxplot for the current time window
    axs_fano[i].boxplot(data_to_plot, labels=labels, patch_artist=True,
                        boxprops=dict(facecolor='skyblue', color='black'),
                        medianprops=dict(color='black'))
    axs_fano[i].set_title(window_name)
    axs_fano[i].set_ylabel("Fano Factor")
    
plt.suptitle("Fano Factor of Spike Counts Across Time Windows and Recordings")
plt.tight_layout(rect=[0, 0, 1, 0.96])
fano_plot_path = os.path.join(figures_dir, "fano_temporal.png")
plt.savefig(fano_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print("Temporal Fano Factor plot saved:", fano_plot_path)

# %% Evoked responses PSTH for water and sugar

# Merge control and experimental dataasets 
# Group datasets into Control vs. Experimental based on file names
control_datasets = {k: ds for k, ds in datasets.items() if "ctrl" in k.lower()}
exp_datasets     = {k: ds for k, ds in datasets.items() if "exp" in k.lower()}

# Merge control and experimental datasets
ctrl_neurons, ctrl_water, ctrl_sugar, ctrl_cta = merge_datasets(control_datasets)
exp_neurons, exp_water, exp_sugar, exp_cta = merge_datasets(exp_datasets)

# Now produce 4 figures total: Pre & Post for Control, Pre & Post for Experimental
plot_group_figures("Control", ctrl_neurons, ctrl_water, ctrl_sugar, ctrl_cta)
plot_group_figures("Experimental", exp_neurons, exp_water, exp_sugar, exp_cta)

# %% Correlograms pre to post CTA
for file_name, ds in datasets.items():
    data = ds["data"]
    neurons_data = ds["neurons"]
    sacc_start = data.get("sacc drinking session start time", 0)
    cta_time = data.get("CTA injection time", 0)
    
    # Determine max spike time among all neurons (to define post-CTA window)
    max_spike_time = max((np.max(neuron[2]) for neuron in neurons_data if len(neuron[2]) > 0), default=0)
    
    # Define the pre-CTA time window: from saccharin drinking start to CTA injection.
    pre_window = (sacc_start, cta_time)
    # Define the post-CTA time window: from 3 hours after CTA until the last spike.
    post_window = (cta_time + 3 * 3600, max_spike_time)
    
    # Plot pre-CTA correlogram using your helper function.
    # This call saves a figure named "<file_name>_pre_correlogram.png" in your figures directory.
    plot_correlogram_matrix(neurons_data, binsize=0.0005, dataset_name=f"{file_name}_pre",
                            limit=0.02, time_window=pre_window)
    
    # Plot post-CTA correlogram similarly.
    plot_correlogram_matrix(neurons_data, binsize=0.0005, dataset_name=f"{file_name}_post",
                            limit=0.02, time_window=post_window)

# %% TIH, Survivor, Hazard
# Per-Neuron ISI Metrics for Each Recording
# Define the base directory for saving per-neuron ISI metric plots (created new one because it was too messy otherwise)
base_isi_dir = os.path.join(figures_dir, "ISI metrics per neuron")
os.makedirs(base_isi_dir, exist_ok=True)

for file_name, ds in datasets.items():
    data = ds["data"]
    neurons = ds["neurons"]
    
    # Define the time window for analysis (e.g., non-stimulated phase)
    time_window = ds["non_stimuli_time"]
    
    # Create a subfolder for this recording
    save_dir = os.path.join(base_isi_dir, file_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Processing {file_name} with {len(neurons)} neurons.")
    
    # Loop over each neuron and save its ISI metrics plot
    for i, neuron in enumerate(neurons):
        neuron_title = f"{file_name} - Neuron {i+1} ISI Metrics"
        neuron_save_path = os.path.join(save_dir, f"neuron_{i+1}_isi_metrics.png")
        plot_isi_metrics_single_neuron(neuron, time_window, bins=50,
                                       figure_title=neuron_title,
                                       save_path=neuron_save_path)
#%% import numpy as np
import matplotlib.pyplot as plt
import os
from functions.compute_firing_rates import compute_firing_rates  # if needed

def plot_firing_rate_paired(neurons, baseline_window, evoked_window, save_path=None,
                            title="Paired Firing Rates per Neuron", xlabel="Baseline Firing Rate (Hz)",
                            ylabel="Evoked Firing Rate (Hz)"):
    """
    For each neuron, compute the firing rate in two time windows (baseline and evoked)
    and create a paired dot plot. Each neuron is represented by two points (baseline and evoked)
    connected by a line.
    
    Parameters:
      neurons : list
          List of neurons (spike times expected at index 2).
      baseline_window : tuple
          (start, end) in seconds for baseline.
      evoked_window : tuple
          (start, end) in seconds for the evoked period.
      save_path : str or None
          If provided, saves the figure to this path.
      title : str
          Title for the plot.
      xlabel : str
          Label for the x-axis.
      ylabel : str
          Label for the y-axis.
          
    Returns:
      baseline_rates, evoked_rates : np.array
          Arrays of firing rates for each condition.
    """
    baseline_rates = []
    evoked_rates = []
    baseline_duration = baseline_window[1] - baseline_window[0]
    evoked_duration = evoked_window[1] - evoked_window[0]
    
    for neuron in neurons:
        spikes = np.array(neuron[2])
        # Filter spikes for baseline and evoked windows
        baseline_spikes = spikes[(spikes >= baseline_window[0]) & (spikes <= baseline_window[1])]
        evoked_spikes = spikes[(spikes >= evoked_window[0]) & (spikes <= evoked_window[1])]
        # Compute firing rates (Hz)
        baseline_rate = len(baseline_spikes) / baseline_duration if baseline_duration > 0 else 0
        evoked_rate = len(evoked_spikes) / evoked_duration if evoked_duration > 0 else 0
        baseline_rates.append(baseline_rate)
        evoked_rates.append(evoked_rate)
    
    baseline_rates = np.array(baseline_rates)
    evoked_rates = np.array(evoked_rates)
    
    # Create the paired dot plot
    n = len(baseline_rates)
    x = np.array([0, 1])
    plt.figure(figsize=(8, 6))
    for i in range(n):
        plt.plot(x, [baseline_rates[i], evoked_rates[i]], color='gray', alpha=0.7)
    plt.scatter(np.zeros(n), baseline_rates, color='blue', s=50, label='Baseline')
    plt.scatter(np.ones(n), evoked_rates, color='red', s=50, label='Evoked')
    
    plt.xticks([0, 1], ['Baseline', 'Evoked'])
    plt.ylabel("Firing Rate (Hz)")
    
    # Compute Pearson correlation coefficient (optional)
    corr = np.corrcoef(baseline_rates, evoked_rates)[0, 1]
    plt.title(title + f"\nPearson r = {corr:.2f}")
    
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Paired firing rate plot saved: {save_path}")
        plt.close()
    else:
        plt.show()
    
    return baseline_rates, evoked_rates

# ---------------- Integration into Main Pipeline -------------------

# For each recording in your datasets, compute and plot the paired firing rate changes.
# Each figure is one per rat (plotting all neurons on the same scatter plot).

for file_name, ds in datasets.items():
    data = ds["data"]
    neurons = ds["neurons"]
    
    # Define the time windows:
    # Baseline: Use the non-stimulated period (from 0 to saccharin session start)
    baseline_window = ds["non_stimuli_time"]
    # Evoked: For example, a 120-second window starting at the saccharin drinking session start time.
    evoked_window = (data["sacc drinking session start time"],
                     data["sacc drinking session start time"] + 120)
    
    save_path = os.path.join(figures_dir, f"{file_name}_firing_rate_paired.png")
    plot_firing_rate_paired(neurons, baseline_window, evoked_window, save_path=save_path,
                            title=f"{file_name}: Baseline vs. Evoked Firing Rates",
                            xlabel="Baseline Firing Rate (Hz)",
                            ylabel="Evoked Firing Rate (Hz)")
