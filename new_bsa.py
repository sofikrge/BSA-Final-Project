#%%
import pickle
import re
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys

#%% 
"""
===========================================================
BSA Final Assignment - Denise Jaeschke & Sofia Karageorgiou
===========================================================
"""

# Import all of our helper functions which can be found under the functions folder
from functions.load_dataset import load_dataset
from functions.correlogram import correlogram
from functions.plot_correlogram_matrix import plot_correlogram_matrix
from functions.compute_firing_rates import compute_firing_rates
from functions.compute_firing_rate_std import compute_firing_rate_std
from functions.get_spike_times import get_spike_times
from functions.process_and_plot_dataset import process_and_plot_dataset

#%% Step 1 Inspect data
"""
===========================================================
Step 1: Inspect data
===========================================================
First we will check whether the data matches the documentation we were provided with.
Thus, we looked at the data types, the keys of the dictionary, and the content of each key.
TODO (Sofia): Maybe comment this whole part out before submission? 

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
===========================================================
Step 2: Data exclusion
===========================================================
Next, our main goal will be to check whether we are actually 
dealing with separate neuronal spiking data or whether some 
spiking activity that is supposed to come from a single 
neuron, is actually a result of multiple neurons being
treated as one.

For that, we will plot the correlogram of each data set and 
for now focus on the auto-correlograms.

TODO
We have the following gameplan:
- only look at unstimulated phase
- 0.5ms bin size as that is the absolute refractory period -> one spike is happening so look at 2 bins
- no immediate peak next to absolute refractory period
- 2ms relative refractory period -> look at 4 bins, there should be close to none as we are looking at the unstimulated 
phase and a very strong stimulus would be needed for a new spike
- chose a conservative criterion because our biggest enemy too high is data loss

""" 
#%% Extract data + run correlograms
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
print(ctrl_rat_1_neurons_data[:5])  # Print first 5 neurons of ctrl_rat_1

"""
print("sacc_start_1:", sacc_start_1)
print("sacc_start_2:", sacc_start_2)
print("sacc_start_3:", sacc_start_3)
print("sacc_start_4:", sacc_start_4)
"""

# Check if data is loaded properly
print(ctrl_rat_1_neurons_data[:5])  # Print first 5 neurons of ctr_rat_1

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

TODO (Sofia): Create it cause what we have rn is not functional :/
The calculations showed that ___ is the optimal bin size for the dataset ___.
(For details look at functions/calculateidealbinsize.py)

"""
#%% Correlogram calculation + plotting 
"""
TODO Make correlograms nicer looking: titles + legends etc
"""
# Define optimal bin sizes for each dataset
optimal_bin_sizes = {
    "ctrl_rat_1": 0.0005,  # Replace with the actual optimal bin size
    "ctrl_rat_2": 0.0005,  # Replace with the actual optimal bin size
    "exp_rat_2": 0.0005,   # Replace with the actual optimal bin size
    "exp_rat_3": 0.0005    # Replace with the actual optimal bin size
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

# TODO Discuss with Denise

What if we do find a pause in cross-correlograms?
- inspect subclusters, exploit fact that they are not symmetric and find out when they fire
- maybe also check adaptation over time as that might explain that
""" 
#%% Step 3: Descriptive metrics
"""
===========================================================
Step 3: Descriptive metrics
===========================================================
"""
# Let's check out the data we have with some descriptive measures at crucial time points: the Pre-CTA and Post-CTA window
"""
TODO
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
"""

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
    
# Create a figure with 2x2 subplots and shared axes
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()

# Collect y-axis limits from each subplot
y_mins, y_maxes = [], []

# Loop through each dataset and process it
for ax, (file_name, ds) in zip(axes, datasets.items()):
    try:
        local_ymin, local_ymax = process_and_plot_dataset(ds, file_name, ax)
        y_mins.append(local_ymin)
        y_maxes.append(local_ymax)
        
        # Print summary statistics
        data = ds["data"]
        sacc_start = data.get("sacc drinking session start time", 0)
        cta_time = data.get("CTA injection time", 0)
        spike_times_list = get_spike_times(data)
        non_stimuli_rates = compute_firing_rates(spike_times_list, ds["non_stimuli_time"])
        pre_CTA_rates = compute_firing_rates(spike_times_list, (sacc_start, cta_time))
        post_CTA_rates = compute_firing_rates(spike_times_list, (cta_time + 3 * 3600, max(np.max(times) for times in spike_times_list if len(times) > 0)))
        
        print(f"Summary statistics for {file_name}:")
        print("Mean non-stimuli firing rate:", np.mean(non_stimuli_rates))
        print("Mean pre-CTA firing rate:", np.mean(pre_CTA_rates))
        print("Mean post-CTA firing rate:", np.mean(post_CTA_rates))
        print("-" * 50)
        
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
# %%
