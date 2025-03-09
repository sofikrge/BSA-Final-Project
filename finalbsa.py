"""
=================================================================================================================================================================================
BSA Final Assignment - Denise Jaeschke & Sofia Karageorgiou
=================================================================================================================================================================================
"""

#%% Imports
import pickle
import re
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys
from tqdm.auto import tqdm

from functions.load_dataset import load_dataset
from functions.plot_correlogram_matrix import plot_correlogram_matrix

#%%
"""
=================================================================================================================================================================================
Preliminary steps: Understanding our dataset + Extracting the data we need
=================================================================================================================================================================================
"""
# Inspect the data using the helper function
base_dir = os.path.dirname(os.path.abspath(__file__))
pkl_dir = os.path.join(base_dir, 'data', 'raw')
file_path = os.path.join(pkl_dir, "exp rat 2.pkl")

# Use the helper function to load the dataset
data, neurons, non_stimuli_time = load_dataset(file_path)

# print(type(data))  # It should be a dictionary
# print(data.keys())  # Check the keys of the dictionary

# Look at content of each key
# print("Event Times:", data["event_times"].keys())
# print("Event Times:", data["event_times"])
# print("Saccharin drinking start time:", data["sacc drinking session start time"])
# print("CTA injection time:", data["CTA injection time"])
# print("Number of neurons recorded:", len(data["neurons"]))
# print("Example neuron data:", data["neurons"][0])  # Checking the first neuron

"""
We got the following correctly and as expected:
- the times when saccharin and water were given.
- the start of the saccharin drinking session and the time of LiCl/saline injection.
- count the number of neurons recorded.
- print one neuron's data to understand its format.

Now, we'll extract the spike data
"""

# Extract data + determine save folder for all figures
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

# Save folder for all figures
save_folder = os.path.join(os.getcwd(), "reports", "figures")

"""
# Debug prints
print(ctrl_rat_1_neurons_data[:5])  # Print first 5 neurons of ctrl_rat_1
print("sacc_start_1:", sacc_start_1)
print("sacc_start_2:", sacc_start_2)
print("sacc_start_3:", sacc_start_3)
print("sacc_start_4:", sacc_start_4)

# Number of neurons per dataset
print("Number of neurons in ctrl_rat_1:", len(ctrl_rat_1_neurons_data))
print("Number of neurons in ctrl_rat_2:", len(ctrl_rat_2_neurons_data))
print("Number of neurons in exp_rat_2:", len(exp_rat_2_neurons_data))
print("Number of neurons in exp_rat_3:", len(exp_rat_3_neurons_data)) 

Now we know we have 27 neurons recorded for ctr rat 1, 4 for ctr rat 2, 13 for exp rat 2 and 25 for exp rat 3
"""
#%%
"""
=================================================================================================================================================================================
1.1 Exclusion criterion: MUA vs SUA - Correlogram
=================================================================================================================================================================================
Notes on process
- first did 0-tolerance plotting, almost all neurons would have to be excluded
- thought about noise, distant neurons affecting the recording etc. -> decided to say that mean + 2 stds of central bins of autocorrelogram are the threshold for autocorrelograms
and for cross-correlograms we check whether the central bins are the minima within that correlogram. if they are, the correlogram is considered to be problematic

"""
# Bin sizes dictionary
# We chose 0.0004s for all datasets because of our exclusion criteria -> elsevier "2 Define refractory period.
# Immediately after a nerve impulse is triggered, an ordinary stimulus is not able to generate another impulse. This brief period is termed the refractory period. 
# The refractory period consists of two phases—the absolute refractory period and the relative refractory period. The absolute refractory period lasts about 1/2500 of a second and is followed by the relative refractory period. 
# During the relative refractory period, a higher intensity stimulus can trigger an impulse."

# https://www.sciencedirect.com/topics/medicine-and-dentistry/refractory-period

# TODO
# - Ask Denise: Is the way global threshold was set good = take mean + 2 * stds of all bin counts of the autocorrelograms

# File path for processed data
processed_dir = os.path.join(base_dir, 'data', 'processed')
os.makedirs(processed_dir, exist_ok=True)

binsizes = { 
    "ctrl_rat_1": 0.0004,
    "ctrl_rat_2": 0.0004,
    "exp_rat_2": 0.0004,
    "exp_rat_3": 0.0004
}

# Example dictionary of datasets:
datasets = {
    "ctrl_rat_1": (ctrl_rat_1_neurons_data, non_stimuli_time_1),
    "ctrl_rat_2": (ctrl_rat_2_neurons_data, non_stimuli_time_2),
    "exp_rat_2":  (exp_rat_2_neurons_data, non_stimuli_time_3),
    "exp_rat_3":  (exp_rat_3_neurons_data, non_stimuli_time_4)
}

# Loop over each dataset and compute/check the correlogram matrix.
for dataset_name, (neurons_data, time_window) in datasets.items():
    # Plot and store correlogram data for this dataset.
    correlogram_data = plot_correlogram_matrix(
        neurons_data=neurons_data,
        binsize=binsizes[dataset_name],
        dataset_name=dataset_name,
        time_window=time_window,
        save_folder=save_folder,
        store_data=True
    )
    
    # Retrieve problematic indices from the returned dictionary.
    problematic_neuron_indices = correlogram_data.get("problematic_neuron_indices", set())
    print(f"Problematic indices for {dataset_name}: {problematic_neuron_indices}")
    
    # Filter out problematic neurons.
    filtered_neurons_data = [neuron for idx, neuron in enumerate(neurons_data)
                             if idx not in problematic_neuron_indices]
    
    # Update the original data dictionary with the filtered neurons.
    data["neurons"] = filtered_neurons_data
    
    # Build the output filename with "_filtered" suffix.
    output_filename = dataset_name + "_filtered.pkl"
    output_path = os.path.join(processed_dir, output_filename)
    
    # Save the full dictionary (with the updated "neurons" key) as a pickle file.
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Saved filtered data for {dataset_name} to {output_path}")

#%% Inspect new files DELETE BEFORE SUBMISSION TODO DOES NOT WORK

# Use the base directory to build a relative path to the processed folder.
base_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(base_dir, 'data', 'processed')
file_path = os.path.join(processed_dir, "exp_rat_2_filtered.pkl")
raw_dir = os.path.join(base_dir, 'data', 'raw')

# Option 1: If your load_dataset helper function is designed to work on the full dictionary,
# you can try using it:
data, neurons, non_stimuli_time = load_dataset(file_path)

# Inspect the filtered dataset
print(type(data))          # Should be a dictionary
print(data.keys())         # Check the keys of the dictionary
print("Event Times:", data["event_times"])  # for example, if present
print("Saccharin drinking start time:", data["sacc drinking session start time"])
print("CTA injection time:", data["CTA injection time"])
print("Number of neurons recorded:", len(data["neurons"]))
print("Example neuron data:", data["neurons"][0])  # Checking the first neuron

# Option 2: If your load_dataset helper doesn't work with the filtered files, simply load with pickle:
with open(file_path, "rb") as f:
    data = pickle.load(f)

print(type(data))          # Should be a dictionary
print(data.keys())         # Check the keys of the dictionary
print("Event Times:", data["event_times"])
print("Saccharin drinking start time:", data["sacc drinking session start time"])
print("CTA injection time:", data["CTA injection time"])
print("Number of neurons recorded:", len(data["neurons"]))
print("Example neuron data:", data["neurons"][0])

# For example, check for "ctrl_rat_1"
original_file = os.path.join(raw_dir, "ctrl rat 1.pkl")
filtered_file = os.path.join(processed_dir, "ctrl_rat_1_filtered.pkl")

# Load original and filtered data
with open(original_file, "rb") as f:
    original_data = pickle.load(f)

with open(filtered_file, "rb") as f:
    filtered_data = pickle.load(f)

original_neurons = original_data.get("neurons", [])
filtered_neurons = filtered_data.get("neurons", [])

print("Original neuron count:", len(original_neurons))
print("Filtered neuron count:", len(filtered_neurons))

# Suppose problematic_neuron_indices was stored in your filtered data output:
# (Alternatively, you may have printed it during filtering.)
# Here we assume you saved it as part of the returned dictionary.
# For example:
#   correlogram_data["problematic_neuron_indices"] = problematic_neuron_indices
problematic_indices = correlogram_data.get("problematic_neuron_indices", set())
print("Problematic indices:", problematic_indices)
print("Number of problematic neurons:", len(problematic_indices))
print("Difference in count:", len(original_neurons) - len(filtered_neurons))
#%%
"""
=================================================================================================================================================================================
1.2 Exclusion criterion: Remove individual spikes with ISI below absolute refractory period
=================================================================================================================================================================================
As we were quite lenient with our sorting in the previous criterion, we will now sort out the spikes that are too close to each other 
and are likely to be noise. We will use the absolute refractory period of 1/2500s to filter out these spikes.
"""