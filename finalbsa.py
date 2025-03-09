"""
=================================================================================================================================================================================
BSA Final Assignment - Denise Jaeschke & Sofia Karageorgiou
=================================================================================================================================================================================
"""

#%% Imports and settings
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
from functions.isi_tih import isi_tih

save_folder = os.path.join(os.getcwd(), "reports", "figures") # folder for figures
base_dir = os.path.dirname(os.path.abspath(__file__)) # Get the directory of the current file
raw_dir = os.path.join(base_dir, 'data', 'raw') # raw data directory
ctrl_1 = os.path.join(raw_dir, "ctrl rat 1.pkl")
ctrl_2 = os.path.join(raw_dir, "ctrl rat 2.pkl")
exp_2 = os.path.join(raw_dir, "exp rat 2.pkl")
exp_3 = os.path.join(raw_dir, "exp rat 3.pkl")
processed_dir = os.path.join(base_dir, 'data', 'processed') # processed data directory (after exclusions)

#
"""
=================================================================================================================================================================================
Preliminary steps: Understanding our dataset + Extracting the data we need
=================================================================================================================================================================================
"""
# Use the helper function to load a dataset
data, neurons, non_stimuli_time = load_dataset(exp_2)

"""
print(type(data))  # It should be a dictionary
print(data.keys())  # Check the keys of the dictionary
"""

"""
# Look at content of each key
print("Event Times:", data["event_times"].keys())
print("Event Times:", data["event_times"])
print("Saccharin drinking start time:", data["sacc drinking session start time"])
print("CTA injection time:", data["CTA injection time"])
print("Number of neurons recorded:", len(data["neurons"]))
print("Example neuron data:", data["neurons"][0])  # Checking the first neuron
"""

"""
We got the following correctly and as expected:
- the times when saccharin and water were given.
- the start of the saccharin drinking session and the time of LiCl/saline injection.
- count the number of neurons recorded.
- print one neuron's data to understand its format.

Now, we'll extract the spike data
"""

# Load all data using helper function we defined
data1, ctrl_rat_1_neurons_data, non_stimuli_time_1 = load_dataset(ctrl_1)
data2, ctrl_rat_2_neurons_data, non_stimuli_time_2 = load_dataset(ctrl_2)
data3, exp_rat_2_neurons_data, non_stimuli_time_3 = load_dataset(exp_2)
data4, exp_rat_3_neurons_data, non_stimuli_time_4 = load_dataset(exp_3)

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
"""
"""
Notes on process
- first did 0-tolerance plotting, almost all neurons would have to be excluded
- thought about noise, distant neurons affecting the recording etc. -> decided to say that mean + 2 stds of central bins of autocorrelogram are the threshold for autocorrelograms
and for cross-correlograms we check whether the central bins are the minima within that correlogram. if they are, the correlogram is considered to be problematic

Bin sizes decision
- We chose 0.0004s for all datasets because of our exclusion criteria -> elsevier "2 Define refractory period.
- Immediately after a nerve impulse is triggered, an ordinary stimulus is not able to generate another impulse. This brief period is termed the refractory period. 
- The refractory period consists of two phases—the absolute refractory period and the relative refractory period. The absolute refractory period lasts about 1/2500 of a second and is followed by the relative refractory period. 
- During the relative refractory period, a higher intensity stimulus can trigger an impulse."
- https://www.sciencedirect.com/topics/medicine-and-dentistry/refractory-period

Definition of problematic correlograms:
- for autocorrelograms: if either center bin exceeds global_threshold (mean + 2 stds of all correlogram center bins), or if the global peak is immediately outside the center bins.
- for cross-correlograms: if both center bins are the minima for the correlogram
"""

# File path for processed data
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

#%% Loop over each dataset and compute/check the correlogram matrix.
for dataset_name, (neurons_data, time_window) in datasets.items():
    print(f"\nProcessing dataset: {dataset_name}")
    
    # Plot and store correlogram data for this dataset
    correlogram_data = plot_correlogram_matrix(
        neurons_data=neurons_data,
        binsize=binsizes[dataset_name],
        dataset_name=dataset_name,
        time_window=time_window,
        save_folder=os.path.join(save_folder, "Correlograms"),
        store_data=True
    )
    
    # Retrieve problematic indices from the returned dictionary
    problematic_neuron_indices = correlogram_data.get("problematic_neuron_indices", set())
    print(f"Problematic indices for {dataset_name}: {problematic_neuron_indices}")
    
    # Filter out problematic neurons.
    filtered_neurons_data = [neuron for idx, neuron in enumerate(neurons_data)
                             if idx not in problematic_neuron_indices]
    
    # Update the original data dictionary with the filtered neurons
    data["neurons"] = filtered_neurons_data
    
    # Build the output filename with "_filtered" suffix.
    output_filename = dataset_name + "_filtered.pkl"
    output_path = os.path.join(processed_dir, output_filename)
    
    # Save the full dictionary (with the updated "neurons" key) as a pickle file
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Saved filtered data for {dataset_name} to {output_path}")

#%%
"""
=================================================================================================================================================================================
1.2 Exclusion criterion: Remove individual spikes with ISI below absolute refractory period
=================================================================================================================================================================================
As we were quite lenient with our sorting in the previous criterion, we will now sort out the spikes that are too close to each other 
and are likely to be noise. We will use the absolute refractory period of 1/2500s to filter out these spikes.
"""

# Define a dictionary mapping dataset names to filtered file names.
filtered_files = {
    "ctrl_rat_1": "ctrl_rat_1_filtered.pkl",
    "ctrl_rat_2": "ctrl_rat_2_filtered.pkl",
    "exp_rat_2":  "exp_rat_2_filtered.pkl",
    "exp_rat_3":  "exp_rat_3_filtered.pkl"
}

filtered_datasets = {}
# Build a dictionary of filtered datasets using the load_dataset helper function
filtered_datasets = {}
for name, filename in filtered_files.items():
    file_path = os.path.join(processed_dir, filename)
    data, neurons, non_stimuli_time = load_dataset(file_path)
    filtered_datasets[name] = (neurons, non_stimuli_time)

# Loop over each filtered dataset and plot ISI histograms for all neurons.
for dataset_name, (neurons_data, non_stimuli_time) in filtered_datasets.items():
    print(f"\nProcessing ISI histograms for filtered dataset: {dataset_name}")
    total_neurons = len(neurons_data)
    problematic_count = 0  # Initialize counter for this dataset.
    
    # Process each neuron in the filtered dataset.
    for idx, neuron in enumerate(neurons_data):
        spike_times = neuron[2]  # Adjust according to your data structure.
        _, problematic_isis = isi_tih(
            spike_times,
            binsize=0.0004,
            min_interval=1/2500,
            neuron_id=idx,
            bins=50,
            dataset_name=dataset_name,
            save_folder="reports/figures/TIH",
            time_window=non_stimuli_time
        )
        # If there are any problematic ISIs, count this neuron as problematic.
        if problematic_isis.size > 0:
            problematic_count += 1
    
    # After processing all neurons in the dataset, print the summary.
    print(f"Filtered dataset {dataset_name}: {problematic_count} out of {total_neurons} neurons are problematic.")
"""
This check showed us that based on our criterion: 
- Filtered dataset ctrl_rat_1: 14 out of 15 neurons are problematic.
- Filtered dataset ctrl_rat_2: 1 out of 1 neurons are problematic.
- Filtered dataset exp_rat_2: 12 out of 12 neurons are problematic.
- Filtered dataset exp_rat_3: 7 out of 7 neurons are problematic.

That would force us to remove a lot of neurons. 
We were thinking about removing spikes that are too close to each other, 
but we decided to keep them for now as we did not find this mentioned in similar studies. 
"""

# %%
