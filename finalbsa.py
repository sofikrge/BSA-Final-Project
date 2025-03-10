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
from functions.load_dataset import load_dataset
from functions.plot_correlogram_matrix import plot_correlogram_matrix
from functions.isi_tih import isi_tih
from functions.analyze_firing_rates import analyze_firing_rates
from functions.cv_fano import analyze_variability
from functions.apply_manual_fusion import apply_manual_fusion
from functions.isi_tih import save_filtered_isi_datasets

# Loading files and folders
base_dir = os.path.dirname(os.path.abspath(__file__))  # Base directory of the script

save_folder = os.path.join(base_dir, "reports", "figures")  # Correct path
os.makedirs(save_folder, exist_ok=True)  # Ensure it exists
raw_dir = os.path.join(base_dir, 'data', 'raw') # raw data directory

processed_dir = os.path.join(base_dir, 'data', 'processed') # processed data directory (after exclusions)
os.makedirs(processed_dir, exist_ok=True)

# Define dataset file paths
dataset_paths = {
    "ctrl_rat_1": os.path.join(raw_dir, "ctrl rat 1.pkl"),
    "ctrl_rat_2": os.path.join(raw_dir, "ctrl rat 2.pkl"),
    "exp_rat_2": os.path.join(raw_dir, "exp rat 2.pkl"),
    "exp_rat_3": os.path.join(raw_dir, "exp rat 3.pkl")
}

# Load datasets once at the beginning
datasets = {}
for name, path in dataset_paths.items():
    data, neurons, non_stimuli_time = load_dataset(path)
    datasets[name] = {"data": data, "neurons": neurons, "non_stimuli_time": non_stimuli_time}

# Set binsize for each dataset
binsizes = {key: 0.0004 for key in datasets.keys()}
#%%
"""
=================================================================================================================================================================================
Preliminary steps: Understanding our dataset + Extracting the data we need
=================================================================================================================================================================================
"""

"""
print(type(data3))  # It should be a dictionary
print(data3.keys())  # Check the keys of the dictionary
"""

"""
# Look at content of each key
print("Event Times:", data3["event_times"].keys())
print("Event Times:", data3["event_times"])
print("Saccharin drinking start time:", data3["sacc drinking session start time"])
print("CTA injection time:", data3["CTA injection time"])
print("Number of neurons recorded:", len(data3["neurons"]))
print("Example neuron data3:", data3["neurons"][0])  # Checking the first neuron
"""

"""
We got the following correctly and as expected:
- the times when saccharin and water were given.
- the start of the saccharin drinking session and the time of LiCl/saline injection.
- count the number of neurons recorded.
- print one neuron's data to understand its format.

Now, we'll extract the spike data
"""

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

# Loop over each dataset and compute/check the correlogram matrix.
for dataset_name, dataset in datasets.items():
    neurons_data = dataset["neurons"]
    time_window = dataset["non_stimuli_time"] # because we we want to see a clearer relative refrac period as well
    print(f"\nProcessing correlogram for dataset: {dataset_name}. Please be patient, this will take a while.")
    
    # Plot and store correlogram data for this dataset
    correlogram_data = plot_correlogram_matrix(neurons_data=neurons_data,binsize=binsizes[dataset_name],dataset_name=dataset_name,time_window=time_window,save_folder=os.path.join(save_folder, "Correlograms"),store_data=True)
    
    # Retrieve problematic indices from the returned dictionary
    problematic_neuron_indices = correlogram_data.get("problematic_neuron_indices", set())
    print(f"Problematic indices for {dataset_name}: {problematic_neuron_indices}")
#%% Apply manual filter
"""
Define a manual filter: specify which neuron indices to fuse for each dataset
As the autocorrelograms don't look faulty, we decided to fuse neurons that are likely to be the same neuron
"""
fusion_file_mapping = {"ctrl_rat_1": ("ctrl rat 1.pkl", "ctrl_rat_1_filtered.pkl"),"ctrl_rat_2": ("ctrl rat 2.pkl", "ctrl_rat_2_filtered.pkl"),"exp_rat_2":  ("exp rat 2.pkl", "exp_rat_2_filtered.pkl"),"exp_rat_3":  ("exp rat 3.pkl", "exp_rat_3_filtered.pkl")}

manual_fusion = {
    "ctrl_rat_1": [{0, 2}, {21, 22, 23, 24}], 
    "ctrl_rat_2": [{0, 1, 2}], # e.g. meaning: fuse 0 1 and 2 into one neuron
    "exp_rat_2": [],  
    "exp_rat_3": [{0, 1}, {2, 6, 20}, {9, 10}, {11,12}, {13,14,}] 
}

apply_manual_fusion(datasets, manual_fusion, fusion_file_mapping, raw_dir, processed_dir)

#%%
"""
=================================================================================================================================================================================
1.2 Exclusion criterion: Remove individual spikes with ISI below absolute refractory period
=================================================================================================================================================================================
"""
"""
TODO Discuss with Denise

As we were quite lenient with our sorting in the previous criterion, we will now sort out the spikes that are too close to each other 
and are likely to be noise. We will use the absolute refractory period of 1/2500s to filter out these spikes.

We searched literature but could not find a paper that explicitly mentioned fitlering out spikes occurring in the absolute refractory period
so we did both and compared the results.

Note: We went with both, the correlogram and the ISI check because the correlogram gives us a better sense of firing patters as it is not just about consecutive spikes
"""

# Define a dictionary mapping dataset names to filtered file names.
filteredCC_files = {"ctrl_rat_1": "ctrl_rat_1_filteredCC.pkl","ctrl_rat_2": "ctrl_rat_2_filteredCC.pkl","exp_rat_2":  "exp_rat_2_filteredCC.pkl","exp_rat_3":  "exp_rat_3_filteredCC.pkl"}

filteredCC_datasets = {
    name: (datasets[name]["neurons"], datasets[name]["non_stimuli_time"])
    for name in filteredCC_files.keys()
}

# Set this flag to enable or disable filtering
apply_filtering = True  # Change to False if you want raw ISI histograms without filtering

# Loop over each filtered dataset and plot ISI histograms for all neurons.
for dataset_name, (neurons_data, non_stimuli_time) in filteredCC_datasets.items():
    print(f"\nProcessing ISI histograms for filtered dataset: {dataset_name}")
    
    # Process each neuron in the filtered dataset.
    for idx, neuron in enumerate(neurons_data):
        spike_times = neuron[2]  # Extract spike times
        
        isi_tih(
            spike_times,
            binsize=0.0004,
            min_interval=0.0004,
            neuron_id=idx,
            bins=50,
            dataset_name=dataset_name,
            save_folder=os.path.join(save_folder, "TIH"),
            time_window=non_stimuli_time,
            apply_filter=apply_filtering
        )

# Call the function to filter and save ISI-filtered datasets
save_filtered_isi_datasets(
    {name: (neurons, non_stimuli_time) for name, (neurons, non_stimuli_time) in filteredCC_datasets.items()},
    processed_dir,
    raw_dir,
    apply_filter=apply_filtering
)

"""
This check showed us that based on our criterion: 
- Filtered dataset ctrl_rat_1: 20 out of 23 neurons problematic
- Filtered dataset ctrl_rat_2: 2 out of 2 neurons are problematic.
- Filtered dataset exp_rat_2: 13 out of 13 neurons are problematic.
- Filtered dataset exp_rat_3: 11 out of 19 neurons are problematic.

That would force us to remove a lot of neurons. 
We were thinking about removing spikes that are too close to each other, 
but we decided to keep them for now as we did not find this mentioned in similar studies. 
"""

# %%
"""
=================================================================================================================================================================================
2. Firing parameters in non-stimulated epochs across time - Compare pre-CTA to post-CTA
=================================================================================================================================================================================
"""

"""
Overview of this section: 
1. Firing Rates Across Time Windows for Each Recording + Group level firing rates
3. Fano factor + CV
4. PSTH
5. Correlograms Pre and Post CTA
6. TIH, Survivor function and Hazard function
"""

# Define a dictionary mapping dataset names to filtered file names. TODO DELETE BEFORE SUBMISSION AS DUPLICATE
final_filtered_files = {"ctrl_rat_1": "ctrl_rat_1_ISIfiltered.pkl","ctrl_rat_2": "ctrl_rat_2_ISIfiltered.pkl","exp_rat_2":  "exp_rat_2_ISIfiltered.pkl","exp_rat_3":  "exp_rat_3_ISIfiltered.pkl"}
final_filtered_datasets = {}
for name, filename in final_filtered_files.items():
    file_path = os.path.join(processed_dir, filename)
    data, neurons, non_stimuli_time = load_dataset(file_path) 
    final_filtered_datasets[name] = (neurons, non_stimuli_time)

# Firing rates
os.makedirs(save_folder, exist_ok=True)
analyze_firing_rates(final_filtered_datasets, final_filtered_files, processed_dir, save_folder)

# Fano factor and CV
analyze_variability(final_filtered_datasets, processed_dir, final_filtered_files, save_folder)

# %%
"""
=================================================================================================================================================================================
3. Changes in Evoked Responses
=================================================================================================================================================================================
"""