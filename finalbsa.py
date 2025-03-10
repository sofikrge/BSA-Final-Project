"""
=================================================================================================================================================================================
BSA Final Assignment - Denise Jaeschke & Sofia Karageorgiou
=================================================================================================================================================================================
TODO
- how did we define the timestamps?
- when do we want to look at which time window? 
- compare our calculations with what was done in tirgulim to be on the safe side
- do we want firing rates over the entire time window or just the mean? 

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
from tqdm import tqdm

from functions.load_dataset import load_dataset
from functions.plot_correlogram_matrix import plot_correlogram_matrix
from functions.isi_tih import isi_tih
from functions.analyze_firing_rates import analyze_firing_rates
from functions.cv_fano import analyze_variability
from functions.apply_manual_fusion import apply_manual_fusion
from functions.isi_tih import save_filtered_isi_datasets
from functions.plot_survivor_hazard import plot_survivor_hazard
from functions.psth_rasterplot import psth_raster
from functions.group_psth_plots import group_psth_plots

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
Preliminary steps: Understanding our dataset
=================================================================================================================================================================================
"""

"""
# Chose a dataset to inspect ctrl_rat_1
dataset_name = "ctrl_rat_1"

# Access dataset
data = datasets[dataset_name]["data"]
neurons = datasets[dataset_name]["neurons"]

print(f"\nDataset: {dataset_name}")
print("Type:", type(data))  # Should be a dictionary
print("Keys:", data.keys())  # Check dataset structure

# Check specific details for a single dataset
print("Event Times:", data["event_times"].keys())

print("Water event times:", data["event_times"].get("water", "No water events found"))
print("Sugar event times:", data["event_times"].get("sugar", "No sugar events found"))

print("Saccharin drinking start time:", data.get("sacc drinking session start time"))
print("CTA injection time:", data.get("CTA injection time"))

# Print total neurons per dataset
print("Number of neurons per dataset:")
for dataset_name, dataset in datasets.items():
    print(f"{dataset_name}: {len(dataset['neurons'])} neurons")

# Check only one neuron
print("\nExample neuron:")
print(neurons[0])  # Print only the first neuron

# Now we know we have 27 neurons recorded for ctr rat 1, 4 for ctr rat 2, 13 for exp rat 2 and 25 for exp rat 3
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
2. Fano factor + CV = Variability
3. TIH, Survivor function and Hazard function
"""

# Define a dictionary mapping dataset names to filtered file names
final_filtered_files = {"ctrl_rat_1": "ctrl_rat_1_ISIfiltered.pkl","ctrl_rat_2": "ctrl_rat_2_ISIfiltered.pkl","exp_rat_2":  "exp_rat_2_ISIfiltered.pkl","exp_rat_3":  "exp_rat_3_ISIfiltered.pkl"}
final_filtered_datasets = {}
for name, filename in final_filtered_files.items():
    file_path = os.path.join(processed_dir, filename)
    data, neurons, non_stimuli_time = load_dataset(file_path) 
    final_filtered_datasets[name] = (neurons, non_stimuli_time)

#%% Firing rates
os.makedirs(save_folder, exist_ok=True)
analyze_firing_rates(final_filtered_datasets, final_filtered_files, processed_dir, save_folder)

# Fano factor and CV
analyze_variability(final_filtered_datasets, processed_dir, final_filtered_files, save_folder)

#%% Survivor function and Hazard function
for dataset_name, (neurons, non_stimuli_time) in final_filtered_datasets.items():
    # Load the associated data to extract sacc_start and cta_time.
    data = load_dataset(os.path.join(processed_dir, final_filtered_files[dataset_name]))[0]
    sacc_start = data.get("sacc drinking session start time", 0)
    cta_time = data.get("CTA injection time", 0)
    
    # Compute the maximum spike time across all neurons for the Post-CTA window.
    dataset_max_time = max((np.max(neuron[2]) for neuron in neurons if len(neuron[2]) > 0), default=0)
    
    # Use the dataset name as the subfolder.
    dataset_subfolder = dataset_name
    
    for idx, neuron in enumerate(neurons):
        neuron_label = f"{dataset_name}_neuron{idx+1}"
        plot_survivor_hazard(
            neuron,
            non_stimuli_time=non_stimuli_time,
            sacc_start=sacc_start,
            cta_time=cta_time,
            dataset_max_time=dataset_max_time,
            figure_title=f"Survivor & Hazard Functions for {neuron_label}",
            save_folder="reports/figures",  # Base folder.
            subfolder=dataset_subfolder,    # One folder per dataset.
            neuron_label=neuron_label
        )

# %%
"""
=================================================================================================================================================================================
3. Changes in Evoked Responses
=================================================================================================================================================================================
"""
"""
1. PSTH
2. Correlograms Pre and Post CTA
"""

# PSTH for each dataset
# Set the PSTH figures folder to reports/figures/psth.
psthfigures_dir = os.path.join("reports", "figures", "psth")
os.makedirs(psthfigures_dir, exist_ok=True)

import functions.psth_rasterplot as prp  # to override its figures_dir

# Set the desired PSTH folder to "reports/figures/psth" and ensure it exists.
desired_psth_folder = os.path.join("reports", "figures", "psth")
os.makedirs(desired_psth_folder, exist_ok=True)

# Loop over all datasets/files with a progress bar.
for dataset_name, (neurons, non_stimuli_time) in tqdm(final_filtered_datasets.items(), desc="Processing datasets", ncols=100):
    # Load the associated data.
    data = load_dataset(os.path.join(processed_dir, final_filtered_files[dataset_name]))[0]
    
    # Extract water and sugar events (adjust the key names if needed).
    water_events = np.array(data.get("event_times", {}).get("water", []))
    sugar_events = np.array(data.get("event_times", {}).get("sugar", []))
    
    # Debug: Print out event counts and CTA time.
    print(f"{dataset_name}: water_events: {len(water_events)}, sugar_events: {len(sugar_events)}")
    cta_time = data.get("CTA injection time", None)
    print(f"{dataset_name}: CTA time: {cta_time}")
    
    # Use the dataset name as the group name.
    group_name = dataset_name
    
    # Call psth_raster for this dataset.
    psth_raster(group_name, neurons, water_events, sugar_events, cta_time)

# group_psth_plots(final_filtered_datasets, final_filtered_files, processed_dir)
# %%
