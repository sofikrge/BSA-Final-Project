
"""
=================================================================================================================================================================================
BSA Final Assignment - Denise Jaeschke & Sofia Karageorgiou
=================================================================================================================================================================================
"""

#%% Imports, loading data, and setting up directories
import numpy as np
import os
from tqdm import tqdm
import glob
from scipy.stats import ks_2samp, ttest_rel, wilcoxon
import pandas as pd
import pickle

from functions.load_dataset import load_dataset
from functions.plot_correlogram_matrix import plot_correlogram_matrix
from functions.isi_tih import isi_tih
from functions.analyze_firing_rates import analyze_firing_rates 
from functions.cv_fano import analyze_variability
from functions.apply_manual_fusion import apply_manual_modification
from functions.isi_tih import save_filtered_isi_datasets
from functions.plot_survivor import plot_survivor, plot_survivor_dataset_summary
from functions.psth_rasterplot import psth_raster
from functions.psth_twobytwo import plot_neuron_rasters_2x2

# Loading files and folders
base_dir = os.path.abspath(os.path.dirname(__file__))  # folder of this script
save_folder = os.path.join(base_dir, "reports", "figures")
os.makedirs(save_folder, exist_ok=True)  # Ensure directory exists
raw_dir = os.path.join(base_dir, "data", "raw")
processed_dir = os.path.join(base_dir, 'data', 'processed')
os.makedirs(processed_dir, exist_ok=True)

# Define dataset file paths
dataset_paths = {
    "ctrl_rat_1": os.path.join(raw_dir, "ctrl rat 1.pkl"),
    "ctrl_rat_2": os.path.join(raw_dir, "ctrl rat 2.pkl"),
    "exp_rat_2": os.path.join(raw_dir, "exp rat 2.pkl"),
    "exp_rat_3": os.path.join(raw_dir, "exp rat 3.pkl"),
}

# Load datasets once at the beginning
datasets = {}
for name, path in dataset_paths.items():
    data, neurons, non_stimuli_time = load_dataset(path)
    datasets[name] = {"data": data, "neurons": neurons, "non_stimuli_time": non_stimuli_time}

# Set binsize in s for first operations
binsizes = {key: 0.0004 for key in datasets.keys()} # bc it is half of the absolute refractory period
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

print(f"Dataset: {dataset_name}")
print("Type:", type(data)) 
print("Keys:", data.keys())

print("Event Times:", data["event_times"].keys())

print("Water event times:", data["event_times"].get("water", "No water events found"))
print("Sugar event times:", data["event_times"].get("sugar", "No sugar events found"))

print("Saccharin drinking start time:", data.get("sacc drinking session start time"))
print("CTA injection time:", data.get("CTA injection time"))

print("Number of neurons per dataset:")
for dataset_name, dataset in datasets.items():
    print(f"{dataset_name}: {len(dataset['neurons'])} neurons")

print("Example neuron:")
print(neurons[0])

# Now we know we have 27 neurons recorded for ctr rat 1, 4 for ctr rat 2, 13 for exp rat 2 and 25 for exp rat 3
"""
#%%
"""
=================================================================================================================================================================================
1.1 Exclusion criterion: MUA vs SUA - Correlogram
=================================================================================================================================================================================
"""
"""

Overview of the next steps: 

1. Correlogram
Notes on process
- first did 0-tolerance plotting, almost all neurons would have to be excluded
- thought about noise, distant neurons affecting the recording etc. 
Our final criteria:
- for autocorrelograms: problematic if the center bins count is above the threshold of local mean - 2stdevs (95% of distribution) or if the bins immediately adjacent to the center are the global maximum.
- for cross-correlograms: if either center bins is below the defined threshold
    
Bin sizes & time-related decisions
- We chose 0.0004s because of our exclusion criteria 
    -> Elsevier "The absolute refractory period lasts about 1/2500 of a second and is followed by the relative refractory period. 
    -> During the relative refractory period, a higher intensity stimulus can trigger an impulse."
    -> Link: https://www.sciencedirect.com/topics/medicine-and-dentistry/refractory-period
- Only look at unstimulated phase to be able to see the relative refrac period as well

2. Manual fusion based on correlogram results
- Fused neurons that were flagged as problematic in the crosscorrelograms + if we felt like their plots looked somewhat symmetrical and they weren't just flagged as problematic bc of generally low firing rates
- We were quite lenient at this stage as we did want to be balanced regarding how much data we lose

3. Double-checking the fusion and deletion made an impact by rendering new correlograms 

"""
# 1. Correlogram
for dataset_name, dataset in datasets.items():
    neurons_data = dataset["neurons"]
    time_window = dataset["non_stimuli_time"]
    print(f"\nProcessing Correlogram for Raw Dataset: {dataset_name}. Please be patient, this might take a while.")
    correlogram_data = plot_correlogram_matrix(neurons_data=neurons_data,binsize=binsizes[dataset_name],dataset_name=dataset_name,time_window=time_window,save_folder=os.path.join(save_folder, "Correlograms"),store_data=True)
    
    # Prints to aid with exclusion decision
    # problematic_neuron_indices = correlogram_data.get("problematic_neuron_indices", set())
    # print(f"Problematic indices for {dataset_name}: {problematic_neuron_indices}")

print("\nAll correlograms have been plotted and saved.")

# 2. Manual fusion + deletion
file_mapping = {
    "ctrl_rat_1": "ctrl rat 1.pkl",
    "ctrl_rat_2": "ctrl rat 2.pkl",
    "exp_rat_2": "exp rat 2.pkl",
    "exp_rat_3": "exp rat 3.pkl"
}
manual_fusion = {
    "ctrl_rat_1": [{0, 2}, {13, 14}],
    "ctrl_rat_2": [{0, 1}],  # e.g. fuse neurons 1 and 2
    "exp_rat_2": [],
    "exp_rat_3": [{2, 3}, {4, 5}, {11, 12}, {20, 21}]
}
manual_deletion = {
    "ctrl_rat_1": [{25}],  # e.g. delete neuron 26
    "ctrl_rat_2": [],   
    "exp_rat_2": [{1,2,4,5,6,7,10}], 
    "exp_rat_3": []  
}
apply_manual_modification(datasets, manual_fusion, manual_deletion, file_mapping, raw_dir, processed_dir)
print("\nAll manually specified fusions and deletions have been processed and filtered datasets have been saved.")

# 3. Plotting Correlograms of filtered data
save_folder_processed = os.path.join(save_folder, "ProcessedCorrelograms")
filtered_files = glob.glob(os.path.join(processed_dir, "*_CCFiltered.pkl"))
for filtered_file in filtered_files:
    dataset_name = os.path.basename(filtered_file).replace("_CCFiltered.pkl", "")
    
    with open(filtered_file, "rb") as f:
        filtered_dataset = pickle.load(f)

    neurons_data_filtered = filtered_dataset["neurons"]

    # Reuse binsize and time_window from original definitions
    binsize_filtered = binsizes[dataset_name]
    time_window_filtered = filtered_dataset["non_stimuli_time"]

    print(f"\nProcessing Correlogram for Processed Dataset: {dataset_name}. Please wait...")
    correlogram_data_filtered = plot_correlogram_matrix(
        neurons_data=neurons_data_filtered,
        binsize=binsize_filtered,
        dataset_name=f"{dataset_name}_filtered",
        time_window=time_window_filtered,
        save_folder=save_folder_processed,
        store_data=True
    )

print("\nAll processed correlograms have been plotted and saved.")

#%%
"""
=================================================================================================================================================================================
1.2 Exclusion criterion: Remove individual spikes with ISI below absolute refractory period
=================================================================================================================================================================================
"""
"""

As we were quite lenient with our sorting in the previous criterion, we will now sort out the spikes that are too close to each other and must be due to be noise or 
incorrect spike sorting. We will use the absolute refractory period of 1/2500s to filter out these spikes.

Note: 
    We went with both, the correlogram and the ISI filtering because the correlogram gives us a better sense of firing patters as it is not just about consecutive spikes, 
    while the ISI focuses on the latter.

"""

# Dictionary mapping dataset names to filtered file names
filteredCC_files = {"ctrl_rat_1": "ctrl_rat_1_CCFiltered.pkl","ctrl_rat_2": "ctrl_rat_2_CCFiltered.pkl","exp_rat_2":  "exp_rat_2_CCFiltered.pkl","exp_rat_3":  "exp_rat_3_CCFiltered.pkl"}
filteredCC_datasets = {
    name: (datasets[name]["neurons"], datasets[name]["non_stimuli_time"])
    for name in filteredCC_files.keys()
}

# Toggle filtering on or off
apply_filtering = True

for dataset_name, (neurons_data, non_stimuli_time) in filteredCC_datasets.items():
    print(f"Creating TIHs for filtered dataset: {dataset_name}")
    
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

# Filter and save ISI-filtered datasets
save_filtered_isi_datasets(
    {name: (neurons, non_stimuli_time) for name, (neurons, non_stimuli_time) in filteredCC_datasets.items()},
    processed_dir,
    raw_dir,
    apply_filter=apply_filtering
)
print("\nAll TIHs have been plotted and ISI-filtered datasets have been saved.")

"""
Checking the debug prints showed us that based on our criterion: 
- Filtered dataset ctrl_rat_1: 20 out of 23 neurons problematic
- Filtered dataset ctrl_rat_2: 2 out of 2 neurons are problematic.
- Filtered dataset exp_rat_2: 13 out of 13 neurons are problematic.
- Filtered dataset exp_rat_3: 11 out of 19 neurons are problematic.

That would force us to remove a lot of neurons. 
So we decided to keep the neurons but remove biologically impossible spikes.
For this we had to make the arbitrary decision unfortunately, to always keep the first spike and remove the spikes that were too close to the first one.
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
2. Fano factor + CV = Variability -> for Poisson process CV = 1, Fano = 1
3. Survivor function
"""

# Define a dictionary mapping dataset names to filtered file names
final_filtered_files = {"ctrl_rat_1": "ctrl_rat_1_ISIfiltered.pkl","ctrl_rat_2": "ctrl_rat_2_ISIfiltered.pkl","exp_rat_2":  "exp_rat_2_ISIfiltered.pkl","exp_rat_3":  "exp_rat_3_ISIfiltered.pkl"}
final_filtered_datasets = {}
for name, filename in final_filtered_files.items():
    file_path = os.path.join(processed_dir, filename)
    data, neurons, non_stimuli_time = load_dataset(file_path) 
    final_filtered_datasets[name] = (neurons, non_stimuli_time)

# Firing rates
os.makedirs(save_folder, exist_ok=True)
analyze_firing_rates(final_filtered_datasets, final_filtered_files, processed_dir, save_folder)
print("Firing Rates have been plotted and saved.")

# Fano factor and CV
analyze_variability(final_filtered_datasets, processed_dir, final_filtered_files, save_folder)
print("Fano Factor and CV have been plotted and saved.")

# Survivor function to check for potential burst activity
for dataset_name, (neurons, non_stimuli_time) in tqdm(final_filtered_datasets.items(), desc="Computing the Survivor Functions"):
    
    data = load_dataset(os.path.join(processed_dir, final_filtered_files[dataset_name]))[0] 
    sacc_start = data.get("sacc drinking session start time", 0)
    cta_time = data.get("CTA injection time", 0)

    dataset_max_time = max((np.max(neuron[2]) for neuron in neurons if len(neuron[2]) > 0), default=0) # compute max spike time
    dataset_subfolder = dataset_name # set dataset as subfolder
    metrics_list = []  # list to store results for all neurons

    for idx, neuron in enumerate(neurons):
        neuron_label = f"{dataset_name}_neuron{idx+1}"
        metrics = plot_survivor(  # Store the returned metrics
            neuron,
            non_stimuli_time=non_stimuli_time,
            sacc_start=sacc_start,
            cta_time=cta_time,
            dataset_max_time=dataset_max_time,
            figure_title=f"Survivor Function for {neuron_label}",
            save_folder=os.path.join(base_dir,"reports", "figures"),
            subfolder=dataset_subfolder,  
            neuron_label=neuron_label
        )
        metrics_list.append(metrics)

    # Summary function using the collected metrics
    plot_survivor_dataset_summary(
        metrics_list, dataset_name, os.path.join(base_dir, "reports", "figures")
    )

print("\nAll Survivor Functions have been plotted and saved.")

# %%
"""
=================================================================================================================================================================================
3. Changes in Evoked Responses
=================================================================================================================================================================================
"""
"""

1. Rasterplots + smoothed PSTH (with moving average filter)
2. 2x2 bar plots PSTHs + dataset summaries, with baseline subtraction and moving average filter = Y-Axis shows the diff btw baseline and evoked response

"""

# 1. Rasterplots + smoothed PSTH
psthfigures_dir = os.path.join(base_dir, "reports", "figures", "PSTH_Raster")
os.makedirs(psthfigures_dir, exist_ok=True)

# store precomputed PSTH data
psth_data_map = {}
psth_group_data = {}

for dataset_name, (neurons, non_stimuli_time) in tqdm(final_filtered_datasets.items(), desc="Processing datasets", ncols=100):
    data = load_dataset(os.path.join(processed_dir, final_filtered_files[dataset_name]))[0]
    
    # Extract water and sugar events
    water_events = np.array(data.get("event_times", {}).get("water", []))
    sugar_events = np.array(data.get("event_times", {}).get("sugar", []))
    
    # print(f"{dataset_name}: water_events: {len(water_events)}, sugar_events: {len(sugar_events)}")
    cta_time = data.get("CTA injection time", None)
    # print(f"{dataset_name}: CTA time: {cta_time}")
    
    psth_results = psth_raster(
        dataset_name,
        neurons,
        water_events,
        sugar_events,
        cta_time,
        save_folder=psthfigures_dir
    )
    
    psth_data_map[dataset_name] = psth_results
    psth_group_data[dataset_name] = psth_results

# Group the datasets
control_group = [psth_group_data[name] for name in psth_group_data if "ctrl" in name]
experimental_group = [psth_group_data[name] for name in psth_group_data if "exp" in name]

# Combine data within each group
def combine_group_data(group_data, key):
    combined_data = []
    for data in group_data:
        combined_data.extend(data[key])
    return combined_data

control_water_pre = combine_group_data(control_group, "psth_water_pre")
control_water_post = combine_group_data(control_group, "psth_water_post")
control_sugar_pre = combine_group_data(control_group, "psth_sugar_pre")
control_sugar_post = combine_group_data(control_group, "psth_sugar_post")

experimental_water_pre = combine_group_data(experimental_group, "psth_water_pre")
experimental_water_post = combine_group_data(experimental_group, "psth_water_post")
experimental_sugar_pre = combine_group_data(experimental_group, "psth_sugar_pre")
experimental_sugar_post = combine_group_data(experimental_group, "psth_sugar_post")

# Perform statistical testing (paired t-test or Wilcoxon signed-rank test)
def test_significance(pre, post, label):
    if len(pre) == len(post) and len(pre) > 1:
        stat, p = ttest_rel(pre, post)  # Paired t-test
    else:
        stat, p = wilcoxon(pre, post) if len(pre) > 0 and len(post) > 0 else (np.nan, np.nan)  # Wilcoxon if possible
    print(f"{label}: Statistic={stat:.3f}, p-value={p:.5f}")

# Perform distribution test (Kolmogorov-Smirnov test)
def test_distribution(pre, post, label):
    if len(pre) > 0 and len(post) > 0:
        stat, p = ks_2samp(pre, post)  # Kolmogorov-Smirnov test
    else:
        stat, p = np.nan, np.nan
    print(f"{label} (KS Test): Statistic={stat:.3f}, p-value={p:.5f}")

# Run the statistical tests for each condition
print("Control Group Test Results:")
print("t-test / Wilcoxon Test Results:")
test_significance(control_water_pre, control_water_post, "Control Water Pre vs. Post")
test_significance(control_sugar_pre, control_sugar_post, "Control Sugar Pre vs. Post")

print("KS Test Results:")
test_distribution(control_water_pre, control_water_post, "Control Water Pre vs. Post")
test_distribution(control_sugar_pre, control_sugar_post, "Control Sugar Pre vs. Post")

print("\nExperimental Group Test Results:")
print("t-test / Wilcoxon Test Results:")
test_significance(experimental_water_pre, experimental_water_post, "Experimental Water Pre vs. Post")
test_significance(experimental_sugar_pre, experimental_sugar_post, "Experimental Sugar Pre vs. Post")

print("KS Test Results:")
test_distribution(experimental_water_pre, experimental_water_post, "Experimental Water Pre vs. Post")
test_distribution(experimental_sugar_pre, experimental_sugar_post, "Experimental Sugar Pre vs. Post")


print("\nAll Raster Plots and Smoothed PSTHs have been saved.")

# 2. 2x2 bar plots PSTHs + dataset summaries
raster_figures_dir = os.path.join(base_dir, "reports", "figures", "PSTH_TwoByTwo")
os.makedirs(raster_figures_dir, exist_ok=True)

for dataset_name, (neurons, non_stimuli_time) in final_filtered_datasets.items():
    print(f"Generating 2x2 PSTH plots with baseline subtraction for {dataset_name}...")

    data = load_dataset(os.path.join(processed_dir, final_filtered_files[dataset_name]))[0]
    cta_time = data.get("CTA injection time", None)

    water_events = np.array(data.get("event_times", {}).get("water", []))
    sugar_events = np.array(data.get("event_times", {}).get("sugar", []))

    plot_neuron_rasters_2x2(
        group_name=dataset_name,
        neurons=neurons,
        water_events=water_events,
        sugar_events=sugar_events,
        cta_time=cta_time,
        save_folder=os.path.join(raster_figures_dir, dataset_name),
        summary_folder=os.path.join(raster_figures_dir, dataset_name, "Summary"),
        window=(-1, 2),
        bin_width=0.05 #ms
    )

print("\nAll 2x2 PSTH plots have been saved successfully!")
print("\nAnalysis completed! Thanks a bunch for your patience :)")
# %%
