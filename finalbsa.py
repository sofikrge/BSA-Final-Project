
"""
=================================================================================================================================================================================
BSA Final Assignment - Denise Jaeschke & Sofia Karageorgiou
=================================================================================================================================================================================
"""
"""
TODO
    - how did we define the timestamps?
    - when do we want to look at which time window? 
    - compare our calculations with what was done in tirgulim to be on the safe side
    - do we want firing rates over the entire time window or just the mean?
    - psth & survivor hazard are saving to the wrong folder
    - our correlogram calculations are different than those in class

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
"""
"""
Class notes on metrics:
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
        => we decided not to look into bc we do not a lot of neurons to begin with, might get very noisy at the end because there are only few neurons that spike with such long ISI

Initial plan for exclusion:
    - only look at unstimulated phase
    - 0.5ms bin size as that is the absolute refractory period -> one spike is happening so look at 2 bins
    - no immediate peak next to absolute refractory period
    - 2ms relative refractory period -> look at 4 bins, there should be close to none as we are looking at the unstimulated 
    phase and a very strong stimulus would be needed for a new spike
    - chose a conservative criterion because our biggest enemy too high is data loss
"""

#%% Imports, loading data, and setting up directories
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

# Compute the correlogram matrix.
for dataset_name, dataset in datasets.items():
    neurons_data = dataset["neurons"]
    time_window = dataset["non_stimuli_time"] # because we we want to see a clearer relative refrac period as well
    print(f"\nProcessing Correlogram for Dataset: {dataset_name}. Please be patient, this might take a while.")
    
    correlogram_data = plot_correlogram_matrix(neurons_data=neurons_data,binsize=binsizes[dataset_name],dataset_name=dataset_name,time_window=time_window,save_folder=os.path.join(save_folder, "Correlograms"),store_data=True)
    
    # Retrieve problematic indices from the returned dictionary
    # problematic_neuron_indices = correlogram_data.get("problematic_neuron_indices", set())
    # print(f"Problematic indices for {dataset_name}: {problematic_neuron_indices}")
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
print("Manual fusion has been applied and filtered datasets have been saved.")

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
apply_filtering = True

# Plot ISI histograms for all neurons
for dataset_name, (neurons_data, non_stimuli_time) in filteredCC_datasets.items():
    print(f"\Creating TIHs for filtered dataset: {dataset_name}")
    
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
print("TIHs have been plotted and ISI-filtered datasets have been saved.")

"""
Checking the debug prints showed us that based on our criterion: 
- Filtered dataset ctrl_rat_1: 20 out of 23 neurons problematic
- Filtered dataset ctrl_rat_2: 2 out of 2 neurons are problematic.
- Filtered dataset exp_rat_2: 13 out of 13 neurons are problematic.
- Filtered dataset exp_rat_3: 11 out of 19 neurons are problematic.

That would force us to remove a lot of neurons. 
So we decided to keep the neurons but remove biologically impossible spikes.
For this we had to make some arbitrary decision unfortunately. 
We decided to keep the first spike and remove all spikes that are too close to the first spike.
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
3. Survivor function
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
print("Firing Rates have been plotted and saved.")

# Fano factor and CV
analyze_variability(final_filtered_datasets, processed_dir, final_filtered_files, save_folder)
print("Fano Factor and CV have been plotted and saved.")

#% Survivor function to check for potential burst activity
for dataset_name, (neurons, non_stimuli_time) in tqdm(final_filtered_datasets.items(), desc="Computing the Survivor Functions"):
    # Load the associated data to extract sacc_start and cta_time.
    data = load_dataset(os.path.join(processed_dir, final_filtered_files[dataset_name]))[0]
    sacc_start = data.get("sacc drinking session start time", 0)
    cta_time = data.get("CTA injection time", 0)
    
    # Compute the maximum spike time across all neurons for the Post-CTA window.
    dataset_max_time = max((np.max(neuron[2]) for neuron in neurons if len(neuron[2]) > 0), default=0)
    
    # Use the dataset name as the subfolder.
    dataset_subfolder = dataset_name
    
    metrics_list = []  # List to store results for all neurons

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

        metrics_list.append(metrics)  # Collect the output for summary

    # Now call the summary function using the collected metrics:
    plot_survivor_dataset_summary(
        metrics_list, dataset_name, os.path.join(base_dir, "reports", "figures")
    )

print("Survivor functions have been plotted and saved.")

# %%
"""
=================================================================================================================================================================================
3. Changes in Evoked Responses
=================================================================================================================================================================================
"""
"""
1. Rasterplots + smoothed PSTH
2. 2x2 bar plots PSTHs + dataset summaries, with baseline subtraction = Y-Axis shows the diff btw baseline and evoked response
3. Cross-correlograms Pre & Post CTA
"""

# 1. Rasterplots + smoothed PSTH
psthfigures_dir = os.path.join(base_dir, "reports", "figures", "psth")
os.makedirs(psthfigures_dir, exist_ok=True)

# store precomputed PSTH data
psth_data_map = {}

for dataset_name, (neurons, non_stimuli_time) in tqdm(final_filtered_datasets.items(), desc="Processing datasets", ncols=100):
    data = load_dataset(os.path.join(processed_dir, final_filtered_files[dataset_name]))[0]
    
    # Extract water and sugar events
    water_events = np.array(data.get("event_times", {}).get("water", []))
    sugar_events = np.array(data.get("event_times", {}).get("sugar", []))
    
    # print(f"{dataset_name}: water_events: {len(water_events)}, sugar_events: {len(sugar_events)}")
    cta_time = data.get("CTA injection time", None)
    # print(f"{dataset_name}: CTA time: {cta_time}")
    
    psth_data = psth_raster(
        dataset_name,
        neurons,
        water_events,
        sugar_events,
        cta_time,
        save_folder=psthfigures_dir
    )
    
    psth_data_map[dataset_name] = psth_data

print("Raster plots and smoothed PSTHs have been saved.")

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
        window=(-1, 2),
        bin_width=0.05 #ms
    )

print("All 2x2 PSTH plots have been saved successfully!")
#%% 3. Cross-correlograms Pre & Post CTA
#%% YAY, we're done!
print("Analysis completed! Thanks a bunch for your patience :)")