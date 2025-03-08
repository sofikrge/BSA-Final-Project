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
from functions.check_correlogram_conditions import check_correlogram_conditions

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
Preliminary steps: Understanding our dataset + Extracting the data we need
=================================================================================================================================================================================
"""
# Inspect the data using the helper function
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

#%% Extract data + determine save folder
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
TODO
- Do not plot the mirror image, only the bottom left triangle
- Write code that tells us which neurons to exclude to not just base it on our visual inspection

"""
# Bin sizes dictionary
binsizes = { # We chose 0.0005s for all datasets because of our exclusion criteria
    "ctrl_rat_1": 0.00025,
    "ctrl_rat_2": 0.00025,
    "exp_rat_2": 0.00025,
    "exp_rat_3": 0.00025
}

# Example dictionary of datasets:
datasets = {
    "ctrl_rat_1": (ctrl_rat_1_neurons_data, non_stimuli_time_1),
    "ctrl_rat_2": (ctrl_rat_2_neurons_data, non_stimuli_time_2),
    "exp_rat_2":  (exp_rat_2_neurons_data, non_stimuli_time_3),
    "exp_rat_3":  (exp_rat_3_neurons_data, non_stimuli_time_4)
}

# Loop over each dataset
# for dataset_name, (neurons_data, time_window) in datasets.items():
#     # Plot and store correlogram data for this dataset
#     correlogram_data = plot_correlogram_matrix(
#         neurons_data=neurons_data,
#         binsize=binsizes[dataset_name],
#         dataset_name=dataset_name,
#         time_window=time_window,
#         save_folder=save_folder,
#         store_data=True
#     )
    
#     # Loop over each stored correlogram and check conditions.
#     for key, data in correlogram_data.items():
#         # Determine correlogram type: if "vs" is not in the key, assume autocorrelogram.
#         corr_type = "auto" if "vs" not in key else "cross"
#         result = check_correlogram_conditions(neuron_id=key, corr_type=corr_type, hist_data=data)
#         if result["problematic"]:
#             print(f"{dataset_name} - {key} is problematic: {result['reason']}")

correlogram_data = plot_correlogram_matrix(
    neurons_data=ctrl_rat_2_neurons_data,
    binsize=binsizes["ctrl_rat_2"],
    dataset_name="ctrl_rat_2",
    time_window=non_stimuli_time_2,
    save_folder=save_folder,
    store_data=True
)

# Loop over each stored correlogram and print only those flagged as problematic.
for key, data in correlogram_data.items():
    # Determine correlogram type: if "vs" is not in the key, assume autocorrelogram.
    corr_type = "auto" if "vs" not in key else "cross"
    result = check_correlogram_conditions(neuron_id=key, corr_type=corr_type, hist_data=data)
    if result["problematic"]:
        print(f"{key} is problematic: {result['reason']}")
# %% Exclude neurons based on the correlogram

