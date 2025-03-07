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
    "ctrl_rat_1": 0.0005,
    "ctrl_rat_2": 0.0005,
    "exp_rat_2": 0.0005,
    "exp_rat_3": 0.0005
}

save_folder = os.path.join(os.getcwd(), "reports", "figures")

# Plot all 4 correlograms with our correlogram helper function using their respective bin sizes
#plot_correlogram_matrix(ctrl_rat_1_neurons_data, binsize=binsizes["ctrl_rat_1"], dataset_name="ctrl_rat_1", time_window=non_stimuli_time_1, save_folder=save_folder)
# plot_correlogram_matrix(ctrl_rat_2_neurons_data, binsize=binsizes["ctrl_rat_2"], dataset_name="ctrl_rat_2", time_window=non_stimuli_time_2, save_folder=save_folder)
#plot_correlogram_matrix(exp_rat_2_neurons_data, binsize=binsizes["exp_rat_2"], dataset_name="exp_rat_2", time_window=non_stimuli_time_3, save_folder=save_folder)
#plot_correlogram_matrix(exp_rat_3_neurons_data, binsize=binsizes["exp_rat_3"], dataset_name="exp_rat_3", time_window=non_stimuli_time_4, save_folder=save_folder)

# Plot the correlogram matrix for ctrl_rat_2_neurons_data and store the computed histogram data.
correlogram_data = plot_correlogram_matrix(
    neurons_data=ctrl_rat_2_neurons_data,
    binsize=binsizes["ctrl_rat_2"],
    dataset_name="ctrl_rat_2",
    time_window=non_stimuli_time_2,
    save_folder=save_folder,
    store_data=True
)

# Now, you can use 'correlogram_data' with your check_correlogram_conditions helper.
for key, data in correlogram_data.items():
    result = check_correlogram_conditions(data["counts"], neuron_id=key, corr_type="auto" if "vs" not in key else "cross")
    print(f"{key} analysis:", result)

# %%
