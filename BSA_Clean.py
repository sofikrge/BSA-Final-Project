"""
===========================================================
BSA Final Assignment - Denise Jaeschke & Sofia Karageorgiou
===========================================================
"""

#%% Imports
import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys
from functions.plot_correlogram_matrix import plot_correlogram_matrix
from functions.correlogram import correlogram

#%% 
"""
===========================================================
Step 1: Inspect data
===========================================================
"""
#%% Inspect data to check if it matches Doc
# Define the correct path to your file
file_path = os.path.join("data", "raw", "exp rat 2.pkl")

# Open the file with the correct path
with open(file_path, "rb") as f:
    data = pickle.load(f) 

print(type(data))  # --> it is a dictionary
print(data.keys())  # check the keys of this dictionary
#print(data) # check out what the data looks like

# Look at content of each key
# Check the event times (stimuli times)
print("Event Times:", data["event_times"].keys())
print("Event Times:", data["event_times"])

# Check the drinking session start time and CTA injection time
print("Saccharin drinking start time:", data["sacc drinking session start time"])
print("CTA injection time:", data["CTA injection time"])

# Check how many neurons were recorded
print("Number of neurons recorded:", len(data["neurons"]))

# Display an example neuron
print("Example neuron data:", data["neurons"][0])  # Checking the first neuron

"""
This matches our expectations, namely:
- the times when saccharin and water were given.
- the start of the saccharin drinking session and the time of LiCl/saline injection.
- count the number of neurons recorded.
- print one neuron’s data to understand its format.

Now, we'll extract the spike data

"""

#%% Extract spike data (only the third element of each neuron list)
neurons_data = [np.sort(neuron[2]) for neuron in data["neurons"]]
#%% 
"""
===========================================================
Step 2: Data exclusion
===========================================================
""" 
#%% Inspect correlograms to identify bad sorting of neurons
# Run correlogram on all unique neuron pairs
# Store results in a dictionary
correlogram_results = {}
num_neurons = len(neurons_data)
print(f"Total neurons: {num_neurons}")

# Run correlogram on all neuron pairs
for i in range(num_neurons):
    for j in range(i, num_neurons):  # Avoid redundant calculations
        t1 = neurons_data[i]
        t2 = neurons_data[j]

        auto = i == j  # Set auto=True if computing for the same neuron
        counts, bins = correlogram(t1, t2, binsize=0.001, limit=0.02, auto=auto)

        # Store the results
        correlogram_results[(i, j)] = {"counts": counts, "bins": bins}

        # Print progress
        print(f"Computed correlogram for neuron {i} and neuron {j}")

# %%Plot the correlogram results
plot_correlogram_matrix(correlogram_results, neurons_data)


# %%
