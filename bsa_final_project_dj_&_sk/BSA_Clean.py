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

#%% Set working directory to current folder 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Current directory:", os.getcwd())  

#%% Load one file to inspect the data
import pickle
with open("exp rat 2.pkl", "rb") as f:
	data = pickle.load(f) 


print(type(data))  # --> it is a dictionary
print(data.keys())  # check the keys of this dictionary
#print(data) # check out what the data looks like

"""
Let's dig deeper and inspect the data further by looking behind each key
"""

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
Therewith we get:
- the times when saccharin and water were given.
- the start of the saccharin drinking session and the time of LiCl/saline injection.
- count the number of neurons recorded.
- print one neuron's data to understand its format.
"""

# %% Now that we have understood the neuron's data format we can rearrange it for better reading for us:

# Extract neuron spike data
neurons_data = data['neurons']

# Create a DataFrame for neurons
neurons_list = []
for neuron in neurons_data:
    electrode, cluster, spike_times = neuron[:3]  # Extract first 3 elements
    neurons_list.append({
        "Electrode": electrode,
        "Cluster": cluster,
        "Spike_Count": len(spike_times),  # Number of spikes
        "First_Spike": spike_times[0],  # First spike time
        "Last_Spike": spike_times[-1]  # Last spike time
    })

# Convert to a Pandas DataFrame
df_neurons = pd.DataFrame(neurons_list)
print(df_neurons.head())
