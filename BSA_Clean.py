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

#%% Extract spike data
neurons_data = data['neurons']

#%% 
"""
===========================================================
Step 2: Data exclusion
===========================================================
""" 
