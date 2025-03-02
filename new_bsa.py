"""
===========================================================
BSA Final Assignment - Denise Jaeschke & Sofia Karageorgiou
===========================================================
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

#%%
"""
===========================================================
Step 1: Inspect data
===========================================================
First we will check whether the data matches the documentation we were provided with.
Thus, we looked at the data types, the keys of the dictionary, and the content of each key.
TODO (Sofia): Maybe comment this whole part out before submission? 

"""
#%% Inspect first file
file_path = os.path.join("data", "raw", "exp rat 2.pkl")

with open(file_path, "rb") as f: # inspect first file
    data = pickle.load(f) 

print(type(data))  # --> it is a dictionary
print(data.keys())  # check the keys of this dictionary
# print(data) # check out what the data looks like

#%% Look at content of each key
# 1. Event times
print("Event Times:", data["event_times"].keys())
print("Event Times:", data["event_times"])

# 2. Drinking session start time and CTA injection time
print("Saccharin drinking start time:", data["sacc drinking session start time"])
print("CTA injection time:", data["CTA injection time"])

# 3. How many neurons were recorded
print("Number of neurons recorded:", len(data["neurons"]))

# 4. Display example neuron
print("Example neuron data:", data["neurons"][0])  # Checking the first neuron

"""
This matches our expectations, namely:
- the times when saccharin and water were given.
- the start of the saccharin drinking session and the time of LiCl/saline injection.
- count the number of neurons recorded.
- print one neuron's data to understand its format.

Now, we'll extract the spike data
"""

#%%
"""
===========================================================
Step 2: Data exclusion
===========================================================
Next, our main goal will be to check whether we are actually 
dealing with separate neuronal spiking data or whether some 
spiking activity that is supposed to come from a single 
neuron, is actually a result of multiple neurons being
treated as one.

For that, we will plot the correlogram of each data set and 
for now focus on the auto-correlograms.

""" 
#%% Extract spiking data from each pkl file and save it as its own variable

base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current script's directory
pkl_dir = os.path.join(base_dir, 'data', 'raw')  # Join with the 'data/raw' relative path

# Define file paths
file_1 = os.path.join(pkl_dir, "ctrl rat 1.pkl")
file_2 = os.path.join(pkl_dir, "ctrl rat 2.pkl")
file_3 = os.path.join(pkl_dir, "exp rat 2.pkl")
file_4 = os.path.join(pkl_dir, "exp rat 3.pkl")

# Load data manually
with open(file_1, "rb") as f:
    ctrl_rat_1_neurons_data = pickle.load(f)["neurons"]

with open(file_2, "rb") as f:
    ctrl_rat_2_neurons_data = pickle.load(f)["neurons"]

with open(file_3, "rb") as f:
    exp_rat_2_neurons_data = pickle.load(f)["neurons"]

with open(file_4, "rb") as f:
    exp_rat_3_neurons_data = pickle.load(f)["neurons"]

# Check if data is loaded properly
print(ctrl_rat_1_neurons_data[:5])  # Print first 5 neurons of ctr_rat_1

#%% Before we work with correlograms, we want to check which bin size is the most optimal one per dataset
"""
With some research, we found out about the Cn(Delta) function which quantifies how well a particular bin size captures spike train information.
Our goal is to find a delta that minimises the Cn. 

Why care about bin size?
- if bins are too small, the histograms will be too noisy with too many empty bins
- if bins are too large, we might miss important information
The optimal bin size achieves the best balance between these two extremes.

We computed Cn(Delta) based on the formula from this video: https://youtu.be/VJGtyeR87R4?si=wsTlEeRorVug9kJC

TODO (Sofia): RUN IT
The calculations showed that ___ is the optimal bin size for the dataset ___.
(For details look at functions/calculateidealbinsize.py)

"""

#%% Correlogram + plotting correlogram functions
def correlogram(t1, t2=None, binsize=.001, limit=.02, auto=False,
                density=False):
    """Return crosscorrelogram of two spike trains.
    Essentially, this algorithm subtracts each spike time in t1
    from all of t2 and bins the results with np.histogram, though
    several tweaks were made for efficiency.
    Originally authored by Chris Rodger, copied from OpenElectrophy, licenced
    with CeCill-B. Examples and testing written by exana team.

    Parameters
    ---------
    t1 : np.array
        First spiketrain, raw spike times in seconds.
    t2 : np.array
        Second spiketrain, raw spike times in seconds.
    binsize : float
        Width of each bar in histogram in seconds.
    limit : float
        Positive and negative extent of histogram, in seconds.
    auto : bool
        If True, then returns autocorrelogram of ⁠ t1 ⁠ and in
        this case ⁠ t2 ⁠ can be None. Default is False.
    density : bool
        If True, then returns the probability density function.
    See also
    --------
    :func:⁠ numpy.histogram ⁠ : The histogram function in use.

    Returns
    -------
    (count, bins) : tuple
        A tuple containing the bin right edges and the
        count/density of spikes in each bin.
    Note
    ----
    ⁠ bins ⁠ are relative to ⁠ t1 ⁠. That is, if ⁠ t1 ⁠ leads ⁠ t2 ⁠, then
    ⁠ count ⁠ will peak in a positive time bin.

    Examples
    --------
    >>> t1 = np.arange(0, .5, .1)
    >>> t2 = np.arange(0.1, .6, .1)
    >>> limit = 1
    >>> binsize = .1
    >>> counts, bins = correlogram(t1=t1, t2=t2, binsize=binsize,
    ...                            limit=limit, auto=False)
    >>> counts
    array([0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0, 0])

    The interpretation of this result is that there are 5 occurences where
    in the bin 0 to 0.1, i.e.

    # TODO fix
    # >>> idx = np.argmax(counts)
    # >>> '%.1f, %.1f' % (abs(bins[idx - 1]), bins[idx])
    # '0.0, 0.1'

    The correlogram algorithm is identical to, but computationally faster than
    the histogram of differences of each timepoint, i.e.

    # TODO Fix the doctest
    # >>> diff = [t2 - t for t in t1]
    # >>> counts2, bins = np.histogram(diff, bins=bins)
    # >>> np.array_equal(counts2, counts)
    # True
    """
    if auto: t2 = t1
    # For auto-CCGs, make sure we use the same exact values
    # Otherwise numerical issues may arise when we compensate for zeros later
    if not int(limit * 1e10) % int(binsize * 1e10) == 0:
        raise ValueError(
            'Time limit {} must be a '.format(limit) +
            'multiple of binsize {}'.format(binsize) +
            ' remainder = {}'.format(limit % binsize))
    # For efficiency, ⁠ t1 ⁠ should be no longer than ⁠ t2 ⁠
    swap_args = False
    if len(t1) > len(t2):
        swap_args = True
        t1, t2 = t2, t1

    # Sort both arguments (this takes negligible time)
    t1 = np.sort(t1)
    t2 = np.sort(t2)

    # Determine the bin edges for the histogram
    # Later we will rely on the symmetry of ⁠ bins ⁠ for undoing ⁠ swap_args ⁠
    limit = float(limit)

    # The numpy.arange method overshoots slightly the edges i.e. binsize + epsilon
    # which leads to inclusion of spikes falling on edges.
    bins = np.arange(-limit, limit + binsize, binsize)

    # Determine the indexes into ⁠ t2 ⁠ that are relevant for each spike in ⁠ t1 ⁠
    ii2 = np.searchsorted(t2, t1 - limit)
    jj2 = np.searchsorted(t2, t1 + limit)

    # Concatenate the recentered spike times into a big array
    # We have excluded spikes outside of the histogram range to limit
    # memory use here.
    big = np.concatenate([t2[i:j] - t for t, i, j in zip(t1, ii2, jj2)])

    # Actually do the histogram. Note that calls to np.histogram are
    # expensive because it does not assume sorted data.
    count, bins = np.histogram(big, bins=bins, density=density)

    if auto:
        # Compensate for the peak at time zero that results in autocorrelations
        # by subtracting the total number of spikes from that bin. Note
        # possible numerical issue here because 0.0 may fall at a bin edge.
        c_temp, bins_temp = np.histogram([0.], bins=bins)
        bin_containing_zero = np.nonzero(c_temp)[0][0]
        count[bin_containing_zero] = 0#-= len(t1)

    # Finally compensate for the swapping of t1 and t2
    if swap_args:
        # Here we rely on being able to simply reverse ⁠ counts ⁠. This is only
        # possible because of the way ⁠ bins ⁠ was defined (bins = -bins[::-1])
        count = count[::-1]

    return count, bins[1:]
def plot_correlogram_matrix(neurons_data, binsize, dataset_name, limit=0.02):
    num_neurons = len(neurons_data)
    fig, axes = plt.subplots(num_neurons, num_neurons, figsize=(num_neurons * 3, num_neurons * 3))
    
    for i, neuron_i in enumerate(neurons_data):
        t1 = neuron_i[:3][2]  # Extract spike times

        for j, neuron_j in enumerate(neurons_data):
            t2 = neuron_j[:3][2]  # Extract spike times

            # Compute correlogram
            counts, bins = correlogram(t1, t2=t2, binsize=binsize, limit=limit, auto=(i == j), density=False)

            # Ensure counts and bins align correctly
            if len(counts) > len(bins) - 1:
                counts = counts[:-1]

            bin_centers = (bins[:-1] + bins[1:]) / 2

            # Plot in the matrix
            ax = axes[i, j] if num_neurons > 1 else axes
            ax.bar(bin_centers, counts, width=np.diff(bins), align='center', color='b', alpha=0.7, edgecolor='k')
            ax.set_xlim(-limit, limit)
            ax.set_xticks([])
            ax.set_yticks([])

            # Labels for the first row and first column
            if i == 0:
                ax.set_title(f"Neuron {j+1}")
            if j == 0:
                ax.set_ylabel(f"Neuron {i+1}")

    plt.suptitle(f"Cross-correlogram with (Bin Size = {binsize:.4f}s)", fontsize=16)  # Show bin size in title
    plt.tight_layout()
    
    # Define relative save path
    save_dir = os.path.join(os.getcwd(), "reports", "figures")  # Relative path
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
    save_path = os.path.join(save_dir, f"{dataset_name}_correlogram.png")

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Free memory

    print(f"Correlogram saved: {save_path}")  # Confirm save location
#%% Define optimal bin sizes for each dataset
optimal_bin_sizes = {
    "ctrl_rat_1": 0.001,  # Replace with the actual optimal bin size
    "ctrl_rat_2": 0.001,  # Replace with the actual optimal bin size
    "exp_rat_2": 0.001,   # Replace with the actual optimal bin size
    "exp_rat_3": 0.001    # Replace with the actual optimal bin size
}
#%% Plot all 4 correlograms using their respective bin sizes
plot_correlogram_matrix(ctrl_rat_1_neurons_data, binsize=optimal_bin_sizes["ctrl_rat_1"], dataset_name="ctrl_rat_1")
plot_correlogram_matrix(ctrl_rat_2_neurons_data, binsize=optimal_bin_sizes["ctrl_rat_2"], dataset_name="ctrl_rat_2")
plot_correlogram_matrix(exp_rat_2_neurons_data, binsize=optimal_bin_sizes["exp_rat_2"], dataset_name="exp_rat_2")
plot_correlogram_matrix(exp_rat_3_neurons_data, binsize=optimal_bin_sizes["exp_rat_3"], dataset_name="exp_rat_3")

"""
Interpretation of the correlograms:
Features to note
- how flat is it?
- does it show a pause in the middle? we expect one for auto-correlograms but not for cross-correlograms

What if we do find a pause in cross-correlograms?
- inspect subclusters, exploit fact that they are not symmetric and find out when they fire
- maybe also check adaptation over time as that might explain that
"""



# %%
