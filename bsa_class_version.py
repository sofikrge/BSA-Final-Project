#%% Descriptive metrics
# To simplify this section, and make it more modular, we created a SpikeTrain class 
# that contains all the neuron-level metrics.

import numpy as np

class SpikeTrain:
    def __init__(self, spikes, duration=None):
        """
        Initialize a SpikeTrain object.
        
        Parameters:
          spikes : array-like
              Array of spike times in seconds.
          duration : float, optional
              Total duration of the recording in seconds. If not provided,
              it is set to the last spike time.
        """
        self.spikes = np.sort(np.array(spikes))
        self.duration = duration if duration is not None else self.spikes[-1] if len(self.spikes) > 0 else 0

    def interspike_intervals(self):
        """
        Compute inter-spike intervals (ISIs).
        
        Returns:
          np.array of ISIs in seconds.
        """
        if len(self.spikes) < 2:
            return np.array([])
        return np.diff(self.spikes)
    
    def coefficient_variation(self):
        """
        Compute the Coefficient of Variation (CV) of the ISIs.
        
        Returns:
          float : CV, defined as the standard deviation divided by the mean of ISIs.
        """
        isis = self.interspike_intervals()
        if len(isis) == 0 or np.mean(isis) == 0:
            return np.nan
        return np.std(isis) / np.mean(isis)
    
    def spike_counts(self, interval_ms):
        """
        Compute the spike counts in bins of a specified interval (in ms).
        
        Parameters:
          interval_ms : float
              Bin width in milliseconds.
        
        Returns:
          np.array: Array of spike counts per bin.
        """
        interval_sec = interval_ms / 1000.
        # Create bin edges from 0 to duration (inclusive)
        bins = np.arange(0, self.duration + interval_sec, interval_sec)
        counts, _ = np.histogram(self.spikes, bins=bins)
        return counts
    
    def fano_factor(self, interval_ms):
        """
        Compute the Fano factor (variance/mean of spike counts) using the given bin width.
        
        Parameters:
          interval_ms : float
              Bin width in milliseconds.
        
        Returns:
          float: Fano factor.
        """
        counts = self.spike_counts(interval_ms)
        mean_count = np.mean(counts)
        if mean_count == 0:
            return np.nan
        return np.var(counts) / mean_count
    
    def psth(self, window=(-1, 2), bin_width=0.05):
        """
        Compute the Peri-Stimulus Time Histogram (PSTH) for this spike train.
        This method assumes that you have aligned the spike times relative to a stimulus.
        
        Parameters:
          window : tuple
              (start, end) time in seconds relative to the stimulus.
          bin_width : float
              Bin width in seconds.
        
        Returns:
          tuple: (bin_centers, psth) where psth is the spike rate (spikes/s).
        """
        bins = np.arange(window[0], window[1] + bin_width, bin_width)
        # Filter spikes to the specified window
        rel_spikes = self.spikes[(self.spikes >= window[0]) & (self.spikes <= window[1])]
        counts, edges = np.histogram(rel_spikes, bins=bins)
        # Normalize by bin width (to get a rate) and number of trials if applicable (here, assume one trial)
        psth_rate = counts / bin_width
        bin_centers = (edges[:-1] + edges[1:]) / 2
        return bin_centers, psth_rate
