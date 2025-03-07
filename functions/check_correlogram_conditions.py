import numpy as np
from functions.correlogram import correlogram

def check_correlogram_conditions(t1, t2=None, binsize=0.0005, limit=0.02, auto=False,
                         density=False, neuron_id="unknown", corr_type="auto"):
    """
    Computes a correlogram using your existing 'correlogram' function,
    then checks conditions on the central adjacent bins.
    
    For autocorrelograms (corr_type="auto"):
      - It checks if the bins immediately left and right of center are non-empty.
      - It also checks if one of these adjacent bins holds the global peak.
    
    For cross-correlograms (corr_type="cross"):
      - It only checks that the bins immediately left and right of center are empty.
    
    Parameters:
      t1 : np.array
          Spike times for the first neuron.
      t2 : np.array or None
          Spike times for the second neuron. If None and auto is True, t1 is used.
      binsize : float
          Width of each histogram bin in seconds.
      limit : float
          Extent (in seconds) on each side of zero lag.
      auto : bool
          If True, compute an autocorrelogram (t2 is set to t1).
      density : bool
          If True, the histogram is normalized to form a density.
      neuron_id : str or int
          Identifier (name or number) for the neuron (or neuron pair).
      corr_type : str
          "auto" for autocorrelogram or "cross" for cross-correlogram.
    
    Returns:
      results : dict
          Dictionary with keys describing the check results and the computed histogram.
    """
    # Compute the correlogram using your existing function.
    counts, bins = correlogram(t1, t2=t2, binsize=binsize, limit=limit, auto=auto, density=density)
    
    # Assuming the returned bins array is for the right bin edges, the number of bins is len(counts).
    center = len(counts) // 2  # index of the center bin
    left_count = counts[center - 1]
    right_count = counts[center + 1] if center + 1 < len(counts) else 0

    results = {}
    
    if corr_type == "auto":
        # Check if the adjacent bins are non-empty.
        if left_count > 0 and right_count > 0:
            results["central_bins_non_empty"] = True
            print(f"Autocorrelogram: Neuron {neuron_id} has non-empty adjacent bins "
                  f"(left: {left_count}, right: {right_count}).")
        else:
            results["central_bins_non_empty"] = False
        
        # Check if one of the adjacent bins holds the global peak.
        global_peak_index = int(np.argmax(counts))
        if global_peak_index in [center - 1, center + 1]:
            results["global_peak_adjacent"] = True
            print(f"Autocorrelogram: Neuron {neuron_id} has its global peak in an adjacent bin "
                  f"(global peak index: {global_peak_index}).")
        else:
            results["global_peak_adjacent"] = False
            
    elif corr_type == "cross":
        # For cross-correlograms, only check that the adjacent bins are empty.
        if left_count == 0 and right_count == 0:
            results["central_bins_empty"] = True
            print(f"Cross-correlogram: Neuron pair {neuron_id} has empty adjacent bins "
                  f"(left: {left_count}, right: {right_count}).")
        else:
            results["central_bins_empty"] = False

    # Return the results along with the computed histogram for further use if needed.
    results["counts"] = counts
    results["bins"] = bins
    return results
