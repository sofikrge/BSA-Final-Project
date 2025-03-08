import numpy as np
from functions.correlogram import correlogram

def check_correlogram_conditions(t1=None, t2=None, binsize=0.0005, limit=0.02, auto=False,
                                 density=False, neuron_id="unknown", corr_type="auto", hist_data=None):
    """
    Checks conditions on the correlogram histogram to flag problematic neurons.

    If 'hist_data' is provided (a dictionary with keys "counts" and "bins"),
    it uses that data directly; otherwise, it computes the correlogram from the provided spike times.

    For an autocorrelogram (corr_type="auto"), a neuron is flagged as problematic if:
      - (A) At least one of the two center bins (the middle two bins for an even number of bins)
          is non-empty, OR
      - (B) The bin immediately to the left of the left center bin or the bin immediately to the right
          of the right center bin contains the global maximum.
          
    For a cross-correlogram (corr_type="cross"), a neuron pair is flagged as problematic if:
      - Both center bins are empty.

    Parameters:
      t1 : np.array or None
          Spike times for the first neuron (used if hist_data is not provided).
      t2 : np.array or None
          Spike times for the second neuron (used if hist_data is not provided).
      binsize : float
          Width of each histogram bin in seconds.
      limit : float
          Extent (in seconds) on each side of zero lag.
      auto : bool
          If True, compute an autocorrelogram (t2 is set to t1).
      density : bool
          If True, compute a density histogram.
      neuron_id : str or int
          Identifier for the neuron (or neuron pair).
      corr_type : str
          "auto" for autocorrelogram or "cross" for cross-correlogram.
      hist_data : dict or None
          Precomputed histogram data with keys "counts" and "bins".
    
    Returns:
      results : dict
          A dictionary with:
            "problematic": True/False,
            "reason": A string explanation of which condition was met.
    """
    # Use precomputed histogram data if available.
    if hist_data is not None:
        counts = hist_data.get("counts")
        bins = hist_data.get("bins")
    else:
        counts, bins = correlogram(t1, t2=t2, binsize=binsize, limit=limit, auto=auto, density=density)
        if len(counts) > len(bins) - 1:
            counts = counts[:-1]
    
    # Compute bin centers for information (not used for center calculation below)
    n_bins = len(counts)
    # Define center bins: if even, use the two middle bins; if odd, use the same middle bin for both.
    center_left = n_bins // 2 - 1
    center_right = n_bins // 2

    # For autocorrelograms: condition A is if either center bin is non-empty.
    # Condition B is if the bin immediately to the left of center_left or right of center_right holds the global maximum.
    left_center_val = counts[center_left]
    right_center_val = counts[center_right]
    
    results = {"problematic": False, "reason": ""}
    
    if corr_type == "auto":
        condition_A = (left_center_val > 0 or right_center_val > 0)
        # Check neighbors if they exist.
        global_peak_index = int(np.argmax(counts))
        condition_B = (global_peak_index == center_left - 1 or global_peak_index == center_right + 1)
        
        if condition_A or condition_B:
            results["problematic"] = True
            reasons = []
            if condition_A:
                reasons.append(f"center bins non-empty (left: {left_center_val}, right: {right_center_val})")
            if condition_B:
                reasons.append(f"global peak in neighbor bin (index: {global_peak_index})")
            results["reason"] = "; ".join(reasons)
            
    elif corr_type == "cross":
        # For cross-correlograms, problematic if both center bins are empty.
        if left_center_val == 0 and right_center_val == 0:
            results["problematic"] = True
            results["reason"] = "both center bins are empty"
    
    return results
