import numpy as np

def correlogram(t1, t2=None, binsize=.0004, limit=.02, auto=False,
                density=False, normalize=True):
    """Return crosscorrelogram of two spike trains.
    Essentially, this algorithm subtracts each spike time in t1
    from all of t2 and bins the results with np.histogram, though
    several tweaks were made for efficiency.
    Originally authored by Chris Rodger, copied from OpenElectrophy, licenced
    with CeCill-B. Examples and testing written by exana team.
    
    
    Sofia notes on this difference based approach: 
    - A spike train can be thought of as a sum of delta functions at the spike times.
    - When f and g are sums of delta functions this integral becomes a sum over the differences btw spike times t2 and t1
    - Instead of computing the full convolution over a long, mostly-zero time series, the difference-based method calculates the differences
        for all spike pairs where the difference is within a speicified window
        -> done for each spik in t1, finding only those spikes in t2 that lie within the window
        code uses np.searchsorted to easily identify range of indices in t2 that are relevant for t1
    - once relevant time differences are collected into large array, histogram is computer over those differences
    - for autocorrelogram every spike would match itself at t=0 (time lag zero) so that the bin would be artifiically high
        common approach we learned about is to remove the center bin to focus on the correlation structure beyond trivial self-match
    - normalise counts by dividing by total number of spike pairs so they are easier to compare
    
    In simpler terms
    - for every spike in first list, look at spikes in second list that happened around the same time 
    - then subtract the time of the spike in first list from time of each nearby spike in second list, gives you list of time difference lags
    then group time differences into small time bins: for each bin you count how many spike pairs had a time difference that falls into that bin
    - histogram shows for each time delay how many spikes occurred with that delay
    
    Why this code is efficient
    - instead of creating huge vector of zeros and then performing operations, we work directly with spike times
    - only compute differences for spikes that are actually close enough, reducing no of operations compared to full convolution
        
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
    
    Returns
    -------
    (count, bins) : tuple
        A tuple containing the bin right edges and the
        count/density of spikes in each bin.
    Note
    ----
    ⁠ bins ⁠ are relative to ⁠ t1 ⁠. That is, if ⁠ t1 ⁠ leads ⁠ t2 ⁠, then
    ⁠ count ⁠ will peak in a positive time bin.

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

    # Generate bin edges with np.linspace to avoid floating-point issues
    num_bins = int(2 * limit / binsize)
    bins = np.linspace(-limit, limit, num_bins + 1)

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
        # Peak at zero because: sum of all spikes in a train
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

    # Convert to float for safe division
    count = count.astype(float)

    # Optional normalization: divide by total no of spike pairs
    #  * For cross-correlation: divide by len(t1)*len(t2) 
    #  * For auto-correlation: divide by (len(t1)*len(t1)) if you want the same logic
    #    or some other factor (like total spike pairs).
    if normalize:
        if auto:
            denom = float(len(t1)) * float(len(t1))  # or len(t1) choose 2, depends on your convention
        else:
            denom = float(len(t1)) * float(len(t2))
        count /= denom

    return count, bins[1:]