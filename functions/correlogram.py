import numpy as np

def correlogram(t1, t2=None, binsize=.0004, limit=.02, auto=False, normalize=True):
    """
    Computes correlogram
    
    We chose a difference based approach bc the convolution approach required creating huge vectors consisting of 0s and 1s
    and that took up way too much memory such that we were not able to do it with our laptops
    
    Logic behind it:
    - compute all pairwise differences but only within a specific window between spikes t1 and t2
    - in a convolution for discrete signals we check 'at a given time shift t, how many spikes in the first spike train line up with the second spike trian?'
    - here we ask 'for each spike in the first spike train, how many spikes in the second spike train are within a certain window of it and what 
    are their time lags'? => so each of these differences are like a line up at that particular shift
    => adding up these line ups should give us the same result as the convolution
    - bin differences to create histogram
    - for autocorrelogram remove central bin to avoid self-count bias
    
    Efficiency highlights: 
    - swap spike trains if one is larger
    - use np.searchsorted to find only relevant spikes
    
    """
    
    if auto: t2 = t1 # for auto-ccgs, both inputs are same spike train
    
    # ensure total range is exact multiple of binsize
    if not int(limit * 1e10) % int(binsize * 1e10) == 0: # scaling values to avoid imprecision
        raise ValueError(
            'Time limit {} must be a '.format(limit) +
            'multiple of binsize {}'.format(binsize) +
            ' remainder = {}'.format(limit % binsize))
    
    swap_args = False 
    if len(t1) > len(t2): # for efficiency, algorithm iterates over shorter spike train
        swap_args = True
        t1, t2 = t2, t1

    # sort spike times
    t1 = np.sort(t1)
    t2 = np.sort(t2)

    # Create histogram bins
    num_bins = int(2 * limit / binsize) # total width spans from -limit to + limit so 2*limit
    if num_bins % 2 == 0: # force odd number of bins for single center bin
        num_bins += 1
    bins = np.linspace(-limit, limit, num_bins + 1) 

    # Find relevant spikes -> only those that fall in the histogram range
    ii2 = np.searchsorted(t2, t1 - limit)
    jj2 = np.searchsorted(t2, t1 + limit)

    big = np.concatenate([t2[i:j] - t for t, i, j in zip(t1, ii2, jj2)]) # for every spike take corresponding segment, compute differences to get time lags between spike t and nearby spikes

    # bin differences
    count, bins = np.histogram(big, bins=bins)

    # in auto-corr each spike has a zero lag with itself, creating artifical peak at zero
    # fix this by setting the zero bin to zero
    # Remove self-spike bias for autocorrelograms.
    if auto:
        self_differences = np.zeros(len(t1))
        self_hist, _ = np.histogram(self_differences, bins=bins)
        bin_containing_zero = np.nonzero(self_hist)[0][0]
        # Subtract the self-spike count (one per spike)
        count[bin_containing_zero] = max(count[bin_containing_zero] - len(t1), 0)


    # swap t1 and t2 back if needed
    if swap_args:
        count = count[::-1]

    count = count.astype(float) # convert to float for safe division

    # normalisation (optional), divide by total number of spike pairs to make comparable
    # we tried to did it but it made the plots in the matrix partly very difficult to see
    if normalize:
        if auto:
            denom = float(len(t1)) * float(len(t1))
        else:
            denom = float(len(t1)) * float(len(t2))
        count /= denom

    return count, bins[1:]