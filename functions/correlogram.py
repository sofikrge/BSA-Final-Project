import numpy as np

def correlogram(t1, t2=None, binsize=.0004, limit=.02, auto=False,
                density=False, normalize=True):
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
    counts = counts.astype(float)

    # Optional normalization: divide by total no of spike pairs
    #  * For cross-correlation: divide by len(t1)*len(t2) 
    #  * For auto-correlation: divide by (len(t1)*len(t1)) if you want the same logic
    #    or some other factor (like total spike pairs).
    if normalize:
        if auto:
            denom = float(len(t1)) * float(len(t1))  # or len(t1) choose 2, depends on your convention
        else:
            denom = float(len(t1)) * float(len(t2))
        counts /= denom

    return count, bins[1:]