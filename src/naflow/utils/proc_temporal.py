import scipy

def round_edge(x, Fs, len_transition):
    """
    Parameters
    ==========
    x : raw
    len_transition : float
        length of rise/fall in seconds. This value will be used for both rise and fall.
    """

    length = x.size / Fs
    alpha = len_transition / length * 2 
    w = scipy.signal.windows.tukey(M = x.size, alpha = alpha)
    
    return x * w