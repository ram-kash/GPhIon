"""Utility functions for the TAOG analysis package."""

import numpy as np

def calc_dist_sqrd_frac(pos1_frac, pos2_frac, lattice_matrix):
    """Calculates squared distance between fractional coordinates with PBC."""
    frac_diff = pos1_frac - pos2_frac
    frac_diff -= np.round(frac_diff)
    cart_diff = np.dot(lattice_matrix.T, frac_diff)
    return np.sum(cart_diff**2)

def smooth(data, window_len=11):
    """
    Smooth data using a moving average filter.
    
    Args:
        data (np.array): The input data array.
        window_len (int): The size of the moving window.
        
    Returns:
        np.array: The smoothed data, having the same length as the input.
    """
    if window_len < 3:
        return data

    # Pad the data at the edges to handle boundary effects
    s = np.r_[data[window_len-1:0:-1], data, data[-2:-window_len-1:-1]]
    
    # Use convolution for the moving average
    w = np.ones(window_len, 'd')
    y = np.convolve(w/w.sum(), s, mode='valid')
    
    # This slicing ensures the output array has the same length as the input data.
    return y[(window_len//2):-(window_len//2)]

