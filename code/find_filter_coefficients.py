import numpy as np

def lowpass_filter(position):
    """
    Compute the coefficients for a low-pass filter based on the given positions(angular parameters).
    
    Parameters:
        position (list or array-like): A list or array of positions(angular parameters) used to compute the filter coefficients.
        The length of this list determines the length of the filter.
                                    
    Returns:
        numpy.ndarray: An array containing the computed filter coefficients.
    """
    
    N = 2 * len(position)
    h = np.zeros(N)
    lo = N // 2 - 1
    hi = lo + 1
    
    h[lo] = np.cos(position[0])
    h[hi] = np.sin(position[0])
    
    nstages = N // 2
    
    for stage in range(1, nstages):
        c = np.cos(position[stage])
        s = np.sin(position[stage])
        
        h[lo-1] = c * h[lo]
        h[lo]   = s * h[lo]
        h[hi+1] = c * h[hi]
        h[hi]   = -s * h[hi]
        
        M = stage - 1
        q = lo + 1
        
        for _ in range(M):
            hlo = h[q]
            hhi = h[q + 1]
            h[q]     = c * hhi - s * hlo
            h[q + 1] = s * hhi + c * hlo
        
        lo -= 1
        hi += 1
    
    return h

def high_pass_filter(h):
    """
    Generate high-pass filter coefficients from low-pass filter coefficients.
    This function takes an array of low-pass filter coefficients `h` and generates the corresponding high-pass filter coefficients `g` using spectral inversion.
    
    Parameters:
        h (array-like): Low-pass filter coefficients. It is assumed that `h` has an even length.
    
    Returns:
        numpy.ndarray: High-pass filter coefficients.
    """
    N = len(h) // 2  # Assuming h has even length and corresponds to low-pass filter coefficients
    g = np.zeros(len(h))
    
    for i in range(len(h)):
        g[i] = (-1) ** (i + 1) * h[2*N - 1 - i]
    
    return g

def generate_filter_bank(positions):
    """
    Generates a bank of low-pass and high-pass filter coefficients for a given set of positions(angular parameters).
    
    Parameters:
        positions (numpy.ndarray): A 2D array of shape (S, d) where S is the number of particles and d is the dimensionality of each particle's position(# of angular parameters).
    
    Returns:
        tuple: A tuple containing two 2D arrays:
                - lowpass_filter_bank (numpy.ndarray): A 2D array of shape (S, 2*d) containing the low-pass filter coefficients for each particle.
                - highpass_filter_bank (numpy.ndarray): A 2D array of shape (S, 2*d) containing the high-pass filter coefficients for each particle.
    """
    
    S, d = positions.shape[0], positions.shape[1]
    
    # Initializ an array of size [S,2*d]
    lowpass_filter_bank = np.zeros((S,2*d))
    highpass_filter_bank = np.zeros((S,2*d))

    # find the low-pass filter coefficients for each particle   
    for particle in range(S):
        lp_fc = lowpass_filter(positions[particle])
        lowpass_filter_bank[particle] = lp_fc

    # find high-pass filter coefficients for each particle from low-pass filter coefficients
    for particle in range(S):
        hp_fc = high_pass_filter(lowpass_filter_bank[particle])
        highpass_filter_bank[particle] = hp_fc 
    
    return lowpass_filter_bank, highpass_filter_bank