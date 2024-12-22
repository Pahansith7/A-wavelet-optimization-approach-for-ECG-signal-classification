import numpy as np
import pywt
import matplotlib.pyplot as plt
import find_filter_coefficients as find_fc

def wavelet_coefficients(signal, dec_lo, dec_hi, level=1):
    """
    Compute the wavelet coefficients for a given signal using custom low-pass and high-pass filters.

    Parameters:
        signal (array-like): The input signal to be decomposed.
        dec_lo (array-like): The low-pass decomposition filter.
        dec_hi (array-like): The high-pass decomposition filter.
        level (int): The level of decomposition. Default is 1.

    Returns:
        tuple: A tuple containing:
            - coeffs (list): List of wavelet coefficients for each level.
            - coeffs_concat (array-like): Concatenated wavelet coefficients.
    """
    # Time-reverse low_dec for low_rec
    rec_lo = dec_lo[::-1]
    # Time-reverse hi_dec and change signs for every other element to get hi_rec
    rec_hi = [(-1)**i * dec_hi[::-1][i] for i in range(len(dec_hi))]
    
    filter_bank = [dec_lo, dec_hi, rec_lo, rec_hi]
    custom_wavelet = pywt.Wavelet(name="custom_wavelet", filter_bank=filter_bank)
    
    # Perform the Discrete Wavelet Transform
    coeffs = pywt.wavedec(signal, custom_wavelet, level=level, mode='symmetric')
    
    # coeffs[0]: Approximation coefficients at level 3 (the coarsest representation).
    # coeffs[1]: Detail coefficients at level 3.
    # coeffs[2]: Detail coefficients at level 2.
    # coeffs[3]: Detail coefficients at level 1.
    
    coeffs_concat = np.concatenate(coeffs)
    # Return the approximation (low-pass) and detail (high-pass) coefficients , concatenated coefficients
    return coeffs, coeffs_concat

   
def update_wavelet_coefficients(X_train, lowpass_filter_bank, highpass_filter_bank, levels, coeff_type):
    """
    Update with the wavelet coefficients for the given training data.
    
    Parameters:
        X_train (pd.DataFrame): The training data containing ECG beats and temporal features.
        lowpass_filter_bank (np.ndarray): The lowpass filter bank used for wavelet transformation.
        highpass_filter_bank (np.ndarray): The highpass filter bank used for wavelet transformation.
        levels (int): The number of decomposition levels for the wavelet transformation.
        coeff_type (str): The type of wavelet coefficients to extract ('a' for approximation, 'd' for detail).
    
    Returns:
        np.ndarray: The combined training data with wavelet coefficients and temporal features.
    """

    S = lowpass_filter_bank.shape[0]
    
    if coeff_type == 'a':
        var = 0
    elif coeff_type == 'd':
        var = 1

    # Prepare the dataset for the fitness function evaluation
    X_train_combined =[]
    
    #training datawith morphological features
    for j in range(S):
        features = []
        for i in range(X_train.shape[0]):
            # 1-3 temporal features
            # 3-303 features ECG beats  
            wc, _ = wavelet_coefficients(X_train.iloc[i,3:], lowpass_filter_bank[j], highpass_filter_bank[j], levels)
            temporal_features= X_train.iloc[i,1:3]
            features.append(np.concatenate((wc[var],temporal_features)))
        X_train_combined.append(features)

    X_train_combined =np.array(X_train_combined)
    return X_train_combined