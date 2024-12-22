from wfdb import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import find_filter_coefficients as find_fc
import find_wavelet_coefficients as fwc
from scipy.signal import butter, filtfilt, resample

def read_data(dir, record_number, ch=0):
    """
    Read ECG signals and annotations

    parameters:
    dir(str): Path to the dataset
    record_number(int): Record Number or Name
    ch(int): channel number

    Returns:
    ecg_record(object), ecg_ann(object) : ECG signal and it's annotations.

    """

    record_name = dir + str(record_number)
    ecg_record = rdrecord(record_name, channels=[ch])
    ecg_ann = rdann(record_name, extension='atr')
    return ecg_record, ecg_ann

def period_normaliztion(y, num_samples):
    """
    Normalize the period of the ECG beat to a fixed length
    
    parameters:
    y(array-like): ECG beat
    num_samples(int): Number of samples to normalize to
        
    Returns:
    x(np.ndarray): Normalized ECG beat
        
    """
    
    n = num_samples
    n_ = len(y)
    x = np.zeros(n)
    for j in range(n):
        rj = (j-1)*(n_-1)/(n-1) 
        j_ = int(rj)
        x[j] = y[j_] + (y[j_+1]- y[j_])*(rj-j_)
    return x
    
def mean_rr_interval(r_peaks):
    # find the RR intervals
    rr_intervals = np.diff(r_peaks)
    
    return np.mean(rr_intervals)


def lowpass(signal, cutoff, fs, order):
    # Butterworth low-pass filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', output='ba', analog=False)[:2]
    y = filtfilt(b, a, signal)
    return np.array(y)


def highpass(signal, cutoff, fs, order):
    # Butterworth high-pass filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', output='ba', analog=False)[:2]
    y = filtfilt(b, a, signal)
    return np.array(y)


def filter_signal(signal, low_cut, high_cut, fs):
    """
    Find the signal without frequency components higher than 40Hz and less than 0.5Hz.

    parameters:
    signal(array-like): Raw ECG Signal
    fs(int): Sampling rate of the given raw ECG signal

    Returns:
    filtered_ecg(np.ndarray) : Signal without frequency components higher than 40Hz and less than 0.5Hz as an array

    """
    filtered_signal1 = lowpass(signal, cutoff=low_cut, fs=fs, order=5)
    filtered_signal2 = highpass(signal, cutoff=high_cut, fs=fs, order=5)

    filtered_ecg = signal - filtered_signal1 - filtered_signal2  # Subtract low frequency components and high
    # frequency components from the original signal
    return np.array(filtered_ecg)

def load_mit_datset(records, classes):

    """
    Prepare the dataset for the classification task
    
    parameters:
    records(array-like): List of record numbers
    classes(array-like): List of classes to consider
    
    Returns:
    dataset(pd.DataFrame): Dataset with 303 features (3 temporal features + ecg beat with 300 samples) per beat and the label
    
    """
    fs = 360  # sampling rate of the MIT-BIH records

    directory =  "C:\\Users\\pahan\\Desktop\\pahansith\\Semester 7\\Bio Signal Processing\\Paper Implementation\\code\\dataset\\mit-bih-arrhythmia-database-1.0.0\\" # path to the database
    dataset = pd.DataFrame()
    X = []
    y = []
    
    for rec_no in records:
        try:
            if rec_no == 114:  # MLII is channel 1 not 0
                ecg_record, ecg_annotations = read_data(directory, rec_no, ch=1)
            else:
                ecg_record, ecg_annotations = read_data(directory, rec_no)

        except FileNotFoundError:
            print(f"Record {rec_no} not found")
            continue

        samples = ecg_annotations.sample
        ecg_signal = ecg_record.p_signal[:, 0]  # only take the first channel ( lead 1)
        ecg_signal = filter_signal(ecg_signal, 0.5, 40, fs)  # filter the signal
        symbols = ecg_annotations.symbol

        beat_symbols = []
        R_peaks = []

        # Remove non-beat annotations
        for i in range(len(samples)):
            if symbols[i] not in ["[", "]", "x", "(", ")", "p", "t", "u", "`", "'", "^", "|", "~", "+", "s", "T",
                                      "*", "D", "=", "@"]:
                beat_symbols.append(symbols[i])
                R_peaks.append(samples[i])
                
        # Convert to numpy arrays
        beat_symbols = np.array(beat_symbols)
        R_peaks = np.array(R_peaks)
        
        for i, r_peak in enumerate(R_peaks):
            
            x = np.zeros(303) # 303 features per beat
            
            # neglect first 10 ecg beats and last 5 ecg beats
            if i < 10  or i > len(R_peaks) - 5:
                continue

            ## TF?
            else:
                left_index = r_peak - (R_peaks[i]-R_peaks[i-1])//2
                right_index = r_peak + (R_peaks[i+1]-R_peaks[i])//2
                ecg_beat = ecg_signal[left_index:right_index]
                
                ecg_beat = period_normaliztion(ecg_beat, 300)
                x[0] = None # QRS duration
                x[1] = (R_peaks[i] - R_peaks[i-1])/fs # Previous RR interval
                x[2] = (mean_rr_interval(R_peaks[i-10:i]))/fs # mean RR interval of the last 10 beats
                x[3:303] = ecg_beat
                if beat_symbols[i] in classes:
                    X.append(x)
                    y.append(beat_symbols[i])
        
        dataset = pd.DataFrame(X)
        dataset['label'] = y
    
    return dataset
    
