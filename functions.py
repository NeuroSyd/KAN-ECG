import numpy as np
from scipy import signal
from scipy.signal import stft
from scipy.signal import butter, filtfilt, resample
import pywt
import torch



def apply_bandpass_filter(ecg_data, fs=500, lowcut=0.5, highcut=40, order=4):
    """
    Applies a bandpass filter to each lead in ECG data.

    Args:
    ecg_data (numpy.ndarray): numpy array of shape [N, 4096, 12], where N is the number of ECG recordings
    fs (float): Sampling frequency in Hz (default: 500 Hz)
    lowcut (float): Lower cutoff frequency in Hz (default: 0.5 Hz)
    highcut (float): Upper cutoff frequency in Hz (default: 40 Hz)
    order (int): Filter order (default: 4)

    Returns:
    numpy.ndarray: a numpy array of shape [N, 4096, 12], containing the denoised ECG data
    """
    nyq = 0.5*fs
    lowcut = lowcut/nyq
    highcut = highcut/nyq

    # Create an empty array to store the denoised ECG data
    denoised_ecg_data = np.zeros_like(ecg_data)

    # Loop through each lead in the ECG data
    for i in range(ecg_data.shape[0]):
        for j in range(ecg_data.shape[2]):
            # Extract the ECG data for the current lead
            lead_data = ecg_data[i, :, j]

            # Design the bandpass filter
            b, a = butter(order,[lowcut,highcut], btype='band')

            # Apply the bandpass filter to the lead data
            denoised_lead_data = filtfilt(b, a, lead_data)

            # Store the denoised lead data in the denoised ECG data array
            denoised_ecg_data[i, :, j] = denoised_lead_data

    return denoised_ecg_data

def filter_ecg_signal(data, wavelet='db4', level=8, fs=500, fc=0.1, order=6):
    """
    Filter ECG signals using wavelet denoising.

    Args:
        data (numpy array): ECG signal data with shape (n_samples, n_samples_per_lead, n_leads).
        wavelet (str, optional): Wavelet type for denoising. Default is 'db4'.
        level (int, optional): Decomposition level for wavelet denoising. Default is 8.
        fs (float, optional): Sampling frequency of ECG signals. Default is 500 Hz.
        fc (float, optional): Cutoff frequency for lowpass filter. Default is 0.1 Hz.
        order (int, optional): Filter order for Butterworth filter. Default is 6.

    Returns:
        numpy array: Filtered ECG signals.
    """
    nyquist = 0.5 * fs
    cutoff = fc / nyquist
    b, a = signal.butter(order, cutoff, btype='lowpass')

    filtered_signals = np.zeros_like(data)

    for n in range(data.shape[0]):
        for i in range(data.shape[2]):
            ecg_signal = data[n, :, i]
            coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)
            cA = coeffs[0]
            filtered_cA = signal.filtfilt(b, a, cA)
            filtered_coeffs = [filtered_cA] + coeffs[1:]
            filtered_signal = pywt.waverec(filtered_coeffs, wavelet)
            filtered_signals[n, :, i] = filtered_signal

    return filtered_signals

# resampling ECG data
def resample_ecg_data(ecg_data, origianl_rate, target_rate, samples):
    """
    Resamples ECG data from 400 Hz to 500 Hz.

    Args:
        ecg_data (np.ndarray): ECG data with shape [N, 4096, 12].

    Returns:
        np.ndarray: Resampled ECG data with shape [N, M, 12], where M is the new number of samples after resampling.
    """
    # Compute the resampling ratio
    resampling_ratio = target_rate / origianl_rate

    # Compute the new number of samples after resampling
    M = int(ecg_data.shape[1] * resampling_ratio)

    # Initialize an array to store the resampled data
    ecg_data_resampled = np.zeros((ecg_data.shape[0], M, ecg_data.shape[2]))

    # Iterate over each channel and resample independently
    for i in range(ecg_data.shape[2]):
        for j in range(ecg_data.shape[0]):
            ecg_data_resampled[j, :, i] = resample(ecg_data[j, :, i], M)
    # Trim the resampled data to the last 4096 samples
    ecg_data_resampled = ecg_data_resampled[:, -samples:, :]
    return ecg_data_resampled

def set_channels_to_zero(ecg_data, n):
    """
    Randomly selects a number of ECG channels to set to zero for each group in the data.

    Args:
    - ecg_data: numpy array of shape (N, 4096, 12) containing ECG data
    - n: maximum number of channels that can be set to zero (up to n-1 channels can be left non-zero)

    Returns:
    - numpy array of shape (N, 4096, 12) with selected channels set to zero for each group
    """

    num_groups = 100
    # Choose number of channels to set to zero (up to n-1)
    num_channels_to_set_zero = n
    group_size = ecg_data.shape[0] // num_groups

    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size

        group_data = ecg_data[start_idx:end_idx, :, :]

        # Choose which channels to set to zero
        channels_to_set_zero = np.random.choice(group_data.shape[-1], num_channels_to_set_zero, replace=False)

        # Set selected channels to zero
        ecg_data[start_idx:end_idx, :, channels_to_set_zero] = 0

    # Handle the last group separately to avoid going beyond the shape
    start_idx = num_groups * group_size
    group_data = ecg_data[start_idx:, :, :]

    # Choose which channels to set to zero
    channels_to_set_zero = np.random.choice(group_data.shape[-1], num_channels_to_set_zero, replace=False)

    # Set selected channels to zero
    ecg_data[start_idx:, :, channels_to_set_zero] = 0

    return ecg_data



def STFT_ECG_all_channels(sampling_rate, ecg_data):
    # Define the STFT parameters
    window = 'hann'
    nperseg = int(sampling_rate*0.5)
    noverlap = int(sampling_rate*0.5*0.5)
    number_of_signals = ecg_data.shape[-1]
    Zxx_overall = []
    for k in range(ecg_data.shape[0]):
        Zxx_all = []
        for i in range(number_of_signals):
            channel = i
            arr1 = ecg_data[k, :, i]
            # Compute the STFT
            f, t, Zxx = stft(arr1, fs=sampling_rate, window=window, nperseg=256, noverlap=None)

            Zxx_all.append(np.abs(Zxx))
        Zxx_all = np.array(Zxx_all)
        Zxx_overall.append(Zxx_all)
    Zxx_overall = np.array(Zxx_overall).transpose(0, 2, 3, 1)
    return Zxx_overall


# Data normalisation
def min_max_normalize(x):

    nonzero_indices = np.any(x != 0, axis=(1, 2))

    if np.any(nonzero_indices):
        x_nonzero = x[nonzero_indices]
        min_values = np.min(x_nonzero, axis=(1, 2), keepdims=True)
        max_values = np.max(x_nonzero, axis=(1, 2), keepdims=True)

        normalized_x_nonzero = (x_nonzero - min_values) / (max_values - min_values)

        normalized_x_array = np.zeros_like(x)
        normalized_x_array[nonzero_indices] = normalized_x_nonzero

        return normalized_x_array