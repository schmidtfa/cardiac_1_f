from mne.time_frequency import psd_welch
from neurodsp.spectral import compute_spectrum, trim_spectrum
import numpy as np
import mne
from warnings import warn

def compute_spectra_mne(mne_data, freq_range):

    '''
    Computes power spectra mne style using a sensible setup based on Gao et al. 2017 
    '''
    
    # Grab the sampling rate, signal and time from the data
    #fs = mne_data.info['sfreq']

    # Calculate power spectra across the the continuous data
    #spectra, freqs = psd_welch(mne_data, fmin=freq_range[0], fmax=freq_range[1],)
                               #n_overlap=time_window*fs*overlap, n_per_seg=fs*time_window)

    spectrum = mne_data.compute_psd(method="multitaper", fmin=freq_range[0], fmax=freq_range[1])
    spectra, freqs = spectrum.get_data(return_freqs=True)
        
    return freqs, spectra, mne_data.info #need cur_data for channel plotting info


def compute_spectra_ndsp(mne_data, method='welch', chan_type=True, freq_range=[1, 25], 
                         time_window=1, overlap=0.5):
    
    '''
    This is a bit faster than the mne implementation.  
    Standard parameters are picked based on Gao et al. 2017 Neuroimage
    '''
    
    # Grab the sampling rate, signal and time from the data
    fs = mne_data.info['sfreq']
    
    if type(mne_data) == mne.epochs.Epochs:
        sig = mne_data.get_data()
    else:
        sig,_ = mne_data.get_data(return_times=True)

 
    psd_kwargs = {'method': method,
                  'avg_type':'mean',
                  'nperseg': fs*time_window if method == 'welch' else None,
                  'noverlap': 0 if method == 'welch' else None,}
        
    if type(mne_data) == mne.epochs.Epochs:

        freqs, psd = zip(*[compute_spectrum(cur_sig, fs, **psd_kwargs) for cur_sig in sig])

        if method == 'welch':
            freqs, psd = zip(*[trim_spectrum(cur_freq, cur_psd, freq_range) for cur_freq, cur_psd in zip(freqs,psd)])
            freqs, psd = freqs[0], np.array(psd)
    else:
        # set frequency resolution
        if method == 'welch' :  
            psd_kwargs['noverlap'] = time_window*fs*overlap 
        else:
            None
        freqs, psd = compute_spectrum(sig, fs, **psd_kwargs)
        
        if method == 'welch':
            freqs, psd = trim_spectrum(freqs, psd, freq_range)
        
                              
    return freqs, psd, mne_data.info #need raw_data.info for channel plotting info


def compute_spectra_ndsp_src(raw_src_data, method='welch', freq_range=[1, 50]):
    
    '''
    This is a bit faster than the mne implementation. 
    Standard parameters are picked based on Gao et al. 2017 Neuroimage
    TODO: This only works for continuous raw src data so far..
    '''
    
    # Grab the sampling rate, signal and time from the data
    fs = raw_src_data.sfreq
    
    nperseg, noverlap, spg_outlier_pct = int(fs), int(fs/2), 5

    psd_settings = {'avg_type':'median',
                    'nperseg': nperseg,
                    'noverlap': noverlap, 
                    'f_range': freq_range,
                    'outlier_percent':spg_outlier_pct}
        
    # set frequency resolution   
    freqs, psd = compute_spectrum(raw_src_data.data, fs, method=method, **psd_settings)
        
    raw_src_data.data = psd
    raw_src_data.tstep = freqs[1] - freqs[0] #should allow visualization as fake time
    raw_src_data.tmin = np.min(freqs)

    return raw_src_data, freqs


def interpolate_line_freq(signal, line_freq, freqs, n_hz_prior):
    '''
    This function takes a power spectrum and interpolates the powerline noise.
    This is done by replacing the value at the powerline freq with the value n Hz beforehand.
    '''
    freq_steps = freqs[1] - freqs[0]
    idx_steps = int(n_hz_prior // freq_steps)
    if idx_steps < 1:
        raise ValueError('Interpolation is not done correctly as the to be replaced values are equal to the values that are used for the replacement')

    interpol = signal.copy()
    for idx, cur_freq in enumerate(freqs):
        if cur_freq % line_freq == 0 and idx > 0:
            if freqs[-1] > cur_freq + (freq_steps * idx_steps):
                interpol[idx-idx_steps:idx+idx_steps] = np.median([signal[idx-idx_steps], signal[idx+idx_steps]])
            else:
                message = f'The Frequency {cur_freq} is too close to the highest frequency in the spectrum. Only indices prior to {cur_freq} are used for the {cur_freq}'
                warn(message, UserWarning, stacklevel=2)
                interpol[idx-idx_steps:idx+idx_steps] = signal[idx-idx_steps]
    return interpol


def create_psd_ave(psd, freqs, info, window_size):

    #window_size should be time in seconds (effective window size from mne)
    
    new_info = mne.Info(info.copy(), sfreq=window_size)
    return mne.EpochsArray(psd, new_info, tmin=freqs.min())


def create_psd_epoch_arr(psd, freqs, info, window_size):

    #window_size should be time in seconds (effective window size from mne)
    
    new_info = mne.Info(info.copy(), sfreq=window_size)
    return mne.EpochsArray(psd, new_info, tmin=freqs.min())
    