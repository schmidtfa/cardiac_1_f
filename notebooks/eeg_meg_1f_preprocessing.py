#%%
import pandas as pd
import mne
from os.path import join
from autoreject import Ransac 
from autoreject.utils import interpolate_bads
import numpy as np
import joblib

#%%
df = pd.read_csv('./data/resting_lists_sbg/resting_list.csv')
df_sel = pd.concat([df.loc[idx] for idx, p in enumerate(df['path']) if 'gw_sleep_pred' in p], axis=1).T

df_sel.drop_duplicates(subset='subject_id', 
                       keep='first',
                       inplace=True)
df_sel.reset_index()

def run_ica(raw, n_comps=50):
    print('Running ICA. Data is copied and the copy is high-pass filtered at 1Hz')
    raw_copy = raw.copy().filter(l_freq=1, h_freq=None)
    
    ica = mne.preprocessing.ICA(n_components=n_comps, max_iter='auto')
    ica.fit(raw_copy)
    
    # reject components by explained variance
    # find which ICs match the EOG pattern using correlation
    ch_types = raw.get_channel_types()

    if 'eog' in ch_types:
        eog_indices, eog_scores = ica.find_bads_eog(raw_copy, measure='correlation', threshold=eye_threshold)
    if 'ecg' in ch_types:
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw_copy, measure='correlation', threshold=heart_threshold)
    
    #%% select heart activity
    raw_heart = raw.copy()
    ica.apply(raw_heart, include=ecg_indices, n_pca_components=len(ecg_indices))
    #%% select everything but heart stuff
    if 'eog' in ch_types:
        ica.apply(raw, exclude=ecg_indices + eog_indices)
    else:
        ica.apply(raw, exclude=ecg_indices)

    return raw, raw_heart, ecg_scores

# %%
#test for eeg -> '19970503pthn'
#test for meg -> '19991118efkm' <- makes sense as this one is broken
h_pass, l_pass = 0.1, 45.
notch, powerline = False, 50
eye_threshold, heart_threshold = 0.5, 0.5


corrs_no_ica_eeg, corrs_no_ica_meg, corrs_ica_eeg, corrs_ica_meg, corrs_ecg_eeg, corrs_ecg_meg = [],[],[],[],[],[]

for idx in range(df_sel['subject_id'].shape[0]):

    cur_subject = df_sel['subject_id'].to_numpy()[idx]
    cur_path = df_sel['path'].to_numpy()[idx]
    raw = mne.io.read_raw_fif(cur_path)

    #do bad data correction if requested
    max_settings_path = '/mnt/obob/staff/fschmidt/meeg_preprocessing/meg/maxfilter_settings/'
    #cal & cross talk files specific to system
    calibration_file = join(max_settings_path, 'sss_cal.dat')
    cross_talk_file = join(max_settings_path, 'ct_sparse.fif')
                
    #find bad channels first in meg
    noisy_chs, flat_chs = mne.preprocessing.find_bad_channels_maxwell(raw,
                                                                  calibration=calibration_file,
                                                                  cross_talk=cross_talk_file)

    #Load data
    raw.load_data()

    raw.info['bads'] = noisy_chs + flat_chs

    raw.interpolate_bads()

    #%% find bad channels also in eeg
    raw_eeg = raw.copy().pick_types(eeg=True, ecg=True, eog=True)
    raw_eeg = raw_eeg.set_eeg_reference(ref_channels='average',
                                              projection=False, verbose=False)

    epo4ransac = mne.make_fixed_length_epochs(raw_eeg, duration=4)
    epo4ransac.load_data().pick_types(eeg=True, ecg=False, eog=False)
    ransac = Ransac(verbose=True)

    ransac.fit(epo4ransac)
    bad_chs_eeg = ransac.bad_chs_
    print(f'RANSAC detected the following bad channels: {bad_chs_eeg}')#

    raw_eeg.info['bads'] = bad_chs_eeg
    raw_eeg = interpolate_bads(raw_eeg, raw_eeg.info['bads'])
            
    #%% Apply filters
    raw_meg = raw.copy().pick_types(meg='mag', ecg=True, eog=True, eeg=False)

    raw_meg.filter(l_freq=h_pass, h_freq=l_pass)
    raw_eeg.filter(l_freq=h_pass, h_freq=l_pass)

    #%% downsample data
    raw_meg.resample(100)
    raw_eeg.resample(100)

    if notch:
        nyquist = raw.info['sfreq'] / 2
        print(f'Running notch filter using {powerline}Hz steps. Nyquist is {nyquist}')
        raw.notch_filter(np.arange(powerline, nyquist, powerline), filter_length='auto', phase='zero')
        raw_eeg.notch_filter(np.arange(powerline, nyquist, powerline), filter_length='auto', phase='zero')   
    
    # %% select meg, eeg
    raw_meg_no_ica = raw_meg.copy()
    raw_eeg_no_ica = raw_eeg.copy()

    #%% Do the ica
    raw_meg, raw_heart_meg, ecg_scores_meg = run_ica(raw_meg)
    raw_eeg, raw_heart_eeg, ecg_scores_eeg = run_ica(raw_eeg, n_comps=None)
             
    data2save = {'eeg': {'no_ica_eeg': raw_eeg_no_ica.to_data_frame(),
                         'heart_eeg': raw_heart_eeg.to_data_frame(),
                         'ica_eeg': raw_eeg.to_data_frame(),
                         'labels': raw_eeg_no_ica.ch_names,
                         'fs': raw_eeg_no_ica.info['sfreq']},
                 'ecg_scores': {'eeg': ecg_scores_eeg,
                                'meg': ecg_scores_meg},
                 'meg': {'no_ica_meg': raw_meg_no_ica.to_data_frame(),
                         'heart_meg': raw_heart_meg.to_data_frame(),
                         'ica_meg': raw_meg.to_data_frame(),
                         'labels': raw_meg_no_ica.ch_names,
                         'fs': raw_meg_no_ica.info['sfreq']}}

    joblib.dump(data2save, join('/mnt/obob/staff/fschmidt/cardiac_1_f/data/data_sim_meeg', f'{cur_subject}.dat'))