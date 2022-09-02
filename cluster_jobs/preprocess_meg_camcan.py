#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 12:58:40 2022

@author: schmidtfa
"""
#%%
from plus_slurm import Job
from os.path import join

import mne
import numpy as np
import joblib

import sys
sys.path.append('/mnt/obob/staff/fschmidt/cardiac_1_f/utils/')
from cleaning_utils import run_potato
from psd_utils import compute_spectra_ndsp
from fooof_utils import fooof2aperiodics

from mne_bids import BIDSPath, read_raw_bids

#%%

class Preprocessing(Job):

    def run(self, 
            subject,
            outdir,
            l_pass = None,
            h_pass = 0.1,
            notch = True,
            eye_threshold = 0.6,
            heart_threshold = 0.6,
            powerline = 50, #in hz
            pick_channel = True,
            is_3d=True,
            freq_range = [1, 150],
            pick_dict = {'meg': 'mag', 'eog':True, 'ecg':True}):
    
        #%% DEBUG 
        # l_pass = None
        # h_pass = 0.1
        # notch = False
        # eye_threshold = 0.5
        # heart_threshold = 0.5
        # powerline = 50 #in hz
        # pick_channel = True
        # pick_dict = {'meg': 'mag', 'eog':True, 'ecg':True}
        # from os import listdir
        # freq_range = [1, 150]
        # base_dir = '/mnt/obob/camcan/cc700/meg/pipeline/release005/BIDSsep/rest'
        # subject = listdir(base_dir)[0][4:]
        #%%
        base_dir = '/mnt/obob/camcan/cc700/meg/pipeline/release005/BIDSsep/rest'

        bids_path = BIDSPath(root=base_dir, subject=subject, session='rest', task='rest',
                                 extension='.fif')
        raw = read_raw_bids(bids_path=bids_path)
                
        #find bad channels and interpolate
        noisy_chs, flat_chs = mne.preprocessing.find_bad_channels_maxwell(raw)

        raw.load_data()
        raw.info['bads'] = noisy_chs + flat_chs
        raw.interpolate_bads()

        #%% if time is below 5mins breaks function here this is mainly needed for 
        if raw.times.max() / 60 < 4.9:
            raise ValueError(f'The total duration of the recording is below 5min. Recording duration is {raw.times.max() / 60} minutes')
        
        #%%
        if pick_channel:
            if 'BIO003' in raw.ch_names:
                raw.set_channel_types({'BIO001': 'eog',
                                       'BIO002': 'eog',
                                       'BIO003': 'ecg',})
            try:
                raw.pick_types(**pick_dict)
            except ValueError:
                pass

        #Apply filters
        raw.filter(l_freq=h_pass, h_freq=l_pass)

        if notch:
            nyquist = raw.info['sfreq'] / 2
            print(f'Running notch filter using {powerline}Hz steps. Nyquist is {nyquist}')
            raw.notch_filter(np.arange(powerline, nyquist, powerline), filter_length='auto', phase='zero')
            
        #Do the ica
        print('Running ICA. Data is copied and the copy is high-pass filtered at 1Hz')
        raw_no_ica = raw.copy()

        raw_copy = raw.copy().filter(l_freq=1, h_freq=None)
    
        ica = mne.preprocessing.ICA(n_components=50, #selecting 50 components here -> fieldtrip standard in our lab
                                    max_iter='auto')
        ica.fit(raw_copy)
        ica.exclude = []
    
        # reject components by explained variance
        # find which ICs match the EOG pattern using correlation
        eog_indices, eog_scores = ica.find_bads_eog(raw_copy, measure='correlation', threshold=eye_threshold)
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw_copy, measure='correlation', threshold=heart_threshold)
    
        #%% select heart activity
        raw_heart = raw.copy()
        ica.apply(raw_heart, include=ecg_indices, 
                  n_pca_components=len(ecg_indices)) #only project back my ecg components
    
        #%% select everything but heart stuff
        ica.apply(raw, exclude=ecg_indices + eog_indices)
            
        #%%clean epochs using potato
        epochs_brain = mne.make_fixed_length_epochs(raw, duration=2, preload=True)
        epochs_no_ica = mne.make_fixed_length_epochs(raw_no_ica, duration=2, preload=True)
        epochs_heart = mne.make_fixed_length_epochs(raw_heart, duration=2, preload=True)
            
            
        epochs_brain = run_potato(epochs_brain)
        epochs_no_ica = run_potato(epochs_no_ica)
        epochs_heart = run_potato(epochs_heart)            
        #%% fooof the data
        def compute_spectra_and_fooof(epochs, freq_range, run_on_ecg=False):
            
            mags = epochs.copy().pick_types(meg='mag')
            freqs, psd_mag, _ = compute_spectra_ndsp(mags,
                                                    method='welch',
                                                    freq_range=freq_range,
                                                    time_window=2)

            if not is_3d:
                psd_mag = psd_mag.mean(axis=0)
            exponents_mag, offsets_mag,  aps_mag = fooof2aperiodics(freqs, freq_range, psd_mag, is_3d=is_3d)
            
            data = {'mag': {'psd': psd_mag,
                            'freqs': freqs,
                            'offsets': offsets_mag,
                            'exponents': exponents_mag,
                            'aps_mag': aps_mag},}

            if run_on_ecg:
                ecg = epochs.copy().pick_types(ecg=True)
            
                freqs, psd_ecg, _ = compute_spectra_ndsp(ecg,
                                                    method='welch',
                                                    freq_range=freq_range,
                                                    time_window=2)
                if not is_3d:
                    psd_ecg = psd_ecg.mean(axis=0) #average for smoother spectra
                exponents_ecg, offsets_ecg,  aps_ecg = fooof2aperiodics(freqs, freq_range, psd_ecg, is_3d=is_3d)
                 
                data.update({'ecg': {'psd': psd_ecg,
                                    'freqs': freqs,
                                    'offsets': offsets_ecg,
                                    'exponents': exponents_ecg,
                                    'aps_ecg': aps_ecg},})                
            
            return data
        
        
        #%% compute spectra for all conditions
        
        data_no_ica = compute_spectra_and_fooof(epochs_no_ica, freq_range)
        data_brain = compute_spectra_and_fooof(epochs_brain, freq_range)
        data_heart = compute_spectra_and_fooof(epochs_heart, freq_range, run_on_ecg=True)
        
        #%%
        
        data = {'data_no_ica': data_no_ica,
                'data_brain': data_brain,
                'data_heart': data_heart,
                'subject_id': subject,
                'ecg_scores': ecg_scores,
                'age': float(raw.info['subject_info']['age'])}
        
        save_string = subject + f"_frange_{freq_range[0]}_{freq_range[1]}_thr_{heart_threshold}.dat"
        
        joblib.dump(data, join(outdir, save_string))
