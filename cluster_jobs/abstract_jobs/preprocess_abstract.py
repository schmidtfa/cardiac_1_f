from cluster_jobs.abstract_jobs.meta_job import Job
import mne
import numpy as np
import joblib
from utils.cleaning_utils import run_potato
from utils.psd_utils import compute_spectra_ndsp
from utils.fooof_utils import fooof2aperiodics

import random
random.seed(42069) #make it reproducible - sort of

class AbstractPreprocessingJob(Job):
    def run(self,
            subject_id,
            l_pass = None,
            h_pass = 1,
            notch = False,
            eye_threshold = 0.5,
            heart_threshold = 0.5,
            powerline = 50, #in hz
            is_3d=False,
            freq_range = [1, 150],
            pick_dict = {'meg': 'mag', 'eog':True, 'ecg':True}):

        self.raw = self._data_loader(subject_id)

        self.raw.pick_types(**pick_dict)
        #Apply filters
        self.raw.filter(l_freq=h_pass, h_freq=l_pass)

        if notch:
            nyquist = self.raw.info['sfreq'] / 2
            print(f'Running notch filter using {powerline}Hz steps. Nyquist is {nyquist}')
            self.raw.notch_filter(np.arange(powerline, nyquist, powerline), filter_length='auto', phase='zero')
            
        #Do the ica
        print('Running ICA. Data is copied and the copy is high-pass filtered at 1Hz')
        raw_no_ica = self.raw.copy()
        raw_copy = self.raw.copy().filter(l_freq=1, h_freq=None)
    
        ica = mne.preprocessing.ICA(n_components=50, #selecting 50 components here -> fieldtrip standard in our lab
                                    max_iter='auto')
        ica.fit(raw_copy)
        ica.exclude = []
    
        # reject components by explained variance
        # find which ICs match the EOG pattern using correlation
        eog_indices, eog_scores = ica.find_bads_eog(raw_copy, measure='correlation', threshold=eye_threshold)
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw_copy, measure='correlation', threshold=heart_threshold)

        explained_variance_ecg = ica.get_explained_variance_ratio(raw_copy, components=ecg_indices)
    
        #%% select heart activity
        raw_heart = self.raw.copy()

        brain2exclude = np.delete(np.arange(ica.n_components), ecg_indices)
        ica.apply(raw_heart, include=ecg_indices, exclude=brain2exclude) #only project back my ecg components

        #%% select everything but heart stuff
        ica.apply(self.raw, exclude=ecg_indices + eog_indices)

        #%% select only eyes
        ica.apply(raw_no_ica, exclude=eog_indices)
            
        #%%clean epochs using potato
        epochs_brain = mne.make_fixed_length_epochs(self.raw, duration=2, preload=True)
        epochs_no_ica = mne.make_fixed_length_epochs(raw_no_ica, duration=2, preload=True)
        epochs_heart = mne.make_fixed_length_epochs(raw_heart, duration=2, preload=True)
            
        epochs_brain = run_potato(epochs_brain)
        epochs_no_ica = run_potato(epochs_no_ica)
        epochs_heart = run_potato(epochs_heart)            
        
        #%% compute spectra and fooof the data
        data_no_ica = self._compute_spectra_and_fooof(epochs_no_ica, freq_range, run_on_ecg=False, 
                                                      is_3d=is_3d)
        data_brain = self._compute_spectra_and_fooof(epochs_brain, freq_range, run_on_ecg=False, 
                                                     is_3d=is_3d)
        data_heart = self._compute_spectra_and_fooof(epochs_heart, freq_range, run_on_ecg=True, 
                                                     is_3d=is_3d)
        
        #%%
        data = {'data_no_ica': data_no_ica,
                'data_brain': data_brain,
                'data_heart': data_heart,
                'subject_id': subject_id,
                'ecg_scores': ecg_scores,
                'age': self._get_age(),
                'explained_variance_ecg': explained_variance_ecg,
                #'explained_variance': ica._get_infos_for_repr().fit_explained_variance
                }

        joblib.dump(data, self.full_output_path)


    def _compute_spectra_and_fooof(self, epochs, freq_range, run_on_ecg, is_3d):

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
            exponents_ecg, offsets_ecg,  aps_ecg = fooof2aperiodics(freqs, freq_range, psd_ecg, is_3d=is_3d, fit_knee=False)
            
            data.update({'ecg': {'psd': psd_ecg,
                                 'freqs': freqs,
                                 'offsets': offsets_ecg,
                                 'exponents': exponents_ecg,
                                 'aps_ecg': aps_ecg},})                

        return data

    #safety methods
    def _data_loader(self):
        raise NotImplementedError

    def _get_age(self):
        raise NotImplementedError