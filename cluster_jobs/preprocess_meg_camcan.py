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
import numpy as np

from mne_bids import BIDSPath, read_raw_bids

class Preprocessing(Job):
    
    job_data_folder = 'data_cam_can'

    def _get_age(self):
        return self.raw.info['subject_info']['age']

    def _data_loader(self, subject):
        #%%
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

        return raw