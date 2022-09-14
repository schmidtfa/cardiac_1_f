#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 12:58:40 2022

@author: schmidtfa
"""
#%%
from cluster_jobs.abstract_jobs.preprocess_abstract import AbstractPreprocessingJob
import mne
from mne_bids import BIDSPath, read_raw_bids

class Preprocessing(AbstractPreprocessingJob):
    
    job_data_folder = 'data_cam_can'

    def _get_age(self):
        return self.raw.info['subject_info']['age']

    def _data_loader(self, subject_id):

        base_dir = '/mnt/obob/camcan/cc700/meg/pipeline/release005/BIDSsep/rest'

        bids_path = BIDSPath(root=base_dir, subject=subject_id, session='rest', task='rest',
                                 extension='.fif')
        raw = read_raw_bids(bids_path=bids_path)
                
        #find bad channels and interpolate
        noisy_chs, flat_chs = mne.preprocessing.find_bad_channels_maxwell(raw)

        raw.load_data()
        raw.info['bads'] = noisy_chs + flat_chs
        raw.interpolate_bads()

        return raw


#if __name__ == '__main__':
 #   job = Preprocessing(subject_id='19800616mrgu')
  #  job.run_private()