#%
from plus_slurm import Job
#%
from os import listdir
from mne_bids import BIDSPath, read_raw_bids
import mne
import numpy as np
import pandas as pd

#%%
class MovementJob(Job):
    def run(self,
            subject_id):

        #% read data
        base_dir = '/mnt/obob/camcan/cc700/meg/pipeline/release005/BIDSsep/rest'
        bids_path = BIDSPath(root=base_dir, subject=subject_id, session='rest', task='rest',
                                    extension='.fif')
        raw = read_raw_bids(bids_path=bids_path)

        #% get head position
        chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
        chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
        head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=True)

        #% calculate distance moved per sample 
        xyz = head_pos[:,4:7]
        dist = []

        for idx, row in enumerate(xyz):
            if idx > 0:
                squared_dist = np.sum((xyz[idx-1]-row)**2, axis=0)
                dist.append(np.sqrt(squared_dist))

        dist_arr = np.array(dist)
        #% put in dataframe
        df_move = pd.DataFrame({'subject_id': subject_id,
                                'age': raw.info['subject_info']['age'],
                                'distance': dist_arr,
                                'time (s)': head_pos[1:,0]})

        #and save
        df_move.to_csv(f'/mnt/obob/staff/fschmidt/cardiac_1_f/data/movement_cc/{subject_id}.csv')
