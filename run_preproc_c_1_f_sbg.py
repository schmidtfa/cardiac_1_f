#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 13:33:47 2022

@author: schmidtfa
"""
#%% imports
from cluster_jobs.preprocess_meg_sbg import Preprocessing
from plus_slurm import JobCluster, PermuteArgument
import pandas as pd

#%% get jobcluster
job_cluster = JobCluster(required_ram='2G',
                         request_time=100,
                         request_cpus=2,
                         python_bin='/home/b1059770/miniconda3/envs/ml/bin/python')

OUTDIR = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/c_1_f_resting_sbg'
df_all = pd.read_csv('./data/resting_lists_sbg/resting_list_single.csv').query('fs_1k == True')
df_all.reset_index(inplace=True)

freq_ranges = [(0.1, 50), (0.1, 100), (0.1, 200)]

#%% put in jobs...
job_cluster.add_job(Preprocessing,
                    idx=PermuteArgument(df_all.index),
                    outdir=OUTDIR,
                    l_pass = None,
                    h_pass = 0.1,
                    is_3d=False,
                    freq_range = (1, 200),#PermuteArgument(freq_ranges), -> current cluster cant handle too many jobs
                    notch = False,
                    eye_threshold = 0.3,
                    heart_threshold = 0.3,
                    powerline = 50, #in hz
                    pick_channel = True,
                    pick_dict = {'meg': 'mag', 'eog':True, 'ecg':True},
                    )
#%% submit...
job_cluster.submit(do_submit=True)
# %%
