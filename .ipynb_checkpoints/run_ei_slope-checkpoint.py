#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 13:33:47 2022

@author: schmidtfa
"""
#%% imports
from cluster_jobs.preprocess_meg import Preprocessing
from obob_condor import JobCluster, PermuteArgument
import pandas as pd

#%% get jobcluster
job_cluster = JobCluster(required_ram='2G',
                         owner='schmidtfa',
                         request_cpus=2,
                         python_bin='/mnt/obob/staff/fschmidt/miniconda3/envs/ml/bin/python')

OUTDIR = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/ei_resting'
df_all = pd.read_csv('./data/resting_list_single.csv').query('fs_1k == True')
df_all.reset_index(inplace=True)

#%% put in jobs...
job_cluster.add_job(Preprocessing,
                    idx=PermuteArgument(df_all.index),
                    outdir=OUTDIR,
                    l_pass = None,
                    h_pass = 0.1,
                    notch = False,
                    do_ica = True,
                    eye_threshold = 0.5,
                    heart_threshold = 0.5,
                    powerline = 50, #in hz
                    pick_channel = True,
                    pick_dict = {'meg': 'mag', 'eog':True, 'ecg':True},
                    )
#%% submit...
job_cluster.submit(do_submit=True)