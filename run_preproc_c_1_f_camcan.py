#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:05:26 2022

@author: schmidtfa
"""
#%% imports
from cluster_jobs.preprocess_meg_camcan import Preprocessing
from plus_slurm import JobCluster, PermuteArgument

from os import listdir

#%% get jobcluster
job_cluster = JobCluster(required_ram='4G',
                         request_time=100,
                         request_cpus=2,
                         python_bin='/home/b1059770/miniconda3/envs/ml/bin/python')

OUTDIR = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/c_1_f_resting_cam_can'
INDIR =  '/mnt/obob/camcan/cc700/meg/pipeline/release005/BIDSsep/rest'

all_files = [file[4:] for file in listdir(INDIR) if 'sub' in file]

#%% put in jobs...
job_cluster.add_job(Preprocessing,
                    subject=PermuteArgument(all_files),
                    outdir=OUTDIR,
                    l_pass = None,
                    is_3d=True,
                    h_pass = 0.1,
                    notch = False,
                    eye_threshold = 0.5,
                    heart_threshold = 0.5,
                    powerline = 50, #in hz
                    freq_range = [1, 150],
                    pick_channel = True,
                    pick_dict = {'meg': 'mag', 'eog':True, 'ecg':True},
                    )
#%% submit...
job_cluster.submit(do_submit=True)
# %%