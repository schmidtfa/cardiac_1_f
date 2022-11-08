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
                         request_time=400,
                         request_cpus=2,
                         python_bin='/mnt/obob/staff/fschmidt/conda_cache/envs/ml/bin/python')

INDIR =  '/mnt/obob/camcan/cc700/meg/pipeline/release005/BIDSsep/rest'

all_files = [file[4:] for file in listdir(INDIR) if 'sub' in file]

#% put in jobs...
job_cluster.add_job(Preprocessing,
                    subject_id=PermuteArgument(all_files),
                    freq_range = (0.1, 45),#PermuteArgument(freq_ranges), -> current cluster cant handle too many jobs
                    eye_threshold = 0.8,
                    heart_threshold = 0.4,
                    )
#% submit...
job_cluster.submit(do_submit=True)
# %%
