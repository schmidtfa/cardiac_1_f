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
                         request_time=200,
                         request_cpus=2,
                         python_bin='/home/b1059770/miniconda3/envs/ml/bin/python')

df_all = pd.read_csv('./data/resting_lists_sbg/resting_list_single.csv').query('fs_1k == True')
df_all.reset_index(inplace=True)
subject_ids = df_all['subject_id'].unique()

#%% put in jobs...
job_cluster.add_job(Preprocessing,
                    subject=PermuteArgument(subject_ids),
                    freq_range = (1, 200),#PermuteArgument(freq_ranges), -> current cluster cant handle too many jobs
                    eye_threshold = 0.5,
                    heart_threshold = 0.5,
                    )
#%% submit...
job_cluster.submit(do_submit=True)

# %%
