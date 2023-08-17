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
job_cluster = JobCluster(required_ram='4G',
                         request_time=4000,
                         request_cpus=2,
                         max_jobs_per_jobcluster=900,
                         python_bin='/mnt/obob/staff/fschmidt/conda_cache/envs/ml/bin/python')

df_all = pd.read_csv('./data/resting_lists_sbg/resting_list_single.csv').query('fs_1k == True')
df_all.reset_index(inplace=True)
subject_ids = df_all['subject_id'].unique()

sss = [True, False]

#%% put in jobs...
job_cluster.add_job(Preprocessing,
                    subject_id=PermuteArgument(subject_ids),
                    eye_threshold = 0.8,
                    heart_threshold = 0.4,
                    irasa=True,
                    sss=False,
                    interpolate=True,
                    )
#%% submit...
job_cluster.submit(do_submit=True)

# %%
