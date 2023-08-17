#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:05:26 2022

@author: schmidtfa
"""
#%% imports
from cluster_jobs.head_movement_camcan import MovementJob
from plus_slurm import JobCluster, PermuteArgument

from os import listdir

#% get jobcluster#
job_cluster = JobCluster(required_ram='4G',
                         request_time=4000,
                         request_cpus=2,
                         max_jobs_per_jobcluster=900,
                         python_bin='/mnt/obob/staff/fschmidt/conda_cache/envs/ml/bin/python')

INDIR =  '/mnt/obob/camcan/cc700/meg/pipeline/release005/BIDSsep/rest'

all_files = [file[4:] for file in listdir(INDIR) if 'sub' in file]

#% put in jobs...
job_cluster.add_job(MovementJob,
                    subject_id=PermuteArgument(all_files),
                    )
#%% submit...
job_cluster.submit(do_submit=True)
# %%
