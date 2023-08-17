#%% imports
import sys
sys.path.append('/mnt/obob/staff/fschmidt/neurogram/cluster_jobs')

from cluster_jobs.stats_across_irasa import StatsAcross
from plus_slurm import SingularityJobCluster, PermuteArgument
import os
import numpy as np

#%% get jobcluster
job_cluster = SingularityJobCluster(required_ram='4G',
                         request_time=6000,
                         request_cpus=4,
                         exclude_nodes='scs1-1,scs1-3,scs1-6,scs1-7,scs1-8,scs1-9,scs1-12,scs1-13,scs1-14,scs1-15,scs1-16,scs1-20,scs1-23,scs1-25,scs1-26',
                         singularity_image='oras://ghcr.io/thht/obob-singularity-container/xfce_desktop:latest',
                         python_bin='/mnt/obob/staff/fschmidt/conda_cache/envs/ml/bin/python')

OUTDIR = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/stats_across_irasa_sss/'

if not os.path.isdir(OUTDIR):
    os.makedirs(OUTDIR)

#%%
sample_kwargs = {'draws': 2000,
               'tune': 2000,
               'chains': 4,
               'target_accept': 0.9,}

lower_freqs = np.arange(1,11) - 0.5
upper_freqs =  [40.,  45.,  50.,  55.,  60., 
                65.,  70.,  75.,  80.,  85.,  90.,
                95., 100., 105., 110., 115., 120., 
                125., 130., 135., 140., 145.]

#%% put in jobs...
job_cluster.add_job(StatsAcross,
                    upper_thresh = PermuteArgument(upper_freqs), 
                    lower_thresh = PermuteArgument(lower_freqs), 
                    outdir=OUTDIR,
                    sss=True,
                    brms_kwargs=sample_kwargs
                    )
#%% submit...
job_cluster.submit(do_submit=True)
