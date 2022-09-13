#%% imports
import sys
sys.path.append('/mnt/obob/staff/fschmidt/neurogram/cluster_jobs')

from cluster_jobs.bay_cor_cam_can import BayesCorrelation
from plus_slurm import JobCluster, PermuteArgument
import os

#%% get jobcluster
job_cluster = JobCluster(required_ram='2G',
                         request_cpus=1,
                         python_bin='/mnt/obob/staff/fschmidt/miniconda3/envs/ml/bin/python')


OUTDIR = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/bay_corr/'

if not os.path.isdir(OUTDIR):
    os.makedirs(OUTDIR)

all_ch_idx = list(range(102))
all_conditions = ['brain_slope', 'brain_no_ica', 'heart_slope_mag']

sample_kwargs = {'tune': 2000, 
                 'draws': 2000,
                 'chains': 4,
                 #'return_inferencedata': True, 
                 'target_accept': 0.9}

#%% put in jobs...
job_cluster.add_job(BayesCorrelation,
                    key2corr = PermuteArgument(all_conditions),
                    channel=PermuteArgument(all_ch_idx),
                    posterior_checks=True,
                    outdir=OUTDIR,
                    **sample_kwargs
                    )
#%% submit...
job_cluster.submit(do_submit=True)
# %%