#%% imports
import sys
sys.path.append('/mnt/obob/staff/fschmidt/cardiac_1_f/')

from cluster_jobs.bay_pred_cam_can_irasa import BayesPred
from plus_slurm import SingularityJobCluster, PermuteArgument
import os

#%% get jobcluster
job_cluster = SingularityJobCluster(required_ram='4G',
                         request_time=600,
                         request_cpus=2,
                         exclude_nodes='scs1-1,scs1-8,scs1-9,scs1-12,scs1-13,scs1-14,scs1-15,scs1-16,scs1-20,scs1-25',
                         singularity_image='oras://ghcr.io/thht/obob-singularity-container/xfce_desktop:latest',
                         python_bin='/mnt/obob/staff/fschmidt/conda_cache/envs/ml/bin/python')


OUTDIR = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/bay_corr_irasa/'

if not os.path.isdir(OUTDIR):
    os.makedirs(OUTDIR)

#%%
all_ch_idx = list(range(102))
all_conditions_fooof = ['brain_slope', 'brain_no_ica', 'heart_slope_mag']
all_conditions_irasa = ['ECG_not_rejected', 'ECG_rejected', 'ECG_components']

#%%
sample_kwargs = {'tune': 2000, 
                 'draws': 2000,
                 'chains': 4,
                 'target_accept': 0.9}

#%% put in jobs...
job_cluster.add_job(BayesPred,
                    key2corr = PermuteArgument(all_conditions_irasa),
                    channel = PermuteArgument(all_ch_idx),
                    outdir=OUTDIR,
                    **sample_kwargs
                    )
#%% submit...
job_cluster.submit(do_submit=True)
# %%