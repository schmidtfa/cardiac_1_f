#%% imports
import sys
sys.path.append('/mnt/obob/staff/fschmidt/neurogram/cluster_jobs')

from cluster_jobs.stats_across_irasa_s_chan import StatsAcross #control
from plus_slurm import SingularityJobCluster, PermuteArgument
import os
import numpy as np

#% get jobcluster
job_cluster = SingularityJobCluster(required_ram='4G',
                         request_time=6000,
                         request_cpus=4,
                         exclude_nodes='scs1-6,scs1-7,scs1-15,scs1-25,scs1-26',
                         singularity_image='oras://ghcr.io/thht/obob-singularity-container/xfce_desktop:latest',
                         python_bin='/mnt/obob/staff/fschmidt/conda_cache/envs/ml/bin/python')

OUTDIR = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/stats_across_irasa_s_chans_final/'

if not os.path.isdir(OUTDIR):
    os.makedirs(OUTDIR)

#%
sample_kwargs = {'draws': 2000,
               'tune': 2000,
               'chains': 4,
               'target_accept': 0.9,}

lower_freqs = np.arange(1,11) - 0.5
upper_freqs =  [40.,  45.,  50.,  55.,  60., 
                65.,  70.,  75.,  80.,  85.,  90.,
                95., 100., 105., 110., 115., 120., 
                125., 130., 135., 140., 145.]


from os import listdir
outfiles = listdir(OUTDIR)

fname_list = []
sss=True

for lower_thresh in lower_freqs:
    for upper_thresh in upper_freqs:
        fname_list.append(f'stats_across_lower_{lower_thresh}_upper_{upper_thresh}_sss_{sss}.dat')


final_files = list(set(fname_list) - set(outfiles))


lower_f_list, upper_f_list = [],[]
for f in final_files:
    lower_f_list.append(float(f.split('_')[3]))
    upper_f_list.append(float(f.split('_')[5]))

lower_freqs = list(set(lower_f_list))
upper_freqs = list(set(upper_f_list))


#% put in jobs...
job_cluster.add_job(StatsAcross,
                    lower_thresh = PermuteArgument(lower_freqs),
                    upper_thresh = PermuteArgument(upper_freqs), 
                    outdir=OUTDIR,
                    sss=sss,
                    brms_kwargs=sample_kwargs
                    )
#% submit...
job_cluster.submit(do_submit=True)
# %%