#%% imports
import sys
sys.path.append('/mnt/obob/staff/fschmidt/neurogram/cluster_jobs')

from cluster_jobs.stats_across_eog import StatsAcross
from plus_slurm import SingularityJobCluster, PermuteArgument

#% get jobcluster
job_cluster = SingularityJobCluster(required_ram='4G',
                         request_time=6000,
                         request_cpus=4,
                         exclude_nodes='scs1-1,scs1-3,scs1-6,scs1-7,scs1-8,scs1-9,scs1-12,scs1-13,scs1-14,scs1-15,scs1-16,scs1-20,scs1-23,scs1-25,scs1-26',
                         singularity_image='oras://ghcr.io/thht/obob-singularity-container/xfce_desktop:latest',
                         python_bin='/mnt/obob/staff/fschmidt/conda_cache/envs/ml/bin/python')

#%
sample_kwargs = {'draws': 2000,
               'tune': 2000,
               'chains': 4,
               'target_accept': 0.9,}

#% put in jobs...
job_cluster.add_job(StatsAcross,
                    eog_n=PermuteArgument([0, 1]),
                    brms_kwargs=sample_kwargs
                    )
#% submit...
job_cluster.submit(do_submit=True)

# %%
