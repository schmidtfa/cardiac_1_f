#%% imports
import sys
sys.path.append('/mnt/obob/staff/fschmidt/neurogram/cluster_jobs')

from cluster_jobs.cam_can_single_channel_slopes import SlopesAcross
from plus_slurm import JobCluster, PermuteArgument

#% get jobcluster
job_cluster = JobCluster(required_ram='2G',
                         request_time=600,
                         request_cpus=2,
                         python_bin='/mnt/obob/staff/fschmidt/conda_cache/envs/ml/bin/python')

OUTDIR = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/stats_across_irasa/'

chs = list(range(102))      

#% put in jobs...
job_cluster.add_job(SlopesAcross,
                    cur_channel=PermuteArgument(chs),
                    sss=True
                    )
#% submit...
job_cluster.submit(do_submit=True)


# %%
