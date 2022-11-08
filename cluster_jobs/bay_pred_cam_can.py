from plus_slurm import Job
#%%
import pandas as pd
#import pymc as pm
import bambi as bmb
import arviz as az
#import aesara.tensor as at
from scipy.stats import zscore

import os
from os.path import join
#%%

class BayesPred(Job):

    #%% the run method starts here
    def run(self, key2corr, channel, outdir, **sample_kwargs):

        #%% create the outdir if it doesnt exist yet
        real_outdir = join(outdir, key2corr)

        if os.path.exists(join(real_outdir, f'channel_{channel}')):
             print('Warning file already exists. No need to rerun')
        else:
             if not os.path.isdir(real_outdir):
                 os.makedirs(real_outdir)

             # %% select data for a given channel
             df4bay_cor = pd.read_csv('/mnt/obob/staff/fschmidt/cardiac_1_f/data/cam_can_1_f_dataframe_1_145.csv')
             cur_df = zscore(df4bay_cor.query('channel == %d' % channel)[[key2corr, 'age']], axis=0)

             trace = bmb.Model(f'{key2corr} ~ 1 + age', 
                              cur_df, 
                              dropna=True, 
                              family='t').fit(**sample_kwargs)

             #%% safe data as netcdf
             az.to_netcdf(trace, join(real_outdir, f'channel_{channel}'))
