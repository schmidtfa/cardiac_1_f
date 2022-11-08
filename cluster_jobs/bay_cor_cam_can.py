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

class BayesCorrelation(Job):

    #%% the run method starts here
    def run(self, key2corr, channel, outdir, **sample_kwargs):

        # %%
        # outdir = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/bay_corr/'
        # channel = 70
        # key2corr = 'brain_no_ica'
        # posterior_checks = False
        # sample_kwargs = {'tune': 2000, 
        #                  'draws': 2000,
        #                  'chains': 4,
        #                  'target_accept': 0.9}
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

            # #%%
            # with pm.Model() as model:

            #     # set some more or less informative priors here 
            #     #mu_age = pm.Normal('mu_age', mu=0, sigma=1.)
            #     mu = pm.Normal('mu_meg', mu=0, sigma=1.)

            #     #prior on correlation
            #     chol, corr, stds = pm.LKJCholeskyCov("chol", n=2, eta=2.0, 
            #                     sd_dist=pm.HalfCauchy.dist(2.5), compute_corr=True)
            #     #stack data together
            #     #mu = at.stack((mu_meg, mu_age), axis=1)
            #     #observed data
            #     y = pm.MvNormal('y', mu=mu, chol=chol, observed=cur_df[[key2corr, 'age']])

            #     trace = pm.sample(**sample_kwargs)

            # #do some posterior checks
            # with model:
            #     if posterior_checks:
            #         pm.sample_posterior_predictive(trace, extend_inferencedata=True,)

             #%% safe data as netcdf
             az.to_netcdf(trace, join(real_outdir, f'channel_{channel}'))
