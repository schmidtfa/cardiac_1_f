from plus_slurm import Job
#%%
import pandas as pd
import pymc as pm
import arviz as az
import aesara.tensor as at

import os
from os.path import join


#%%

class BayesCorrelation(Job):

    #%% the run method starts here
    def run(self, key2corr, channel, posterior_checks, outdir, **sample_kwargs):

        # %%
        outdir = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/bay_corr/'
        all_ch_idx = list(range(102))
        all_conditions = ['heart_slope_mag'] # 'brain_no_ica',
        posterior_checks = False
        sample_kwargs = {'tune': 2000, 
                 'draws': 2000,
                 'chains': 2,
                 #'return_inferencedata': True, 
                 'target_accept': 0.9}

        for key2corr in all_conditions:
            #%% create the outdir if it doesnt exist yet
            real_outdir = join(outdir, key2corr)

            if not os.path.isdir(real_outdir):
                os.makedirs(real_outdir)

            for channel in all_ch_idx:
                # %% select data for a given channel
                df4bay_cor = pd.read_csv('/mnt/obob/staff/fschmidt/cardiac_1_f/data/cam_can_1_f_dataframe.csv')
                cur_df = df4bay_cor.query('channel == %d' % channel)

                #%%
                with pm.Model() as model:

                    # set some more or less informative priors here
                    mu_age = pm.TruncatedNormal('mu_age', mu=40, sigma=10., lower=18, upper=90)
                    mu_meg = pm.Normal('mu_meg', mu=0, sigma=1.)

                    #prior on correlation
                    chol, corr, stds = pm.LKJCholeskyCov("chol", n=2, eta=4.0, 
                                    sd_dist=pm.HalfCauchy.dist(2.5), compute_corr=True)
                    #stack data together
                    mu = at.stack((mu_meg, mu_age), axis=1)
                    #observed data
                    y = pm.MvNormal('y', mu=mu, chol=chol, observed=cur_df[[key2corr, 'age']])

                    trace = pm.sample(**sample_kwargs)

                #do some posterior checks
                with model:
                    if posterior_checks:
                        pm.sample_posterior_predictive(trace, extend_inferencedata=True,)

                #%% safe data as netcdf
                az.to_netcdf(trace, join(real_outdir, f'channel_{channel}'))
# %%
