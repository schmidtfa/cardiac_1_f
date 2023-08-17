#%%
import pandas as pd
import bambi as bmb
import pymc as pm
import joblib
from scipy.stats import zscore
from os.path import join
import arviz as az
from plus_slurm import Job

import sys
sys.path.append('/mnt/obob/staff/fschmidt/cardiac_1_f')

from utils.pymc_utils import coefficients2pcorrs



#%%

#a seed for reproducibility
import random
random.seed(42069)


#%%

class StatsAcross(Job):

    #%% the run method starts here
    def run(self, lower_thresh, upper_thresh, outdir, sss, brms_kwargs):

        #%%
        df_slopes = pd.read_csv(f'/mnt/obob/staff/fschmidt/cardiac_1_f/data/slopes_across_all_cam_can_irasa_sss_{sss}.csv').query(f'upper_freqs == {upper_thresh}').query(f'lower_freqs == {lower_thresh}')

        #%%
        cur_df_z = zscore(df_slopes[['ECG_not_rejected', 'ECG_rejected', 'ECG_components', 'age']].dropna(), axis=0)


        #%%
        mdf_comp_ecg_z = bmb.Model(data=cur_df_z, 
                            formula='ECG_components ~ 1 + age', 
                            dropna=True,
                            family='t').fit(**brms_kwargs)

        mdf_no_ica_z = bmb.Model(data=cur_df_z, 
                            formula='ECG_not_rejected ~ 1 + age', 
                            dropna=True,
                            family='t').fit(**brms_kwargs)

        mdf_ica_z = bmb.Model(data=cur_df_z, 
                            formula='ECG_rejected ~ 1 + age', 
                            dropna=True,
                            family='t').fit(**brms_kwargs)

        #%%
        df2density = pd.DataFrame({'ECG not rejected': mdf_no_ica_z.posterior['age'].to_numpy().flatten(),
                                   'ECG rejected': mdf_ica_z.posterior['age'].to_numpy().flatten(),
                                   'ECG component': mdf_comp_ecg_z.posterior['age'].to_numpy().flatten(),
                                    })

        #%% compute effect combining ecg comps and rejected data
          
        with pm.Model() as multi_regression:

            # set some more or less informativ priors
            b0 = pm.Normal("Intercept", 50, 20)
            b1 = pm.Normal('ECG_components', 0, 10)
            b2 = pm.Normal('ECG_rejected', 0, 10)
            sigma = pm.HalfCauchy("sigma", beta=2.5)

            #regression
            mu = (b0 + b1 * df_slopes['ECG_components'].values 
                    + b2 * df_slopes['ECG_rejected'].values)

            # likelihood -> we are predicting age here with is uncommon but gives us the chance to control for contributions of different predictors
            y = pm.TruncatedNormal('age', mu=mu, lower=18, upper=90, sigma=sigma, 
                                    observed=df_slopes['age'].values)

            mdf = pm.sample(**brms_kwargs)

        summary_multi = az.summary(mdf)
    
        pcorrs = coefficients2pcorrs(df4mdf=df_slopes, mdf=mdf, response_var='age', predictor_vars=['ECG_components', 'ECG_rejected'])

        #%% save data

        data = {'single_effects': df2density,
                'ecg_eff': az.summary(mdf_comp_ecg_z),
                'no_ica_eff': az.summary(mdf_no_ica_z),
                'ica_eff': az.summary(mdf_ica_z),
                'summary_multi': summary_multi,
                'partial_corr': pcorrs}

        joblib.dump(data, join(outdir, f'stats_across_lower_{lower_thresh}_upper_{upper_thresh}.dat'))
# %%
