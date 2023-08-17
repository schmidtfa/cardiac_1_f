#%%

import sys
sys.path.append('/mnt/obob/staff/fschmidt/cardiac_1_f')

from utils.pymc_utils import coefficients2pcorrs

from os import listdir
from os.path import join

import pandas as pd
import bambi as bmb
import pymc as pm
import joblib
from scipy.stats import zscore
import arviz as az
from plus_slurm import Job
import numpy as np



#%%

class StatsAcross(Job):

    #%% the run method starts here
    def run(self, lower_thresh, upper_thresh, outdir, sss, brms_kwargs):

        #%%
        
        outfiles = listdir(outdir)
        fname2save = f'stats_across_lower_{lower_thresh}_upper_{upper_thresh}_sss_{sss}.dat'

        if fname2save not in outfiles:

                INDIR = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/slopes_across_all_chs'
                df_all = pd.concat([pd.read_csv(join(INDIR, file)) for file in listdir(INDIR) if f'sss_{sss}' in file])

                #%%
                df_slopes = df_all.query(f'upper_freqs == {upper_thresh}').query(f'lower_freqs == {lower_thresh}')

                #%%

                ch_df_list = [] 

                for cur_ch in sorted(df_slopes['channel'].unique()):
                        #%
                        cur_df_slopes = df_slopes.query(f'channel == {cur_ch}')

                        cur_df_z = zscore(cur_df_slopes[['ECG_not_rejected', 'ECG_rejected', 'ECG_components', 'age']].dropna(), axis=0)

                        #%
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

                        #%
                        def info_from_mdf(mdf, cur_ch, lower_thresh, upper_thresh, condition_key):
                                sum = az.summary(mdf)
                                age_eff = sum.loc['age']

                                df_eff = pd.DataFrame({
                                        'average': age_eff['mean'],
                                        'negative_effect': age_eff['hdi_97%'] < -0.1,
                                        'positive_effect': age_eff['hdi_3%'] > 0.1,
                                        'null_effect': np.logical_and(age_eff['hdi_3%'] > -0.1, age_eff['hdi_97%'] < 0.1),
                                        'channel': cur_ch,
                                        'lower_thresh': lower_thresh,
                                        'upper_thresh': upper_thresh,
                                        'condition': condition_key
                                }, index=[cur_ch])

                                return df_eff


                        ch_df_list.append(pd.concat([info_from_mdf(mdf_comp_ecg_z, cur_ch, lower_thresh, upper_thresh, 'ECG_components'),
                                                info_from_mdf(mdf_no_ica_z, cur_ch, lower_thresh, upper_thresh, 'ECG_not_rejected'),
                                                info_from_mdf(mdf_ica_z, cur_ch, lower_thresh, upper_thresh, 'ECG_rejected'),
                        ]))


                ch_eff_df = pd.concat(ch_df_list)

                chs_ecg_eff_neg = list(ch_eff_df.query('condition == "ECG_components"').query('negative_effect == True')['channel'])
                chs_brain_eff_neg = list(ch_eff_df.query('condition == "ECG_rejected"').query('negative_effect == True')['channel'])
                
                chs_ecg_eff_pos = list(ch_eff_df.query('condition == "ECG_components"').query('positive_effect == True')['channel'])
                chs_brain_eff_pos = list(ch_eff_df.query('condition == "ECG_rejected"').query('positive_effect == True')['channel'])

                ecg_effs = [chs_ecg_eff_neg, chs_ecg_eff_pos]
                brain_effs = [chs_brain_eff_neg, chs_brain_eff_pos]
                eff_direction = ['negative', 'positive']

                #%% aggregate
                summary_multi, pcorrs = [], []

                for chs_ecg_eff, chs_brain_eff, cur_direction in zip(ecg_effs, brain_effs, eff_direction):

                        best_ecg = df_slopes.query(f'channel == {chs_ecg_eff}').groupby('subject_id').mean().reset_index()[['ECG_components', 'age', 'subject_id']]
                        best_brain = df_slopes.query(f'channel == {chs_brain_eff}').groupby('subject_id').mean().reset_index()[['ECG_rejected', 'subject_id']]

                        #TODO: if no brain or ecg channels significant. -> safe some info
                        if len(best_ecg) > 0 and len(best_brain) > 0:

                                df_best_cmb = best_ecg.merge(best_brain, on='subject_id')
                                #%% compute effect combining ecg comps and rejected data
                                
                                with pm.Model() as multi_regression:

                                        # set some more or less informativ priors
                                        b0 = pm.Normal("Intercept", 50, 20)
                                        b1 = pm.Normal('ECG_components', 0, 10)
                                        b2 = pm.Normal('ECG_rejected', 0, 10)
                                        sigma = pm.HalfCauchy("sigma", beta=2.5)

                                        #regression
                                        mu = (b0 + b1 * df_best_cmb['ECG_components'].values 
                                                 + b2 * df_best_cmb['ECG_rejected'].values)

                                        # likelihood -> we are predicting age here with is uncommon but gives us the chance to control for contributions of different predictors
                                        y = pm.TruncatedNormal('age', mu=mu, lower=18, upper=90, sigma=sigma, 
                                                                observed=df_best_cmb['age'].values)

                                        mdf = pm.sample(**brms_kwargs)

                                summary_multi_tmp = az.summary(mdf)
                                summary_multi_tmp['effect_direction'] = cur_direction
                                summary_multi.append(summary_multi_tmp)
                                pcorrs_tmp = coefficients2pcorrs(df4mdf=df_best_cmb, mdf=mdf, response_var='age', predictor_vars=['ECG_components', 'ECG_rejected'])
                                pcorrs_tmp['effect_direction'] = cur_direction
                                pcorrs.append(pcorrs_tmp)

                        else: 
                                summary_multi.append(pd.DataFrame({'ecg_eff': len(best_ecg) > 0,
                                                                   'brain_eff': len(best_brain) > 0,
                                                                   'effect_direction': cur_direction,},
                                                                   index=[0]))
                                pcorrs.append(None)

                        #%% save data

                        data = {'single_effects': ch_eff_df,
                                'summary_multi': summary_multi,
                                'effect_direction': cur_direction,
                                'partial_corr': pcorrs}

                joblib.dump(data, join(outdir, f'stats_across_lower_{lower_thresh}_upper_{upper_thresh}_sss_{sss}.dat'))
