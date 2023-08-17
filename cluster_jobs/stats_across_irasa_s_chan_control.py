#%%

import sys
sys.path.append('/mnt/obob/staff/fschmidt/cardiac_1_f')

from os import listdir
from os.path import join

import pandas as pd
import bambi as bmb
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
        #%%
                INDIR = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/slopes_across_all_chs'
                df_all = pd.concat([pd.read_csv(join(INDIR, file)) for file in listdir(INDIR) if f'sss_{sss}' in file])

                df_movement = pd.read_csv('/mnt/obob/staff/fschmidt/cardiac_1_f/data/movement_cc_df.csv')
                #%%
                df_slopes = df_all.query(f'upper_freqs == {upper_thresh}').query(f'lower_freqs == {lower_thresh}')
                df_slopes_cmb = df_slopes.merge(df_movement[['subject_id', 'distance_median']], on='subject_id')
                #%%

                ch_df_list = [] 

                for cur_ch in sorted(df_slopes_cmb['channel'].unique()):
                        #%
                        cur_df_slopes = df_slopes_cmb.query(f'channel == {cur_ch}')

                        cur_df_z = zscore(cur_df_slopes[['ECG_not_rejected', 'ECG_rejected', 'ECG_components', 'age', 'distance_median']].dropna(), axis=0)

                        #%
                        mdf_comp_ecg_z = bmb.Model(data=cur_df_z, 
                                                formula='ECG_components ~ 1 + age + distance_median', 
                                                dropna=True,
                                                family='t').fit(**brms_kwargs)

                        mdf_no_ica_z = bmb.Model(data=cur_df_z, 
                                                formula='ECG_not_rejected ~ 1 + age + distance_median', 
                                                dropna=True,
                                                family='t').fit(**brms_kwargs)

                        mdf_ica_z = bmb.Model(data=cur_df_z, 
                                                formula='ECG_rejected ~ 1 + age + distance_median', 
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

                data = {'single_effects': ch_eff_df,}

                joblib.dump(data, join(outdir, f'stats_across_lower_{lower_thresh}_upper_{upper_thresh}_sss_{sss}.dat'))
