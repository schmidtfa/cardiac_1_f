import pandas as pd
import numpy as np
from scipy.stats import zscore
import bambi as bmb
import arviz as az
from plus_slurm import Job

class StatsAcross(Job):

    #%% the run method starts here
    def run(self, eog_n, brms_kwargs):

        eog = pd.read_csv('/mnt/obob/staff/fschmidt/cardiac_1_f/data/eog_slopes_camcan.csv')
        lower_freqs = np.arange(1,11) - 0.5
        upper_freqs = np.arange(4, 15, 0.5) * 10

        corr_list = []

        for low_thresh in lower_freqs:
            for up_thresh in upper_freqs: 

                cur_eog = (eog.query(f'upper_freqs == {up_thresh}')
                            .query(f'lower_freqs == {low_thresh}')[[f'EOG_{eog_n}', 'age', 'subject_id']]
                            )

                cur_eog.columns = ['eog', 'age', 'subject_id']
                cur_eog_z = zscore(cur_eog[['eog', 'age']])

                mdf = bmb.Model(formula='eog ~ 1 + age', data=cur_eog_z).fit(**brms_kwargs)
                cur_corr = az.summary(mdf)
                cur_corr['lower_thresh'] = low_thresh
                cur_corr['upper_thresh'] = up_thresh

                corr_list.append(cur_corr)

        # %%

        df_corr = pd.concat(corr_list)
        df_corr.to_csv(f'/mnt/obob/staff/fschmidt/cardiac_1_f/data/cam_can_eog_{eog_n}_across_freqs.csv')