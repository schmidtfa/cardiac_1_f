#%%
from pathlib import Path
import sys
sys.path.append('/mnt/obob/staff/fschmidt/cardiac_1_f')
import pandas as pd
import joblib
import numpy as np
from plus_slurm import Job


#%%
class SlopesAcross(Job):

    #%% the run method starts here
    def run(self, cur_channel, sss):


        heart_thresh, eye_thresh = 0.4, 0.8
        path2data = Path('/mnt/obob/staff/fschmidt/cardiac_1_f/data/data_cam_can_irasa_final')

        irasa = True

        def get_log_slope(psd_aperiodic, freqs, freq_range=[0.5, 45]):

            '''
            Based on the slope fitting for IRASA implemented in Yasa. 
            '''

            freq_logical = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])
            freqs, psd_aperiodic = freqs[freq_logical], psd_aperiodic[freq_logical]

            # Aperiodic fit in semilog space for each channel
            from scipy.optimize import curve_fit

            def _func(t, a, b):
                # See https://github.com/fooof-tools/fooof
                return a + np.log(t**b)

            y_log = np.log(psd_aperiodic)
            # Note that here we define bounds for the slope but not for the
            # intercept.
            popt, pcov = curve_fit(
                _func, freqs, y_log, p0=(2, -1), bounds=((-np.inf, -10), (np.inf, 2))
            )
            intercept, slope = popt[0], popt[1]

            return slope


        def meg2df(key):
            cur_df_meg = pd.DataFrame(cur_data[key]['aperiodic'][meg_idcs,:]).T
            cur_df_meg['Frequency(Hz)'] = cur_data[key]['freqs']
            df_meg_tidy = cur_df_meg.melt(id_vars='Frequency(Hz)')
            df_meg_tidy.columns = ['Frequency(Hz)', 'channel', 'power']
            df_meg_tidy['condition'] = key
            return df_meg_tidy


        #%%
        my_path_ending = f'*/*__eye_threshold_{eye_thresh}__heart_threshold_{heart_thresh}__irasa_{irasa}__sss_{sss}__interpolate_True.dat'

        all_files = [str(sub_path) for sub_path in path2data.glob(my_path_ending) if sub_path.is_file()]

        # %%
        meg_list, slope_list = [], []

        for file in all_files:

            cur_data = joblib.load(file)
            
            meg_idcs = [True if 'MEG' in chan else False for chan in cur_data['data_heart']['fit_params']['Chan']]
            ecg_idcs = [True if 'ECG' in chan else False for chan in cur_data['data_heart']['fit_params']['Chan']]

            #% get meg data
            cur_meg = pd.concat([meg2df('data_no_ica'), meg2df('data_brain'), meg2df('data_heart')])
            cur_meg['age'] = float(cur_data['age'])
            cur_meg['subject_id'] = cur_data['subject_id']

            meg_list.append(cur_meg)

            #% extract slopes in 10hz steps
            freqs = cur_data['data_brain']['freqs']

            lower_freqs = np.arange(1,11) - 0.5
            upper_freqs = np.arange(4, 15, 0.5) * 10


            for lower_freq in lower_freqs:

                slopes_no_ica = [get_log_slope(cur_meg.query(f'channel == {cur_channel}').query('condition == "data_no_ica"')['power'].to_numpy(), freqs, freq_range=[lower_freq, cur_upper]) for cur_upper in upper_freqs]
                slopes_ica = [get_log_slope(cur_meg.query(f'channel == {cur_channel}').query('condition == "data_brain"')['power'].to_numpy(), freqs, freq_range=[lower_freq, cur_upper]) for cur_upper in upper_freqs]
                slopes_ecg = [get_log_slope(cur_meg.query(f'channel == {cur_channel}').query('condition == "data_heart"')['power'].to_numpy(), freqs, freq_range=[lower_freq, cur_upper]) for cur_upper in upper_freqs]


                df_slopes = pd.DataFrame({'ECG_not_rejected': slopes_no_ica,
                                            'ECG_rejected': slopes_ica,
                                            'ECG_components': slopes_ecg,
                                            'upper_freqs': upper_freqs})

                df_slopes['age'] = float(cur_data['age'])
                df_slopes['lower_freqs'] = lower_freq
                df_slopes['subject_id'] = cur_data['subject_id']
                df_slopes['channel'] = cur_channel

                slope_list.append(df_slopes)
        # %%
        df_slope_all = pd.concat(slope_list)

        df_slope_all.to_csv(f'/mnt/obob/staff/fschmidt/cardiac_1_f/data/slopes_across_all_chs/cam_can_irasa_sss_{sss}_ch_{cur_channel}.csv')
