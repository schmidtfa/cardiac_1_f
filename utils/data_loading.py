import joblib
import numpy as np
import pandas as pd



def data_loader(path2data, freqs, heart_thresh, eye_thresh, peaks, fit_knee, sss, interpolate=False, get_psd=False):

    '''Data loading utility function can be used for loading preprocessed salzburg and camcan data'''

    #my_path_ending = f'*/*[[]{freqs[0]}, {freqs[1]}[]]__eye_threshold_{eye_thresh}__heart_threshold_{heart_thresh}__n_peaks_{peaks}__fit_knee_{fit_knee}.dat'
    #irasa=False
    if interpolate:
        my_path_ending = f'*/*lower_freq_fooof_{freqs[0]}__upper_freq_fooof_{freqs[1]}__eye_threshold_{eye_thresh}__heart_threshold_{heart_thresh}__n_peaks_{peaks}__fit_knee_{fit_knee}__sss_{sss}__interpolate_True.dat'
    else:
        my_path_ending = f'*/*lower_freq_fooof_{freqs[0]}__upper_freq_fooof_{freqs[1]}__eye_threshold_{eye_thresh}__heart_threshold_{heart_thresh}__n_peaks_{peaks}__fit_knee_{fit_knee}__sss_{sss}.dat'

    #print(my_path_ending)
    all_files = [str(sub_path) for sub_path in path2data.glob(my_path_ending) if sub_path.is_file()]
    print(len(all_files))

    all_df, all_psd = [], []
    for idx, file in enumerate(all_files):

        print(f'cur index is {idx}/{len(all_files)}')

        cur_data = joblib.load(file)

        s_id = cur_data['subject_id']
        print(f'cur subject is {s_id}')
    
        if 'ecg_scores' in cur_data.keys() and (cur_data['ecg_scores'] > heart_thresh).sum() > 0:

            if get_psd:
    
                def _make_data_meg(cur_data, key):
                    cur_psd = pd.DataFrame(cur_data['psd'])
                    cur_psd['channel'] = np.arange(102)
                    psd_melt = cur_psd.melt(id_vars='channel')
                    psd_melt['Frequency (Hz)'] = psd_melt['variable'].replace(dict(zip(np.arange(len(cur_data['freqs'])), cur_data['freqs'])))
                    psd_melt.drop(labels='variable', axis=1, inplace=True)
                    psd_melt.columns = ['channel', key, 'Frequency (Hz)']
                    return psd_melt

                mags_df = _make_data_meg(cur_data['data_brain']['mag'], 'Magnetometers (ECG removed)')
                mags_heart_df = _make_data_meg(cur_data['data_heart']['mag'], 'ECG Component Magnetometers')
                mags_no_ica_df = _make_data_meg(cur_data['data_no_ica']['mag'], 'Magnetometers (ECG present)')

                df_meg_cmb = mags_df.merge(mags_heart_df, on=['channel', 'Frequency (Hz)'])
                df_meg_cmb = df_meg_cmb.merge(mags_no_ica_df, on=['channel', 'Frequency (Hz)'])                    
                df_ecg = pd.DataFrame({'ECG Electrode' : cur_data['data_heart']['ecg']['psd'][0,:],
                                       'Frequency (Hz)': cur_data['data_heart']['ecg']['freqs'],
                                      })
                
                df_psd = df_meg_cmb.merge(df_ecg, on='Frequency (Hz)')
                df_psd['subject_id'] = cur_data['subject_id']
                df_psd['age'] = cur_data['age']
                all_psd.append(df_psd)
                

            all_df.append(pd.DataFrame({'heart_slope_mag': cur_data['data_heart']['mag']['exponents'],
                                        'brain_slope': cur_data['data_brain']['mag']['exponents'],
                                        'brain_no_ica': cur_data['data_no_ica']['mag']['exponents'],
                                        'heart_slope': cur_data['data_heart']['ecg']['exponents'][0],
                                        'heart_slope_avg': cur_data['data_heart']['mag']['aps_mag']['Exponent'].mean(),
                                        'brain_slope_avg': cur_data['data_brain']['mag']['aps_mag']['Exponent'].mean(),
                                        'no_ica_slope_avg': cur_data['data_no_ica']['mag']['aps_mag']['Exponent'].mean(),
                                        'channel': np.arange(102),
                                        'n_components': (cur_data['ecg_scores'] > heart_thresh).sum(),
                                        'explained_variance_ratio': cur_data['explained_variance_ecg']['mag'],
                                        'subject_id': cur_data['subject_id'],
                                        'r2_no_ica': cur_data['data_no_ica']['mag']['r2'].mean(),
                                        'error_no_ica': cur_data['data_no_ica']['mag']['error'].mean(),
                                        'r2_ica': cur_data['data_brain']['mag']['r2'].mean(),
                                        'error_ica': cur_data['data_brain']['mag']['error'].mean(),
                                        'r2_heart': cur_data['data_heart']['mag']['r2'].mean(),
                                        'error_heart': cur_data['data_heart']['mag']['error'].mean(),
                                        'age': float(cur_data['age'])}))

            #pick fooof freq range
            freq_res = cur_data['data_heart']['mag']['freqs'][1] - cur_data['data_heart']['mag']['freqs'][0]
            min_freq, max_freq = freqs[0], freqs[1]

            if min_freq > 0.1:

                fooof_freqs = np.arange(min_freq, max_freq + freq_res, freq_res)

            else:

                fooof_freqs = np.arange(min_freq, max_freq, freq_res)
        

    if get_psd:
        df_cmb = pd.concat(all_df)
        df_cmb_psd = pd.concat(all_psd)

        return df_cmb, df_cmb_psd

    else:
        df_cmb = pd.concat(all_df)
        return df_cmb