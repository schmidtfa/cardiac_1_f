#%%
from cluster_jobs.abstract_jobs.preprocess_abstract import AbstractPreprocessingJob
from os.path import join
import mne
import pandas as pd

#%%
class Preprocessing(AbstractPreprocessingJob):

    job_data_folder = 'data_sbg'

    def _get_age(self):
        return self.raw.info['subject_info']['age']

    def _data_loader(self, subject):
        #%%
        df_all = pd.read_csv('/mnt/obob/staff/fschmidt/cardiac_1_f/data/resting_lists_sbg/resting_list_single.csv').query('fs_1k == True')
        df_all.reset_index(inplace=True)

        print(subject)
        df = df_all.query(f'subject_id == "{subject}"')

        cur_path = list(df['path'])[0]

        print(f'The path is: {cur_path}')
        raw = mne.io.read_raw_fif(cur_path)

        #set age
        raw.info['subject_info']['age'] = df['measurement_age']

        #do bad data correction if requested
        max_settings_path = '/mnt/obob/staff/fschmidt/meeg_preprocessing/meg/maxfilter_settings/'
        #cal & cross talk files specific to system
        calibration_file = join(max_settings_path, 'sss_cal.dat')  # TH: Use pathlib for this
        cross_talk_file = join(max_settings_path, 'ct_sparse.fif')
                
        #find bad channels first
        noisy_chs, flat_chs = mne.preprocessing.find_bad_channels_maxwell(raw,
                                                                          calibration=calibration_file,
                                                                          cross_talk=cross_talk_file)
        #Load data
        raw.load_data()
        raw.info['bads'] = noisy_chs + flat_chs

        raw.interpolate_bads() # remove

        #%% if time is below 5mins breaks function here -> this is because some people in salzburg recorded ~1min resting states
        if raw.times.max() / 60 < 4.9:
            raise ValueError(f'The total duration of the recording is below 5min. Recording duration is {raw.times.max() / 60} minutes')
        
        #%% make sure that if channels are set as bio that they get added correctly
        if 'BIO003' in raw.ch_names:
            raw.set_channel_types({'BIO001': 'eog',
                                   'BIO002': 'eog',
                                   'BIO003': 'ecg',})
        return raw

#if __name__ == '__main__':
 #   job = Preprocessing(subject='19800616mrgu')
  #  job.run_private()
