import numpy as np
import eelbrain as eb
from sklearn.decomposition import PCA
import pandas as pd




def do_boosting(avg, fwd, boosting_kwargs):
    
    #%get channels post ecg
    start_idx =  np.where(avg.keys() =='ECG003')[0][0] + 1

    tmin = 0
    tstep = 0.01
    nsamples = avg.shape[0]
    time_course = eb.UTS(tmin, tstep, nsamples)
    chans = eb.Scalar('channel', range(len(avg.iloc[:,start_idx:].T))) #

    ecg_tc = eb.NDVar(avg['ECG003'], time_course, name='ECG')
    eeg_tc = eb.NDVar(avg.iloc[:,start_idx:].T, (chans, time_course,), name='EEG')

    if fwd:
        trf = eb.boosting(eeg_tc, ecg_tc, **boosting_kwargs)
    else:
        trf = eb.boosting(ecg_tc, eeg_tc, **boosting_kwargs)
    return trf



def get_prediction(avg, trf):
        #%get channels post ecg
    start_idx =  np.where(avg.keys() =='ECG003')[0][0] + 1

    tmin = 0
    tstep = 0.01
    nsamples = avg.shape[0]
    time_course = eb.UTS(tmin, tstep, nsamples)
    chans = eb.Scalar('channel', range(len(avg.iloc[:,start_idx:].T))) #

    ecg_tc = eb.NDVar(avg['ECG003'], time_course, name='ECG')
    eeg_tc = eb.NDVar(avg.iloc[:,start_idx:].T, (chans, time_course,), name='EEG')

    return eb.convolve(trf.h_scaled, ecg_tc, eeg_tc)



def trf2pandas(cur_trf, key, do_pca=True):

    if do_pca:
        pca = PCA(n_components=1)
        trf_pca = pca.fit_transform(cur_trf[key].h.x.T)

        df_trf = pd.DataFrame({'amplitude (a.u.)': trf_pca.flatten(),
                              'time': list(cur_trf[key].h_time)})
        df_trf['condition'] = key

    else:
        df_trf_tmp = pd.DataFrame(cur_trf[key].h_scaled.x.T)
        df_trf_tmp['time'] = list(cur_trf[key].h_time)
        df_trf = df_trf_tmp.melt(id_vars='time')
        df_trf.columns= ['time', 'channel', 'amplitude (a.u.)']
        df_trf['condition'] = key
    return df_trf





def get_max_amp(data):

    time_data = []

    trf_times = data.query(f'time > -0.10 and time < 0.10') #to avoid including edge artifacts

    for subject in trf_times['subject_id'].unique():

        cur_trf_subject = trf_times.query(f'subject_id == "{subject}"')

        for channel in cur_trf_subject['channel'].unique():

            cur_trf_channel = cur_trf_subject.query(f'channel == {channel}')

            for condition in cur_trf_channel['condition'].unique():

                cur_trf = cur_trf_channel.query(f'condition == "{condition}"')
                time_data.append(pd.DataFrame(cur_trf.iloc[np.abs(cur_trf['amplitude (a.u.)']).argmax()]).T)
    
    return pd.concat(time_data)
