## eelbrain_utils
import eelbrain as eb
import numpy as np
import mne


def ds_from_evoked_depsampt(data_1, data_2, key_data_1, key_data_2):


    for d_1, d_2 in zip(data_1, data_2):
        if type(d_1) != mne.evoked.EvokedArray or type(d_2) != mne.evoked.EvokedArray:
            raise TypeError('Data needs to be of type "mne.evoked.EvokedArray"')
    
    if np.shape(data_1) != np.shape(data_2):
        raise ValueError('Unequal sample sizes in both groups. Equalize before running.')
    
    
    
    ds = eb.Dataset()
    data = [eb.load.fiff.evoked_ndvar(evoked) for evoked in data_1] + [eb.load.fiff.evoked_ndvar(evoked) for evoked in data_2]

    subjects = eb.Factor(np.arange(len(data_1)), random=True)
    
    ds['subjects'] = eb.combine([subjects, subjects])
    ds['category'] = eb.Factor([key_data_1, key_data_2], repeat=len(subjects))
    ds['eeg'] = eb.combine(data)
                         
    return ds



def gen_tfr_eb_avg(cur_tfr):

    cur_tfr_data = np.expand_dims(cur_tfr.data, axis=0)#

    time_kwargs = {'tmin': cur_tfr.times[0],
                   'tstep': cur_tfr.times[1] - cur_tfr.times[0],
                   'nsamples': cur_tfr.times.shape[0]}

    case = eb.Case(1)
    sensor = eb.load.fiff.sensor_dim(cur_tfr.info)
    frequency = eb.Scalar(name='frequency', values=cur_tfr.freqs, unit='Hz')
    time = eb.UTS(**time_kwargs)

    return eb.NDVar(cur_tfr_data, (case, sensor, frequency, time), name='tfr')



def create_tfr_avg_dataset(data_1, data_2, key_1, key_2):

    ds = eb.Dataset()
    data_1_eb = [gen_tfr_eb_avg(cur_tfr) for cur_tfr in data_1]
    data_2_eb = [gen_tfr_eb_avg(cur_tfr) for cur_tfr in data_2]
    data = data_1_eb + data_2_eb
    
    subjects = eb.Factor(np.arange(len(data_1_eb)), random=True)

    ds['subjects'] = eb.combine([subjects, subjects])
    ds['category'] = eb.Factor([key_1, key_2], repeat=len(subjects))
    ds['tfr'] = eb.combine(data)
    
    return ds

