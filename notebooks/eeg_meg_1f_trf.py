#%%
from os import listdir
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import pandas as pd
import joblib
import eelbrain as eb
from sklearn.decomposition import PCA

from neurodsp.spectral import compute_spectrum
from neurodsp.utils import create_times
from neurodsp.plts.spectral import plot_power_spectra
import pymc as pm
import arviz as az

from fooof import FOOOFGroup

import matplotlib as mpl
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)


import sys
sys.path.append('/mnt/obob/staff/fschmidt/meeg_preprocessing/utils/')
from psd_utils import compute_spectra_ndsp


sns.set_style('ticks')
sns.set_context('poster')

INDIR = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/data_sim_meeg'
all_files = listdir(INDIR)
all_files
#%%
def do_boosting(avg, boosting_kwargs):
    
    #%get channels post ecg
    start_idx =  np.where(avg.keys() =='ECG003')[0][0] + 1

    tmin = 0
    tstep = 0.01
    nsamples = avg.shape[0]
    time_course = eb.UTS(tmin, tstep, nsamples)
    chans = eb.Scalar('channel', range(len(avg.iloc[:,start_idx:].T))) #

    ecg_tc = eb.NDVar(avg['ECG003'], time_course, name='ECG')
    eeg_tc = eb.NDVar(avg.iloc[:,start_idx:].T, (chans, time_course,), name='EEG')

    trf = eb.boosting(eeg_tc, ecg_tc, **boosting_kwargs)
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


def comp_fooof(cur_data, fs):
    nperseg = fs*5
    start_idx =  np.where(cur_data.keys() =='ECG003')[0][0]
    data2psd = cur_data.iloc[:,start_idx:]
    freqs, psd = compute_spectrum(data2psd.T, fs, method='welch', 
                        avg_type='median', nperseg=nperseg, noverlap=nperseg/2)

    fg = FOOOFGroup(max_n_peaks=0)#, aperiodic_mode='knee')
    fg.fit(freqs, psd, freq_range=(1., 45), progress='tqdm')              
    aps = pd.DataFrame(fg.get_params('aperiodic_params'))
    aps.columns = ['Offset', 'Exponent'] #'Knee',
    return aps


#%%
boosting_kwargs = {'tstart': -0.25,
                   'tstop': 0.25,
                   'basis':0.05,
                   'test':True,
                   'scale_data': True,}

all_eeg, all_meg = [], []

exps_heart_eeg, exps_heart_meg = [], []

OUTDIR = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/data_sim_meeg_trf'

run_boosting = False

for idx, file in enumerate(all_files):
    cur_subject = joblib.load(join(INDIR, file))

    if cur_subject['ecg_scores']['eeg'].max() > 0.5 and cur_subject['ecg_scores']['meg'].max() > 0.5:
        FS = cur_subject['eeg']['fs']
        #Compute power instead of correlation
        aps_eeg = comp_fooof(cur_subject['eeg']['heart_eeg'], FS)
        aps_meg = comp_fooof(cur_subject['meg']['heart_meg'], FS)
        exps_heart_eeg.append(aps_eeg['Exponent'])
        exps_heart_meg.append(aps_meg['Exponent'])

        if run_boosting:
            trf_data = {'trf_eeg_heart': do_boosting(cur_subject['eeg']['heart_eeg'], boosting_kwargs),
                        'trf_eeg_ica': do_boosting(cur_subject['eeg']['ica_eeg'], boosting_kwargs),
                        'trf_eeg_no_ica': do_boosting(cur_subject['eeg']['no_ica_eeg'], boosting_kwargs),
                        'trf_meg_heart': do_boosting(cur_subject['meg']['heart_meg'], boosting_kwargs),
                        'trf_meg_ica': do_boosting(cur_subject['meg']['ica_meg'], boosting_kwargs),
                        'trf_meg_no_ica': do_boosting(cur_subject['meg']['no_ica_meg'], boosting_kwargs),
                      }
            joblib.dump(trf_data, join(OUTDIR, file))

#%%
corr_eeg = pd.concat(exps_heart_eeg,axis=1).corr().iloc[1:,0]
corr_meg = pd.concat(exps_heart_meg,axis=1).corr().iloc[1:,0]

plt.hist(corr_eeg)
plt.hist(corr_meg)
#%%
df_corr = pd.DataFrame({'corr_eeg': corr_eeg,
                        'corr_meg': corr_meg,
                        'subject': np.arange(len(corr_eeg))})


sns.catplot(data=df_corr.melt(id_vars='subject'), x='variable', y='value')

pg.ttest(corr_eeg, 0)#, paired=True)
#%%
pg.ttest(corr_meg, 0)

#%%
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

all_trfs = listdir(OUTDIR)

r_eeg, r_meg, trf_list_eeg, trf_list_meg = [],[],[],[]

for cur_trf_file in all_trfs:

    cur_trf = joblib.load(join(OUTDIR, cur_trf_file))

    r_eeg.append(pd.DataFrame({'pure': cur_trf['trf_eeg_heart'].r.x.mean(),
                               'removed': cur_trf['trf_eeg_ica'].r.x.mean(),
                               'present': cur_trf['trf_eeg_no_ica'].r.x.mean(),
                               'subject_id': cur_trf_file[:-4],
                               'device': 'eeg'}, index=[0]))

    r_meg.append(pd.DataFrame({'pure': cur_trf['trf_meg_heart'].r.x.mean(),
                               'removed': cur_trf['trf_meg_ica'].r.x.mean(),
                               'present': cur_trf['trf_meg_no_ica'].r.x.mean(),
                               'subject_id': cur_trf_file[:-4],
                               'device': 'meg'}, index=[0]))

    trf_cmb_eeg = pd.concat([trf2pandas(cur_trf, 'trf_eeg_heart'),
                              trf2pandas(cur_trf, 'trf_eeg_ica'),
                              trf2pandas(cur_trf, 'trf_eeg_no_ica')], axis=0)
    trf_cmb_eeg['subject_id'] = cur_trf_file[:-4]  
    trf_list_eeg.append(trf_cmb_eeg)

    trf_cmb_meg = pd.concat([trf2pandas(cur_trf, 'trf_meg_heart'),
                              trf2pandas(cur_trf, 'trf_meg_ica'),
                              trf2pandas(cur_trf, 'trf_meg_no_ica')], axis=0)
    trf_cmb_meg['subject_id'] = cur_trf_file[:-4]  
    trf_list_meg.append(trf_cmb_meg)


df_r_eeg = pd.concat(r_eeg)
df_r_meg = pd.concat(r_meg)
df_trf_cmb_eeg = pd.concat(trf_list_eeg)
df_trf_cmb_meg = pd.concat(trf_list_meg)
# %%
plot_order = ['pure', 'present', 'removed']
colors = ['#8da0cb', '#fc8d62', '#66c2a5']

tidy_r_meg = df_r_meg.melt(id_vars=['subject_id', 'device'])
g = sns.swarmplot(data=tidy_r_meg, 
            x='variable', y='value', 
            hue='variable', size=10, alpha=0.5,
            order=plot_order, palette=colors)
g = sns.pointplot(data=tidy_r_meg, 
            x='variable', y='value', 
            markers="_", scale=1.7,
            hue='variable',
            order=plot_order, palette=colors)
g.set_xlabel('ECG Component')
g.set_ylabel('Correlation Coefficient (r)')
g.figure.set_size_inches(6,6)
g.figure.savefig('./results/meg_boost_r.svg')

# %%
tidy_r_eeg = df_r_eeg.melt(id_vars=['subject_id', 'device'])
g = sns.swarmplot(data=tidy_r_eeg, 
            x='variable', y='value', 
            hue='variable', size=10, alpha=0.5,
            order=plot_order, palette=colors)
g = sns.pointplot(data=tidy_r_eeg, 
            x='variable', y='value', 
            markers="_", scale=1.7,
            hue='variable',
            order=plot_order, palette=colors)
g.set_xlabel('ECG Component')
g.set_ylabel('Correlation Coefficient (r)')
g.figure.set_size_inches(6,6)
g.figure.savefig('./results/eeg_boost_r.svg')
# %%
pg.ttest(tidy_r_eeg.query('variable == "present"')['value'],
         tidy_r_meg.query('variable == "present"')['value'],
         paired=True)
#%%
eeg_ecg_present = tidy_r_eeg.query('variable == "present"').copy()
meg_ecg_present = tidy_r_meg.query('variable == "present"').copy()

df_ecg_present = pd.concat([eeg_ecg_present, meg_ecg_present])

g = sns.swarmplot(data=df_ecg_present, 
            x='device', y='value', 
            hue='device', size=10, alpha=0.5,
            palette='tab20')
g = sns.pointplot(data=df_ecg_present, 
            x='device', y='value', 
            markers="_", scale=1.7,
            hue='device',
            palette='tab20')
g.set_ylabel('Correlation Coefficient (r)')
g.set_xlabel('')
g.figure.set_size_inches(3,6)
sns.despine()
g.figure.savefig('./results/eeg_meg_boost_r_comp.svg')
#%%
trf2plot = df_trf_cmb_eeg.reset_index()
g = sns.lineplot(data=trf2plot.query('condition != "trf_eeg_heart"'),
            x='time', 
            y='amplitude (a.u.)',
            hue='condition',
            legend=False,
            palette=['#fc8d62', '#8da0cb'])
g.set_xlim(-.2,.2)
g.set_xlabel('Time (s)')
g.set_ylabel('Amplitude (a.u.)')
g.figure.set_size_inches(4,4)
sns.despine()
g.figure.savefig('./results/eeg_boost_trf.svg')
# %%
trf2plot = df_trf_cmb_meg.reset_index()
g = sns.lineplot(data=trf2plot.query('condition != "trf_meg_heart"'), 
            x='time', 
            y='amplitude (a.u.)',
            hue='condition',
            #legend=False,
            palette=['#fc8d62', '#8da0cb'])
g.set_xlim(-.2,.2)
g.set_xlabel('Time (s)')
g.set_ylabel('Amplitude (a.u.)')
g.figure.set_size_inches(4,4)
sns.despine()
g.figure.savefig('./results/meg_boost_trf.svg')
# %%


