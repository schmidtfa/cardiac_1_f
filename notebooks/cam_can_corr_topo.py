#%%
from os import listdir 
from os.path import join
import arviz as az
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from natsort import natsorted

import matplotlib as mpl
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)


sns.set_style('ticks')
sns.set_context('talk')
# %%
INDIR = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/bay_corr'
sub_dirs = ['brain_slope', 'brain_no_ica', 'heart_slope_mag']

cur_sub_dir = sub_dirs[1]
files_by_feature = natsorted(listdir(join(INDIR, cur_sub_dir)))

def get_corrs(cur_sub_dir):

    corr_mean, pos_effect, neg_effect, cor_trace = [], [], [], []

    for file in files_by_feature:

        trace = az.from_netcdf(join(INDIR, cur_sub_dir, file))
        cor_trace.append(trace.posterior['age'].to_numpy().flatten()) # [:,:,0,1]
        cur_corr = az.summary(trace).loc['age']
        corr_mean.append(cur_corr['mean'])
        pos_effect.append(cur_corr['hdi_3%'] > 0.)
        neg_effect.append(cur_corr['hdi_97%'] < -0.)

    trace_array = np.array(cor_trace)
    return corr_mean, pos_effect, neg_effect, trace_array

# %%
corr_ica, pos_ica, neg_ica, trace_ica = get_corrs(sub_dirs[0])
#%%
corr_no_ica, pos_no_ica, neg_no_ica, trace_no_ica = get_corrs(sub_dirs[1])

#%%
corr_heart, pos_heart, neg_heart, trace_heart = get_corrs(sub_dirs[2])

#%%
df_corrs = pd.DataFrame({'corr_ica': corr_ica,
                         'pos_ica': pos_ica,
                         'neg_ica': neg_ica,
                         'corr_no_ica': corr_no_ica,
                         'pos_no_ica': pos_no_ica,
                         'neg_no_ica': neg_no_ica,
                         'corr_heart': corr_heart,
                         'pos_heart': pos_heart,
                         'neg_heart': neg_heart,})

df_corrs.to_csv('../results/cam_can_pred_bayes.csv')
# %%
def plot_corr_topo(corr, mask, info, title, vmin=None, vmax=None, cmap='RdBu_r'):
    
    sns.set_style('ticks')
    sns.set_context('talk')
    corr = np.expand_dims(corr, axis=1) #matrix.shape = n_subjects, n_channels
    
    evoked = mne.EvokedArray(corr, info, tmin=0.)

    mask = np.expand_dims(mask, axis=1)
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=10)

    topo = evoked.plot_topomap(times=[0], scalings=1,
                        time_format=None, cmap=cmap,
                        vmin=vmin, vmax=vmax,
                        units='beta', cbar_fmt='%0.3f', mask=mask, 
                        mask_params=mask_params, title=title,
                        size=3, time_unit='s');
    return topo
# %% get some info structure for plotting
info = mne.io.read_info('/mnt/sinuhe/data_raw/ss_cocktailparty/subject_subject/210503/19930422eibn_resting.fif',
                        verbose=False)
mag_adjacency = mne.channels.find_ch_adjacency(info, 'mag')

info_mags = mne.pick_info(info, mne.pick_types(info, meg='mag'))

# %%
topo_ica_corr = plot_corr_topo(corr_ica, pos_ica, info_mags,'', vmax=0.4, vmin=-0.4);
topo_ica_corr.figure.savefig('../results/topo_ica_corr.svg')

# %%
topo_no_ica_corr = plot_corr_topo(corr_no_ica, pos_no_ica, info_mags,'', vmax=0.4, vmin=-0.4);
topo_no_ica_corr.figure.savefig('../results/topo_no_ica_corr.svg')
# %%
topo_heart_corr = plot_corr_topo(corr_heart, pos_heart, info_mags,'', vmax=0.4, vmin=-0.4);
topo_heart_corr.figure.savefig('../results/topo_heart_corr.svg')
# %% Lets get the probability of a reduction in explained variance after rejecting the heart  

proba_ica_no_ica = np.mean(trace_ica**2 < trace_no_ica**2, axis=1)
topo_proba_ica_no_ica = plot_corr_topo(proba_ica_no_ica, (proba_ica_no_ica > .95), info_mags,'', vmax=1, vmin=0, cmap='Reds');
topo_proba_ica_no_ica.figure.savefig('../results/topo_proba_ica_no_ica.svg')

#%%
proba_ica_no_ica = np.mean(trace_ica**2 > trace_no_ica**2, axis=1)
topo_proba_ica_no_ica = plot_corr_topo(proba_ica_no_ica, (proba_ica_no_ica > .95), info_mags,'', vmax=1, vmin=0, cmap='Reds');
topo_proba_ica_no_ica.figure.savefig('../results/topo_proba_ica_no_ica_opposed.svg')

# %%
sign_ica_heart = np.concatenate(trace_heart[pos_ica])
sign_ica = np.concatenate(trace_ica[pos_ica])
sign_ica_no_ica = np.concatenate(trace_no_ica[pos_ica])
# %%