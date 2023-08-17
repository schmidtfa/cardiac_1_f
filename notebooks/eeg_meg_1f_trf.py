#%%
from os import listdir
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib

import sys
sys.path.append('/mnt/obob/staff/fschmidt/cardiac_1_f')

from utils.plot_utils import plot_ridge
from utils.trf_utils import do_boosting, trf2pandas, get_max_amp

import bambi as bmb
import arviz as az
import matplotlib as mpl
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)


import sys
sys.path.append('/mnt/obob/staff/fschmidt/meeg_preprocessing/utils/')

sns.set_style('ticks')
sns.set_context('poster')

INDIR = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/data_sim_meeg'
all_files = listdir(INDIR)
all_files = [file for file in all_files if len(file) == 16]
#%%

def combine_trf_dfs(dev_type, do_pca):
    trf_cmb = pd.concat([trf2pandas(cur_trf, f'trf_{dev_type}_heart', do_pca=do_pca),
                         trf2pandas(cur_trf, f'trf_{dev_type}_ica', do_pca=do_pca),
                         trf2pandas(cur_trf, f'trf_{dev_type}_no_ica', do_pca=do_pca)], axis=0)

    trf_cmb['subject_id'] = cur_trf_file[:-4]  
    return trf_cmb


boosting_kwargs = {'tstart': -0.25,
                   'tstop': 0.25,
                   'basis':0.05,
                   'test':True,
                   'scale_data': True,}

all_eeg, all_meg = [], []

exps_heart_eeg, exps_heart_meg = [], []

meg_scores, eeg_scores = [], [] 
heart_threshold = 0.4

OUTDIR = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/data_sim_meeg_trf'

run_boosting = False

if run_boosting:
    for idx, file in enumerate(all_files):
        cur_subject = joblib.load(join(INDIR, f'{file[:12]}_bp_freq_0.1_45.0_h_thresh_{heart_threshold}.dat'))

        if cur_subject['ecg_scores']['eeg'].max() > heart_threshold and cur_subject['ecg_scores']['meg'].max() > heart_threshold:

            #Compute power instead of correlation
            trf_data = {'trf_eeg_heart': do_boosting(cur_subject['eeg']['heart_eeg'], fwd=True, boosting_kwargs=boosting_kwargs),
                        'trf_eeg_ica': do_boosting(cur_subject['eeg']['ica_eeg'], fwd=True, boosting_kwargs=boosting_kwargs),
                        'trf_eeg_no_ica': do_boosting(cur_subject['eeg']['no_ica_eeg'], fwd=True, boosting_kwargs=boosting_kwargs),
                        'trf_meg_heart': do_boosting(cur_subject['meg']['heart_meg'], fwd=True, boosting_kwargs=boosting_kwargs),
                        'trf_meg_ica': do_boosting(cur_subject['meg']['ica_meg'], fwd=True, boosting_kwargs=boosting_kwargs),
                        'trf_meg_no_ica': do_boosting(cur_subject['meg']['no_ica_meg'], fwd=True, boosting_kwargs=boosting_kwargs),
                        'bwd_eeg_heart': do_boosting(cur_subject['eeg']['heart_eeg'], fwd=False, boosting_kwargs=boosting_kwargs),
                        'bwd_eeg_ica': do_boosting(cur_subject['eeg']['ica_eeg'], fwd=False, boosting_kwargs=boosting_kwargs),
                        'bwd_eeg_no_ica': do_boosting(cur_subject['eeg']['no_ica_eeg'], fwd=False, boosting_kwargs=boosting_kwargs),
                        'bwd_meg_heart': do_boosting(cur_subject['meg']['heart_meg'], fwd=False, boosting_kwargs=boosting_kwargs),
                        'bwd_meg_ica': do_boosting(cur_subject['meg']['ica_meg'], fwd=False, boosting_kwargs=boosting_kwargs),
                        'bwd_meg_no_ica': do_boosting(cur_subject['meg']['no_ica_meg'], fwd=False, boosting_kwargs=boosting_kwargs),
                        }
            joblib.dump(trf_data, join(OUTDIR, file))

#%%

all_trfs = listdir(OUTDIR)

r_eeg, r_meg, trf_list_eeg, trf_list_eeg_pca, trf_list_meg, trf_list_meg_pca = [],[],[],[],[],[]

for cur_trf_file in all_trfs:

    cur_trf = joblib.load(join(OUTDIR, cur_trf_file))

    r_eeg.append(pd.DataFrame({'pure': cur_trf['bwd_eeg_heart'].r,
                               'removed': cur_trf['bwd_eeg_ica'].r,
                               'present': cur_trf['bwd_eeg_no_ica'].r,
                               'subject_id': cur_trf_file[:-4],
                               'device': 'eeg'}, index=[0]))

    r_meg.append(pd.DataFrame({'pure': cur_trf['bwd_meg_heart'].r,
                               'removed': cur_trf['bwd_meg_ica'].r,
                               'present': cur_trf['bwd_meg_no_ica'].r,
                               'subject_id': cur_trf_file[:-4],
                               'device': 'meg'}, index=[0]))

    trf_cmb_eeg = combine_trf_dfs(dev_type='eeg', do_pca=False)
    trf_cmb_eeg_pca = combine_trf_dfs(dev_type='eeg', do_pca=True)

    trf_list_eeg.append(trf_cmb_eeg)
    trf_list_eeg_pca.append(trf_cmb_eeg_pca)

    trf_cmb_meg = combine_trf_dfs(dev_type='meg', do_pca=False)
    trf_cmb_meg_pca = combine_trf_dfs(dev_type='meg', do_pca=True)

    trf_list_meg.append(trf_cmb_meg)
    trf_list_meg_pca.append(trf_cmb_meg_pca)

df_r_eeg = pd.concat(r_eeg)
df_r_meg = pd.concat(r_meg)

df_trf_cmb_eeg = pd.concat(trf_list_eeg)
df_trf_cmb_meg = pd.concat(trf_list_meg)

df_trf_cmb_eeg_pca = pd.concat(trf_list_eeg_pca)
df_trf_cmb_meg_pca = pd.concat(trf_list_meg_pca)

#%% get maximum amplitude time per subject, channel and condition
#%%
plot_order = ['pure', 'present', 'removed']
colors = ['#66c2a5', '#fc8d62', '#8da0cb',]


colors_density = ['#fc8d62','#8da0cb', '#66c2a5']
hue_order_m = ['trf_meg_ica', 'trf_meg_no_ica', 'trf_meg_heart']
hue_order_e = ['trf_eeg_ica', 'trf_eeg_no_ica', 'trf_eeg_heart']

#%%
#%% model dispersion of peaks around 0
df_time_meg = get_max_amp(df_trf_cmb_meg)
df_time_eeg = get_max_amp(df_trf_cmb_eeg)

meg_mean = df_time_meg.groupby(['condition', 'subject_id']).mean().reset_index()
eeg_mean = df_time_eeg.groupby(['condition', 'subject_id']).mean().reset_index()

eeg_mean['abs_time'] = np.abs(eeg_mean['time'])
meg_mean['abs_time'] = np.abs(meg_mean['time'])

mdf_disp_eeg = bmb.Model('abs_time ~ 0 + condition', dropna=True, family='t', data=eeg_mean).fit()
mdf_disp_meg = bmb.Model('abs_time ~ 0 + condition', dropna=True, family='t', data=meg_mean).fit()


#%%

g = sns.kdeplot(data=meg_mean, x="time", hue="condition", 
            palette=colors_density, fill=True, common_norm=False,
            hue_order=hue_order_m,
            legend=False, alpha=.8, linewidth=0, bw=2)
sns.despine()
g.set_xlabel('Time (s)')

g.set_xlim(-0.2, 0.2)
g.figure.set_size_inches(4,4)
#g.figure.savefig('../results/meg_boost_kde_all.svg')
#%%
g = sns.kdeplot(data=eeg_mean, x="time", hue="condition", 
            palette=colors_density, fill=True, common_norm=False,
            hue_order=hue_order_e,
            legend=False, alpha=.8, linewidth=0, bw=2)
sns.despine()
g.set_xlabel('Time (s)')
g.set_xlim(-0.2, 0.2)
g.figure.set_size_inches(4,4)
#g.figure.savefig('../results/eeg_boost_kde_all.svg')
# %%
tidy_r_meg = df_r_meg.melt(id_vars=['subject_id', 'device'])
tidy_r_eeg = df_r_eeg.melt(id_vars=['subject_id', 'device'])

#%%
eeg_ecg_present = tidy_r_eeg.query('variable != "pure"').copy()
meg_ecg_present = tidy_r_meg.query('variable != "pure"').copy()

sns.set_style('ticks')
sns.set_context('poster')

df_ecg_present = pd.concat([tidy_r_eeg, tidy_r_meg])

fig, axes = plt.subplots(1, 2, sharey=True)

g = sns.swarmplot(data=tidy_r_eeg.query('device == "eeg"'), 
            x='variable', y='value', order=plot_order,
            hue='variable', size=10, alpha=0.5, #col='device',
            palette=colors,#['#fc8d62', '#8da0cb'],
            ax=axes[0])

g = sns.pointplot(data=tidy_r_eeg.query('device == "eeg"'), 
            x='variable', y='value', order=plot_order,
            markers="_", scale=1.7,
            hue='variable',
            palette=colors,#['#fc8d62', '#8da0cb'],
            ax=axes[0])
g.set_ylabel('Decoding Accuracy (r)')
g.set_xlabel('')
g.figure.set_size_inches(5,4)
g.legend_.remove()

g = sns.swarmplot(data=tidy_r_meg.query('device == "meg"'), 
            x='variable', y='value', order=plot_order,
            hue='variable', size=10, alpha=0.5, #col='device',
            palette=colors,#['#fc8d62', '#8da0cb'],
            ax=axes[1])
g = sns.pointplot(data=tidy_r_meg.query('device == "meg"'), 
            x='variable', y='value', order=plot_order,
            markers="_", scale=1.7,
            hue='variable',
            palette=colors,#['#fc8d62', '#8da0cb'],
            ax=axes[1])
g.legend_.remove()

g.set_ylabel('')
g.set_xlabel('')
g.figure.set_size_inches(5,4)
sns.despine()
#fig.savefig('../results/eeg_meg_boost_r_comp_all.svg')
#%%
trf2plot = df_trf_cmb_eeg_pca.reset_index()
g = sns.relplot(data=trf2plot,#.query('condition != "trf_eeg_heart"'),
            x='time', 
            y='amplitude (a.u.)',
            hue='condition',
            col="condition",
            kind="line",
            legend=False,
            palette=colors,
            facet_kws={'sharey': False,
                       'sharex': False,
                       'xlim': (-.2,.2)})
g.set_xlabels('Time (s)')
g.set_ylabels('Amplitude (a.u.)')
sns.despine()
#g.figure.savefig('../results/eeg_boost_trf_all.svg')
# %%
trf2plot = df_trf_cmb_meg_pca.reset_index()
g = sns.relplot(data=trf2plot,#.query('condition != "trf_eeg_heart"'),
            x='time', 
            y='amplitude (a.u.)',
            hue='condition',
            col="condition",
            kind="line",
            legend=False,
            palette=colors,
            facet_kws={'sharey': False,
                       'sharex': False,
                       'xlim': (-.2,.2)})
g.set_xlabels('Time (s)')
g.set_ylabels('Amplitude (a.u.)')
sns.despine()
#g.figure.savefig('../results/meg_boost_trf_all.svg')


# %% compare eeg and meg after removal of the ecg and before

df_rem = df_ecg_present.query('variable == "removed"')
df_pres = df_ecg_present.query('variable == "present"')
df_pure = df_ecg_present.query('variable == "pure"')

#%%
md_rem = bmb.Model('value ~ 0 + device', data=df_rem)
mdf_rem = md_rem.fit()

md_pres = bmb.Model('value ~ 0 + device', data=df_pres)
mdf_pres = md_pres.fit()

md_pure = bmb.Model('value ~ 0 + device', data=df_pure)
mdf_pure = md_pure.fit()


#%% put posteriors in df
df_mdf_rem = pd.DataFrame({'eeg': mdf_rem.posterior['device'][:,:,0].to_numpy().flatten(),
                           'meg': mdf_rem.posterior['device'][:,:,1].to_numpy().flatten(),
                          }).melt()

df_mdf_pres = pd.DataFrame({'eeg': mdf_pres.posterior['device'][:,:,0].to_numpy().flatten(),
                           'meg': mdf_pres.posterior['device'][:,:,1].to_numpy().flatten(),
                          }).melt()

df_mdf_pure = pd.DataFrame({'eeg': mdf_pure.posterior['device'][:,:,0].to_numpy().flatten(),
                           'meg': mdf_pure.posterior['device'][:,:,1].to_numpy().flatten(),
                          }).melt()
                   

# %%
pal_rem = ['#fc8d62', '#fc8d62']
g_rem = plot_ridge(df_mdf_rem, 'variable', 'value', pal=pal_rem, aspect=5, xlim=(0., .4))
#g_rem.figure.savefig('../results/recon_ecg_rem_eff.svg')
# %%
pal_pres = ['#8da0cb', '#8da0cb']
g_pres = plot_ridge(df_mdf_pres, 'variable', 'value', pal=pal_pres, aspect=5, xlim=(0.25, .65))
#g_pres.figure.savefig('../results/recon_ecg_pres_eff.svg')

#%%
pal_pure = ['#66c2a5', '#66c2a5']
g_pure = plot_ridge(df_mdf_pure, 'variable', 'value', pal=pal_pure, aspect=5, xlim=(0.45, .85))
#g_pure.figure.savefig('../results/recon_ecg_pure_eff.svg')


# %% to lazy to substract
lvl = ["meg", "eeg"]
from scipy.stats import zscore

df_rem['z'] = zscore(df_rem['value'])
md_rem_4stat = bmb.Model('z ~ 1 + C(device, levels=lvl)', data=df_rem)
mdf_rem_4stat = md_rem_4stat.fit()

az.summary(mdf_rem_4stat)
# %%
lvl = ["meg", "eeg"]
df_pure['z'] = zscore(df_pure['value'])
md_pure_4stat = bmb.Model('z ~ 1 + C(device, levels=lvl)', data=df_pure)
mdf_pure_4stat = md_pure_4stat.fit()

az.summary(mdf_pure_4stat)
# %%
lvl = ["meg", "eeg"]
df_pres['z'] = zscore(df_pres['value'])
md_pres_4stat = bmb.Model('z ~ 1 + C(device, levels=lvl)', data=df_pres)
mdf_pres_4stat = md_pres_4stat.fit()

az.summary(mdf_pres_4stat)
# %%
