#%% Imports
import pandas as pd
import bambi as bmb
import pymc as pm
import aesara.tensor as at
import pingouin as pg
import joblib
from os import listdir
from os.path import join
import numpy as np
import mne
#import eelbrain as eb

import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

import matplotlib as mpl
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

sns.set_context('poster')
sns.set_style('ticks')

my_freqs = '_1_150'

INDIR = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/c_1_f_resting_cam_can'
all_files = [file for file in listdir(INDIR) if my_freqs in file]

#%% Load data
all_df, all_psd = [], []
for file in all_files:
    cur_data = joblib.load(join(INDIR, file))
    
    if 'ecg_scores' in cur_data.keys() and (cur_data['ecg_scores'] > 0.5).sum() > 0:
    
        def make_data_meg(cur_data, key):
            cur_psd = pd.DataFrame(cur_data['psd'].mean(axis=0))
            cur_psd['channel'] = np.arange(102)
            psd_melt = cur_psd.melt(id_vars='channel')
            psd_melt['Frequency (Hz)'] = psd_melt['variable'].replace(dict(zip(np.arange(300), cur_data['freqs'])))
            psd_melt.drop(labels='variable', axis=1, inplace=True)
            psd_melt.columns = ['channel', key, 'Frequency (Hz)']
            return psd_melt

        mags_df = make_data_meg(cur_data['data_brain']['mag'], 'Magnetometers (ECG removed)')
        mags_heart_df = make_data_meg(cur_data['data_heart']['mag'], 'ECG Component Magnetometers')
        mags_no_ica_df = make_data_meg(cur_data['data_no_ica']['mag'], 'Magnetometers (ECG present)')

        df_meg_cmb = mags_df.merge(mags_heart_df, on=['channel', 'Frequency (Hz)'])
        df_meg_cmb = df_meg_cmb.merge(mags_no_ica_df, on=['channel', 'Frequency (Hz)'])                    
        df_ecg = pd.DataFrame({'ECG Electrode' : cur_data['data_heart']['ecg']['psd'][0,:][0],
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
                                    'subject_id': cur_data['subject_id'],
                                    'age': cur_data['age']}))
df_cmb = pd.concat(all_df)
df_cmb_psd = pd.concat(all_psd)
#%% Check average psds across subjects
avg_psd = df_cmb_psd.groupby(['Frequency (Hz)', 'channel']).mean().reset_index()

def interpolate_line_freq(signal, line_freq, freqs, freq_res):
    '''
    This function takes a power spectrum and interpolates the powerline noise.
    This is done by replacing the value at the powerline freq with the values beforehand.
    '''
    freq_steps = int(1 / freq_res)
    interpol = signal.copy()
    for idx, cur_freq in enumerate(freqs):
        if cur_freq % line_freq == 0 and idx > 0:
            interpol[idx-freq_steps:idx+freq_steps] = signal[idx-freq_steps]

    return interpol


#%%
fig, ax = plt.subplots()
sns.lineplot(x='Frequency (Hz)', y='Magnetometers (ECG removed)',
             hue="channel",
             data=avg_psd, ax=ax)
fig.set_size_inches(6,6)
plt.xscale('log')
plt.yscale('log')
fig.savefig(f'../results/mags_ga_ecg_removed{my_freqs}.svg')
#%%
fig, ax = plt.subplots()
sns.lineplot(x='Frequency (Hz)', y='ECG Component Magnetometers',
             hue="channel",
             data=avg_psd, ax=ax)
fig.set_size_inches(6,6)
plt.xscale('log')
plt.yscale('log')
fig.savefig(f'../results/mags_ga_ecg_component{my_freqs}.svg')
#%%
fig, ax = plt.subplots()
sns.lineplot(x='Frequency (Hz)', y='Magnetometers (ECG present)',
             hue="channel",
             data=avg_psd, ax=ax)
fig.set_size_inches(6,6)
plt.xscale('log')
plt.yscale('log')
fig.savefig(f'../results/mags_ga_ecg_present{my_freqs}.svg')

# %%
subject_list = df_cmb_psd['subject_id'].unique()

all_r_heart, all_r_brain = [], []

# correlation of powerspectra with and without heartcomponent with the ecg electrode (grandaverage across channels)
for subject in subject_list:
    cur_df = df_cmb_psd.query(f'subject_id == "{subject}"')

    all_r_brain.append(float(pg.corr(cur_df['Magnetometers (ECG present)'], cur_df['ECG Electrode'])['r']))
    all_r_heart.append(float(pg.corr(cur_df['ECG Component Magnetometers'], cur_df['ECG Electrode'])['r']))

for subject in subject_list:
    cur_df = df_cmb_psd.query(f'subject_id == "{subject}"')

#%%
df_ecg_corr = pd.DataFrame({'no ECG Comps.': all_r_brain,
                            'ECG Comps.': all_r_heart,
                            'subject': np.arange(len(all_r_heart))}).melt(id_vars='subject')

df_ecg_corr.columns = ['subject', 'Magnetometers', 'Correlation Coefficient (r)']

g = sns.catplot(data = df_ecg_corr, x='Magnetometers', y='Correlation Coefficient (r)', aspect=1)
#g.set_ylabels('')
g.figure.set_size_inches(5, 5, forward=True)
g.figure.savefig('../results/corr_comp_ecg_mags.svg')

#%%
pg.pairwise_ttests(data=df_ecg_corr, dv='Correlation Coefficient (r)', within='Magnetometers', subject='subject')

# %%
cur_df_cmb = df_cmb.query('channel == 0')

#%%
def plot_slope_age_corr(key, x, y, color):
    corr = pg.corr(cur_df_cmb['age'], cur_df_cmb[key])

    g = sns.lmplot(data=cur_df_cmb, x='age', y=key, line_kws={'color': color},
                   scatter_kws={"s": 40, 'color': '#888888', 'alpha': 0.25})

    r = round(float(corr['r']), 2)
    p = round(float(corr['p-val']), 3)

    if p == 0.0:
        p = 'p < 0.001'
    else:
        p = f'p = {p}'

    plt.annotate(text=f'r = {r}', xy=(x, y))
    plt.annotate(text=p, xy=(x, y - 0.2))

    g.set_xlabels('age (years)')
    g.set_ylabels('1/f slope')
    g.ax.figure.savefig(f'../results/corr_{key}_{my_freqs}.svg', )
#%%
plot_slope_age_corr('no_ica_slope_avg', 20, 2., '#e78ac3')
#%%
plot_slope_age_corr('brain_slope_avg', 20, 2., '#66c2a5')
#%%
plot_slope_age_corr('heart_slope_avg', 20, 2.6, '#fc8d62')
#%%
plot_slope_age_corr('heart_slope', 20, 2.2, '#8da0cb')

#%%
pg.corr(cur_df_cmb['heart_slope'], cur_df_cmb['heart_slope_avg'])

# %%
df_cmb.to_csv('../data/cam_can_1_f_dataframe_1_200.csv')

# %% visualize per channel
import fnmatch

corrs_no_ica, corrs_ica, corrs_ecg_comp, corrs_ecg = [], [], [], []

for channel in df_cmb.channel.unique():
    cur_df = df_cmb.query(f'channel == {channel}')
    corrs_no_ica.append(pg.corr(cur_df['brain_no_ica'], cur_df['age']))
    corrs_ica.append(pg.corr(cur_df['brain_slope'], cur_df['age']))
    corrs_ecg_comp.append(pg.corr(cur_df['heart_slope_mag'], cur_df['age']))


    corrs_ecg.append(pg.corr(cur_df['heart_slope_mag'], cur_df['heart_slope']))
    
    

corr_ica_df = pd.concat(corrs_ica)
corr_no_ica_df = pd.concat(corrs_no_ica)
corr_ecg_df = pd.concat(corrs_ecg)
corr_ecg_comp_df = pd.concat(corrs_ecg_comp)

info = mne.io.read_info('/mnt/sinuhe/data_raw/ss_cocktailparty/subject_subject/210503/19930422eibn_resting.fif',
                        verbose=False)
mag_adjacency = mne.channels.find_ch_adjacency(info, 'mag')
grad_adjacency = mne.channels.find_ch_adjacency(info, 'grad')


info_meg = mne.pick_info(info, mne.pick_types(info, meg=True))

info_mags = mne.pick_info(info, mne.pick_types(info, meg='mag'))

info_grads = mne.pick_info(info, mne.pick_types(info, meg='grad'))

labels_meg = info['ch_names'][15:321]
get_feature = lambda feature: np.array([True if fnmatch.fnmatch(label, feature) else False for label in labels_meg])
mag_idx = get_feature('MEG*1')
grad_idx = ~mag_idx

def plot_corr_topo(corr, mask, info, title, vmin=None, vmax=None):
    
    sns.set_style('ticks')
    sns.set_context('talk')
    corr = np.expand_dims(corr, axis=1) #matrix.shape = n_subjects, n_channels
    
    evoked = mne.EvokedArray(corr, info, tmin=0.)

    mask = np.expand_dims(mask, axis=1)
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=10)

    topo = evoked.plot_topomap(times=[0], scalings=1,
                        time_format=None, #cmap='Reds',
                        vmin=vmin, vmax=vmax,
                        units='r', cbar_fmt='%0.3f', mask=mask, 
                        mask_params=mask_params, title=title,
                        size=3, time_unit='s');
    return topo


# %%
plot_corr_topo(corr_no_ica_df['r'], corr_no_ica_df['p-val'] < 0.05, info_mags, 'Correlation age per chan (no ica)', vmin=-0.35, vmax=0.35);
# %%
plot_corr_topo(corr_ica_df['r'], corr_ica_df['p-val'] < 0.05, info_mags, 'Correlation age per chan (ica)', vmin=-0.35, vmax=0.35);
# %%
plot_corr_topo(corr_ecg_comp_df['r'], corr_ecg_comp_df['p-val'] < 0.05, info_mags, 'Correlation age per chan (ica)', vmin=-0.35, vmax=0.35);
# %%
plot_corr_topo(corr_ecg_df['r'], corr_ecg_df['p-val'] < 0.05, info_mags, 'Correlation ECG per chan (ica)')# vmin=-0.35, vmax=0.35);

# %% throw in eelbrain
def aggregate_sign_feature(feature_key, pos_mask):
    df_brain = df_cmb[[feature_key, 'subject_id', 'heart_slope', 'channel']].reset_index()
    feature_by_age = []


    for subject in df_brain['subject_id'].unique():
            cur_subject = df_brain.query(f'subject_id == "{subject}"')
            feature_by_age.append(cur_subject
                                         .sort_values(by='channel')[pos_mask]
                                         .mean()[[feature_key,'heart_slope']])

    df = pd.concat(feature_by_age, axis=1).T
    return df

#%%
feature_key = 'heart_slope_mag'
pos_mask = np.array(corr_ecg_df['r'] < -0.14)
df = aggregate_sign_feature(feature_key, pos_mask)
#%%
pg.corr(df[feature_key], df['heart_slope'])

#%% combined model
brms_kwargs = {'draws': 1000,
               'tune': 1000,
               'chains': 2,
               'target_accept': 0.98,}

md = bmb.Model(data=df_cmb, 
                    formula='age ~ 1 + brain_slope + heart_slope_mag + (1 + brain_slope + heart_slope_mag | channel)', 
                    dropna=True,
                    family='t')
#  + brain_no_ica
md.build()

#%%              
mdf = md.fit(**brms_kwargs)

#%%
az.to_netcdf(mdf, '../results/cam_can_trace_regr_bmb_1_200.ncdf')
#%%
mdf_summary = az.summary(mdf)
# %%
az.plot_trace(mdf)

#%%
mask = [True if 'brain_slope' in id else False for id in mdf_summary.index]
#%%
(mdf_summary[mask]['hdi_3%'] > 0).sum()
#%%
(mdf_summary[mask]['hdi_97%'] < 0).sum()
# %% try to put everything in one pymc model

#%% all in one pymc
def do_bayesian_correlation(df, key_a, key_b, sampling_kwargs, 
                            do_prior_checks=False, do_posterior_checks=False, sample=True):

    '''
    This function computes a bayesian correlation over a set of coordinates between two variables. 
    This is mainly a function for documentation purposes and ease of use. 
    When applying this to different data most parts would need to be changed.
    '''

    channel_idxs, channel = pd.factorize(df['channel'])
    coords = {'channel': channel,
              'channel_ids': np.arange(len(channel_idxs))}

    with pm.Model(coords=coords) as correlation_model:
        channel_idx = pm.Data("channel_idx", channel_idxs, dims="channel_ids", mutable=False)
        channel_num = pm.Data('channel_num', channel, dims='channel', mutable=False)

        # set some more or less informative priors here
        mu_age = pm.TruncatedNormal('mu_age', mu=40, sigma=10., lower=18, upper=90, dims='channel') #
        mu_meg = pm.Normal('mu_meg', mu=0, sigma=1., dims='channel')

        #prior on correlation
        chol, corr, stds = pm.LKJCholeskyCov("chol", n=2, eta=4.0, 
                       sd_dist=pm.HalfCauchy.dist(2.5), compute_corr=True)
    
        #stack data together
        mu = at.stack((mu_meg[channel_idx], mu_age[channel_idx]), axis=1)
        #save correlation
        chan_corr = pm.Deterministic('chan_corr', corr[0, 1])
        #observed data
        y = pm.MvNormal('y', mu=mu, chol=chol, observed=df[[key_a, key_b]])

        #do some prior checks
        if do_prior_checks:
            idata = pm.sample_prior_predictive(50)

        #do the actual sampling
        if sample:
            trace = pm.sample(**sampling_kwargs)

        #do some posterior checks
        if do_posterior_checks:
            pm.sample_posterior_predictive(trace, extend_inferencedata=True,)

    if do_prior_checks:
        return idata
    else:
        return trace

#%% run the model
sampling_kwargs = {
    'draws': 500, 
    'tune': 1000, 
    'chains': 2,
    'init': 'adapt_diag',
    'target_accept': 0.95
}
#%%
brain_ica_x_age_trace = do_bayesian_correlation(df_cmb, 'brain_slope', 'age', sampling_kwargs)
az.to_netcdf(brain_ica_x_age_trace, '../results/cam_can_brain_ica_x_age_corr.ncdf')
#%%
brain_no_ica_x_age_trace = do_bayesian_correlation(df_cmb, 'brain_no_ica', 'age', sampling_kwargs)
az.to_netcdf(mdf, '../results/cam_can_brain_no_ica_x_age_corr.ncdf')
#%%
brain_ecg_x_age_trace = do_bayesian_correlation(df_cmb, 'heart_slope_mag', 'age', sampling_kwargs)
az.to_netcdf(mdf, '../results/cam_can_brain_ecg_x_age_trace_corr.ncdf')