#%% Imports
import pandas as pd
import bambi as bmb
import pymc as pm
import joblib
from pathlib import Path
import numpy as np#
import mne
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import matplotlib.patheffects as pe

import sys
sys.path.append('/mnt/obob/staff/fschmidt/cardiac_1_f')

from utils.pymc_utils import coefficients2pcorrs, aggregate_sign_feature
from utils.plot_utils import plot_ridge, plot_bayes_linear_regression

import matplotlib as mpl
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

sns.set_context('poster')
sns.set_style('ticks')

#a seed for reproducibility
import random
random.seed(42069)


brms_kwargs = {'draws': 2000,
               'tune': 2000,
               'chains': 4,
               'target_accept': 0.9,}

#%%
heart_thresh, eye_thresh = 0.4, 0.8

indir = Path('/mnt/obob/staff/fschmidt/cardiac_1_f/data/data_cam_can')
my_path_ending = f'*/*[[]0.1, 45[]]__eye_threshold_{eye_thresh}__heart_threshold_{heart_thresh}.dat'

all_files = [str(sub_path) for sub_path in indir.glob(my_path_ending) if sub_path.is_file()]
print(len(all_files))
#
#%% Load data
all_df, all_psd = [], []
for idx, file in enumerate(all_files):

    print(f'cur index is {idx}/{len(all_files)}')

    cur_data = joblib.load(file)
    
    if 'ecg_scores' in cur_data.keys() and (cur_data['ecg_scores'] > heart_thresh).sum() > 0:
    
        def make_data_meg(cur_data, key):
            cur_psd = pd.DataFrame(cur_data['psd'])
            cur_psd['channel'] = np.arange(102)
            psd_melt = cur_psd.melt(id_vars='channel')
            psd_melt['Frequency (Hz)'] = psd_melt['variable'].replace(dict(zip(np.arange(len(cur_data['freqs'])), cur_data['freqs'])))
            psd_melt.drop(labels='variable', axis=1, inplace=True)
            psd_melt.columns = ['channel', key, 'Frequency (Hz)']
            return psd_melt

        mags_df = make_data_meg(cur_data['data_brain']['mag'], 'Magnetometers (ECG removed)')
        mags_heart_df = make_data_meg(cur_data['data_heart']['mag'], 'ECG Component Magnetometers')
        mags_no_ica_df = make_data_meg(cur_data['data_no_ica']['mag'], 'Magnetometers (ECG present)')

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
                                    'age': float(cur_data['age'])}))
df_cmb = pd.concat(all_df)
#df_cmb.to_csv('../data/cam_can_1_f_dataframe_1_145.csv')
df_cmb_psd = pd.concat(all_psd)
#df_cmb_psd.to_csv('../data/cam_can_1_f_dataframe_1_145_psd.csv')

#%%
cur_df_cmb = df_cmb.query('channel == 0')

#%%get basic demographics
dem_dict = {'n_subj': len(cur_df_cmb['subject_id'].unique()),
            'mean_age': cur_df_cmb['age'].mean(),
            'std_age': cur_df_cmb['age'].std(),
            'min': cur_df_cmb['age'].min(),
            'max': cur_df_cmb['age'].max(),}

print(dem_dict)

#%% Check average psds across subjects
avg_psd = df_cmb_psd.groupby(['Frequency (Hz)', 'channel']).mean().reset_index()

#%%
my_freqs = '_1_145'

fig, ax = plt.subplots()
sns.lineplot(x='Frequency (Hz)', y='Magnetometers (ECG removed)',
             hue="channel",
             data=avg_psd, ax=ax)
fig.set_size_inches(6,6)
plt.xlim(0.5, 145)
plt.xscale('log')
plt.yscale('log')
fig.savefig(f'../results/mags_ga_ecg_removed{my_freqs}.svg')
#%%
fig, ax = plt.subplots()
sns.lineplot(x='Frequency (Hz)', y='ECG Component Magnetometers',
             hue="channel",
             data=avg_psd, ax=ax)
fig.set_size_inches(6,6)
plt.xlim(0.5, 145)
plt.xscale('log')
plt.yscale('log')
fig.savefig(f'../results/mags_ga_ecg_component{my_freqs}.svg')
#%%
fig, ax = plt.subplots()
sns.lineplot(x='Frequency (Hz)', y='Magnetometers (ECG present)',
             hue="channel",
             data=avg_psd, ax=ax)
fig.set_size_inches(6,6)
plt.xlim(0.5, 145)
plt.xscale('log')
plt.yscale('log')
fig.savefig(f'../results/mags_ga_ecg_present{my_freqs}.svg')


#%% slope ecg not rejected
md_no_ica = bmb.Model(data=cur_df_cmb, 
                    formula='no_ica_slope_avg ~ 1 + age', 
                    dropna=True,
                    family='t'
                    )
md_no_ica.build()

mdf_no_ica = md_no_ica.fit(**brms_kwargs)
sum_no_ica = az.summary(mdf_no_ica)
md_no_ica.predict(mdf_no_ica) # needed for plotting
md_no_ica.predict(mdf_no_ica, kind='pps') #do posterior pred checks

#%%

g_ecg_present = plot_bayes_linear_regression(df=cur_df_cmb, fitted=mdf_no_ica, 
                                          line_color='#8da0cb',
                                          x_key='age', y_key='no_ica_slope_avg',
                                          add_ppm=True)
g_ecg_present.figure.set_size_inches(4,4)
g_ecg_present.set_xlabel('age (years)')
g_ecg_present.set_ylabel('1/f slope')
g_ecg_present.set_ylim(0.5, 2.5)
sns.despine()
#g_ecg_present.figure.savefig(f'../results/pred_no_ica_avg_{my_freqs}.svg')

#%% ecg component only
md_comp_ecg = bmb.Model(data=cur_df_cmb, 
                    formula='heart_slope_avg ~ 1 + age', 
                    dropna=True,
                    family='t')
md_comp_ecg.build()

mdf_comp_ecg = md_comp_ecg.fit(**brms_kwargs)
sum_comp_ecg = az.summary(mdf_comp_ecg)

md_comp_ecg.predict(mdf_comp_ecg) # needed for plotting
md_comp_ecg.predict(mdf_comp_ecg, kind='pps') #do posterior pred checks

#az.plot_ppc(mdf_comp_ecg)

g_ecg_comp = plot_bayes_linear_regression(df=cur_df_cmb, fitted=mdf_comp_ecg, 
                                          line_color='#66c2a5',
                                          x_key='age', y_key='heart_slope_avg',
                                          add_ppm=True)
g_ecg_comp.figure.set_size_inches(4,4)
g_ecg_comp.set_xlabel('age (years)')
g_ecg_comp.set_ylabel('1/f slope')
g_ecg_comp.set_ylim(0.5, 2.5)
sns.despine()
#g_ecg_comp.figure.savefig(f'../results/pred_heart_slope_avg_{my_freqs}.svg', )

#%% check total variance explained by ecg components and average amount of components extracted
n_comp = cur_df_cmb['n_components'].mean()
n_comp_std = cur_df_cmb['n_components'].std()

print(f'a total of {n_comp} components (STD: {n_comp_std}) were rejected')

#%% slope ecg rejected
md_ica = bmb.Model(data=cur_df_cmb, 
                    formula='brain_slope_avg ~ 1 + age', 
                    dropna=True,
                    family='t'
                    )
md_ica.build()

mdf_ica = md_ica.fit(**brms_kwargs)
sum_ica = az.summary(mdf_ica, round_to=None)

md_ica.predict(mdf_ica) # needed for plotting
md_ica.predict(mdf_ica, kind='pps') #do posterior pred checks

#az.plot_ppc(mdf_ica)
#%
g_ica_comp = plot_bayes_linear_regression(df=cur_df_cmb, fitted=mdf_ica, 
                                          line_color='#fc8d62',
                                          x_key='age', y_key='brain_slope_avg',
                                          add_ppm=True)
g_ica_comp.figure.set_size_inches(4,4)
g_ica_comp.set_xlabel('age (years)')
g_ica_comp.set_ylabel('1/f slope')
g_ica_comp.set_ylim(0.5, 2.5)
sns.despine()
#g_ica_comp.figure.savefig(f'../results/pred_brain_slope_avg_{my_freqs}.svg', )

#%% refit all models with standardized betas to make them comparable
cur_df_z = zscore(cur_df_cmb[['heart_slope_avg', 'brain_slope_avg', 'no_ica_slope_avg', 'age']], axis=0)


mdf_comp_ecg_z = bmb.Model(data=cur_df_z, 
                    formula='heart_slope_avg ~ 1 + age', 
                    dropna=True,
                    family='t').fit(**brms_kwargs)

mdf_no_ica_z = bmb.Model(data=cur_df_z, 
                    formula='no_ica_slope_avg ~ 1 + age', 
                    dropna=True,
                    family='t').fit(**brms_kwargs)

mdf_ica_z = bmb.Model(data=cur_df_z, 
                    formula='brain_slope_avg ~ 1 + age', 
                    dropna=True,
                    family='t').fit(**brms_kwargs)

#%% add standardized betas as stats
az.summary(mdf_no_ica_z)
#%%
az.summary(mdf_ica_z)
#%%
az.summary(mdf_comp_ecg_z)

#%%
df2density = pd.DataFrame({'ECG not rejected': mdf_no_ica_z.posterior['age'].to_numpy().flatten(),
              'ECG rejected': mdf_ica_z.posterior['age'].to_numpy().flatten(),
              'ECG component': mdf_comp_ecg_z.posterior['age'].to_numpy().flatten(),
             })

df2density_tidy = df2density.melt()

#%%
(df2density['ECG not rejected'] > df2density['ECG rejected']).mean()


#%%
sns.set_style('ticks')
sns.set_context('poster')

my_colors = ['#8da0cb', '#fc8d62', '#66c2a5']

g = plot_ridge(df2density_tidy, 'variable', 'value', pal=my_colors, aspect=5, xlim=(-0.1, .5), height=0.6)
g.set_xlabels('β (standardized)')
g.figure.savefig('../results/beta_comp_cam_can.svg')

#%% plot age distribution
fig, ax = plt.subplots(figsize=(4, 4))
cur_df_cmb['age'].plot(kind='hist', color='#777777', density=True)
cur_df_cmb['age'].plot(kind='kde', color='#990F02')
ax.set_xlabel('age (years)')
ax.set_ylabel('Density')
ax.set_xlim(0, 100)
sns.despine()
ax.figure.savefig('../results/age_dist_cam_can.svg')

#%% run combined model
# note i am only using heart components and meg data without heart components here.
# as both originate from the no ica data it makes no sense adding this as additional predictor
with pm.Model() as multi_regression:

    # set some more or less informativ priors
    b0 = pm.Normal("Intercept", 50, 20)
    b1 = pm.Normal('heart_slope_avg', 0, 10)
    b2 = pm.Normal('brain_slope_avg', 0, 10)
    sigma = pm.HalfCauchy("sigma", beta=2.5)

    #regression
    mu = (b0 + b1 * cur_df_cmb['heart_slope_avg'].values 
             + b2 * cur_df_cmb['brain_slope_avg'].values)

    # likelihood -> we are predicting age here with is uncommon but gives us the chance to control for contributions of different predictors
    y = pm.TruncatedNormal('age', mu=mu, lower=18, upper=90, sigma=sigma, 
                            observed=cur_df_cmb['age'].values)

    mdf = pm.sample(**brms_kwargs)


#%%
with multi_regression:
   pm.sample_posterior_predictive(mdf, extend_inferencedata=True)

az.plot_ppc(mdf)            

#%%
az.summary(mdf)

# %% plot complete posterior distribution
slope_brain = mdf.posterior['brain_slope_avg'].values.flatten()
slope_heart = mdf.posterior['heart_slope_avg'].values.flatten()
intercept = mdf.posterior["Intercept"].stack(draws=("chain", "draw")).values

sns.set_context('poster')
sns.set_style('ticks')

x_range = np.array([0.5, 2.5])
#np.array([df_heart_brain['heart_slope_mag'].min(), df_heart_brain['heart_slope_mag'].max()])
#brain_min_max = np.array([df_heart_brain['brain_slope'].min(), df_heart_brain['brain_slope'].max()])

fig = plt.figure()

plt.plot(x_range, intercept.mean() + slope_heart.mean() * x_range, color='#66c2a5',
         path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()])
plt.plot(x_range, intercept + slope_heart * x_range[:, None], 
          color='#66c2a5', zorder=1, lw=0.1, alpha=0.1)

plt.plot(x_range, intercept.mean() + slope_brain.mean() * x_range, color='#FC8D62',
        path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()])
plt.plot(x_range, intercept + slope_brain * x_range[:, None], 
          color='#FC8D62', zorder=1, lw=0.1, alpha=0.1)

plt.xlabel("1/f slope")
plt.ylabel("age (years)");
sns.despine()

fig.set_size_inches(4,4)
fig.savefig('../results/age_pred_slope_ecg_vs_ica_all_sens.svg')

#%% visualize partial correlation
pcorrs = coefficients2pcorrs(df4mdf=cur_df_cmb, mdf=mdf, response_var='age', predictor_vars=['heart_slope_avg', 'brain_slope_avg'])

pal = ['#66c2a5', '#fc8d62']

g = plot_ridge(pcorrs, 'predictors', 'partial correlation coefficient', pal=pal, aspect=5, xlim=(-0.2, .6))

g.figure.savefig('../results/partial_corr_heart_brain_age_all_sens.svg')


# %% run models across significant channels

#% throw in eelbrain
pos_mask_heart = pd.read_csv('../results/cam_can_pred_bayes.csv')['pos_heart'].to_numpy()
pos_mask_no_ica = pd.read_csv('../results/cam_can_pred_bayes.csv')['pos_no_ica'].to_numpy()
pos_mask_brain = pd.read_csv('../results/cam_can_pred_bayes.csv')['pos_ica'].to_numpy()

df2agg_heart = df_cmb[['heart_slope_mag', 'subject_id', 'channel', 'age']].reset_index()
df2agg_no_ica = df_cmb[['brain_no_ica', 'subject_id', 'channel', 'age']].reset_index()
df2agg_brain = df_cmb[['brain_slope', 'subject_id', 'channel', 'age']].reset_index()

df_heart = aggregate_sign_feature(df2agg_heart, 'heart_slope_mag', pos_mask_heart)
df_no_ica = aggregate_sign_feature(df2agg_no_ica, 'brain_no_ica', pos_mask_no_ica)
df_brain = aggregate_sign_feature(df2agg_brain, 'brain_slope', pos_mask_brain)

#%%
df_brain.drop(columns='age', inplace=True)
df_heart_brain = df_heart.merge(df_brain, on='subject_id')

#%%
with pm.Model() as regression:

    # set some more or less informativ priors
    b0 = pm.Normal("Intercept", 50, 20)
    b1 = pm.Normal('heart_slope_mag', 0, 10)
    b2 = pm.Normal('brain_slope', 0, 10)
    sigma = pm.HalfCauchy("sigma", beta=2.5)

    #regression
    mu = (b0 + b1 * df_heart_brain['heart_slope_mag'].values 
             + b2 * df_heart_brain['brain_slope'].values)

    # likelihood
    y = pm.TruncatedNormal('age', mu=mu, lower=18, upper=90, sigma=sigma, 
                            observed=df_heart_brain['age'].values)

    mdf = pm.sample(**brms_kwargs)

#%%
with regression:
   pm.sample_posterior_predictive(mdf,  extend_inferencedata=True)

az.plot_ppc(mdf)

#%%
pcorrs = coefficients2pcorrs(df4mdf=df_heart_brain, mdf=mdf, response_var='age', predictor_vars=['heart_slope_mag', 'brain_slope'])

#%%
import matplotlib.pyplot as plt
pal = ['#66c2a5', '#fc8d62']

g = plot_ridge(pcorrs, 'predictors', 'partial correlation coefficient', pal=pal, aspect=5, xlim=(-0.1, .6))

g.figure.savefig('../results/partial_corr_heart_brain_age.svg')
 # %%

sns.set_style('ticks')
sns.set_context('talk')

slope_brain = mdf.posterior['brain_slope'].values.flatten()
slope_heart = mdf.posterior['heart_slope_mag'].values.flatten()
intercept = mdf.posterior["Intercept"].stack(draws=("chain", "draw")).values

x_range = np.array([0.5, 2.5])
#np.array([df_heart_brain['heart_slope_mag'].min(), df_heart_brain['heart_slope_mag'].max()])
#brain_min_max = np.array([df_heart_brain['brain_slope'].min(), df_heart_brain['brain_slope'].max()])

fig = plt.figure()

plt.plot(x_range, intercept.mean() + slope_heart.mean() * x_range, color='#66c2a5',
         path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()])
plt.plot(x_range, intercept + slope_heart * x_range[:, None], 
          color='#66c2a5', zorder=1, lw=0.1, alpha=0.1)

plt.plot(x_range, intercept.mean() + slope_brain.mean() * x_range, color='#FC8D62',
        path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()])
plt.plot(x_range, intercept + slope_brain * x_range[:, None], 
          color='#FC8D62', zorder=1, lw=0.1, alpha=0.1)

plt.xlabel("1/f slope")
plt.ylabel("age (years)");
sns.despine()

fig.set_size_inches(4,4)
fig.savefig('../results/age_pred_slope_ecg_vs_ica.svg')
# %%
az.summary(mdf)
# %% see whether exponent is actually reduced after removing 1/f
def plot_topo(corr, mask, info, title, vmin=None, vmax=None, cmap='RdBu_r'):
    
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
df_brain_exp = df_cmb[['brain_slope', 'channel', 'subject_id']].pivot(columns='channel', index='subject_id')['brain_slope']
df_no_ica_exp = df_cmb[['brain_no_ica', 'channel', 'subject_id']].pivot(columns='channel', index='subject_id')['brain_no_ica']
#%%

ch_list = []

for cur_ch in df_cmb['channel'].unique():

    print(f'cur channel is: {cur_ch}')

    cur_df2test = pd.DataFrame({'no_ica' : df_no_ica_exp[cur_ch].to_numpy(),
                                'brain' : df_brain_exp[cur_ch].to_numpy(),
                                'subject_id': df_brain_exp[cur_ch]
                                }).melt(id_vars='subject_id')

    cur_mdf_diff = bmb.Model('value ~ 1 + variable + (1|subject_id)', cur_df2test).fit()

    cur_ch_df = pd.DataFrame(az.summary(cur_mdf_diff).loc['variable[no_ica]']).T
    cur_ch_df['channel'] = cur_ch
    ch_list.append(cur_ch_df)
# %%
contrast_exp_df = pd.concat(ch_list).reset_index()


# %%
mask4diff = np.logical_xor((contrast_exp_df['hdi_3%'] > 0).to_numpy(),
                           (contrast_exp_df['hdi_97%'] < 0).to_numpy())
# %%
topo_ica_no_ica = plot_topo(corr=contrast_exp_df['mean'].to_numpy(), 
          mask=mask4diff, info=info_mags, title='', vmin=None, vmax=None, cmap='RdBu_r');
topo_ica_no_ica.figure.savefig('../results/topo_ica_no_ica_exp_diff.svg')
# %%
