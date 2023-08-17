#%% Imports
import pandas as pd
import bambi as bmb
import pymc as pm
import joblib
from pathlib import Path
import numpy as np
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
from utils.data_loading import data_loader

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
peaks = 2
knee = True
irasa=False
sss = False
get_psd = False
freqs = [0.1, 145]
path2data = Path('/mnt/obob/staff/fschmidt/cardiac_1_f/data/data_cam_can')

#%%
df_cmb = data_loader(path2data=path2data, freqs=[0.1, 145], 
                     heart_thresh=heart_thresh, eye_thresh=eye_thresh,
                     fit_knee=knee, peaks=peaks, interpolate=True, sss=sss, get_psd=get_psd)


#%%
df_cmb.to_csv('../data/cam_can_1_f_dataframe_1_145.csv')

#%% plot 
df_tidy_r2 = df_cmb.query('channel == 0')[['r2_no_ica', 'r2_ica', 'r2_heart']].melt()
df_tidy_r2.columns = ['condition', 'R^2']
df_tidy_r2.replace(to_replace={'r2_no_ica': 'ECG not rejected', 
                               'r2_ica': 'ECG rejected', 
                               'r2_heart': 'ECG components'}, inplace=True)


my_colors = ['#8da0cb', '#fc8d62', '#66c2a5']

sns.set_context('poster')
sns.set_style('ticks')

g = sns.stripplot(data=df_tidy_r2, 
            y='condition', x='R^2', #order=plot_order,
            hue='condition', size=10, alpha=0.025,
            palette=my_colors,
            )
g = sns.pointplot(data=df_tidy_r2, 
            y='condition', x='R^2', #order=plot_order,
            markers="+", scale=1.7,
            hue='condition',
            palette=my_colors,
            )
g.legend_.remove()

g.set_ylabel('')
g.set_xlabel('R$^2$ (Fooof)')
g.figure.set_size_inches(5,4)
sns.despine()

g.figure.savefig(f'../results/model_fit_fooof_sss_{sss}_knee_{knee}.svg')


#%%
if get_psd:

    df_cmb_psd.to_csv(f'../data/cam_can_1_f_dataframe_1_145_psd_{sss}.csv')

    #%
    df_cmb_psd['age'] = df_cmb_psd['age'].astype(float)
    psd_age2plot = df_cmb_psd.groupby(['Frequency (Hz)', 'subject_id']).mean().reset_index()


    psd_age2plot['young'] = psd_age2plot['age'] < psd_age2plot['age'].median()



    #%
    psd_age2plot.columns = ['Frequency (Hz)', 'subject_id', 'channel', 'ECG rejected',
        'ECG components', 'ECG not rejected', 'ECG Electrode', 'age', 'young']

    #%
    age2plot_tidy = psd_age2plot[['Frequency (Hz)', 'ECG not rejected', 'ECG rejected',
                                'ECG components', 'age', 'young', 'subject_id']].melt(id_vars=['Frequency (Hz)', 'age', 'subject_id', 'young'], 
                                                        var_name='condition',
                                                        value_name='Power (fT$^2$/Hz)')


    #%
    sns.set_style('ticks')
    sns.set_context('poster')

    g = sns.relplot(data=age2plot_tidy,  x='Frequency (Hz)', y='Power (fT$^2$/Hz)',
                    hue='young', kind='line', col='condition', palette='deep')
    g.axes[0,0].set_yscale('log')
    g.axes[0,0].set_xscale('log')

    g.figure.savefig(f'../results/raw_spectra_age_split_cam_can_{sss}.svg')

    #% Check average psds across subjects
    avg_psd = df_cmb_psd.groupby(['Frequency (Hz)', 'channel']).mean().reset_index()

    #%
    avg_psd.columns = ['Frequency (Hz)', 'channel', 'ECG rejected',
                    'ECG components', 'ECG not rejected', 'age']

    avg_psd_tidy = avg_psd[['Frequency (Hz)', 'channel', 'ECG not rejected', 'ECG rejected',
                            'ECG components', 'age']].melt(id_vars=['Frequency (Hz)', 'channel', 'age'], var_name='condition', value_name='Power (fT$^2$/Hz)')

    #%
    sns.set_style('ticks')
    sns.set_context('poster')

    g = sns.relplot(data=avg_psd_tidy,  x='Frequency (Hz)', y='Power (fT$^2$/Hz)', hue='channel', kind='line', col='condition')
    g.axes[0,0].set_yscale('log')
    g.axes[0,0].set_xscale('log')

    g.figure.savefig(f'../results/aperiodic_slope_channel_split_cam_can_sss_{sss}_knee_{knee}.svg')

    #%
    my_freqs = '_1_145'

    fig, ax = plt.subplots()
    sns.lineplot(x='Frequency (Hz)', y='Magnetometers (ECG removed)',
                hue="channel",
                data=avg_psd, ax=ax)
    fig.set_size_inches(6,6)
    plt.xlim(0.5, 145)
    plt.xscale('log')
    plt.yscale('log')
    fig.savefig(f'../results/mags_ga_ecg_removed{my_freqs}_sss_{sss}_knee_{knee}.svg')
    #%
    fig, ax = plt.subplots()
    sns.lineplot(x='Frequency (Hz)', y='ECG Component Magnetometers',
                hue="channel",
                data=avg_psd, ax=ax)
    fig.set_size_inches(6,6)
    plt.xlim(0.5, 145)
    plt.xscale('log')
    plt.yscale('log')
    fig.savefig(f'../results/mags_ga_ecg_component{my_freqs}_sss_{sss}_knee_{knee}.svg')
    #%
    fig, ax = plt.subplots()
    sns.lineplot(x='Frequency (Hz)', y='Magnetometers (ECG present)',
                hue="channel",
                data=avg_psd, ax=ax)
    fig.set_size_inches(6,6)
    plt.xlim(0.5, 145)
    plt.xscale('log')
    plt.yscale('log')
    fig.savefig(f'../results/mags_ga_ecg_present{my_freqs}_sss_{sss}_knee_{knee}.svg')



#%%
cur_df_cmb = df_cmb.query('channel == 0')


def print_mdf_fooof(key):
    avg_fooof = cur_df_cmb[key].mean()
    std_fooof = cur_df_cmb[key].std()

    print(f'The R2 was {avg_fooof} on average. SD = {std_fooof}')

#%%
print_mdf_fooof('r2_ica')
#%%
print_mdf_fooof('r2_no_ica')
#%%
print_mdf_fooof('r2_heart')


#%% slope 2 exponent for lazy people
cur_df_cmb[['heart_slope_avg', 'brain_slope_avg', 'no_ica_slope_avg']] *= -1


#%% refit all models with standardized betas to make them comparable
cur_df_z = zscore(cur_df_cmb[['heart_slope_avg', 'brain_slope_avg', 'no_ica_slope_avg', 'age']].dropna(), axis=0)


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

#%%get basic demographics
dem_dict = {'n_subj': len(cur_df_cmb['subject_id'].unique()),
            'mean_age': cur_df_cmb['age'].mean(),
            'std_age': cur_df_cmb['age'].std(),
            'min': cur_df_cmb['age'].min(),
            'max': cur_df_cmb['age'].max(),}

print(dem_dict)


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
g_ecg_present.set_ylim(-2.5, -.5)
sns.despine()
g_ecg_present.figure.savefig(f'../results/pred_no_ica_avg_cam_can_sss_{sss}_knee_{knee}.svg')

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
#%%
g_ecg_comp = plot_bayes_linear_regression(df=cur_df_cmb, fitted=mdf_comp_ecg, 
                                          line_color='#66c2a5',
                                          x_key='age', y_key='heart_slope_avg',
                                          add_ppm=True)
g_ecg_comp.figure.set_size_inches(4,4)
g_ecg_comp.set_xlabel('age (years)')
g_ecg_comp.set_ylabel('1/f slope')
g_ecg_comp.set_ylim(-2.5, -0.5)
sns.despine()
g_ecg_comp.figure.savefig(f'../results/pred_heart_slope_avg_cam_can_sss_{sss}_knee_{knee}.svg', )

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
#%%
g_ica_comp = plot_bayes_linear_regression(df=cur_df_cmb, fitted=mdf_ica, 
                                          line_color='#fc8d62',
                                          x_key='age', y_key='brain_slope_avg',
                                          add_ppm=True)
g_ica_comp.figure.set_size_inches(4,4)
g_ica_comp.set_xlabel('age (years)')
g_ica_comp.set_ylabel('1/f slope')
g_ica_comp.set_ylim(-2.5, -0.5)
sns.despine()
g_ica_comp.figure.savefig(f'../results/pred_brain_slope_avg_cam_can_sss_{sss}_knee_{knee}.svg', )


#%%
df2density = pd.DataFrame({'ECG not rejected': mdf_no_ica_z.posterior['age'].to_numpy().flatten(),
              'ECG rejected': mdf_ica_z.posterior['age'].to_numpy().flatten(),
              'ECG component': mdf_comp_ecg_z.posterior['age'].to_numpy().flatten(),
             })

df2density_tidy = df2density.melt()


#%%
sns.set_style('ticks')
sns.set_context('poster')

my_colors = ['#8da0cb', '#fc8d62', '#66c2a5']

g = plot_ridge(df2density_tidy, 'variable', 'value', pal=my_colors, aspect=5, xlim=(-0.5, .1), height=0.6)
g.set_xlabels('β (standardized)')
g.figure.savefig(f'../results/beta_comp_cam_can_sss_{sss}_knee_{knee}.svg')

#%% plot age distribution

sns.set_style('ticks')
sns.set_context('poster')

fig, ax = plt.subplots(figsize=(4, 4))
cur_df_cmb['age'].plot(kind='hist', color='#777777', density=True)
cur_df_cmb['age'].plot(kind='kde', color='#990F02')
ax.set_xlabel('age (years)')
ax.set_ylabel('Density')
ax.set_xlim(0, 100)
sns.despine()
ax.figure.savefig(f'../results/age_dist_cam_can_sss_{sss}_knee_{knee}.svg')

