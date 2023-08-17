#%%
from pathlib import Path
import sys
sys.path.append('/mnt/obob/staff/fschmidt/cardiac_1_f')
from utils.data_loading import data_loader
import bambi as bmb
from scipy.stats import zscore
import arviz as az
import pingouin as pg
import pandas as pd
import joblib
import numpy as np

import pymc as pm
import matplotlib.patheffects as pe

from utils.pymc_utils import coefficients2pcorrs, aggregate_sign_feature_irasa
from utils.plot_utils import plot_ridge, plot_bayes_linear_regression


import matplotlib as mpl
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('poster')
sns.set_style('ticks')




brms_kwargs = {'draws': 2000,
               'tune': 2000,
               'chains': 4,
               'target_accept': 0.9,}


#show that..
#Exponents across frequency ranges without knee are highly related
#show grand average fooof fits (per channel as in the original spectra)

#[[]{freqs[0]}, {freqs[1]}[]]

#%%
heart_thresh, eye_thresh = 0.4, 0.8
path2data = Path('/mnt/obob/staff/fschmidt/cardiac_1_f/data/data_cam_can_irasa_final')
irasa = True
sss = False
fit_slope_ranges=False
get_psd = False

#%%
my_path_ending = f'*/*__eye_threshold_{eye_thresh}__heart_threshold_{heart_thresh}__irasa_{irasa}__sss_{sss}__interpolate_True.dat'

all_files = [str(sub_path) for sub_path in path2data.glob(my_path_ending) if sub_path.is_file()]
print(len(all_files))
# %%
df_list, df_list_cmb, meg_list, meg_list_p = [], [], [], []

for file in all_files:

    cur_data = joblib.load(file)
    
    meg_idcs = [True if 'MEG' in chan else False for chan in cur_data['data_heart']['fit_params']['Chan']]
    ecg_idcs = [True if 'ECG' in chan else False for chan in cur_data['data_heart']['fit_params']['Chan']]

    #% get meg data
    cur_meg = pd.DataFrame({'ECG_not_rejected': cur_data['data_no_ica']['aperiodic'][meg_idcs,:].mean(axis=0),
                            'ECG_rejected': cur_data['data_brain']['aperiodic'][meg_idcs,:].mean(axis=0),
                            'ECG_components': cur_data['data_heart']['aperiodic'][meg_idcs,:].mean(axis=0),
                            'Frequency(Hz)': cur_data['data_brain']['freqs'],})

    cur_meg['age'] = float(cur_data['age'])
    cur_meg['subject_id'] = cur_data['subject_id']


    meg_list.append(cur_meg)
    max_freqs = cur_data['data_brain']['freqs'].shape[0]


    cur_meg_p = pd.DataFrame({'ECG_not_rejected': np.mean(cur_data['data_no_ica']['raw_spectra'][meg_idcs,:max_freqs], axis=0),
                            'ECG_rejected': np.mean(cur_data['data_brain']['raw_spectra'][meg_idcs,:max_freqs], axis=0),
                            'ECG_components': np.mean(cur_data['data_heart']['raw_spectra'][meg_idcs,:max_freqs], axis=0),
                            'Frequency(Hz)': cur_data['data_brain']['freqs'],})

    cur_meg_p['age'] = float(cur_data['age'])
    cur_meg_p['subject_id'] = cur_data['subject_id']

    meg_list_p.append(cur_meg_p)
    

    #% extract slopes in 10hz steps
    freqs = cur_meg['Frequency(Hz)'].to_numpy()

    lower_freqs = np.arange(1,11) - 0.5
    upper_freqs = np.arange(4, 15, 0.5) * 10

    #% average scores
    df = pd.concat([cur_data['data_no_ica']['fit_params'][meg_idcs].mean(),
                    cur_data['data_brain']['fit_params'][meg_idcs].mean(),
                    cur_data['data_heart']['fit_params'][meg_idcs].mean(),
                    cur_data['data_heart']['fit_params'][ecg_idcs][['Intercept', 'Slope', 'R^2', 'std(osc)']].T], 
                    axis=1).T

    df['condition'] = ['ECG_not_rejected', 'ECG_rejected', 'ECG_components', 'heart']
    df['age'] = float(cur_data['age'])
    df['subject_id'] = cur_data['subject_id']
    df_list_cmb.append(df)


    #% non averages
    no_ica = cur_data['data_no_ica']['fit_params'][meg_idcs].copy()
    no_ica['condition'] = 'ECG_not_rejected'

    ica = cur_data['data_brain']['fit_params'][meg_idcs].copy()
    ica['condition'] = 'ECG_rejected'

    ecg = cur_data['data_heart']['fit_params'][meg_idcs].copy()
    ecg['condition'] = 'ECG_components'


    df_all = pd.concat([no_ica,
                        ica,
                        ecg,])

    df_all['age'] = float(cur_data['age'])
    df_all['subject_id'] = cur_data['subject_id']

    all_ch_names_map = dict(zip(no_ica['Chan'].unique(), np.arange(len(no_ica['Chan'].unique()))))

    df_all.replace(all_ch_names_map, inplace=True)
    df_list.append(df_all)
#%% plot raw slope aging differences
df_meg = pd.concat(meg_list)
df_meg_p = pd.concat(meg_list_p)
df_meg.columns = ['ECG not rejected', 'ECG rejected', 'ECG components', 'Frequency (Hz)',
                  'age', 'subject_id']

df_meg_p.columns = ['ECG not rejected', 'ECG rejected', 'ECG components', 'Frequency (Hz)',
                  'age', 'subject_id']

median_age = np.median(df_meg['age'])

df_meg['young'] = df_meg['age'] < median_age
df_meg_p['young'] = df_meg_p['age'] < median_age

df_meg_tidy = df_meg.melt(id_vars=['Frequency (Hz)', 'age', 'subject_id', 'young'], var_name='condition', value_name='Power (fT$^2$/Hz)')
df_meg_tidy_p = df_meg_p.melt(id_vars=['Frequency (Hz)', 'age', 'subject_id', 'young'], var_name='condition', value_name='Power')

#%% run some periodic checks

df_brain = df_meg_tidy_p.query('condition == "ECG rejected"')

#%%

sns.set_style('ticks')
sns.set_context('poster')

g = sns.relplot(data=df_meg_tidy,  x='Frequency (Hz)', y='Power (fT$^2$/Hz)', hue='young', kind='line', col='condition', palette='deep')
g.axes[0,0].set_yscale('log')
#g.axes[0,0].set_xlim(0, 45)
g.axes[0,0].set_xscale('log')

g.figure.savefig(f'../results/aperiodic_slope_age_split_cam_can_irasa_sss_{sss}.svg')


#%%
sns.set_style('ticks')
sns.set_context('poster')
#df_meg_tidy_p['Power'][df_meg_tidy_p['Power'] < 0] = 0

cmap = ['#E7298A', '#E6AB02']

alpha_bool =np.logical_and(df_meg_tidy_p['Frequency (Hz)'] <= 14, df_meg_tidy_p['Frequency (Hz)'] >= 5)

g = sns.relplot(data=df_meg_tidy_p,  x='Frequency (Hz)', y='Power', hue='young', kind='line', col='condition', palette=cmap)
g.axes[0,0].set_yscale('log')
g.axes[0,0].set_xscale('log')

g.figure.savefig(f'../results/raw_spectra_age_split_cam_can_irasa_sss_{sss}.svg')


# %%
df_cmb = pd.concat(df_list_cmb).reset_index()
df_cmb_meg = df_cmb.query('condition != "heart"')

#%% compare irasa and fooof estimates
cols_of_interest = ['heart_slope_avg', 'brain_slope_avg', 'no_ica_slope_avg', 'age', 'subject_id']
df_fooof = pd.read_csv('../data/cam_can_1_f_dataframe_1_145.csv').query('channel == 0')[cols_of_interest]

df_cmb_meg_w = df_cmb_meg[['condition', 'Slope', 'subject_id']].pivot_table(values='Slope', index=['subject_id'], columns='condition').reset_index()


df_irasa_fooof = df_cmb_meg_w.merge(df_fooof, on='subject_id')


#%%
cur_df = df_cmb_meg.query('condition == "heart"')

#%%

df_irasa_fooof.columns = ['subject_id', 'ECG_components_IRASA', 'ECG_not_rejected_IRASA', 'ECG_rejected_IRASA',
                          'ECG_components_FOOOF', 'ECG_rejected_FOOOF', 'ECG_not_rejected_FOOOF', 'age']

df_irasa_fooof_z = zscore(df_irasa_fooof[['ECG_components_IRASA', 'ECG_not_rejected_IRASA', 'ECG_rejected_IRASA', 
                                        'ECG_components_FOOOF', 'ECG_rejected_FOOOF', 'ECG_not_rejected_FOOOF', 'age']].dropna(), axis=0)


#%%
mdf_comp_ecg_z = bmb.Model(data=df_irasa_fooof_z, 
                    formula='ECG_components_FOOOF ~ 1 + ECG_components_IRASA', 
                    dropna=True,).fit(**brms_kwargs)
                    #family='t').fit(**brms_kwargs)

mdf_no_ica_z = bmb.Model(data=df_irasa_fooof_z, 
                    formula='ECG_not_rejected_FOOOF ~ 1 + ECG_not_rejected_IRASA', 
                    dropna=True,).fit(**brms_kwargs)
                    #family='t').fit(**brms_kwargs)

mdf_ica_z = bmb.Model(data=df_irasa_fooof_z, 
                    formula='ECG_rejected_FOOOF ~ 1 + ECG_rejected_IRASA', 
                    dropna=True,).fit(**brms_kwargs)
                    #family='t').fit(**brms_kwargs)

#%% add standardized betas as stats
az.summary(mdf_no_ica_z)
#%%
az.summary(mdf_ica_z)
#%%
az.summary(mdf_comp_ecg_z)


#%% slope ecg not rejected
md_no_ica = bmb.Model(data=df_irasa_fooof, 
                    formula='ECG_not_rejected_FOOOF ~ 1 + ECG_not_rejected_IRASA', 
                    dropna=True,
                    #family='t'
                    )
md_no_ica.build()

mdf_no_ica = md_no_ica.fit(**brms_kwargs)
sum_no_ica = az.summary(mdf_no_ica)
md_no_ica.predict(mdf_no_ica) # needed for plotting
md_no_ica.predict(mdf_no_ica, kind='pps') #do posterior pred checks

#%%

g_ecg_present = plot_bayes_linear_regression(df=df_irasa_fooof, fitted=mdf_no_ica, 
                                          line_color='#8da0cb',
                                          x_key='ECG_not_rejected_IRASA', y_key='ECG_not_rejected_FOOOF',
                                          add_ppm=True)
g_ecg_present.figure.set_size_inches(4,4)
g_ecg_present.set_xlabel('ECG not rejected (IRASA)')
g_ecg_present.set_ylabel('ECG not rejected (FOOOF)')
g_ecg_present.set_ylim(.5, 2.5)
g_ecg_present.set_xlim(-2.5, -0.5)
sns.despine()
g_ecg_present.figure.savefig(f'../results/pred_no_ica_fooof_irasa_sss_{sss}.svg')

#%% ecg component only
md_comp_ecg = bmb.Model(data=df_irasa_fooof, 
                    formula='ECG_components_FOOOF ~ 1 + ECG_components_IRASA', 
                    dropna=True,
                    #family='t',
                    )
md_comp_ecg.build()

mdf_comp_ecg = md_comp_ecg.fit(**brms_kwargs)
sum_comp_ecg = az.summary(mdf_comp_ecg)

md_comp_ecg.predict(mdf_comp_ecg) # needed for plotting
md_comp_ecg.predict(mdf_comp_ecg, kind='pps') #do posterior pred checks

#az.plot_ppc(mdf_comp_ecg)
#%%
g_ecg_comp = plot_bayes_linear_regression(df=df_irasa_fooof, fitted=mdf_comp_ecg, 
                                          line_color='#66c2a5',
                                          x_key='ECG_components_IRASA', y_key='ECG_components_FOOOF',
                                          add_ppm=True)
g_ecg_comp.figure.set_size_inches(4,4)
g_ecg_comp.set_xlabel('ECG components (IRASA)')
g_ecg_comp.set_ylabel('ECG components (FOOOF)')
g_ecg_comp.set_ylim(.5, 2.5)
g_ecg_comp.set_xlim(-2.5, -0.5)
sns.despine()
g_ecg_comp.figure.savefig(f'../results/pred_ecg_slope_fooof_irasa_sss_{sss}.svg', )


#%% slope ecg rejected
md_ica = bmb.Model(data=df_irasa_fooof, 
                    formula='ECG_rejected_FOOOF ~ 1 + ECG_rejected_IRASA', 
                    dropna=True,
                    #family='t'
                    )
md_ica.build()

mdf_ica = md_ica.fit(**brms_kwargs)
sum_ica = az.summary(mdf_ica, round_to=None)

md_ica.predict(mdf_ica) # needed for plotting
md_ica.predict(mdf_ica, kind='pps') #do posterior pred checks

#az.plot_ppc(mdf_ica)
#%%
g_ica_comp = plot_bayes_linear_regression(df=df_irasa_fooof, fitted=mdf_ica, 
                                          line_color='#fc8d62',
                                          x_key='ECG_rejected_IRASA', y_key='ECG_rejected_FOOOF',
                                          add_ppm=True)
g_ica_comp.figure.set_size_inches(4,4)
g_ica_comp.set_xlabel('ECG rejected (IRASA)')
g_ica_comp.set_ylabel('ECG rejected (FOOOF)')
g_ica_comp.set_ylim(.5, 2.5)
g_ica_comp.set_xlim(-2.5, -0.5)
sns.despine()
g_ica_comp.figure.savefig(f'../results/pred_brain_slope_fooof_irasa_sss_{sss}.svg', )

# %%
