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

from utils.pymc_utils import coefficients2pcorrs, aggregate_sign_feature
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


#%%
heart_thresh, eye_thresh = 0.4, 0.8
path2data = Path('/mnt/obob/staff/fschmidt/cardiac_1_f/data/data_sbg_irasa')
irasa = True
sss = False
fit_slope_ranges=False
#%%
my_path_ending = f'*/*__eye_threshold_{eye_thresh}__heart_threshold_{heart_thresh}__irasa_{irasa}__sss_{sss}__interpolate_True.dat'

all_files = [str(sub_path) for sub_path in path2data.glob(my_path_ending) if sub_path.is_file()]
print(len(all_files))
# %%
df_list, df_list_cmb, meg_list = [], [], []

for file in all_files:

    cur_data = joblib.load(file)
    
    meg_idcs = [True if 'MEG' in chan else False for chan in cur_data['data_heart']['fit_params']['Chan']]

    if (cur_data['data_heart']['fit_params']['Chan'] == 'ECG003').sum() > 0:
        ecg_idcs = [True if 'ECG003' in chan else False for chan in cur_data['data_heart']['fit_params']['Chan']]

    if (cur_data['data_heart']['fit_params']['Chan'] == 'BIO003').sum() > 0:
        ecg_idcs = [True if 'BIO003' in chan else False for chan in cur_data['data_heart']['fit_params']['Chan']]

    #% get meg data
    cur_meg = pd.DataFrame({'ECG_not_rejected': cur_data['data_no_ica']['aperiodic'][meg_idcs,:].mean(axis=0),
                            'ECG_rejected': cur_data['data_brain']['aperiodic'][meg_idcs,:].mean(axis=0),
                            'ECG_components': cur_data['data_heart']['aperiodic'][meg_idcs,:].mean(axis=0),
                            'Frequency(Hz)': cur_data['data_brain']['freqs'],})

    cur_meg['age'] = float(cur_data['age'])
    cur_meg['subject_id'] = cur_data['subject_id']

    meg_list.append(cur_meg)

    #% extract slopes in 10hz steps
    freqs = cur_meg['Frequency(Hz)'].to_numpy()

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

#%%

df_cmb = pd.concat(df_list_cmb).reset_index()
df_cmb_meg = df_cmb.query('condition != "heart"')

df_cmb_meg_w = df_cmb_meg[['condition', 'Slope', 'age', 'subject_id']].pivot_table(values='Slope', index=['subject_id', 'age'], columns='condition').reset_index()

#%%
cur_df_z = zscore(df_cmb_meg_w[['ECG_not_rejected', 'ECG_rejected', 'ECG_components', 'age']].dropna(), axis=0)


#%%
mdf_comp_ecg_z = bmb.Model(data=cur_df_z, 
                    formula='ECG_components ~ 1 + age', 
                    dropna=True,
                    family='t').fit(**brms_kwargs)

mdf_no_ica_z = bmb.Model(data=cur_df_z, 
                    formula='ECG_not_rejected ~ 1 + age', 
                    dropna=True,
                    family='t').fit(**brms_kwargs)

mdf_ica_z = bmb.Model(data=cur_df_z, 
                    formula='ECG_rejected ~ 1 + age', 
                    dropna=True,
                    family='t').fit(**brms_kwargs)

#%%
df2density = pd.DataFrame({'ECG not rejected': mdf_no_ica_z.posterior['age'].to_numpy().flatten(),
                           'ECG rejected': mdf_ica_z.posterior['age'].to_numpy().flatten(),
                           'ECG component': mdf_comp_ecg_z.posterior['age'].to_numpy().flatten(),
                        })

df2density_tidy = df2density.melt()
# %%

my_colors = ['#8da0cb', '#fc8d62', '#66c2a5']

g = plot_ridge(df2density_tidy, 'variable', 'value', pal=my_colors, aspect=5, xlim=(-0.5, .1), height=0.6)
g.set_xlabels('β (standardized)')
g.figure.savefig(f'../results/beta_comp_sbg_irasa_sss_{sss}.svg')

#%% plot r2 for all subjects
my_colors = ['#8da0cb', '#fc8d62', '#66c2a5']

sns.set_context('poster')
sns.set_style('ticks')

g = sns.stripplot(data=df_cmb_meg, 
            y='condition', x='R^2', #order=plot_order,
            hue='condition', size=10, alpha=0.025,
            palette=my_colors,
            )
g = sns.pointplot(data=df_cmb_meg, 
            y='condition', x='R^2', #order=plot_order,
            markers="+", scale=1.7,
            hue='condition',
            palette=my_colors,
            )
g.legend_.remove()

g.set_ylabel('')
g.set_xlabel('R$^2$ (IRASA)')
g.figure.set_size_inches(5,4)
sns.despine()

g.figure.savefig(f'../results/sbg_model_fit_irasa_sss_{sss}.svg')

#%% slope ecg not rejected
md_no_ica = bmb.Model(data=df_cmb_meg_w, 
                    formula='ECG_not_rejected ~ 1 + age', 
                    dropna=True,
                    family='t'
                    )
md_no_ica.build()

mdf_no_ica = md_no_ica.fit(**brms_kwargs)
sum_no_ica = az.summary(mdf_no_ica)
md_no_ica.predict(mdf_no_ica) # needed for plotting
md_no_ica.predict(mdf_no_ica, kind='pps') #do posterior pred checks

#%%
g = az.plot_ppc(mdf_no_ica)
g.set_xlim(-3, 0)
g.figure.savefig('../results/sbg_ppc_no_ica.svg')

#%%

g_ecg_present = plot_bayes_linear_regression(df=df_cmb_meg_w, fitted=mdf_no_ica, 
                                          line_color='#8da0cb',
                                          x_key='age', y_key='ECG_not_rejected',
                                          add_ppm=True)
g_ecg_present.figure.set_size_inches(4,4)
g_ecg_present.set_xlabel('age (years)')
g_ecg_present.set_ylabel('1/f slope')
g_ecg_present.set_ylim(-2.5, -0.5)
sns.despine()
g_ecg_present.figure.savefig(f'../results/sbg_pred_no_ica_avg_irasa_sss_{sss}.svg')

#%% ecg component only
md_comp_ecg = bmb.Model(data=df_cmb_meg_w, 
                    formula='ECG_components ~ 1 + age', 
                    dropna=True,
                    family='t')
md_comp_ecg.build()

mdf_comp_ecg = md_comp_ecg.fit(**brms_kwargs)
sum_comp_ecg = az.summary(mdf_comp_ecg)

md_comp_ecg.predict(mdf_comp_ecg) # needed for plotting
md_comp_ecg.predict(mdf_comp_ecg, kind='pps') #do posterior pred checks

#az.plot_ppc(mdf_comp_ecg)
#%%
g = az.plot_ppc(mdf_comp_ecg)
g.set_xlim(-3, 0)
g.figure.savefig('../results/sbg_ppc_ecg_comps.svg')

#%%
g_ecg_comp = plot_bayes_linear_regression(df=df_cmb_meg_w, fitted=mdf_comp_ecg, 
                                          line_color='#66c2a5',
                                          x_key='age', y_key='ECG_components',
                                          add_ppm=True)
g_ecg_comp.figure.set_size_inches(4,4)
g_ecg_comp.set_xlabel('age (years)')
g_ecg_comp.set_ylabel('1/f slope')
g_ecg_comp.set_ylim(-2.5, -0.5)
sns.despine()
g_ecg_comp.figure.savefig(f'../results/sbg_pred_ecg_slope_avg_irasa_sss_{sss}.svg', )


#%% slope ecg rejected
md_ica = bmb.Model(data=df_cmb_meg_w, 
                    formula='ECG_rejected ~ 1 + age', 
                    dropna=True,
                    family='t'
                    )
md_ica.build()

mdf_ica = md_ica.fit(**brms_kwargs)
sum_ica = az.summary(mdf_ica, round_to=None)

md_ica.predict(mdf_ica) # needed for plotting
md_ica.predict(mdf_ica, kind='pps') #do posterior pred checks

#%%
g = az.plot_ppc(mdf_ica)
g.set_xlim(-3, 0)
g.figure.savefig('../results/sbg_ppc_ica.svg')


#az.plot_ppc(mdf_ica)
#%%
g_ica_comp = plot_bayes_linear_regression(df=df_cmb_meg_w, fitted=mdf_ica, 
                                          line_color='#fc8d62',
                                          x_key='age', y_key='ECG_rejected',
                                          add_ppm=True)
g_ica_comp.figure.set_size_inches(4,4)
g_ica_comp.set_xlabel('age (years)')
g_ica_comp.set_ylabel('1/f slope')
g_ica_comp.set_ylim(-2.5, -0.5)
sns.despine()
g_ica_comp.figure.savefig(f'../results/sbg_pred_brain_slope_avg_irasa_sss_{sss}.svg', )

# %%
fig, ax = plt.subplots(figsize=(4, 4))
df_cmb_meg_w['age'].plot(kind='hist', color='#777777', density=True)
df_cmb_meg_w['age'].plot(kind='kde', color='#990F02')
ax.set_xlabel('age (years)')
ax.set_ylabel('Density')
ax.set_xlim(0, 80)
sns.despine()
ax.figure.savefig('../results/age_dist_sbg.svg')
# %%
