#%%
import joblib
from os.path import join
import numpy as np
import pandas as pd

import matplotlib as mpl
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

import matplotlib.pyplot as plt
import seaborn as sns


import sys
sys.path.append('/mnt/obob/staff/fschmidt/cardiac_1_f')
from utils.plot_utils import plot_corr_topo

sns.set_context('poster')
sns.set_style('ticks')

import bambi as bmb
import pymc as pm


# %%
INDIR = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/stats_across_irasa_s_chans_final'

lower_freqs = np.arange(1,11) - 0.5
upper_freqs =  [40.,  45.,  50.,  55.,  60., 
                65.,  70.,  75.,  80.,  85.,  90.,
                95., 100., 105., 110., 115., 120., 
                125., 130., 135., 140., 145.]

# %%

def get_sign_from_az(eff):

    pos_eff= eff['hdi_3%'] > 0.1
    neg_eff = eff['hdi_97%'] < -0.1
    null_eff = np.logical_and(eff['hdi_3%'] > -0.1, eff['hdi_97%'] < 0.1)

    return pos_eff, neg_eff, null_eff


#%%
split_eff_list, split_eff_list_sss = [], []
joined_eff_list = []
unique_effs = []

sss_list = [True, False]
joined = True

for cur_low in lower_freqs:
    for cur_up in upper_freqs:

        #% split effects to pandas
        cur_data = joblib.load(join(INDIR, f'stats_across_lower_{cur_low}_upper_{cur_up}_sss_{sss_list[1]}.dat'))

        #% split effects
        split_eff_list.append(cur_data['single_effects'])

        #%joined effects

        if joined:
            if type(cur_data['summary_multi']) == list:
                for eff_id in [0,1]:
                    if cur_data['summary_multi'][eff_id].shape[0] == 1:
                        cur_data['summary_multi'][eff_id]['lower_thresh'] = cur_low
                        cur_data['summary_multi'][eff_id]['upper_thresh'] = cur_up
                        unique_effs.append(cur_data['summary_multi'][eff_id])

                    else:
                        cur_data['partial_corr'][eff_id]['lower_thresh'] = cur_low
                        cur_data['partial_corr'][eff_id]['upper_thresh'] = cur_up
                        joined_eff_list.append(cur_data['partial_corr'][eff_id])
            else:
                if cur_data['summary_multi'].shape[0] == 1:
                        cur_data['summary_multi']['lower_thresh'] = cur_low
                        cur_data['summary_multi']['upper_thresh'] = cur_up
                        unique_effs.append(cur_data['summary_multi'])

                else:
                        cur_data['partial_corr']['lower_thresh'] = cur_low
                        cur_data['partial_corr']['upper_thresh'] = cur_up
                        joined_eff_list.append(cur_data['partial_corr'])
#%%
unique_brain = pd.concat(unique_effs).pivot_table(columns='lower_thresh', index=['effect_direction', 'upper_thresh'], values='brain_eff')
unique_ecg = pd.concat(unique_effs).pivot_table(columns='lower_thresh', index=['effect_direction', 'upper_thresh'], values='ecg_eff')#.fillna(0)
#%%
def plot_my_mesh(df2plot, ax, my_cmap='RdBu_r', vmin=-0.4, vmax=0.4):

    data2plot = np.flipud(df2plot.to_numpy())

    lower_freqs2plot = df2plot.columns.to_numpy()
    upper_freqs2plot = df2plot.index.to_numpy()

    extent = [lower_freqs2plot.min(), lower_freqs2plot.max(), 
              upper_freqs2plot.min(), upper_freqs2plot.max()]

    mesh = ax.imshow(data2plot, cmap=my_cmap, aspect=0.1,
                     vmin=vmin, vmax=vmax,
                     interpolation='none',
                     extent=extent)

    return mesh


unique_effs = [unique_brain, unique_ecg]
titles = ['unique_brain', 'unique_ecg']

f, axes = plt.subplots(ncols=2, figsize=(15, 6))

mesh = plot_my_mesh(unique_brain.loc['positive'], axes[0], 'Reds', vmin=0, vmax=1)
mesh = plot_my_mesh(unique_ecg.loc['positive'], axes[1], 'Reds', vmin=0, vmax=1)

for ax, title in zip(axes, titles):
    ax.set_title(title)
    ax.set_ylabel('Upper Slope Limit (Hz)')
    ax.set_xlabel('Lower Slope Limit (Hz)')
    #mesh.set_norm(cm.colors.LogNorm(vmin=0.001, vmax=1))

f.tight_layout()
cbar =f.colorbar(mesh,  ax=axes.ravel().tolist(), orientation='vertical')
cbar.set_label('unique positive effects')

f.savefig(f'../results/irasa_unique_pos.svg', format='svg')

#%%
unique_effs = [unique_brain, unique_ecg]
titles = ['unique_brain', 'unique_ecg']

f, axes = plt.subplots(ncols=2, figsize=(15, 6))

mesh = plot_my_mesh(unique_brain.loc['negative'], axes[0], 'Blues', vmin=0, vmax=1)
mesh = plot_my_mesh(unique_ecg.loc['negative'], axes[1], 'Blues', vmin=0, vmax=1)

for ax, title in zip(axes, titles):
    ax.set_title(title)
    ax.set_ylabel('Upper Slope Limit (Hz)')
    ax.set_xlabel('Lower Slope Limit (Hz)')
    #mesh.set_norm(cm.colors.LogNorm(vmin=0.001, vmax=1))

f.tight_layout()
cbar =f.colorbar(mesh,  ax=axes.ravel().tolist(), orientation='vertical')
cbar.set_label('unique negative effects')

f.savefig(f'../results/irasa_unique_neg.svg', format='svg')

#%%
titles = ['positive', 'negative']
f, axes = plt.subplots(ncols=2, figsize=(15, 6))

mesh = plot_my_mesh(np.isnan(unique_brain.loc['positive']), axes[0], 'Reds', vmin=0, vmax=1)
mesh = plot_my_mesh(np.isnan(unique_ecg.loc['negative']), axes[1], 'Blues', vmin=0, vmax=1)

for ax, title in zip(axes, titles):
    ax.set_title(title)
    ax.set_ylabel('Upper Slope Limit (Hz)')
    ax.set_xlabel('Lower Slope Limit (Hz)')
    #mesh.set_norm(cm.colors.LogNorm(vmin=0.001, vmax=1))

f.tight_layout()
cbar =f.colorbar(mesh,  ax=axes.ravel().tolist(), orientation='vertical')
cbar.set_label('joined effects')

f.savefig(f'../results/irasa_joined_eff.svg', format='svg')


#%%
pos_list, neg_list = [],[]

for eff in joined_eff_list:
    
    if eff['effect_direction'].iloc[0] == "positive":
    
        pos_list.append((eff.groupby('predictors').quantile((0.03, 0.97))
                            .reset_index()
                            .query('level_1 == 0.03')))

    elif eff['effect_direction'].iloc[0] == "negative":

        neg_list.append((eff.groupby('predictors').quantile((0.03, 0.97))
                            .reset_index()
                            .query('level_1 == 0.97')))



#%%
positive_joined = pd.concat(pos_list)
negative_joined = pd.concat(neg_list)

#%%
negative_joined['neg_effect'] = negative_joined['partial correlation coefficient'] < -0.1
positive_joined['pos_effect'] = positive_joined['partial correlation coefficient'] > 0.1

# %%
res_pos_eff = positive_joined.groupby('predictors').mean().reset_index()
res_neg_eff = negative_joined.groupby('predictors').mean().reset_index()
res_pos_eff['perc'] = res_pos_eff['pos_effect'] * 100
res_neg_eff['perc'] = res_neg_eff['neg_effect'] * 100

cmap = ['#66c2a5', '#fc8d62']

#%%
g = sns.barplot(data=res_pos_eff, x='predictors', y='perc', palette=cmap)
g.set_ylim(0,100)
g.set_xlabel('')
g.set_ylabel('Residual Flattening Effects (%)')
g.figure.set_size_inches(5, 5)
sns.despine()
g.figure.savefig('../results/residual_pos.svg')
# %%
g = sns.barplot(data=res_neg_eff, x='predictors', y='perc', palette=cmap)
g.set_ylim(0,100)
g.set_xlabel('')
g.set_ylabel('Residual Steepening Effects (%)')
g.figure.set_size_inches(5, 5)
sns.despine()
g.figure.savefig('../results/residual_neg.svg')
# %%
