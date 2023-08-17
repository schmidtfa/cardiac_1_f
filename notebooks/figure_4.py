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
joined = False

for cur_low in lower_freqs:
    for cur_up in upper_freqs:

        #% split effects to pandas
        cur_data = joblib.load(join(INDIR, f'stats_across_lower_{cur_low}_upper_{cur_up}_sss_{sss_list[1]}.dat'))
        #cur_data_sss = joblib.load(join(INDIR, f'stats_across_lower_{cur_low}_upper_{cur_up}_sss_{sss_list[0]}.dat'))

        #% split effects
        split_eff_list.append(cur_data['single_effects'])
        #split_eff_list_sss.append(cur_data_sss['single_effects'])

        #%joined effects

        if joined:
            if type(cur_data['summary_multi']) == list:
                for eff_id in [0,1]:
                    if cur_data['summary_multi'][eff_id].shape[0] > 2:
                        cur_data['summary_multi'][eff_id]['lower_thresh'] = cur_low
                        cur_data['summary_multi'][eff_id]['upper_thresh'] = cur_up
                        cur_data['summary_multi'][eff_id]['effect_direction'] = cur_data['effect_direction']
                        unique_effs.append(cur_data['summary_multi'][eff_id])

                    else:
                        cur_data['partial_corr'][eff_id]['lower_thresh'] = cur_low
                        cur_data['partial_corr'][eff_id]['upper_thresh'] = cur_up
                        cur_data['partial_corr'][eff_id]['effect_direction'] = cur_data['effect_direction']
                        joined_eff_list.append(cur_data['partial_corr'][eff_id])
            else:
                if cur_data['summary_multi'].shape[0] > 2:
                        cur_data['summary_multi']['lower_thresh'] = cur_low
                        cur_data['summary_multi']['upper_thresh'] = cur_up
                        cur_data['summary_multi']['effect_direction'] = cur_data['effect_direction']
                        unique_effs.append(cur_data['summary_multi'])

                else:
                        cur_data['partial_corr']['lower_thresh'] = cur_low
                        cur_data['partial_corr']['upper_thresh'] = cur_up
                        cur_data['partial_corr']['effect_direction'] = cur_data['effect_direction']
                        joined_eff_list.append(cur_data['partial_corr'])
#%%
if joined:
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


if joined:
    unique_effs = [unique_brain, unique_ecg]
    titles = ['unique_brain', 'unique_ecg']

    f, axes = plt.subplots(ncols=2, figsize=(15, 6))

    mesh = plot_my_mesh(unique_brain, axes[0], 'Greys', vmin=0, vmax=1)
    mesh = plot_my_mesh(unique_ecg, axes[1], 'Greys', vmin=0, vmax=1)

    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_ylabel('Upper Slope Limit (Hz)')
        ax.set_xlabel('Lower Slope Limit (Hz)')
        #mesh.set_norm(cm.colors.LogNorm(vmin=0.001, vmax=1))

    f.tight_layout()
    cbar =f.colorbar(mesh,  ax=axes.ravel().tolist(), orientation='vertical')
    cbar.set_label('unique effects')

#f.savefig(f'../results/across_irasa_split_null_{sss}.svg', format='svg')

    #%%
    joined_effs = pd.concat(joined_eff_list)


#%%

df_split = pd.concat(split_eff_list)
#df_split_sss = pd.concat(split_eff_list_sss)
#%%
df_split['no_effect'] = df_split[['negative_effect', 'positive_effect', 'null_effect']].sum(axis=1) == 0
#%%
df_effs = df_split.query('null_effect == False').query('no_effect == False')

fig, axes = plt.subplots(ncols=3, figsize=(18, 6))
colors = ['#8da0cb',  '#fc8d62', '#66c2a5',]
condition_labels = ['ECG not rejected', 'ECG rejected', 'ECG components', ]
condition_list = ['ECG_not_rejected', 'ECG_rejected', 'ECG_components', ]


for ix, ax in enumerate(axes):
    ax.hist(df_effs.query(f'condition == "{condition_list[ix]}"')['average'], 
            label=condition_labels[ix], color=colors[ix])
    ax.set_ylabel('Number of Effects')
    ax.set_xlabel('standardized β')
    ax.set_xlim((-.45, .25))
    ax.legend()
    
fig.tight_layout()
sns.despine()
fig.savefig('../results/total_effect_sizes_by_channel.svg')

#df_split_sss['no_effect'] = df_split_sss[['negative_effect', 'positive_effect', 'null_effect']].sum(axis=1) == 0


#%% plot the undecided split in negative and positive
df_split['no_eff_neg'] = np.logical_and(df_split['no_effect'] == True, df_split['average'] < 0)
df_split['no_eff_pos'] = np.logical_and(df_split['no_effect'] == True, df_split['average'] > 0)

#df_split_sss['no_eff_neg'] = np.logical_and(df_split_sss['no_effect'] == True, df_split_sss['average'] < 0)
#df_split_sss['no_eff_pos'] = np.logical_and(df_split_sss['no_effect'] == True, df_split_sss['average'] > 0)

#%%

def prep_data4stacked(df):
    df2bar = df[['condition', 'negative_effect', 'positive_effect', 'null_effect', 'no_eff_neg', 'no_eff_pos']].melt(id_vars='condition').groupby(['condition', 'variable']).mean().reset_index()


    df2bar['value'] *= 100
    df2bar.set_index('condition')
    df_pivot = df2bar.pivot_table(index='condition', columns='variable', values='value').reset_index()

    return df_pivot

#%%

df_pivot = prep_data4stacked(df_split)
#df_pivot_sss = prep_data4stacked(df_split_sss)


#%%
sns.set_style('ticks')
sns.set_context('poster')

fig, ax = plt.subplots(figsize = (12,6))

col_order = ['condition', 'negative_effect', 'no_eff_neg', 'positive_effect', 'no_eff_pos', 'null_effect',]

pal = sns.color_palette('deep', as_cmap=True)
pal2 = sns.color_palette('pastel', as_cmap=True)
cmap = [pal[0], pal2[0], pal[3], pal2[3], pal[2]]

df_pivot[col_order].plot(
    x = 'condition',
    kind = 'barh',
    color=cmap,
    stacked = True,
    #mark_right = True,
    ax=ax)

sns.despine()

#fig.savefig(f'../results/stacked_bar_maxf.svg')

#%%
# fig, ax = plt.subplots(figsize = (12,6))

# col_order = ['condition', 'negative_effect', 'no_eff_neg', 'positive_effect', 'no_eff_pos', 'null_effect',]

# pal = sns.color_palette('deep', as_cmap=True)
# pal2 = sns.color_palette('pastel', as_cmap=True)
# cmap = [pal[0], pal2[0], pal[3], pal2[3], pal[2]]

# df_pivot_sss[col_order].plot(
#     x = 'condition',
#     kind = 'barh',
#     color=cmap,
#     stacked = True,
#     #mark_right = True,
#     ax=ax)

# sns.despine()

#fig.savefig(f'../results/stacked_bar_maxf.svg')


# %%
ecg_not_rejected = df_split.query('condition == "ECG_not_rejected"').groupby('channel').mean().reset_index()
ecg_rejected = df_split.query('condition == "ECG_rejected"').groupby('channel').mean().reset_index()
ecg_components = df_split.query('condition == "ECG_components"').groupby('channel').mean().reset_index()
sss= False
# %%
# %% get some info structure for plotting
import mne

info = mne.io.read_info('/mnt/sinuhe/data_raw/ss_cocktailparty/subject_subject/210503/19930422eibn_resting.fif',
                        verbose=False)
mag_adjacency = mne.channels.find_ch_adjacency(info, 'mag')

info_mags = mne.pick_info(info, mne.pick_types(info, meg='mag'))


no_eff_neg_cmap = sns.light_palette("#79C", as_cmap=True)
no_eff_pos_cmap = sns.light_palette((20, 60, 50), input="husl", as_cmap=True)
#%%
cur_topo = plot_corr_topo(ecg_components['no_eff_neg'].to_numpy(), np.zeros(102), info_mags,'', 
                vmax=1, vmin=0, cmap=no_eff_neg_cmap);
#cur_topo.figure.set_norm(cm.colors.LogNorm(vmin=0.001, vmax=1))
#cur_topo.figure.savefig(f'../results/topo_ecg_no_eff_neg_irasa_{sss}.svg')


#%%
cur_topo = plot_corr_topo(ecg_components['no_eff_pos'].to_numpy(), np.zeros(102), info_mags,'', 
                vmax=1, vmin=0, cmap=no_eff_pos_cmap);

#cur_topo.figure.savefig(f'../results/topo_ecg_no_eff_pos_irasa_{sss}.svg')


#%%
cur_topo = plot_corr_topo(ecg_rejected['no_eff_neg'].to_numpy(), np.zeros(102), info_mags,'', 
                vmax=1, vmin=0, cmap=no_eff_neg_cmap);

#cur_topo.figure.savefig(f'../results/topo_rej_no_eff_neg_irasa_{sss}.svg')


#%%
cur_topo = plot_corr_topo(ecg_rejected['no_eff_pos'].to_numpy(), np.zeros(102), info_mags,'', 
                vmax=1, vmin=0, cmap=no_eff_pos_cmap);

#cur_topo.figure.savefig(f'../results/topo_rej_no_eff_pos_irasa_{sss}.svg')

#%%
cur_topo = plot_corr_topo(ecg_not_rejected['no_eff_neg'].to_numpy(), np.zeros(102), info_mags,'', 
                vmax=1, vmin=0, cmap=no_eff_neg_cmap);

#cur_topo.figure.savefig(f'../results/topo_nrej_no_eff_neg_irasa_{sss}.svg')


#%%
cur_topo = plot_corr_topo(ecg_not_rejected['no_eff_pos'].to_numpy(), np.zeros(102), info_mags,'', 
                vmax=1, vmin=0, cmap=no_eff_pos_cmap);

#cur_topo.figure.savefig(f'../results/topo_nrej_no_eff_pos_irasa_{sss}.svg')


#%%

cur_topo = plot_corr_topo(ecg_components['average'].to_numpy(), np.zeros(102), info_mags,'', 
                vmax=0.4, vmin=-0.4, cmap='RdBu_r');

#cur_topo.figure.savefig(f'../results/topo_ecg_ave_irasa_{sss}.svg')

#%%

cur_topo = plot_corr_topo(ecg_components['negative_effect'].to_numpy(), np.zeros(102), info_mags,'', 
                vmax=1, vmin=0, cmap='Blues');

#cur_topo.figure.savefig(f'../results/topo_ecg_neg_irasa_{sss}.svg')
#%%
cur_topo = plot_corr_topo(ecg_components['positive_effect'].to_numpy(), np.zeros(102), info_mags,'', 
               vmax=0.5, vmin=0, cmap='Reds');

#cur_topo.figure.savefig(f'../results/topo_ecg_pos_irasa_{sss}.svg')
#%%
cur_topo = plot_corr_topo(ecg_components['null_effect'].to_numpy(), np.zeros(102), info_mags,'', 
            vmax=0.5, vmin=0, cmap='Greens');

#cur_topo.figure.savefig(f'../results/topo_ecg_null_irasa_{sss}.svg')
#%%

cur_topo = plot_corr_topo(ecg_rejected['average'].to_numpy(), np.zeros(102), info_mags,'', 
                vmax=0.4, vmin=-0.4, cmap='RdBu_r');

#cur_topo.figure.savefig(f'../results/topo_rej_ave_irasa_{sss}.svg')

#%%
cur_topo = plot_corr_topo(ecg_rejected['negative_effect'].to_numpy(), 
               np.zeros(102), info_mags,'', vmax=1, vmin=0, cmap='Blues');

#cur_topo.figure.savefig(f'../results/topo_rej_neg_irasa_{sss}.svg')
#%%
cur_topo = plot_corr_topo(ecg_rejected['positive_effect'].to_numpy(), np.zeros(102), 
                info_mags,'', vmax=0.1, vmin=0, cmap='Reds');

#cur_topo.figure.savefig(f'../results/topo_rej_pos_irasa_{sss}.svg')
#%%
cur_topo = plot_corr_topo(ecg_rejected['null_effect'].to_numpy(), np.zeros(102), info_mags,'', 
                vmax=.5, vmin=0, cmap='Greens');

#cur_topo.figure.savefig(f'../results/topo_rej_null_irasa_{sss}.svg')

#%%

cur_topo = plot_corr_topo(ecg_not_rejected['average'].to_numpy(), np.zeros(102), info_mags,'', 
                vmax=0.4, vmin=-0.4, cmap='RdBu_r');

#cur_topo.figure.savefig(f'../results/topo_nrej_ave_irasa_{sss}.svg')

#%%
cur_topo = plot_corr_topo(ecg_not_rejected['negative_effect'].to_numpy(), np.zeros(102), 
                info_mags,'', vmax=1, vmin=0, cmap='Blues');

#cur_topo.figure.savefig(f'../results/topo_nrej_neg_irasa_{sss}.svg')
#%%
cur_topo = plot_corr_topo(ecg_not_rejected['positive_effect'].to_numpy(), np.zeros(102), 
                info_mags,'', vmax=.1, vmin=0, cmap='Reds');

#cur_topo.figure.savefig(f'../results/topo_nrej_pos_irasa_{sss}.svg')
#%%
cur_topo = plot_corr_topo(ecg_not_rejected['null_effect'].to_numpy(), np.zeros(102), 
                info_mags,'', vmax=.5, vmin=0, cmap='Greens');

#cur_topo.figure.savefig(f'../results/topo_nrej_null_irasa_{sss}.svg')

# %%
ecg_not_rejected_fmap = df_split.query('condition == "ECG_not_rejected"').groupby(['lower_thresh', 'upper_thresh']).mean().reset_index()
ecg_rejected_fmap = df_split.query('condition == "ECG_rejected"').groupby(['lower_thresh', 'upper_thresh']).mean().reset_index()
ecg_components_fmap = df_split.query('condition == "ECG_components"').groupby(['lower_thresh', 'upper_thresh']).mean().reset_index()


sns.set_style('ticks')
sns.set_context('poster')


# %%
def plot_my_mesh(df, ax, key, my_cmap='RdBu_r', vmin=-0.4, vmax=0.4):

    df2plot = df[['lower_thresh', 'upper_thresh', key]].pivot(columns='lower_thresh', index='upper_thresh', values=key)

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
# %%
all_split_data = [ecg_not_rejected_fmap, ecg_rejected_fmap, ecg_components_fmap]
titles = ['ecg_not_rejected', 'ecg_rejected', 'ecg_components']

f, axes = plt.subplots(ncols=3, figsize=(15, 6))
for ax, cur_data, cur_title in zip(axes, all_split_data, titles):

    mesh = plot_my_mesh(cur_data, ax, 'average', vmin=-0.4, vmax=0.4)#cur_mask

    ax.set_title(cur_title)
    ax.set_ylabel('Upper Slope Limit (Hz)')
    ax.set_xlabel('Lower Slope Limit (Hz)')
    

f.tight_layout()
cbar =f.colorbar(mesh,  ax=axes.ravel().tolist(), orientation='vertical')
cbar.set_label('β (standardized)')

#f.savefig(f'../results/across_irasa_split_average_{sss}.svg', format='svg')

#%%
import matplotlib.colors as colors
import matplotlib.cm as cm

f, axes = plt.subplots(ncols=3, figsize=(15, 6))
for ax, cur_data, cur_title in zip(axes, all_split_data, titles):

    mesh = plot_my_mesh(cur_data, ax, 'positive_effect', 'Reds', vmin=0, vmax=.1)#cur_mask

    ax.set_title(cur_title)
    ax.set_ylabel('Upper Slope Limit (Hz)')
    ax.set_xlabel('Lower Slope Limit (Hz)')
    #mesh.set_norm(cm.colors.LogNorm(vmin=0.001, vmax=1))

f.tight_layout()
cbar =f.colorbar(mesh,  ax=axes.ravel().tolist(), orientation='vertical')
cbar.set_label('positive effects (%)')

f.savefig(f'../results/across_irasa_split_positive_{sss}.svg', format='svg')

# %%

f, axes = plt.subplots(ncols=3, figsize=(15, 6))
for ax, cur_data, cur_title in zip(axes, all_split_data, titles):

    mesh = plot_my_mesh(cur_data, ax, 'negative_effect', 'Blues', vmin=0, vmax=1)

    ax.set_title(cur_title)
    ax.set_ylabel('Upper Slope Limit (Hz)')
    ax.set_xlabel('Lower Slope Limit (Hz)')
    #mesh.set_norm(cm.colors.LogNorm(vmin=0.001, vmax=1))

f.tight_layout()
cbar =f.colorbar(mesh,  ax=axes.ravel().tolist(), orientation='vertical')
cbar.set_label('negative effects (%)')

f.savefig(f'../results/across_irasa_split_negative_{sss}.svg', format='svg')
# %%
f, axes = plt.subplots(ncols=3, figsize=(15, 6))
for ax, cur_data, cur_title in zip(axes, all_split_data, titles):

    mesh = plot_my_mesh(cur_data, ax, 'null_effect', 'Greens', vmin=0, vmax=.5)

    ax.set_title(cur_title)
    ax.set_ylabel('Upper Slope Limit (Hz)')
    ax.set_xlabel('Lower Slope Limit (Hz)')
    #mesh.set_norm(cm.colors.LogNorm(vmin=0.001, vmax=1))

f.tight_layout()
cbar =f.colorbar(mesh,  ax=axes.ravel().tolist(), orientation='vertical')
cbar.set_label('null effects log10(%)')

f.savefig(f'../results/across_irasa_split_null_{sss}.svg', format='svg')

#%%
# %%
f, axes = plt.subplots(ncols=3, figsize=(15, 6))
for ax, cur_data, cur_title in zip(axes, all_split_data, titles):

    mesh = plot_my_mesh(cur_data, ax, 'no_eff_neg', no_eff_neg_cmap, vmin=0, vmax=0.5)

    ax.set_title(cur_title)
    ax.set_ylabel('Upper Slope Limit (Hz)')
    ax.set_xlabel('Lower Slope Limit (Hz)')
    #mesh.set_norm(cm.colors.LogNorm(vmin=0.001, vmax=1))

f.tight_layout()
cbar =f.colorbar(mesh,  ax=axes.ravel().tolist(), orientation='vertical')
cbar.set_label('undecided effects (leaning negative; %)')

f.savefig(f'../results/across_irasa_split_no_eff_neg_{sss}.svg', format='svg')


# %%

f, axes = plt.subplots(ncols=3, figsize=(15, 6))
for ax, cur_data, cur_title in zip(axes, all_split_data, titles):

    mesh = plot_my_mesh(cur_data, ax, 'no_eff_pos', no_eff_pos_cmap, vmin=0, vmax=0.5)

    ax.set_title(cur_title)
    ax.set_ylabel('Upper Slope Limit (Hz)')
    ax.set_xlabel('Lower Slope Limit (Hz)')
    #mesh.set_norm(cm.colors.LogNorm(vmin=0.001, vmax=1))

f.tight_layout()
cbar =f.colorbar(mesh,  ax=axes.ravel().tolist(), orientation='vertical')
cbar.set_label('undecided effects (leaning positive; %)')

f.savefig(f'../results/across_irasa_split_no_eff_pos_{sss}.svg', format='svg')

