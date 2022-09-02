#%%
import arviz as az
import mne
import seaborn as sns
import numpy as np
import fnmatch
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt

import matplotlib as mpl
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)
#%%
mdf = az.from_netcdf('../results/cam_can_trace_regr_short_bmb_1_200.ncdf')

# %%
summary = az.summary(mdf)

#%% 
def effects_from_stats(summary, feature):
    
    mask = [True if f'{feature}|channel[' in id else False for id in summary.index]
    beta = summary[mask]['mean']
    pos_mask = (summary[mask]['hdi_3%'] > 0).to_numpy()
    neg_mask = (summary[mask]['hdi_97%'] < 0).to_numpy()

    return beta, pos_mask, neg_mask

# %%
#beta_no_ica, pos_no_ica, _ = effects_from_stats(summary, 'brain_no_ica')
beta_ica, pos_ica, _ = effects_from_stats(summary, 'brain_slope')
beta_heart, pos_heart, _ = effects_from_stats(summary, 'heart_slope_mag')

#%%
info = mne.io.read_info('/mnt/sinuhe/data_raw/ss_cocktailparty/subject_subject/210503/19930422eibn_resting.fif',
                        verbose=False)
mag_adjacency = mne.channels.find_ch_adjacency(info, 'mag')
grad_adjacency = mne.channels.find_ch_adjacency(info, 'grad')


info_meg = mne.pick_info(info, mne.pick_types(info, meg=True))

info_mags = mne.pick_info(info, mne.pick_types(info, meg='mag'))


labels_meg = info['ch_names'][15:321]
get_feature = lambda feature: np.array([True if fnmatch.fnmatch(label, feature) else False for label in labels_meg])
mag_idx = get_feature('MEG*1')
grad_idx = ~mag_idx
# %%
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
                        units='beta', cbar_fmt='%0.3f', mask=mask, 
                        mask_params=mask_params, title=title,
                        size=3, time_unit='s');
    return topo
# %%
col_kwargs = {'vmin': -18,#-22,
              'vmax': 18, #22
              }

#g = plot_corr_topo(beta_no_ica, pos_no_ica, info_mags, 'brain_no_ica', **col_kwargs);
#g.figure.savefig('./results/cam_can_regr_brain_no_ica_topo.svg')
# %%
g = plot_corr_topo(beta_ica, pos_ica, info_mags, 'brain_ica', **col_kwargs);
#g.figure.savefig('../results/cam_can_regr_brain_ica_topo.svg')
# %%
g = plot_corr_topo(beta_heart, pos_heart, info_mags, 'heart_ica', **col_kwargs);
#g.figure.savefig('../results/cam_can_regr_heart_ica_topo.svg')
# %% reload real data to plot effect at sign channels
df_cam_can = pd.read_csv('../data/cam_can_1_f_dataframe_1_200.csv')
# %%

def aggregate_sign_feature(feature_key, pos_mask):
    df_brain = df_cam_can[[feature_key, 'subject_id', 'age', 'channel']]
    feature_by_age = []
    for subject in df_brain['subject_id'].unique():
        cur_subject = df_brain.query(f'subject_id == "{subject}"')
        feature_by_age.append(cur_subject
                                        .sort_values(by='channel')[pos_mask]
                                        .mean()[[feature_key,'age']])

    df = pd.concat(feature_by_age, axis=1).T
    return df
# %%
df_brain_ica = aggregate_sign_feature('brain_slope', pos_ica)
df_heart = aggregate_sign_feature('heart_slope_mag', pos_heart)

# %%
pg.corr(df_brain_ica['brain_slope'], df_brain_ica['age'])
# %%
pg.corr(df_heart['heart_slope_mag'], df_brain_ica['age'])
# %%
df_heart['brain_slope'] = df_brain_ica['brain_slope']

# %%
x=2.2
y=1.
corr = pg.corr(df_heart['heart_slope_mag'], df_heart['brain_slope'])

g = sns.lmplot(data=df_heart, x='heart_slope_mag', y='brain_slope',
                   scatter_kws={"s": 40, 'color': '#888888', 'alpha': 0.25})

r = round(float(corr['r']), 2)
p = round(float(corr['p-val']), 3)

if p == 0.0:
    p = 'p < 0.001'
else:
    p = f'p = {p}'

plt.annotate(text=f'r = {r}', xy=(x, y))
plt.annotate(text=p, xy=(x, y - 0.15))

g.set_xlabels('1/f slope (ECG Component)')
g.set_ylabels('1/f slope (ECG Component not removed)')
g.ax.figure.savefig(f'../results/slope_heart_x_slope_no_ica.svg', )
#%%

plt.scatter(df_heart['heart_slope_mag'], df_brain_ica['brain_no_ica'])
# %%
proba_heart_l_ica = (mdf.posterior['brain_slope|channel'] < mdf.posterior['heart_slope_mag|channel']).mean(axis=2)
# %%
proba_heart_l_ica.mean()
# %%
plt.hist(mdf.posterior['brain_slope|channel'].to_numpy().flatten())
plt.hist(mdf.posterior['heart_slope_mag|channel'].to_numpy().flatten())
# %%
