#%%
import pandas as pd
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
import bambi as bmb
import arviz as az
from scipy.stats import zscore
import numpy as np

import seaborn as sns

sns.set_style('ticks')
sns.set_context('poster')

import matplotlib as mpl
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)
from matplotlib.patches import Rectangle

import sys
sys.path.append('/mnt/obob/staff/fschmidt/cardiac_1_f/utils')

from plot_utils import plot_bayes_linear_regression

# %%
INDIR = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/movement_cc'
all_files = listdir(INDIR)
# %%
df_list = []

for file in all_files:
    cur_df = pd.read_csv(join(INDIR, file))
    cur_df['recording_time'] = cur_df['time (s)'].max() - cur_df['time (s)'].min()
    cur_df['distance_median'] = cur_df['distance'].median() * 1000 #get data in mm
    cur_df['distance_total'] = cur_df['distance'].sum() * 1000  / cur_df['recording_time']
    cols_of_interest = ['subject_id', 'age', 'distance_median', 'recording_time', 'distance_total']
    df_list.append(cur_df[cols_of_interest])
# %%
df = pd.concat(df_list).drop_duplicates('subject_id')

#%%
df.to_csv('../data/movement_cc_df.csv')

#%%
dat_z = zscore(df[['age', 'distance_median']])

#%%
md = bmb.Model(formula='distance_median ~ 1 + age',
                    data=df, family='t')
mdf = md.fit()

#%%
md_z = bmb.Model(formula='distance_median ~ 1 + age',
                    data=dat_z, family='t')
                    
mdf_z = md_z.fit()

# %%
md.predict(mdf)

#%%
g = plot_bayes_linear_regression(df, mdf, 'age', 'distance_median', line_color='#333333')
g.set_ylabel('Head Movement Velocity (mm/s)')
g.set_xlabel('age (years)')
g.figure.set_size_inches(4,4)
sns.despine()
g.figure.savefig(f'../results/movement_velocity.svg')
# %%
az.summary(mdf_z)

# %%

# %%
eog_0 = pd.read_csv('../data/cam_can_eog_0_across_freqs.csv')
eog_1 = pd.read_csv('../data/cam_can_eog_1_across_freqs.csv')

#%%
def eog2pivot(eog):
    eog_sel = eog.query('`Unnamed: 0` == "age"')[['mean', 'hdi_3%','hdi_97%', 'lower_thresh', 'upper_thresh']]
    pos = eog_sel.pivot_table(index='lower_thresh', columns='upper_thresh', values='hdi_97%') < -0.1
    neg = eog_sel.pivot_table(index='lower_thresh', columns='upper_thresh', values='hdi_3%') > 0.1
    null = np.logical_and(eog_sel.pivot_table(index='lower_thresh', columns='upper_thresh', values='hdi_3%') > -0.1, 
                          eog_sel.pivot_table(index='lower_thresh', columns='upper_thresh', values='hdi_97%') < 0.1)
    mean = eog_sel.pivot_table(index='lower_thresh', columns='upper_thresh', values='mean')

    return pos.T, neg.T, mean.T, null.T

# %%
eog_0_pos, eog_0_neg, eog_0_mean, eog_0_null = eog2pivot(eog_0)
eog_1_pos, eog_1_neg, eog_1_mean, eog_1_null = eog2pivot(eog_1)

# %%
def plot_my_mesh(df, ax, mask, mask_null):

    data2plot = np.flipud(df.to_numpy())

    lower_freqs2plot = df.columns.to_numpy()
    upper_freqs2plot = df.index.to_numpy()

    #my_cmap = 'Blues_r'
    my_cmap = 'RdBu_r'

    extent = [lower_freqs2plot.min(), lower_freqs2plot.max(), 
              upper_freqs2plot.min(), upper_freqs2plot.max()]

    tfr2plot = data2plot# * np.flipud(mask) * 1


    ax.set_aspect(0.1)
    cell_aspect = (extent[3] - extent[2]) / mask.shape[0]

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != mask_null.to_numpy()[i, j]:
                rect = Rectangle((0.5+j*0.9, extent[3]-cell_aspect-(mask.shape[0]-i-1)*cell_aspect), 0.9, cell_aspect, 
                fill=False, hatch='xxx', edgecolor='#666666', linewidth=0)
                          #linewidth=1)
                ax.add_patch(rect)

    mesh = ax.imshow(tfr2plot, cmap=my_cmap, aspect=0.1,
                     vmin=-.4, vmax=.4,
                     interpolation='none',
                     extent=extent)

    cmap1 = mpl.colors.ListedColormap(['none', 'green'])
    ax.imshow(np.flipud(mask_null), cmap=cmap1, aspect=0.1,
                    #vmin=-.4, vmax=-.0,
                    extent=extent)

    return mesh


eog_0_list = [eog_0_mean, eog_0_neg, eog_0_pos, eog_0_null]
eog_1_list = [eog_1_mean, eog_1_neg, eog_1_pos, eog_1_null]
#%%
f, axes = plt.subplots(ncols=2, figsize=(15, 6))

for ax, eog_l, cur_title in zip(axes, [eog_0_list, eog_1_list], ['EOGV', 'EOGH']):

    cur_mask = (eog_l[1] + eog_l[2]).to_numpy() == False
    mesh = plot_my_mesh(eog_l[0], ax, cur_mask, eog_l[3])
    ax.set_title(cur_title)
    ax.set_ylabel('Upper Slope Limit (Hz)')
    ax.set_xlabel('Lower Slope Limit (Hz)')

f.tight_layout()
cbar =f.colorbar(mesh,  ax=axes.ravel().tolist(), orientation='vertical')
cbar.set_label('β (standardized)')


f.savefig(f'../results/across_irasa_eog.pdf', format='pdf')
f.savefig(f'../results/across_irasa_eog.svg')
# %%
