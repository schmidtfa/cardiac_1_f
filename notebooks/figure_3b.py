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
from matplotlib.patches import Rectangle

sns.set_context('poster')
sns.set_style('ticks')


# %%
sss = False

if sss:
    INDIR = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/stats_across_irasa_sss'
else:
    INDIR = '/mnt/obob/staff/fschmidt/cardiac_1_f/data/stats_across_irasa'


lower_freqs = np.arange(1,11) - 0.5
upper_freqs =  [40.,  45.,  50.,  55.,  60., 
                65.,  70.,  75.,  80.,  85.,  90.,
                95., 100., 105., 110., 115., 120., 
                125., 130., 135., 140., 145.]

def get_sign_from_az(eff):

    pos_eff= eff['hdi_3%'] > 0.1
    neg_eff = eff['hdi_97%'] < -0.1
    null_eff = np.logical_and(eff['hdi_3%'] > -0.1, eff['hdi_97%'] < 0.1)

    return pos_eff, neg_eff, null_eff

#%%

split_eff_list, joined_eff_list = [], []

for cur_low in lower_freqs:
    for cur_up in upper_freqs:


        #% split effects to pandas
        cur_data = joblib.load(join(INDIR, f'stats_across_lower_{cur_low}_upper_{cur_up}.dat'))

        cur_eff = pd.DataFrame(cur_data['single_effects'].mean())

        pos_list, neg_list, null_list = [], [], []

        for cur_key in ['no_ica_eff', 'ica_eff', 'ecg_eff']:

            pos_eff, neg_eff, null_eff = get_sign_from_az(cur_data[cur_key].loc['age'])

            pos_list.append(pos_eff)
            neg_list.append(neg_eff)
            null_list.append(null_eff)


        cur_eff['pos_eff'] = pos_list
        cur_eff['neg_eff'] = neg_list
        cur_eff['null_eff'] = null_list

        cur_eff['lower_thresh'] = cur_low
        cur_eff['upper_thresh'] = cur_up

        eff_pd = cur_eff.reset_index()
        eff_pd.columns = ['condition', 'standardized beta', 'pos_eff', 'neg_eff', 'null_eff', 'lower_thresh', 'upper_thresh']

        split_eff_list.append(eff_pd)


        #% multi effects
        cur_df_joined = cur_data['partial_corr'].groupby('predictors').mean().reset_index()

        pos_eff_ecg, neg_eff_ecg, null_eff_ecg = get_sign_from_az(cur_data['summary_multi'].loc['ECG_components'])
        pos_eff_brain, neg_eff_brain, null_eff_brain = get_sign_from_az(cur_data['summary_multi'].loc['ECG_rejected'])

        cur_df_joined['pos_eff'] = [pos_eff_ecg, pos_eff_brain]
        cur_df_joined['neg_eff'] = [neg_eff_ecg, neg_eff_brain]
        cur_df_joined['null_eff'] = [null_eff_ecg, null_eff_brain]

        cur_df_joined['lower_thresh'] = cur_low
        cur_df_joined['upper_thresh'] = cur_up

        joined_eff_list.append(cur_df_joined)



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

def pivot_my_data(df, key):

    return df[[key, 'lower_thresh', 'upper_thresh']].pivot(values=key, index='lower_thresh', columns='upper_thresh').T



df_split, df_joined = pd.concat(split_eff_list), pd.concat(joined_eff_list)

ecg_nr = df_split.query('condition == "ECG not rejected"')
ecg_r = df_split.query('condition == "ECG rejected"')
ecg_comp = df_split.query('condition == "ECG component"')


ecg_nr_data, ecg_nr_neg_mask = pivot_my_data(ecg_nr, 'standardized beta'), pivot_my_data(ecg_nr, 'neg_eff')
ecg_r_data, ecg_r_neg_mask = pivot_my_data(ecg_r, 'standardized beta'), pivot_my_data(ecg_r, 'neg_eff')
ecg_comp_data, ecg_comp_neg_mask = pivot_my_data(ecg_comp, 'standardized beta'), pivot_my_data(ecg_comp, 'neg_eff')

_, ecg_nr_pos_mask = pivot_my_data(ecg_nr, 'standardized beta'), pivot_my_data(ecg_nr, 'pos_eff')
_, ecg_r_pos_mask = pivot_my_data(ecg_r, 'standardized beta'), pivot_my_data(ecg_r, 'pos_eff')
_, ecg_comp_pos_mask = pivot_my_data(ecg_comp, 'standardized beta'), pivot_my_data(ecg_comp, 'pos_eff')

_, ecg_nr_null_mask = pivot_my_data(ecg_nr, 'standardized beta'), pivot_my_data(ecg_nr, 'null_eff')
_, ecg_r_null_mask = pivot_my_data(ecg_r, 'standardized beta'), pivot_my_data(ecg_r, 'null_eff')
_, ecg_comp_null_mask = pivot_my_data(ecg_comp, 'standardized beta'), pivot_my_data(ecg_comp, 'null_eff')


ecg_comp_mask = ecg_comp_pos_mask + ecg_comp_neg_mask
ecg_r_mask = ecg_r_pos_mask + ecg_r_neg_mask
ecg_nr_mask = ecg_nr_pos_mask + ecg_nr_neg_mask 


#%%
titles = ['ECG not rejected', 'ECG rejected', 'ECG components']
all_split_data = [ecg_nr_data, ecg_r_data, ecg_comp_data]
all_split_mask = [ecg_nr_mask, ecg_r_mask, ecg_comp_mask]
all_split_null_mask = [ecg_nr_null_mask, ecg_r_null_mask, ecg_comp_null_mask]

f, axes = plt.subplots(ncols=3, figsize=(15, 6))
for ax, cur_data2, cur_mask, cur_null, cur_title in zip(axes, all_split_data, all_split_mask, all_split_null_mask, titles):

    cur_mask = cur_mask.to_numpy() == False
    mesh = plot_my_mesh(cur_data2, ax,  cur_mask, cur_null)#cur_mask

    ax.set_title(cur_title)
    ax.set_ylabel('Upper Slope Limit (Hz)')
    ax.set_xlabel('Lower Slope Limit (Hz)')

f.tight_layout()
cbar =f.colorbar(mesh,  ax=axes.ravel().tolist(), orientation='vertical')
cbar.set_label('β (standardized)')

#f.savefig(f'../results/across_irasa_split_{sss}.pdf', format='pdf')
# %%

comp_list = []

for cur_low in lower_freqs:
    for cur_up in upper_freqs:


        #% split effects to pandas
        cur_data = joblib.load(join(INDIR, f'stats_across_lower_{cur_low}_upper_{cur_up}.dat'))

        cur_eff = np.abs(cur_data['single_effects'])

        tmp_df = pd.DataFrame({'P(ECG rejected < ECG Components)': (cur_eff['ECG component'] > cur_eff['ECG rejected']).mean(),
                               'P(ECG not rejected < ECG Components)': (cur_eff['ECG component'] > cur_eff['ECG not rejected']).mean(),
                               'P(ECG rejected < ECG not rejected)': (cur_eff['ECG not rejected'] > cur_eff['ECG rejected']).mean(),
                               'Lower Slope Limit (Hz)': cur_low,
                               'Upper Slope Limit (Hz)': cur_up},
                               index=[0])
        
        comp_list.append(tmp_df)


# %%
df_cmb = pd.concat(comp_list)
# %%
df_cmb
# %%
reject = df_cmb.pivot(index='Lower Slope Limit (Hz)', columns='Upper Slope Limit (Hz)', values='P(ECG rejected < ECG Components)').T# * 100
not_reject = df_cmb.pivot(index='Lower Slope Limit (Hz)', columns='Upper Slope Limit (Hz)', values='P(ECG not rejected < ECG Components)').T
reject_no_reject = df_cmb.pivot(index='Lower Slope Limit (Hz)', columns='Upper Slope Limit (Hz)', values='P(ECG rejected < ECG not rejected)').T# * 100


# %%

def plot_my_mesh_2(df, mask, ax):

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
            if mask[i, j] != False:#mask_null.to_numpy()[i, j]:
                rect = Rectangle((0.5+j*0.9, extent[3]-cell_aspect-(mask.shape[0]-i-1)*cell_aspect), 0.9, cell_aspect, 
                fill=False, hatch='xxx', edgecolor='#666666', linewidth=0)
                          #linewidth=1)
                ax.add_patch(rect)

    mesh = ax.imshow(tfr2plot, cmap=my_cmap, aspect=0.1,
                     vmin=.0, vmax=1,
                     interpolation='none',
                     extent=extent)
    

    return mesh

# %%
f, axes = plt.subplots(ncols=3, figsize=(15, 6))
all_my_data = [reject, not_reject, reject_no_reject]
all_masks = [reject > 0.94, not_reject > 0.94, reject_no_reject > 0.94]
titles = ['P(ECG rejected < \n ECG components)', 'P(ECG not rejected < \n ECG components)', 'P(ECG rejected < \n ECG not rejected)']

for ix, (ax, cur_data2, mask, cur_title) in enumerate(zip(axes, all_my_data, all_masks, titles)):

    cur_mask = mask.to_numpy() == False
    mesh = plot_my_mesh_2(cur_data2, cur_mask, ax)

    ax.set_title(cur_title)
    if ix == 0:
        ax.set_ylabel('Upper Slope Limit (Hz)')
        ax.set_xlabel('Lower Slope Limit (Hz)')

f.tight_layout()
cbar =f.colorbar(mesh,  ax=axes.ravel().tolist(), orientation='vertical')
cbar.set_label('Probability')

f.savefig('../results/p_decrease_avg.pdf', format='pdf')
# %%
