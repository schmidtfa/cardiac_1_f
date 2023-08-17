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
df_cmb_k = data_loader(path2data=path2data, freqs=[0.1, 145], 
                     heart_thresh=heart_thresh, eye_thresh=eye_thresh,
                     fit_knee=True, peaks=peaks, interpolate=True, sss=sss, get_psd=get_psd)

#%%
df_cmb = data_loader(path2data=path2data, freqs=[0.1, 145], 
                     heart_thresh=heart_thresh, eye_thresh=eye_thresh,
                     fit_knee=False, peaks=peaks, interpolate=True, sss=sss, get_psd=get_psd)

# %%
# calculate bic for regression
def calculate_bic(n, mse, num_params):
    bic = n * np.log(mse) + num_params * np.log(n)
    return bic


#%%

n = 145*2 #0.5hz frequency resolution to 145
num_params = 3# i modelled 2 peaks using fooof
num_params_k = num_params + 1# added the knee

#%%
cols_of_interest = ['heart_slope_avg', 'brain_slope_avg', 'no_ica_slope_avg', 'age', 'subject_id', 
                    'r2_heart', 'r2_ica', 'r2_no_ica', 'error_heart', 'error_ica', 'error_no_ica']

k_exps = df_cmb_k.query('channel == 0')[cols_of_interest]
k_exps['ap_mode'] = "Knee"
k_exps['ECG rejected'] = calculate_bic(n, k_exps['error_ica'], num_params_k)
k_exps['ECG not rejected'] = calculate_bic(n, k_exps['error_no_ica'], num_params_k)
k_exps['ECG Components'] = calculate_bic(n, k_exps['error_heart'], num_params_k)


#%%

exps = df_cmb.query('channel == 0')[cols_of_interest]
exps['ap_mode'] = "Fixed"
exps['ECG rejected'] = calculate_bic(n, exps['error_ica'], num_params)
exps['ECG not rejected'] = calculate_bic(n, exps['error_no_ica'], num_params)
exps['ECG Components'] = calculate_bic(n, exps['error_heart'], num_params)



exps_cmb = pd.concat([k_exps, exps])
bic_df = exps_cmb[['ap_mode', 'ECG not rejected', 'ECG rejected',  'ECG Components', 'subject_id']].melt(id_vars=['subject_id', 'ap_mode'],
                                                                                     var_name='condition',
                                                                                     value_name='BIC')

r2_df = exps_cmb[['ap_mode', 'r2_no_ica', 'r2_ica', 'r2_heart', 'subject_id']].melt(id_vars=['subject_id', 'ap_mode'],
                                                                                     var_name='condition',
                                                                                     value_name='R2')

#%%
my_colors = ['#8da0cb', '#fc8d62', '#66c2a5']
my_order = ['ECG not rejected', 'ECG rejected', 'ECG Components',]

g = sns.catplot(bic_df, x='ap_mode', y='BIC', 
            hue='condition', col='condition',
            palette=my_colors, margin_titles=True, 
            kind='point', sharey=False)

g.set_titles(col_template="{col_name}")

for ax in g.axes.flat:
    ax.set_xlabel('')

g.figure.savefig('../results/bic_comparison_knee.svg')

#%%
my_colors = ['#8da0cb', '#fc8d62', '#66c2a5']
my_order = ['r2_no_ica', 'r2_ica', 'r2_heart',]

g = sns.catplot(r2_df, x='ap_mode', y='R2', 
            hue='condition', col='condition',
            palette=my_colors, margin_titles=True, 
            kind='point', sharey=False)

g.set_titles(col_template="{col_name}")

for ax in g.axes.flat:
    ax.set_xlabel('')

g.figure.savefig('../results/r2_comparison_knee.svg')

