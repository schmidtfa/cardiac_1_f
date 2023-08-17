#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

sns.set_style('ticks')
sns.set_context('poster')
# %% This is just a visualization of the results table previously reported

not_rejected = pd.DataFrame({
    'ICA-Threshold (0.4)': [-.235, -.102],
    'ICA-Threshold (0.5)': [-.250, -.120],
    'ICA-Threshold (0.6)': [-.242, -.088],
    'ICA-Threshold (0.7)': [-.232, -.062],
    'ICA-Threshold (0.8)': [-.240, -.025],

})
not_rejected['MEG'] = 'ECG not rejected'

rejected = pd.DataFrame({
    'ICA-Threshold (0.4)': [-.163, -.013],
    'ICA-Threshold (0.5)': [-.172, -.022],
    'ICA-Threshold (0.6)': [-.167, -.011],
    'ICA-Threshold (0.7)': [-.156, .015],
    'ICA-Threshold (0.8)': [-.175, .041],

})
rejected['MEG'] = 'ECG rejected'

comps = pd.DataFrame({
    'ICA-Threshold (0.4)': [-.387, -.244],
    'ICA-Threshold (0.5)': [-.379, -.231],
    'ICA-Threshold (0.6)': [-.365, -.210],
    'ICA-Threshold (0.7)': [-.362, -.195],
    'ICA-Threshold (0.8)': [-.395, -.176],

})
comps['MEG'] = 'ECG components'
# %%
df_ica = pd.concat([not_rejected, rejected, comps]).melt(id_vars='MEG', var_name='ICA', value_name='β (standardized)')
# %%
colors = ['#8da0cb', '#fc8d62', '#66c2a5',]

g = sns.catplot(df_ica, y='ICA', x='β (standardized)', col='MEG', palette=colors,
                 hue='MEG', kind='point', errorbar=('ci', 100))
g.figure.savefig('../results/ica_threshold_comp_fig.svg')
# %%
