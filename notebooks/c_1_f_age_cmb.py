#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg


sns.set_style('ticks')
sns.set_context('talk')
# %%
df_sbg = pd.read_csv('../data/sbg_1_f_dataframe_1_200.csv')
df_sbg['location'] = 'sbg'
df_cam_can = pd.read_csv('../data/cam_can_1_f_dataframe_1_200.csv')
df_cam_can['location'] = 'cam'
# %%
df_cmb = pd.concat([df_sbg, df_cam_can])
# %%
cur_df_cmb = df_cmb.query('channel == 0')

#%%
def plot_slope_age_corr(key, x, y, color):
    corr = pg.corr(cur_df_cmb['age'], cur_df_cmb[key])

    g = sns.lmplot(data=cur_df_cmb, x='age', y=key, 
                    line_kws={'color': color}, 
                    scatter_kws={"s": 40, 'color': '#888888', 'alpha': 0.25})

    r = round(float(corr['r']), 2)
    p = round(float(corr['p-val']), 3)

    if p == 0.0:
        p = 'p < 0.001'
    else:
        p = f'p = {p}'

    plt.annotate(text=f'r = {r}', xy=(x, y))
    plt.annotate(text=p, xy=(x, y - 0.2))

    g.set_xlabels('age (years)')
    g.set_ylabels('1/f slope')
    #g.ax.figure.savefig(f'../results/corr_{key}_{my_freqs}.svg', )
#%%
plot_slope_age_corr('no_ica_slope_avg', 20, 2., '#e78ac3')
#%%
plot_slope_age_corr('brain_slope_avg', 20, 2., '#66c2a5')
#%%
plot_slope_age_corr('heart_slope_avg', 20, 2.6, '#fc8d62')
#%%
plot_slope_age_corr('heart_slope', 20, 2.2, '#8da0cb')
# %%
