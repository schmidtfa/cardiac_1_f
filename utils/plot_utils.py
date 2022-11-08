
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import mne
import numpy as np
import arviz as az


def plot_ridge(df, variable_name, values, pal, plot_order=None, xlim=(-1,1), aspect=15, height=0.5):

      '''Pretty ridge plot function in python. This is based on code from https://seaborn.pydata.org/examples/kde_ridgeplot.html'''

      sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

      plotting_kwargs = {'row':variable_name, 
                         'hue': variable_name, 
                         'aspect': aspect,
                         'height': height,
                         'palette': pal}

      if plot_order is not None:
            plotting_kwargs.update({'row_order': plot_order,
                                    'hue_order': plot_order})
      
      g = sns.FacetGrid(df, **plotting_kwargs)

      g.map(sns.kdeplot, values,
            bw_adjust=.5, clip_on=False,
            fill=True, alpha=1, linewidth=1.5)
      g.map(sns.kdeplot, values, clip_on=False, color="w", lw=2, bw_adjust=.5)

      # passing color=None to refline() uses the hue mapping
      g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

      # Define and use a simple function to label the plot in axes coordinates
      def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color=color,
                  ha="left", va="center", transform=ax.transAxes)

      g.map(label, values)

      # Set the subplots to overlap
      g.figure.subplots_adjust(hspace=-.25)

      # Remove axes details that don't play well with overlap
      g.set_titles("")
      g.set(xlim=xlim)
      g.set(yticks=[], ylabel="")
      g.despine(bottom=True, left=True)

      return g


def plot_slope_corr(df, key_a, key_b, x, y, color):

    '''Utility function to plot a regression line + dots and a correlation coefficient'''

    corr = pg.corr(df[key_a], df[key_b])
    print(corr)
    g = sns.lmplot(data=df, x=key_a, y=key_b, line_kws={'color': color},
                   scatter_kws={"s": 40, 'color': '#888888', 'alpha': 0.25})

    r = round(float(corr['r']), 2)
    p = round(float(corr['p-val']), 3)

    if p == 0.0:
        p = 'p < 0.001'
    else:
        p = f'p = {p}'

    plt.annotate(text=f'r = {r}', xy=(x, y))
    plt.annotate(text=p, xy=(x, y - 0.2))

    return g




def plot_corr_topo(corr, mask, info, title, vmin=None, vmax=None, cmap='RdBu_r'):

    '''Function to dump a numpy array on an meg helmet'''
    
    sns.set_style('ticks')
    sns.set_context('talk')
    corr = np.expand_dims(corr, axis=1) #matrix.shape = n_subjects, n_channels
    
    evoked = mne.EvokedArray(corr, info, tmin=0.)

    mask = np.expand_dims(mask, axis=1)
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=10)

    topo = evoked.plot_topomap(times=[0], scalings=1,
                        time_format=None, cmap=cmap,
                        vmin=vmin, vmax=vmax,
                        units='beta', cbar_fmt='%0.3f', mask=mask, 
                        mask_params=mask_params, title=title,
                        size=3, time_unit='s');
    return topo



def plot_bayes_linear_regression(df, fitted, x_key, y_key, line_color, random_factor=None, add_ppm=True):

    '''This function can be used to visualize the results of a bayesian linear regression.
        
        df: Pandas DataFrame of the data used to build the model
        fitted: InferenceData object of the fitted model
        x_key: String referring to the x-axis label
        y_key: String referring to the y-axis label
        random_factor:  if this is a string use it to get information for plotting
        
        Note: Requires model prediction to be run first to get y mean
    '''
    
    Intercept = fitted.posterior.Intercept.values.mean()
    Slope = fitted.posterior[x_key].values.mean()
    x_range = np.linspace(df[x_key].min(), df[x_key].max(), fitted.posterior.dims['draw'])
    regression_line = Intercept + Slope * x_range

    g = sns.scatterplot(x=x_key, y=y_key, data=df, color='#888888', alpha=0.25)
    plt.plot(x_range, regression_line, color=line_color, linewidth=3)
    
    if add_ppm == True:
        if random_factor == None:
            hdi2plot = fitted.posterior[f"{y_key}_mean"]
            az.plot_hdi(x=df[x_key], y=hdi2plot, color=line_color)
        else:
            #annoying & tedious but safe
            reaction_mean = fitted.posterior[f"{y_key}_mean"].stack(samples=("draw", "chain")).values
            hdi_list = []

            for _, subject in enumerate(df[random_factor].unique()):
                idx = df.index[df[random_factor] == subject].tolist()
                hdi_list.append(reaction_mean[idx].T)
            
            hdi2plot = np.array(hdi_list)
            az.plot_hdi(x=df[x_key].unique(), y=hdi2plot, color=line_color, hdi_prob=0.68)
        
    return g