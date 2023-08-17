#%%
import numpy as np
from lisc import Words, Counts
from lisc.plts.counts import plot_matrix
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from lisc.plts.words import plot_years, plot_wordcloud
from lisc import Counts
from lisc.plts.utils import check_args, check_ax # counts_data_helper
import joblib
import matplotlib as mpl
from matplotlib.colors import ListedColormap
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)


sns.set_style('ticks')
sns.set_context('talk')
#%% set up a database
terms_measure =[['electrocardiography', 'ECG'], 
                ['electroencephalography', 'EEG', 
                 'magnetoencephalography', 'MEG', 
                 ]]

ap = ['power law', 'power-law', 'powerlaw', '1/f', 'aperiodic', 'scale-free']
terms_ap = [ap, ap]

terms_disorder = [['Age', 'ageing'], ['Sleep'], ['Working Memory', 'working memory'],
                  ['ADHD', 'ADD'], ['Autism', 'ASD'], ['Stroke'], ['Epilepsy'], ['Dementia', 'alzheimer', "alzheimer's"],
                  ['Heart failure', 'Cardiac Arrest', 'cardiovascular disease', 'congestive heart failure'],
                   ['Parkinson', "parkinson's"], ['Schizophrenia'], ['Multiple Sclerosis', 'MS'],
                  ]
# %% get co-occurences
counts = Counts()
counts.add_terms(terms_measure, dim='A')
counts.add_terms(terms_disorder, dim='B')
counts.add_terms(terms_ap, 'inclusions')

# Collect co-occurrence data
counts.run_collection(verbose=True)

#%%
print(counts.counts)
counts.compute_score('normalize', dim='A')



def plot_matrix(data, x_labels=None, y_labels=None,
                cmap='purple', square=True, ax=None, **kwargs):
    """Plot a matrix representation of given data.

    Parameters
    ----------
    data : Counts or 2d array
        Data to plot in matrix format.
    x_labels : list of str, optional
        Labels for the x-axis.
    y_labels : list of str, optional
        Labels for the y-axis.
    attribute : {'score', 'counts'}, optional
        Which data attribute from the counts object to plot the data for.
        Only used if the `data` input is a Counts object.
    transpose : bool, optional, default: False
        Whether to transpose the data before plotting.
    cmap : {'purple', 'blue'} or matplotlib.cmap
        Colormap to use for the plot.
        If string, uses a sequential palette of the specified color.
    square : bool
        Whether to plot all the cells as equally sized squares.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **kwargs
        Additional keyword arguments to pass through to seaborn.heatmap.

    Notes
    -----
    This function is a wrapper of the seaborn `heatmap` plot function.

    Examples
    --------
    See the example for the :meth:`~.compute_score` method of the :class:`~.Counts` class.
    """
    #data, x_labels, y_labels = counts_data_helper(data, x_labels, y_labels, attribute, transpose)


    g =sns.heatmap(data, square=square, ax=check_ax(ax, kwargs.pop('figsize', None)), cmap=cmap,
                    **check_args(['xticklabels', 'yticklabels'], x_labels, y_labels), annot=counts.counts,
                    **kwargs)
    plt.tight_layout()
    return g


#%% figure 1A
fig, ax = plt.subplots(figsize=(6,8))

g = plot_matrix(counts.score, 
            y_labels=['Heart (1/f)', 'Brain (1/f)'], 
            x_labels=counts.terms['B'].labels,
            cmap=ListedColormap(['white']), 
            linewidths=2, linecolor='black',
            cbar=False,
            square=True,
            ax=ax
)
ax.xaxis.set_tick_params(labeltop=True, top=True, labelbottom=False, bottom=False, labelrotation=90)


g.figure.set_size_inches(6, 8, forward=True)
g.figure.savefig('../results/article_counts_1_f.svg')

# %%
words = Words()
words.add_terms(terms_measure)
words.add_terms(terms_ap, 'inclusions')
words.run_collection(retmax=500)

# %%
titles_cmb = [True if title in words.results[1].titles else False for title in words.results[0].titles]

n_titles_cmb = np.sum(titles_cmb)
n_titles_heart = len(words.results[0].titles) - n_titles_cmb
n_titles_brain = len(words.results[1].titles) - n_titles_cmb

n_all_titles = np.sum([n_titles_heart, n_titles_brain, n_titles_cmb])

#%% Do a basic pie chart

labels = ['Brain (1/f)', 'Heart (1/f)', 'Heart + Brain (1/f)']
prop = [n_titles_brain, n_titles_heart, n_titles_cmb] / n_all_titles


fig, ax = plt.subplots()
ax.pie(prop, labels=labels, autopct='%1.1f%%', pctdistance=0.5,labeldistance=1.1, 
colors=['#FC8D62', '#66C2A5', '#8DA0CB',])
#['#66c2a5', '#fc8d62', '#8da0cb'])

centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
ax.axis('equal');  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.figure.set_size_inches(6, 6, forward=True)

ax.figure.savefig('./results/paper_overlap.svg')

#%% get titles that overlap
#Note: checked the four articles none consider confounding/mediating influences of ecg on meeg measurements
print(np.array(words.results[0].titles)[titles_cmb])
titles_cmb = [True if title in words.results[1].titles else False for title in words.results[0].titles]

#%%
years_heart = np.array([year for year in words.results[0].years if type(year) == int])
years_brain = np.array([year for year in words.results[1].years if type(year) == int])

#%%
sns.set_style('ticks')
sns.set_context('poster')

g = sns.kdeplot(years_heart, color='#66C2A5')
sns.kdeplot(years_brain, color='#FC8D62')

g.set_xlabel('Year of Publication')
g.set_ylabel('Number of Articles (relative)')
g.figure.set_size_inches(10, 5, forward=True)
sns.despine()
g.figure.savefig('../results/heart_brain_time.svg')
#%%
fig, axes = plt.subplots(nrows=2)
axes[0].hist(years_brain, color='#FC8D62')
axes[1].hist(years_heart, color='#66C2A5')

for ax in axes:
    ax.set_xlim(1975, 2023)

fig.tight_layout()
fig.supylabel('Number of Articles')
fig.supxlabel('Year of Publication')
sns.despine()
fig.savefig('../results/heart_brain_time.svg')


#%% fetch all the text data
#Note for this to run properly it is necessary to adjust the terms measure or select the word results more decisively

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import joblib
driver = webdriver.Chrome()

all_ecg, all_text_list, bad_idcs = [], [], []

DOI_BASE = 'https://doi.org/'

for idx, doi in enumerate(words.results[1].dois[:-1]):

    #if its a list just try the first one 
    if type(doi) == list:
        cur_doi = doi[0]
    else:
        cur_doi = doi

    try:
        driver.get(DOI_BASE + cur_doi)
        time.sleep(10) #add some time in between website queries: TODO: Maybe add some jitter to trick cloudflare

        all_text = driver.find_element(By.XPATH, "/html/body").text #grab the whole html body

        all_ecg.append(cur_ecg)
        all_text_list.append(all_text)
    
    except TypeError:
        bad_idcs.append(idx)

#%% dump it all (two different files one for meg and one for eeg)

data = {'bad_idcs': bad_idcs,
        'txt_data': all_text_list,
        'doi': doi,
        'word_list': words.results[1],
        'all_ecg': all_ecg}

#joblib.dump(data, 'text_data_eeg.dat')
joblib.dump(data, 'text_data_meg.dat')
