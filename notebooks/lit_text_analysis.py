#%%
import numpy as np
from lisc import Words, Counts
from lisc.plts.counts import plot_matrix
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from lisc.plts.words import plot_years, plot_wordcloud
import joblib
import matplotlib as mpl
import joblib
import textwrap

import pandas as pd
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)


sns.set_style('ticks')
sns.set_context('talk')

#%% we only go back to 1997 texts from before are poorly digitized.
ecg = joblib.load('../data/text_data_eeg.dat')['txt_data']

#%%

min_txt_len_eeg = np.array([len(txt) for txt in ecg]) > 10_000 #drop the very short htmls as the website was probably not accesible
all_full_texts_eeg = np.array(ecg)[min_txt_len_eeg]

#%%
meg_info = joblib.load('../data/text_data_meg.dat')

min_txt_len_meg = np.array([len(txt) for txt in meg_info['txt_data']]) > 10_000
all_full_texts_meg = np.array(meg_info['txt_data'])[min_txt_len_meg]


#%%
def get_keyword_indices(all_full_texts):
    
    search_words_cardiac = ['cardio', ' cardiac', ' heart', 'ecg']
    reject_word_list = ['doi',  'pmid', 'scopus', 'google scholar', 'pubmed'] #content words that lead to a different article (e.g. in references)
    dict_list = []
    n_chars_past = 600

    for idx, txt in enumerate(all_full_texts):
        
        cur_dict = {
                    'valid': False,
                    'word_context': '',
                    'full_text': ''
                    }

        for search_word in search_words_cardiac:
            cur_txt = txt.lower()
            cur_dict_list = []
            while search_word in cur_txt:
                cur_word_idx = cur_txt.find(search_word)
                word_context = cur_txt[cur_word_idx - 400:cur_word_idx + n_chars_past]
                if any(reject_word in word_context for reject_word in reject_word_list) == False:

                    text_split = word_context.split(' ')[1:-2]
                    text_split2  = ' '.join(text_split).splitlines() #remove \n
                    text4nlp = ' '.join(text_split2) #join again
                    cur_dict['word_context'] = text4nlp
                    
                    if text4nlp == '':
                        cur_dict['valid'] = False
                    else:
                        cur_dict['valid'] = True
                        cur_dict['full_text'] = cur_txt
                cur_dict_list.append(pd.DataFrame(cur_dict, index=[idx]))
                cur_txt = cur_txt[cur_word_idx+len(search_word)+n_chars_past:-1]
            if cur_dict_list == []:
                cur_dict['full_text'] = cur_txt
                dict_list.append(pd.DataFrame(cur_dict, index=[idx])) 
            else:
                dict_list.append(pd.concat(cur_dict_list))

    return pd.concat(dict_list)


#%%
eeg_df = get_keyword_indices(all_full_texts_eeg).reset_index()
meg_df = get_keyword_indices(all_full_texts_meg).reset_index()


#%% cleaning keywords
search_words_dss = ['dss', 'denoising source separation']
search_words_svd = ['svd', 'singular value decomposition']
search_words_ecg = ['ecg', 'electrocardiography', 'electrocardiogram']
search_words_ica = ['ica', 'independent component analysis']
search_words_sss = ['sss', 'signal space seperation', 'signal-space seperation']
search_words_ssp = ['ssp', 'signal-space projection', 'signal space projection']

#%%

def process_df(df):

    cur_valid_meg = df.query('valid == True')

    from nltk.stem.snowball import SnowballStemmer

    stemmer = SnowballStemmer(language='english')

    invalid_docs = []
    remove_stems = ['remov', 'discard', 'reject']

    for cur_idx in range(cur_valid_meg.shape[0]):
        removed_present = []
        for cur_stem in remove_stems:
            doc = cur_valid_meg.iloc[cur_idx]['word_context'].split(' ')
            cols = ["stem"]
            rows = []

            
            for token in doc:
                row = stemmer.stem(token)
                rows.append(row)
            #removed_present.append(cur_stem in rows) #check this
            removed_present.append(pd.DataFrame(rows, columns=cols).query(f'stem == "{cur_stem}"').shape[0] > 0)
        #print(removed_present)
        if any(removed_present) == False: #if all
            invalid_docs.append(cur_idx)

    #df.drop(index=invalid_docs, inplace=True)
    for cur_doc_ix in invalid_docs:
        df[df['index'] == cur_doc_ix]['valid'] = False
        #df[df['index'] == cur_doc_ix]['full_text'] = ''
        #df[df['index'] == cur_doc_ix]['word_context'] = ''

    #% rejoin texts for each "original" index and query those for cleaning tools
    list4rejoin = []

    for cur_idx in df['index'].unique():
        cur_meg = df.query(f'index == {cur_idx}')
        if cur_meg.shape[0] > 1:
            list4rejoin.append(pd.DataFrame(cur_meg.sum()).T)
        else:
            list4rejoin.append(cur_meg)

    meg_df_final = pd.concat(list4rejoin)[['valid', 'word_context', 'full_text']]
    meg_df_final['valid'] = meg_df_final['valid'] > 0
    #%

    def word_searcher(df, search_words):

        search_my_words = lambda search_words, txt: any(word in txt for word in search_words) #else False

        feature_vector = []
        for cur_txt in df['word_context']:
            if cur_txt == '':
                feature_vector.append(False)
            else:
                feature_vector.append(search_my_words(search_words=search_words, txt=cur_txt))
        return feature_vector


    meg_df_final['ECG'] = word_searcher(meg_df_final, search_words_ecg)
    meg_df_final['ICA'] = word_searcher(meg_df_final, search_words_ica)
    meg_df_final['SVD'] = word_searcher(meg_df_final, search_words_svd)
    meg_df_final['SSS'] = word_searcher(meg_df_final, search_words_sss)
    meg_df_final['SSP'] = word_searcher(meg_df_final, search_words_ssp)
    meg_df_final['DSS'] = word_searcher(meg_df_final, search_words_dss)

    return meg_df_final
#%%
meg_df_final = process_df(meg_df).reset_index().drop(columns='index')
eeg_df_final = process_df(eeg_df).reset_index().drop(columns='index')

#%%
tmp_eeg_valid = eeg_df_final.query('valid == True')
tmp_meg_valid = meg_df_final.query('valid == True')

#%% find word indices to extract word context
#go through residual indices

# meg_ix = 35
# print(tmp_meg_valid.index[meg_ix])
# tmp_meg_valid.iloc[meg_ix]['word_context']

#%% omit residual studies
omit_idcs_meg = [0, 59]

for ix in omit_idcs_meg:
    meg_df_final.loc[ix, 'valid'] = False

drop_list_meg = [3, 4, 38, 39, 40]
meg_df_final.drop(drop_list_meg, inplace=True)

#%%
# eeg_ix = 81
# print(tmp_eeg_valid.index[eeg_ix])
# tmp_eeg_valid.iloc[eeg_ix]['word_context']

# tmp_eeg_valid.iloc[eeg_ix]['full_text']

#%%
omit_idcs_eeg = [20, 21, 22, 24, 29, 30,
                 34, 41, 55, 63, 79, 81,
                 84, 85, 90, 95, 99, 102,
                 108, 123, 126, 129,
                 136, 155, 170, 175, 180,
                 184, 195, ]

for ix in omit_idcs_eeg:
    eeg_df_final.loc[ix, 'valid'] = False


#%%
drop_list_eeg = [10, 11, 26, 35, 52, 61, 67,
                 130, 137, 153, 166, 178, 179,
                 182, 184, 197, 202, 205, 209,
                 ]
eeg_df_final.drop(drop_list_eeg, inplace=True)

eeg_df_final = eeg_df_final.query('SSS == False')

#%%
meg_df_final.mean()
#%%
meg_df_final.query('valid == True').mean()


#%%
eeg_df_final.mean()
#%%
eeg_df_final.query('valid == True').mean()

#%%


percentages = eeg_df_final.query('valid == True').mean()[['ICA', 'SVD', 'SSS', 'SSP', 'DSS']] * 100
#
wrapped_labels = [textwrap.fill(col, 12) for col in percentages.index]

# Plot the stacked bar chart
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(wrapped_labels, percentages, color='#F5DF4D')
ax.bar(wrapped_labels, percentages-100, color='#939597', bottom=100)

plt.xlabel('')
plt.ylabel('Cleaning Methods (%)')
plt.legend(['True', 'False'])
plt.ylim(0, 100)
sns.despine()

fig.savefig('../results/eeg_cleaning.svg')
#%%
percentages = eeg_df_final.mean()[['valid', 'ECG']] * 100
percentages.index = ['Cardiac (removed)', 'ECG (recorded)']
wrapped_labels = [textwrap.fill(col, 12) for col in percentages.index]

# Plot the stacked bar chart
fig, ax = plt.subplots(figsize=(3, 4))
ax.bar(wrapped_labels, percentages, color='#F5DF4D')
ax.bar(wrapped_labels, percentages-100, color='#939597', bottom=100)

plt.xlabel('')
plt.ylabel('EEG Studies (%)')
plt.legend(['True', 'False'])
plt.ylim(0, 100)
sns.despine()

fig.savefig('../results/heart_keywords_eeg.svg')

#%%


percentages = meg_df_final.query('valid == True').mean()[['ICA', 'SVD', 'SSS', 'SSP', 'DSS']] * 100

wrapped_labels = [textwrap.fill(col, 12) for col in percentages.index]

# Plot the stacked bar chart
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(wrapped_labels, percentages, color='#F5DF4D')
ax.bar(wrapped_labels, percentages-100, color='#939597', bottom=100)

plt.xlabel('')
plt.ylabel('Cleaning Methods (%)')
plt.legend(['True', 'False'])
plt.ylim(0, 100)
sns.despine()

fig.savefig('../results/meg_cleaning.svg')

#%%
percentages = meg_df_final.mean()[['valid', 'ECG']] * 100
percentages.index = ['Cardiac (removed)', 'ECG (recorded)']
#wrapped_labels = [textwrap.fill(col, 12) for col in percentages.index]

# Plot the stacked bar chart

wrapped_labels = [textwrap.fill(col, 12) for col in percentages.index]

# Plot the stacked bar chart
fig, ax = plt.subplots(figsize=(3, 4))
ax.bar(wrapped_labels, percentages, color='#F5DF4D')
ax.bar(wrapped_labels, percentages-100, color='#939597', bottom=100)

plt.xlabel('')
plt.ylabel('MEG Studies (%)')
plt.legend(['True', 'False'])
plt.ylim(0, 100)
sns.despine()

fig.savefig('../results/heart_keywords_meg.svg')