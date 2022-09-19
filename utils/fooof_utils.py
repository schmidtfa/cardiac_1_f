#This is a set of functions i use to interact with the fooof objects
#Imports
import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib import cm
from fooof.analysis import get_band_peak_fg
from fooof.objs import combine_fooofs
from fooof import FOOOFGroup, fit_fooof_3d
import pandas as pd


def check_nans(data, nan_policy='zero'):
    """
    Check an array for nan values, and replace, based on policy.
    """

    # Find where there are nan values in the data
    nan_inds = np.where(np.isnan(data))

    # Apply desired nan policy to data
    if nan_policy == 'zero':
        data[nan_inds] = 0
    elif nan_policy == 'mean':
        data[nan_inds] = np.nanmean(data)
    else:
        raise ValueError('Nan policy not understood.')

    return data

def check_infs(data, inf_policy='zero'):
    """
    Check an array for inf values, and replace, based on policy.
    """

    # Find where there are nan values in the data
    inf_inds = np.where(np.isinf(data))

    # Apply desired nan policy to data
    if inf_policy == 'zero':
        data[inf_inds] = 0
    else:
        raise ValueError('Inf policy not understood.')

    return data

#remove outliers
def check_outliers(data, thresh):
    """
    Calculate indices of outliers, as defined by a standard deviation threshold. Similar as in Donoghue et al. 2020.
    Yet slightly different. Function returns an array of bools indicating whether or not the error or r2 is an outlier.
    True if the parameter is an inlier
    """

    return np.abs(data - np.mean(data)) < thresh * np.std(data)



def get_good_idx(fg, thresh=2.5):

    '''
    Used to identify bad/good model fits based on a threshold in sd
    '''

    r2 = check_outliers(fg.get_params('r_squared'), thresh)
    err = check_outliers(fg.get_params('error'), thresh)
    good_idx = np.logical_and(r2, err)
    return good_idx

def plot_band_peak_topos(fg, chan_type, bands):

    '''
    Plot peak power for different bands as topography.
    '''

    _, axes = plt.subplots(1, 4, figsize=(15, 5))
    for ind, (label, band_def) in enumerate(bands):

        # Get the power values across channels for the current band
        band_power = check_nans(get_band_peak_fg(fg, band_def)[:, 1])

        # Create a topomap for the current oscillation band
        mne.viz.plot_topomap(band_power, chan_type, cmap=cm.viridis, contours=0,
                             axes=axes[ind], show=False);
        
        #cbar = plt.colorbar(sm, orientation='vertical', label='Power')

        # Set the plot title
        axes[ind].set_title(label + ' power', {'fontsize' : 20})
        
        

def check_my_foofing(fgs):

    '''
    Simple wrapper to combine a list of Fooof objects into a single one and display model diagnostics.
    '''

    all_fg = combine_fooofs(fgs)
    # Explore the results from across all model fits
    all_fg.print_results()
    all_fg.plot()



def get_fooof_data(fg, param_type='aperiodic_params', param='exponent', impute=True, thresh=2.5):  # TH: consider pulling the `thresh` parameter up
                                                                                                   # i.e., in the `fooof2aperiodics` function                                                                                          # as this is the one that gets called by the user.
    '''
    Gets fooof data and imputes outlying fits using the median (if wanted)
    '''
    
    params = fg.get_params(param_type, param)
    if impute == True:  # TH: so, this basically replaces the bad elements with
                        # the median of all elements?
        bad_idx = get_good_idx(fg, thresh) == False
        median = np.median(params)  # TH: Please be aware that this median INCLUDES THE BAD ONES! Is this really what you want?
        params[bad_idx] = median
        
    return params

def get_good_aps(fg):
    
    '''
    This function returns "good" aperiodic components. Good is determined by the quality of the model fit.
    '''

    aps = pd.DataFrame(fg.get_params('aperiodic_params'))
    aps.columns = ['Offset' ,'Exponent']

    aps_clean = aps.loc[get_good_idx(fg, thresh=2)].reset_index()
    return aps_clean


def fooof2aperiodics(freqs, freq_range, psd, is_3d=False):

    '''
    fits a fooof model without peaks to extract and return aperiodics only 
    '''
    
    fg = FOOOFGroup(max_n_peaks=0) #fit no peaks to speed-up processing

    if is_3d:
        fgs = fit_fooof_3d(fg, freqs, psd, freq_range=freq_range)
        exponents = np.mean([get_fooof_data(fg, param='exponent') for fg in fgs], axis=0)
        offsets = np.mean([get_fooof_data(fg, param='offset') for fg in fgs], axis=0)

        aps_clean = pd.concat([get_good_aps(fg) for fg in fgs]).groupby('index').mean()

    else:
        fg.fit(freqs, psd, freq_range=freq_range)

        exponents = get_fooof_data(fg, param='exponent')
        offsets = get_fooof_data(fg, param='offset')

        aps_clean = get_good_aps(fg)

    return exponents, offsets, aps_clean
