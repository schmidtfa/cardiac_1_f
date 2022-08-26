import pyriemann


def run_potato(epochs, potato_threshold=2):

    '''
    This function takes an mne.epochs object and identifies outlying ("bad") segments using some riemann magic 
    '''

    # estimate the covariance matrices from our epochs
    covs = pyriemann.estimation.Covariances(estimator="lwf")
    cov_mats = covs.fit_transform(epochs.get_data())
        
    # Fit the Potato
    potato = pyriemann.clustering.Potato(threshold=potato_threshold)
    potato.fit(cov_mats)

    # Get the clean epoch indices and select only these for further processing
    clean_idx = potato.predict(cov_mats).astype(bool)
    return epochs[clean_idx]  