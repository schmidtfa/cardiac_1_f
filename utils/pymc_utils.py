import statsmodels.api as sm
import pandas as pd
import numpy as np
import pymc as pm
import aesara.tensor as at



def aggregate_sign_feature(df_in, feature_key, pos_mask):

    '''
    Aggregate data across channels per subject for each feature of interest.
    '''
    
    feature_by_age = []

    for subject in df_in['subject_id'].unique():
        cur_subject = df_in.query(f'subject_id == "{subject}"')
        feature_by_age.append(cur_subject
                                         .sort_values(by='channel')[pos_mask]
                                         .mean()[[feature_key, 'age']])

    df = pd.concat(feature_by_age, axis=1).T
    df['subject_id'] = df_in['subject_id'].unique()
    return df


def coefficients2pcorrs(df4mdf, mdf, response_var, predictor_vars, md=None):

    '''
    Compute partial correlations based on regression coefficients. 
    This code was taken from this example (https://bambinos.github.io/bambi/main/notebooks/ESCS_multiple_regression.html) 
    and turned into a function for convenience.

    Parameters:
    -------------------------------------
    df4mdf: pandas dataframe used for model fitting
    mdf: trace of the posterior for your multiple regression model
    response_var: string of the dependent variable used in the model
    predictor_vars: list of the independent variables (as strings) used in the model
    md: if you used bambi you can insert your build bambi model here

    '''

    samples = mdf.posterior
    # x_matrix = common effects design matrix (excluding intercept/constant term)
    if md:
        terms = [t for t in md.common_terms.values() if t.name != "Intercept"]
        x_matrix = [pd.DataFrame(x.data, columns=x.levels) for x in terms]
        x_matrix = pd.concat(x_matrix, axis=1)

    else:
        x_matrix = df4mdf[predictor_vars]

    dm_statistics = {
        'r2_x': pd.Series(
            {
                x: sm.OLS(
                    endog=x_matrix[x],
                    exog=sm.add_constant(x_matrix.drop(x, axis=1))
                    if "Intercept" in [i for i in mdf.posterior.data_vars] 
                    else x_matrix.drop(x, axis=1),
                )
                .fit()
                .rsquared
                for x in list(x_matrix.columns)
            }
        ),
        'sigma_x': x_matrix.std(),
        'mean_x': x_matrix.mean(axis=0),
    }

    r2_x = dm_statistics['r2_x']
    sd_x = dm_statistics['sigma_x']
    r2_y = pd.Series([sm.OLS(endog=df4mdf[response_var],
                             exog=sm.add_constant(df4mdf[[p for p in predictor_vars if p != x]])).fit().rsquared
                    for x in predictor_vars], index=predictor_vars)
    sd_y = df4mdf[response_var].std()

    # compute the products to multiply each slope with to produce the partial correlations
    slope_constant = (sd_x[predictor_vars] / sd_y) * ((1 - r2_x[predictor_vars]) / (1 - r2_y)) ** 0.5
    pcorr_samples = samples[predictor_vars] * slope_constant

    df_pcorr = pd.DataFrame(np.array([pcorr_samples[corr].to_numpy().flatten() for corr in pcorr_samples]).T)
    df_pcorr.columns = predictor_vars
    tidy_pcorr = df_pcorr.melt()
    tidy_pcorr.columns = ['predictors', 'partial correlation coefficient']

    return tidy_pcorr
