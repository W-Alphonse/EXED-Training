# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

from valeo.infrastructure.LogManager import LogManager


class Transformer() :
    logger = LogManager.logger(__name__)

    ''' 
    A strategy for imputing missing values by modeling each feature with missing values as a function of other features in a round-robin fashion. 
    Multivariate imputer that estimates each feature from all the others.
    https://scikit-learn.org/stable/modules/impute.html#iterative-imputer
    
    Arg:
    ----
    estimator : The estimator to use at each step of the round-robin imputation.
    
    Returns:
    --------
    A transformed Dataframe containing all the missing values.
    NB: The arguement Dataframe is NOT modified => It stills intact  
    https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7
    '''
    def iterative_imputer_transform(self, df_to_transform : pd.DataFrame,  estimator=BayesianRidge(), missing_values=np.nan,  max_iter=10, initial_strategy = 'median') -> pd.DataFrame :
        cols = df_to_transform.columns
        imputer = IterativeImputer(estimator=estimator, missing_values=missing_values,  max_iter=max_iter, initial_strategy=initial_strategy,  add_indicator=False) # It models each feature with missing values as a function of other features, and uses that estimate for imputation
        df_transformed = pd.DataFrame(imputer.fit_transform(df_to_transform))
        # df_transformed.columns = df_transformed.columns[:-1]
        df_transformed.columns = cols
        return df_transformed

    def robust_scaler_transform(self, df_to_transform : pd.DataFrame, with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0)):
        cols = df_to_transform.columns
        scaler  =  RobustScaler(with_centering=with_centering, with_scaling=with_scaling, quantile_range=quantile_range)
        df_transformed = pd.DataFrame(scaler.fit_transform(df_to_transform))
        df_transformed.columns = cols
        return df_transformed