# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

import pandas as pd
# from valeo.infrastructure import LogManager
from valeo.infrastructure.LogManager import LogManager


class Transformer() :

    def __init__(self):
        lm = LogManager()
        self.logger = lm.logger(__name__)

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
    '''
    def iterative_imputer_transform(self, df_to_transform : pd.DataFrame,  estimator=BayesianRidge()) -> pd.DataFrame :
        cols = df_to_transform.columns
        # print (type(cols))
        imputer = IterativeImputer(BayesianRidge())
        df_transformed = pd.DataFrame(imputer.fit_transform(df_to_transform))
        df_transformed.columns = cols
        return df_transformed