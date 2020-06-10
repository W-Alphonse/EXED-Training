from category_encoders import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute._base import _BaseImputer
import numpy as np
import pandas as pd

from valeo.infrastructure import Const as C

class NumericalImputer(BaseEstimator, TransformerMixin) :
    def __init__(self, variables, imputer:_BaseImputer) :
        self.imputer = imputer
        self.variables =  [variables] if not isinstance(variables, list) else variables

    def fit(self, X, y=None):
        if self.imputer == None :  # If self.imputer not set then use the most frequent imputer as the default one
            self.imputer_dict_ = {}
            for feature in self.variables:
                # The mode of a set of values is the value that appears most often. It can be multiple values.
                # Therefore, we retaine the first most frequent value
                self.imputer_dict_[feature] = X[feature].mode()[0]
        return self

    def transform(self, X):
        X_new = X[self.variables].copy()
        if self.imputer == None :
            for feature in self.variables:
                X_new[feature].fillna(self.imputer_dict_[feature], inplace=True)
        else :
            X_new = self.imputer.fit_transform(X_new[self.variables])
        return X_new

class NumericalScaler(BaseEstimator, TransformerMixin) :
    def __init__(self, variables, scaler:_BaseImputer):
        self.scaler = scaler
        self.variables = [variables] if not isinstance(variables, list) else variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X[:][self.variables].copy()
        X_new = self.scaler.fit_transform(X_new[self.variables])
        return X_new

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, log10Tranformation=True):
        self.log10Tranformation = log10Tranformation

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        X_new = X_new + 1
        X_new = np.log10(X_new) if self.log10Tranformation else np.log2(X_new)
        return X_new


class SqrtTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        X_new = np.sqrt(X_new)
        return X_new


class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(X.columns.to_list(), axis=1)


class ProcDateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        proc_splitted = X_new[C.PROC_TRACEINFO].str.rsplit(pat='-', n=2, expand=True)
        proc_date = pd.to_datetime(proc_splitted[1], format="%y%m%d")
        #   20/04/2019 <= proc_date < 07/09/2019
        X_new[C.proc_month]   = pd.cut( proc_date.dt.month,
                            bins = [-np.inf,4, 5, 6, 7, 8, np.inf], labels=['Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Sept'])
        X_new[C.proc_week]    = pd.cut( proc_date.dt.week,
                            bins = [-np.inf, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, np.inf],
                            labels = ['<= 21/04', '<= 28/04', '<= 05/05', '<= 12/05', '<= 19/05', '<= 26/05', '<= 02/06', '<= 09/06', '<= 16/06', '<= 23/06',
                                      '<= 30/06', '<= 07/07', '<= 14/07', '<= 21/07',
                                      '<= 28/07', '<= 04/08','<= 11/08','<= 18/08','<= 25/08','<= 01/09','<= 08/09'])
        X_new[C.proc_weekday] = pd.cut( proc_date.dt.weekday,
                            bins = [-np.inf, 0, 1, 2, 3, 4, 5, np.inf], labels = ['Lundi', 'Mardi', 'Merc', 'Jeudi', 'Vend', 'Samedi', 'Dim'])
        X_new = X_new.drop([C.PROC_TRACEINFO], axis=1) #, inplace=True)
        return X_new

class ProcDatePcaTransformer(ProcDateTransformer):
    def fit(self, X, y=None):
        return super().fit(X)

    def transform(self, X):
        X_new = super().transform(X)
        print(f'X_new.type:{type(X_new)}')
        #
        m = OneHotEncoder().fit_transform(X_new[C.proc_month])
        pca = PCA(n_components = 0.90).fit_transform(m)
        print(f'X_new[C.proc_month]:{type(pca)} - {pca.shape}')
        #
        month_pca = PCA(n_components = 0.90).fit_transform( OneHotEncoder().fit_transform(X_new[C.proc_month]) )
        print(f'month_pca:{month_pca.shape}')
        week_pca = PCA(n_components = 0.90).fit_transform( OneHotEncoder().fit_transform(X_new[C.proc_week]) )
        print(f'week_pca:{week_pca.shape}')
        weekday_pca = PCA(n_components = 0.90).fit_transform( OneHotEncoder().fit_transform(X_new[C.proc_weekday]) )
        print(f'weekday_pca:{weekday_pca.shape}')
        #
        X_new = X_new.drop([C.proc_month], axis=1)
        X_new = X_new.drop([C.proc_week], axis=1)
        X_new = X_new.drop([C.proc_weekday], axis=1)
        #
        # X_new = pd.concat([X_new, month_pca, week_pca, weekday_pca], axis = 1)
        date_pca = pd.DataFrame.from_records([month_pca, week_pca, weekday_pca])
        # print(date_pca.head())
        # print(date_pca.info())
        # print(date_pca.describe())
        X_new = pd.concat([X_new, date_pca], axis = 1)
        return X_new


class OP100CapuchonInsertionMesureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        # Option-1: Supposer que les valeurs Manquants étaient traitées par Imputer
        X_new["OP100_Capuchon_insertion_mesure_cat"] = pd.cut(X_new[C.OP100_Capuchon_insertion_mesure],
                                                      bins = [-np.inf, 0.2925, 0.335, 0.3775, np.inf], labels = [1, 2, 3, 4])
        #
        # Option-2: Isoler les valeurs manquants dans une catégorie indépendnate, la cat '4'
        # f = X_new[C.OP100_Capuchon_insertion_mesure].isna()
        # X_new.loc[f, "OP100_Capuchon_insertion_mesure_cat"] = '4'
        # X_new.loc[~f, "OP100_Capuchon_insertion_mesure_cat"] =pd.cut(X_new[~f][C.OP100_Capuchon_insertion_mesure],
        #                                              bins = [-np.inf, 0.32, 0.37, np.inf], labels = ['1', '2', '3'])
        #
        X_new = X_new.drop([C.OP100_Capuchon_insertion_mesure], axis=1)
        return X_new


class BucketTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, fld_binsList_lblList):
        self.fld_binsList_lblList = fld_binsList_lblList
        self.fld_to_bucketize = fld_binsList_lblList[0]
        self.bins  = fld_binsList_lblList[1]
        self.label = fld_binsList_lblList[2]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        X_new[f"{self.fld_to_bucketize}_cat"] = pd.cut(X_new[self.fld_to_bucketize], bins = self.bins, labels = self.label)
        X_new = X_new.drop([self.fld_to_bucketize], axis=1)
        return X_new


class EmtpyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X

# class CategoricalEncoder(BaseEstimator, TransformerMixin):
#     """String to numbers categorical encoder."""
#
#     def __init__(self, variables=None):
#         self.variables = [variables] if not isinstance(variables, list) else variables
#
#     def fit(self, X, y):
#         temp = pd.concat([X, y], axis=1)
#         temp.columns = list(X.columns) + ['target']
#
#         # persist transforming dictionary
#         self.encoder_dict_ = {}
#
#         for var in self.variables:
#             t = temp.groupby([var])['target'].mean().sort_values(
#                 ascending=True).index
#             self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}
#
#         return self
#
#     def transform(self, X):
#         # encode labels
#         X = X.copy()
#         for feature in self.variables:
#             X[feature] = X[feature].map(self.encoder_dict_[feature])
#
#         # check if transformer introduces NaN
#         if X[self.variables].isnull().any().any():
#             null_counts = X[self.variables].isnull().any()
#             vars_ = {key: value for (key, value) in null_counts.items()
#                      if value is True}
#             raise ValueError(
#                 f'Categorical encoder has introduced NaN when '
#                 f'transforming categorical variables: {vars_.keys()}')
#
#         return X