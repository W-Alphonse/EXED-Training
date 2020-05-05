from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute._base import _BaseImputer
import numpy as np
import pandas as pd

from valeo.infrastructure import Const as C

class NumericalImputer(BaseEstimator, TransformerMixin) :
    def __init__(self, variables, imputer:_BaseImputer) :
        self.imputer = imputer
        # self.variables = None if variables == None else ( [variables] if not isinstance(variables, list) else variables )
        self.variables =  [variables] if not isinstance(variables, list) else variables

    def fit(self, X, y=None):
        # if self.variables == None :
        #     self.variables = X.select_dtypes('number').columns.to_list()
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
            # print(f'X_new[self.variables]:{X_new[self.variables].shape}')
            # print(f'self.imputer.fit_transform(X_new[self.variables]):{self.imputer.fit_transform(X_new[self.variables]).shape}')

            # X_new[self.variables] = self.imputer.fit_transform(X_new[self.variables])
            X_new = self.imputer.fit_transform(X_new[self.variables])

            # X_new[:][self.variables] = self.imputer.fit_transform(X_new[:][self.variables])
        return X_new

class NumericalScaler(BaseEstimator, TransformerMixin) :
    def __init__(self, variables, scaler:_BaseImputer):
        self.scaler = scaler
        self.variables = [variables] if not isinstance(variables, list) else variables

    def fit(self, X, y=None):
        # if self.variables == None :
        #     self.variables = X.select_dtypes('number').columns.to_list()
        return self

    def transform(self, X):
        X_new = X[:][self.variables].copy()
        X_new = self.scaler.fit_transform(X_new[self.variables])
        return X_new

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, log10Tranformation=True):
        # self.variables = [variables] if not isinstance(variables, list) else variables
        self.log10Tranformation = log10Tranformation

    def fit(self, X, y=None):
        # if self.variables == None :
        #     self.variables = X.select_dtypes('number').columns.to_list()
        return self

    def transform(self, X):
        X_new = X.copy()
        # print(f'*** LogTransformer.columnx : {X.columns}')

        # # check that the values are non-negative for log transform
        # if not (X_new[self.variables] > 0).all().all():
        #     vars_ = self.variables[(X_new[self.variables] <= 0).any()]
        #     raise ValueError(
        #         f"Variables contain zero or negative values, "
        #         f"can't apply log for vars: {vars_}")
        #
        # for feature in self.variables:
        #     X_new[feature] = np.log(X_new[feature])
        X_new = X_new + 1
        # X_new[self.variables] = X_new[self.variables].applymap(np.log10)
        X_new = np.log10(X_new) if self.log10Tranformation else np.log2(X_new)
        return X_new


class SqrtTransformer(BaseEstimator, TransformerMixin):
    # def __init__(self, variables):
    #     self.variables = [variables] if not isinstance(variables, list) else variables

    def fit(self, X, y=None):
        # if self.variables == None :
        #     self.variables = X.select_dtypes('number').columns.to_list()
        return self

    def transform(self, X):
        X_new = X.copy()
        X_new = np.sqrt(X_new)
        return X_new


class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):
    # def __init__(self, variables_to_drop):
    #     self.variables_to_drop = [variables_to_drop] if not isinstance(variables_to_drop, list) else variables_to_drop

    def fit(self, X, y=None):
        return self

    # def transform(self, X):
    #     X_new = X.copy()
    #     print(f'X_new.columns-before:{X_new.columns}')
    #     X_new = X_new.drop([C.PROC_TRACEINFO], axis=1)
    #     print(f'X_new.columns-after:{X_new.columns}')
    #     return X_new

    def transform(self, X):
        # print(f'X[0,:] : {X[0,:]}')
        # print(f'X.columns: {X.columns}')
        return X.drop(X.columns.to_list(), axis=1)
        # return X


class ProcDateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        proc_splitted = X_new[C.PROC_TRACEINFO].str.rsplit(pat='-', n=2, expand=True)
        proc_date = pd.to_datetime(proc_splitted[1], format="%y%m%d")
        #   xx/04/2019 < proc_date < xx/09/2019
        X_new['month']   = pd.cut( proc_date.dt.month,
                                bins = [-np.inf,5, 6, 7, 8, 9, np.inf], labels=['4', '5', '6', '7', '8', '9'])
        X_new['week']    = pd.cut( proc_date.dt.week,
                                bins = [-np.inf, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, np.inf],
                                labels=['16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31','32','33','34','35','36','37'])
        X_new['weekday'] = pd.cut( proc_date.dt.weekday,
                                bins=[-np.inf, 1, 2, 3, 4, 5, 6, np.inf], labels=['1', '2', '3', '4', '5', '6', '7'])
        X_new = X_new.drop([C.PROC_TRACEINFO], axis=1) #, inplace=True)
        return X_new

# Générer un nombre de jour à partir du numéro de série
# z = pd.merge(data, Y_data, on=Const.PROC_TRACEINFO, suffixes=('','_right'))
# zz = z[z['Binar OP130_Resultat_Global_v'] == 1][['PROC_TRACEINFO', 'OP100_Capuchon_insertion_mesure', 'proc_date', 'proc_index']].sort_values(by='proc_index')
# zzz = zz['proc_index'].diff()
# pd.concat([zz, zzz], axis=1)

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