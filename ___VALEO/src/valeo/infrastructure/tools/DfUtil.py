from datetime import datetime

from pandas import Series
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection._search import BaseSearchCV

from valeo.infrastructure.LogManager import LogManager
from valeo.infrastructure import Const as C

import os
import pandas as pd
import numpy as np

class DfUtil() :
    logger = LogManager.logger(__name__)


    # https://stackabuse.com/pythons-classmethod-and-staticmethod-explained/
    @classmethod
    def read_csv(cls, pathAsStrList : [], sep=',') -> pd.DataFrame:
        try:
            return pd.read_csv(os.path.join(pathAsStrList[0], *pathAsStrList[1:]), sep=sep )
        except Exception as ex :
            cls.logger.exception("Error while load data from %s", "/".join(pathAsStrList))

    @classmethod
    def write_y_csv(cls, X_id:Series, y_target: np.ndarray, y_col_name:str, pathAsStrList : [], ts_type=C.ts_sfix):
        DfUtil.write_df_csv( pd.DataFrame(data={X_id.name:X_id, y_col_name:y_target}),  pathAsStrList, ts_type=ts_type)

    @classmethod
    def write_df_csv(cls, df:pd.DataFrame, pathAsStrList : [], ts_type=C.ts_sfix):
        try :
            df.to_csv( C.ts_pathanme(pathAsStrList,ts_type), index = False)
        except Exception as ex:
            cls.logger = LogManager.logger("DfUtil")
            cls.logger.exception(f"Error while writing 'df' to CSV '{pathAsStrList}'")

    @classmethod
    def write_cv_search_history_result(cls, search_type:[str], df_cv_result:pd.DataFrame, base_search_cv: BaseSearchCV) :
        df = pd.DataFrame({'Type'       : [ search_type],
                           'Date'       : [ datetime.now().strftime('%m%d-%H:%M')],
                           'Score Test' : [ float( '%.4f' % base_search_cv.best_score_) ],
                           'Score Train': [ float( '%.4f' % df_cv_result.iloc[base_search_cv.best_index_]['mean_train_score'] ) ],   # df_cv_result['mean_train_score'].mean()
                           'Params'    :  [ base_search_cv.best_params_ ],
                           # 'Estimator' :  [ base_search_cv.best_estimator_]
                           })
        df.to_csv( C.ts_pathanme([C.rootReports(), '__cv_search_history.csv'], ts_type=C.ts_none) , mode = 'a',  header=False)

    @classmethod
    def cv_best_generalized_score_and_param(cls, df_cv_result:pd.DataFrame) -> dict :
        df = pd.concat([ df_cv_result['rank_test_score'],
                         df_cv_result['mean_train_score'] - df_cv_result['mean_test_score'],
                         df_cv_result['mean_train_score'] , df_cv_result['mean_test_score'],
                         df_cv_result['params'] ],
                         axis = 1)
        sorted_df = df.sort_values(by=0) # sort by 'rank_test_score'
        # print(sorted_df.info())
        return  { C.bg_score_test_set : sorted_df.iloc[0]['mean_test_score'],   # --> best_generalized_score_test_set
                  C.bg_score_train_set: sorted_df.iloc[0]['mean_train_score'],  # --> best_generalized_score_train_set
                  C.bg_rank: sorted_df.iloc[0]['rank_test_score'],              # --> best_generalized_rank NB: 'rank_test_score' correspond to the index of the best TestScore in the ResultSet
                  C.bg_score_diff: sorted_df.iloc[0][0],                        # --> best_generalized_score_difference_with_1st # sorted_df.iloc[0][0]
                  C.bg_params: sorted_df.iloc[0]['params']  }                   # --> best_generalized_params


    @classmethod
    def df_imputer(cls, dfToImpute:pd.DataFrame, imputer:BaseEstimator):
        '''This method encodes non-null data and replace it in the original data'''
        # Retains only non-null values. dropna: Remove [rows(default) OR columns] when missing values
        nonulls = np.array(dfToImpute.dropna())
        # Reshapes the data for encoding
        impute_reshape = nonulls.reshape(-1,1)
        #     #encode date
        #     impute_ordinal = imputer.fit_transform(impute_reshape)
        # Assign back encoded values to non-null values
        dfToImpute.loc[dfToImpute.notnull()] = np.squeeze(imputer.fit_transform(impute_reshape))  # np.squeeze: Remove single-dimensional entries from the shape of an array.
        return dfToImpute

    @classmethod
    def outlier_filter(cls, df:pd.DataFrame, qtile1=0.25, qtile3=0.75) -> pd.Series:
        num_col = cls.numerical_cols(df)
        Q1 = df[num_col].quantile(qtile1)
        Q3 = df[num_col].quantile(qtile3)
        IQR = Q3 - Q1
        #
        return ((df[num_col] < (Q1 - 1.5 * IQR)) |(df[num_col] > (Q3 + 1.5 * IQR))).any(axis=1)

    @classmethod
    def outlier_ratio(cls, df:pd.DataFrame, qtile1=0.25, qtile3=0.75) -> float:
        return len(df[cls.outlier_filter(df, qtile1=qtile1, qtile3=qtile3)].index)/len(df.index)


    # https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html
    @classmethod
    def numerical_cols(cls, df:pd.DataFrame) -> list :
        return df.select_dtypes('number').columns.to_list()

    @classmethod
    def object_or_bool_cols(cls, df:pd.DataFrame) -> list :
        return df.select_dtypes(include=['object', 'bool']).columns.to_list()

    @classmethod
    def categorical_cols(cls, df:pd.DataFrame) -> list :
        return df.select_dtypes('category').columns.to_list()

    # # NB: Not used ....
    # @classmethod
    # def complete_df_with_gaussian_data(cls,  df:pd.DataFrame):
    #     f = df[C.OP100_Capuchon_insertion_mesure].isna()
    #     # df[missing-rows, column-to-feed] = sigma_column * np.random.randn(<occurence-count-to-generate>) + mu_column
    #     df.loc[f, C.OP100_Capuchon_insertion_mesure] = 0.024425 * np.random.randn(18627) + 0.388173

    # return : X_train, X_test, y_train, y_test
    @classmethod
    def stratified_shuffle_split(cls, X_df:pd.DataFrame, y_df:pd.DataFrame, stratification_column:str, test_size:float, nsplit=10)  -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame) :
        split = StratifiedShuffleSplit(n_splits=nsplit, test_size=test_size, random_state=42)
        yield cls.stratified_split(split, X_df, y_df, stratification_column, test_size, nsplit=nsplit)


    @classmethod
    def stratified_split(cls, split, X_df:pd.DataFrame, y_df:pd.DataFrame, stratification_column:str, test_size:float, nsplit=10)  -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame) :
        for train_index, test_index in split.split(X_df, X_df[stratification_column]) : # arg2 de 'split.split(...)' est le critère de stratification
            # print(f'train_index:{train_index} - {len(train_index)}')
            # print(f'test_index:{test_index} - {len(test_index)} ')
            X_train = X_df.iloc[train_index]
            X_test = X_df.iloc[test_index]
            y_train = y_df.iloc[train_index]
            y_test = y_df.iloc[test_index]
        yield X_train, X_test, y_train, y_test