from datetime import datetime

from pandas import Series
from sklearn.base import BaseEstimator

from valeo.infrastructure.LogManager import LogManager
from valeo.infrastructure import Const as C

import os
import pandas as pd
import numpy as np

class DfUtil() :
    logger = LogManager.logger(__name__)


    # https://stackabuse.com/pythons-classmethod-and-staticmethod-explained/
    @classmethod
    def read_csv(cls, pathAsStrList : []) -> pd.DataFrame:
        try:
            return pd.read_csv(os.path.join(pathAsStrList[0], *pathAsStrList[1:]) )
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
        # Q1 = df.quantile(0.25)
        # IQR = Q3 - Q1
        # #
        # outliers = ((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)
        return len(df[cls.outlier_filter(df, qtile1=qtile1, qtile3=qtile3)].index)/len(df.index)


    @classmethod
    def numerical_cols(cls, df:pd.DataFrame) -> list :
        return df.select_dtypes('number').columns.to_list()

    @classmethod
    def categorical_cols(cls, df:pd.DataFrame) -> list :
        return df.select_dtypes(include=['object', 'bool']).columns.to_list()

    # # NB: Not used ....
    # @classmethod
    # def complete_df_with_gaussian_data(cls,  df:pd.DataFrame):
    #     f = df[C.OP100_Capuchon_insertion_mesure].isna()
    #     # df[missing-rows, column-to-feed] = sigma_column * np.random.randn(<occurence-count-to-generate>) + mu_column
    #     df.loc[f, C.OP100_Capuchon_insertion_mesure] = 0.024425 * np.random.randn(18627) + 0.388173
