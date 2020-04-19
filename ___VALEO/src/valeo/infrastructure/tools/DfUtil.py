from datetime import datetime

from pandas import Series
from sklearn.base import BaseEstimator

from valeo.infrastructure.LogManager import LogManager
from valeo.infrastructure import Const as C

import os
import pandas as pd
import numpy as np

class DfUtil() :
    logger = LogManager.logger("DfUtil")

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
            DfUtil.logger.exception(f"Error while writing 'df' to CSV '{pathAsStrList}'")

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