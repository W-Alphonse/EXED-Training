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
        try :
            pd.DataFrame(data={X_id.name:X_id, y_col_name:y_target}).to_csv( C.ts_pathanme(pathAsStrList,ts_type), index = False)
        except Exception as ex:
            DfUtil.logger.exception(f"Error while generating writing target values '{X_id.name},{y_col_name}'  to {'/'.join(pathAsStrList)}")

    # @classmethod
    # def write_df_csv(cls, df:pd.DataFrame, pathAsStrList : [], ts_type=C.ts_sfix):
    #     try :
    #         fname_with_ext = os.path.splitext(pathAsStrList[-1])
    #         df.to_csv(os.path.join(pathAsStrList[0], '' if len(pathAsStrList) == 2 else str(*pathAsStrList[1:-1] ),
    #                                 f"{fname_with_ext[0]}{datetime.now().strftime('_%Y_%m_%d-%H.%M.%S')}{fname_with_ext[1]}" if ts_type == DfUtil.ts_sfix else \
    #                                     (f"{datetime.now().strftime('%Y_%m_%d-%H.%M.%S_')}{pathAsStrList[-1]}" if ts_type == DfUtil.ts_pfix  else pathAsStrList[-1])) , index = False)
    #     except Exception as ex:
    #         cls.logger.exception(f"Error while writing 'df' to CSV '{pathAsStrList}'")

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