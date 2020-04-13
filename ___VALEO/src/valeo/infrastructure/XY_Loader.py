import pandas as pd
from sklearn.model_selection import ShuffleSplit

import valeo.infrastructure.XY_metadata as XY_metadata
import valeo.infrastructure.LogManager as LogManager


class XY_Loader:
    logger = LogManager.logger(__name__)

    def get_cv(X, y):
        cv = ShuffleSplit(n_splits=8, test_size=0.5, random_state=57)
        return cv.split(X)


    def load_XY_df(self, mt: XY_metadata, delete_XY_join_cols=True) -> ():
        X_df = pd.read_csv(mt.root_data, mt.X_filename)
        # 1 - Check whether Y is in separate file or in the same as X
        if mt.is_XY_in_separate_file :
            Y_df = pd.read_csv(mt.root_data, mt.Y_filename)
            XY_df = pd.merge(left=X_df, right=Y_df, how='inner', left_on=mt.X_join, right_on=mt.Y_join, suffixes=('',''))
        else :
            XY_df = X_df
        # 2 - When not reading a Test dataset (it means there is a Target dataset) THEN Let X_df group only features and Y_df only target
        if mt.is_training_set :
            Y_df = XY_df[mt.target_col_name]
            X_df = XY_df.drop(mt.target_col_name, axis=1)
        # 3 - Check whether we should remove joining columns
        if delete_XY_join_cols :
            X_df = X_df.drop(mt.X_join, axis=1)
            try :
                X_df = X_df.drop(mt.X_join, axis=1)
            except :
                pass
        #
        return X_df, Y_df

    def load_XY_values(self, mt: XY_metadata, delete_XY_join_cols=True) -> ():
        X_df, Y_df = self.load_XY_df(mt, delete_XY_join_cols)
        return X_df.values if X_df is not None else None, \
               Y_df.values if Y_df is not None else None

    # def get_train_data(self, mt: XY_metadata, delete_XY_join_cols=True) -> ():
    #     return self.load_XY_df(mt, delete_XY_join_cols = delete_XY_join_cols)
    #
    #
    # def get_test_data(self, mt: XY_metadata) -> ():
    #     return self.load_XY_df(mt, delete_XY_join_cols = False)
