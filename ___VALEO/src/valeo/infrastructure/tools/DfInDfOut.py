
import pandas as pd

class DfInDfOut:

    # https://github.com/scikit-learn/scikit-learn/issues/5523 : Pandas in, Pandas out
    def check_output(self, X, ensure_index=None, ensure_columns=None):
        """
        Joins X with ensure_index's index or ensure_columns's columns when avaialble
        """
        if ensure_index is not None:
            if ensure_columns is not None:
                if type(ensure_index) is pd.DataFrame and type(ensure_columns) is pd.DataFrame:
                    X = pd.DataFrame(X, index=ensure_index.index, columns=ensure_columns.columns)
            else:
                if type(ensure_index) is pd.DataFrame:
                    X = pd.DataFrame(X, index=ensure_index.index)
        return X