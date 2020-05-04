from sklearn.impute import SimpleImputer as _SimpleImputer

from valeo.infrastructure.tools.DfInDfOut import DfInDfOut

'''
It is a SimpleImputer wrapped useful to be used while pipelining dataset.
Usually when piplening a dataframe dataset, the dataframe is transformed into dataframe.values ndarray, 
and hence it lost its columns names features which it not handy while debugging.
This imputer inherit the imputation functionality from its parent imputer and preserve the columns names while pipelining.
https://github.com/scikit-learn/scikit-learn/issues/5523 : Pandas in, Pandas out
'''
class SimpleImputer(_SimpleImputer, DfInDfOut):

    def transform(self, X):
        Xt = super().transform(X)
        return super().check_output(Xt, ensure_index=X, ensure_columns=X)