from sklearn.impute import SimpleImputer as _SimpleImputer

from valeo.infrastructure.tools.DfInDfOut import DfInDfOut


class SimpleImputer(_SimpleImputer, DfInDfOut):

    def transform(self, X):
        Xt = super().transform(X)
        return super().check_output(Xt, ensure_index=X, ensure_columns=X)