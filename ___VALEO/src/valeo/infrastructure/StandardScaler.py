from sklearn.preprocessing import StandardScaler as _StandardScaler

from valeo.infrastructure.tools.DfInDfOut import DfInDfOut


class StandardScaler(_StandardScaler, DfInDfOut):

    def transform(self, X):
        Xt = super().transform(X)
        return super().check_output(Xt, ensure_index=X, ensure_columns=X)