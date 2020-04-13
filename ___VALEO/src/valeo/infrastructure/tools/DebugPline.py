
import os
from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from valeo.infrastructure import Const as C


class DebugPline(BaseEstimator, TransformerMixin):
    counter = 1

    def transform(self, X):
        np.savetxt(os.path.join(C.rootProject(), 'log', 'debugPline_' + datetime.now().strftime("%Y_%m_%d-%H.%M.%S_") + str(DebugPline.counter)) + '.txt', X, delimiter=',')
        DebugPline.counter += 1
        return X

    def fit(self, X, y=None, **fit_params):
        return self