
import os
from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from valeo.infrastructure import Const as C


class DebugPipeline(BaseEstimator, TransformerMixin):
    OFFSET = 10
    counter = 1 # -OFFSET

    # def __init__(self):
        # DebugPipeline.counter += DebugPipeline.OFFSET

    def transform(self, X):
        # print(X.columns)
        np.savetxt(os.path.join(C.rootProject(), 'log', 'debugPline_' + datetime.now().strftime("%Y_%m_%d-%H.%M.%S_") + str(DebugPipeline.counter)) + '.txt', X, delimiter=',')
        DebugPipeline.counter += 1
        return X

    def fit(self, X, y=None, **fit_params):
        return self