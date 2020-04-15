
import os
from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from valeo.infrastructure import Const as C


class DebugPipeline(BaseEstimator, TransformerMixin):
    OFFSET  = 10
    counter = -OFFSET

    def __init__(self):
        DebugPipeline.counter = ( (DebugPipeline.counter + DebugPipeline.OFFSET) // DebugPipeline.OFFSET) * DebugPipeline.OFFSET

    def transform(self, X):
        # %f : print micro seconds
        np.savetxt(os.path.join(C.rootProject(), 'log', 'debugPline_' + datetime.now().strftime("%Y_%m_%d-%H.%M.%S_") + str(DebugPipeline.counter)) + '.txt', X, delimiter=',')
        DebugPipeline.counter += 1
        return X

    def fit(self, X, y=None, **fit_params):
        return self