import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import sys

from valeo.infrastructure.LogManager import LogManager as lm
# NB: Initializing logger here allows "class loaders of application classes" to benefit from the global initialization
logger = lm().logger(__name__)

from valeo.infrastructure import Const
from valeo.infrastructure.generic.DfUtil import DfUtil
from valeo.infrastructure.generic.ImgUtil import ImgUtil
import valeo.infrastructure.Transformer as transf


if __name__ == "__main__" :
    # dfUtil.loadCsvData(["aa","bb"])
    # df = DfUtil.loadCsvData(["..//data", "train", "traininginputs.csv"])
    # DfUtil.save_df_as_hist(df,"Hello")

    data = DfUtil.loadCsvData([Const.rootData() , "train", "traininginputs.csv"])
    if data is not None:
        data.info()


# l = ["data", "train", "traininginputs.csv"]
# print(*l[1:])