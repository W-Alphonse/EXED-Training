import os
import logging
import numpy as np
# %matplotlib.pyplot inline
import pandas as pd
import matplotlib.pyplot as plt

from valeo.infrastructure.LogManager import LogManager as lm
# NB: Initializing logger here allows "class loaders of application classes" to benefit from the global initialization
logger = lm().logger(__name__)

from valeo.infrastructure import Const
from valeo.infrastructure.generic.DfUtil import DfUtil as dfUtil
from valeo.infrastructure.generic.ImgUtil import ImgUtil as imgUtil


if __name__ == "__main__" :
    # dfUtil.loadCsvData(["aa","bb"])
    df = dfUtil.loadCsvData(["..//data", "train", "traininginputs.csv"])
    imgUtil.save_df_as_hist(df,"Hello")
    # if data is not None:
    #     data.info()
    #     # On constate que plus de la moitié des valeur de la feature 7 sont manquants
    #     # 7   OP100_Capuchon_insertion_mesure  15888 non-null  float64
    #     data.describe()
    #
    #     data([["OP070_V_1_angle_value","OP070_V_1_torque_value","OP070_V_2_angle_value","OP070_V_2_torque_value",
    #            "OP110_Vissage_M8_angle_value","OP110_Vissage_M8_torque_value"]]).describe()

    # print(Const.APP_RESOURCE_PATH )
    # print(Const.rootSrc())
    # print(Const.rootProject())
    # print(Const.rootData())
    # print(Const.rootImages())


# l = ["data", "train", "traininginputs.csv"]
# print(*l[1:])