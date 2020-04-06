import os
import logging
import numpy as np
# %matplotlib.pyplot inline
import pandas as pd
import matplotlib.pyplot as plt


logger = logging.getLogger('valeo.main')

def loadCsvData(pathAsStrList):
    try:
        return pd.read_csv(os.path.join(pathAsStrList[0], *pathAsStrList[1:]) )
    except (Exception):
        logger.exception("Error while load data from %s", "/".join(pathAsStrList))

data = loadCsvData(["..//data", "train", "traininginputs.csv"])
if data is not None:
    data.info()
    # On constate que plus de la moitié des valeur de la feature 7 sont manquants
    # 7   OP100_Capuchon_insertion_mesure  15888 non-null  float64
    data.describe()

    data([["OP070_V_1_angle_value","OP070_V_1_torque_value","OP070_V_2_angle_value","OP070_V_2_torque_value",
           "OP110_Vissage_M8_angle_value","OP110_Vissage_M8_torque_value"]]).describe()


# l = ["data", "train", "traininginputs.csv"]
# print(*l[1:])