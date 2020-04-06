from valeo.infrastructure.LogManager import LogManager
import os
import pandas as pd

class DfUtil() :
    logger = None

    def __init__(self):
        DfUtil.logger = LogManager.logger(__name__)

    # https://stackabuse.com/pythons-classmethod-and-staticmethod-explained/
    @classmethod
    def loadCsvData(cls, pathAsStrList):
        try:
            return pd.read_csv(os.path.join(pathAsStrList[0], *pathAsStrList[1:]) )
        except (Exception):
            cls.logger.exception("Error while load data from %s", "/".join(pathAsStrList))