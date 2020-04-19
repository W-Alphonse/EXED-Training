import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.impute._iterative import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from valeo.domain.DefectPredictor import DefectPredictor
from valeo.domain.ValeoPipeline import ValeoPipeline as ValeoPipeline
from valeo.domain.ValeoPreprocessor import ValeoPreprocessor
from valeo.infrastructure.LogManager import LogManager as lm
# NB: Initializing logger here allows "class loaders of application classes" to benefit from the global initialization
from valeo.infrastructure.XY_Loader import XY_Loader
from valeo.infrastructure.tools.DfUtil import DfUtil

logger = lm().logger(__name__)

from valeo.infrastructure import Const as C
from valeo.infrastructure.XY_metadata import XY_metadata as XY_metadata
# import valeo.infrastructure.XY_metadata as XY_metadata
# import valeo.infrastructure.XY_Loader as XY_Loader

if __name__ == "__main__" :
    logger.info("Début .....")
    # data = DfUtil.loadCsvData([C.rootData() , "train", "traininginputs.csv"])
    # if data is not None:
    #     data.info()

    mt_train = XY_metadata([C.rootDataTrain(), 'traininginputs.csv'], [C.rootDataTrain(), 'trainingoutput.csv'], [C.PROC_TRACEINFO], [C.PROC_TRACEINFO], C.Binar_OP130_Resultat_Global_v)
    xy_loader = XY_Loader();
    X_df, y_df = xy_loader.load_XY_df(mt_train)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.25, random_state=48, stratify=y_df)  # shuffle=True,
    #
    # vp = ValeoPipeline()
    # vp.execute(X_df, y_df, C.smote_over_sampling)
    #

    # vproc.build_column_preprocessor().fit_transform(X_train)
    # from imblearn.pipeline import Pipeline
    # ppl = Pipeline([('column_preprocessor', vproc.build_column_preprocessor())])
    # ppl.fit_transform(X_train)

    # vproc = ValeoPreprocessor()
    # X1 = vproc.execute(X_train)

    # ------------
    # Transformer
    # ------------
    pred = DefectPredictor()

    # ------------
    # Predictor
    # ------------
    # a. Fit and predict on X_train, X_test
    # pred.fit(X_train, y_train, C.smote_over_sampling)
    fitted_model = pred.fit_and_plot(X_train, y_train, X_test, y_test,C.smote_over_sampling)
    # a.2 - GridSearchCV
    # pred.fit_grid_search_cv(X_df, y_df, C.smote_over_sampling)

    #
    # b. Fit X and using CV
    # fitted_model, cv_results = pred.fit_cv(X_df, y_df, C.smote_over_sampling)
    #
    # c. Test suning ENS data
    # X_ens = DfUtil.read_csv([C.rootDataTest() , "testinputs.csv"])
    # y_ens = fitted_model.predict(X_ens.drop(columns=[C.PROC_TRACEINFO]))
    # DfUtil.write_y_csv(X_ens[C.PROC_TRACEINFO], y_ens, C.Binar_OP130_Resultat_Global_v, [C.rootDataTest() , "testoutput.csv"])

'''
    vproc = ValeoPreprocessor()
    imp = vproc.build_iterative_preprocessor()
    transf =imp.fit(X_df, y_df)
    logger.debug(f"type(a):{type(transf)}")
    logger.debug(f"transf:{transf}")

    mt_test = XY_metadata([C.rootData(), 'test','testinputs.csv'], None, [C.PROC_TRACEINFO], None, None)
    logger.debug(f"mt_test:{mt_test}")
    x, y = xy_loader.load_XY_df(mt_test)
    logger.debug(f"X_train_df.isna(): {x.isna()}" )
    logger.debug(f"len(X_train_df): {len(x[x.isna()])}" )

    results = imp.transform(x)
    # logger.debug(f"type(results): {type(results)}")      # type(results): <class 'numpy.ndarray'>
    # logger.debug(f"np.isnan(results):{np.isnan(results)}")
    logger.debug(f"np.count_nonzero(np.isnan(results)):{np.count_nonzero(np.isnan(results))}")
'''
    # logger.debug(f"results.isna(): {results.isna()}" )   # 8001 dont 3641 non-null
    # logger.debug(f"len(results): {len(results[results.isna()])}" )
# l = ["data", "train", "traininginputs.csv"]
# print(*l[1:])