import datetime
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.model_selection._search import BaseSearchCV

from valeo.domain.ValeoPredictor import ValeoPredictor
from valeo.infrastructure.LogManager import LogManager as lm
from valeo.infrastructure.XY_Loader import XY_Loader
from valeo.infrastructure.tools.DfUtil import DfUtil

logger = lm().logger(__name__)

from valeo.infrastructure import Const as C
from valeo.infrastructure.XY_metadata import XY_metadata as XY_metadata

def generate_y_ens(fitted_model : BaseEstimator,  clfTypes) :
    X_ens = DfUtil.read_csv([C.rootDataTest() , "testinputs.csv"])
    y_ens = fitted_model.predict( X_ens )
    DfUtil.write_y_csv(X_ens[C.PROC_TRACEINFO], y_ens, C.Binar_OP130_Resultat_Global_v, [C.rootDataTest() , f"testoutput_{clfTypes[0]}.csv"])

if __name__ == "__main__" :
    clfTypes = [C.LRC_SMOTEEN]  # BRFC, BBC_ADABoost, BBC_GBC, BBC_HGBC,
                            # RFC_SMOTEEN, RFC_SMOTETOMEK, RFC_BLINESMT_RUS, RUSBoost_ADABoost,
                            # LRC_SMOTEEN, KNN_SMOTEEN, SVC_SMOTEEN, GNB_SMOTENN

    logger.info(f'Starting ***{clfTypes[0]}*** at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} .....')

    # 1 - Load the data
    mt_train = XY_metadata([C.rootDataTrain(), 'traininginputs.csv'], [C.rootDataTrain(), 'trainingoutput.csv'], [C.PROC_TRACEINFO], [C.PROC_TRACEINFO], C.Binar_OP130_Resultat_Global_v)
    xy_loader = XY_Loader();
    X_df, y_df = xy_loader.load_XY_df(mt_train, delete_XY_join_cols=False)

    # 2 - Instantiate ValeoPredictor
    pred = ValeoPredictor()

    # 2.a - Fit and predict on X_train, X_test
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=48, stratify=y_df)

    # 2.b - Fit using CV
    fitted_model = pred.fit_cv_best_score(X_df, y_df, clfTypes, n_splits=8)

            # # Extract feature importances
            # import pandas as pd
            # fi = pd.DataFrame({'feature': list(X_train.columns), 'importance': fitted_model.feature_importances_}).sort_values('importance', ascending = False)
            # print(fi.head())

    # 2.c - Fit using GridSearchCV or RandomizedSearchCV
    # fitted_model = pred.fit_cv_grid_or_random_or_opt_search(X_train, y_train, clfTypes, cv_type= C.grid,  n_splits=8)
    # fitted_model = pred.fit_cv_grid_or_random_or_opt_search(X_train, y_train, clfTypes, cv_type= C.rand, n_iter=20, n_splits=8)
    # fitted_model = pred.fit_cv_grid_or_random_or_opt_search(X_train, y_train, clfTypes, cv_type= C.opt, n_iter=20, n_splits=8)
    #

    # 3 - Predit and Plot For ALL using the TestSet
    if isinstance(fitted_model, BaseSearchCV) :
        pred.predict_and_plot(fitted_model.best_estimator_, X_test, y_test, clfTypes)
        generate_y_ens(fitted_model.best_estimator_ ,  clfTypes)
    else :
        pred.predict_and_plot(fitted_model, X_test, y_test, clfTypes)
        # 4 - Fit on all available DATA and THEN Test using ENS dataset. https://machinelearningmastery.com/train-final-machine-learning-model/
        generate_y_ens(pred.fit(X_df, y_df, clfTypes),  clfTypes)

    logger.info(f'Ending ***{clfTypes[0]}*** at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} .....')
