import datetime
from typing import Union

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from valeo.domain.ValeoPredictor import ValeoPredictor
from valeo.infrastructure.LogManager import LogManager as lm
from valeo.infrastructure.XY_Loader import XY_Loader
from valeo.infrastructure.tools.DfUtil import DfUtil

logger = lm().logger(__name__)

from valeo.infrastructure import Const as C
from valeo.infrastructure.XY_metadata import XY_metadata as XY_metadata


def generate_ens_prediction(clfTypes:[str],
                            clfSelection:Union[C.simple_train_test, C.cross_validation,  C.grid_cv, C.rand_cv, C.optim_cv, C.view_hyp],
                            rand_or_optim_iteration_count:int = 0) :
    # 1 - Load the data
    mt_train = XY_metadata([C.rootDataTrain(), 'traininginputs.csv'], [C.rootDataTrain(), 'trainingoutput.csv'], [C.PROC_TRACEINFO], [C.PROC_TRACEINFO], C.Binar_OP130_Resultat_Global_v)
    xy_loader = XY_Loader()
    X_df, y_df = xy_loader.load_XY_df(mt_train, delete_XY_join_cols=False)

    # 2 - Instantiate ValeoPredictor
    pred = ValeoPredictor()

    # 2.a - Split, fit and validate on  X_train, X_test
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=48)
    fitted_model =  pred.fit(X_train, y_train, clfTypes) if clfSelection == C.simple_train_test else \
                    pred.fit_cv_best_score(X_train, y_train, clfTypes, n_splits=8) if clfSelection == C.cross_validation else \
                    pred.view_model_params_keys(X_train, clfTypes) if clfSelection == C.view_hyp else \
                    pred.fit_cv_grid_or_random_or_opt_search(X_train, y_train, clfTypes, cv_type=clfSelection, n_iter=rand_or_optim_iteration_count, n_splits=8)

    # 3 - Predict, plot and generate ENS artefact
    if fitted_model != None:
        pred.predict_and_plot(fitted_model, X_test, y_test, clfTypes)
        generate_y_ens(pred.fit(X_df, y_df, clfTypes),  f'{clfTypes[0]}_{clfSelection}')  # NB: Produce a smooth better roc_auc than 'generate_y_ens(fitted_model.fit(X_df, y_df),  ....)'
        # generate_y_ens(fitted_model.fit(X_df, y_df),  f'{clfTypes[0]}_{clfSelection}')

def generate_y_ens(fitted_model:BaseEstimator, clf_type_and_mselection:str) :
    X_ens = DfUtil.read_csv([C.rootDataTest() , "testinputs.csv"])
    y_ens = fitted_model.predict( X_ens )
    DfUtil.write_y_csv(X_ens[C.PROC_TRACEINFO], y_ens, C.Binar_OP130_Resultat_Global_v, [C.rootDataTest() , f"testoutput_{clf_type_and_mselection}.csv"])



if __name__ == "__main__" :
    # 1 - Set the arguments : The classifier algorithm type + The model train/test selection + Iteration count in case of RandomizedSearchCV or BayesSearchCV
    # 1.a - clfTypes :  C.BRFC(20_iter), C.BBC_ADABoost, C.BBC_GBC, C.BBC_HGBC, C.RUSBoost_ADABoost,
    #                   C.RFC_SMOTEEN, C.RFC_SMOTETOMEK, C.RFC_BLINESMT_RUS,
    #                   C.LRC_SMOTEEN, C.SVC_SMOTEEN, C.KNN_SMOTEEN, C.GNB_SMOTENN
    clfTypes = [C.BRFC]
    clfSelection =  C.cross_validation  # 1.b - clfSelection : Union[C.simple_train_test, C.cross_validation, C.grid_cv, C.rand_cv, C.optim_cv, C.view_hyp]
    rand_or_optim_iteration_count = 0   # 1.c - Number of iteration while performing RandomizedSearchCV or BayesSearchCV. It's useless for other model seelction

    # 2 - Perform the prediction
    logger.info(f'Starting *** {clfTypes[0]} - {clfSelection} *** at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} .....')
    generate_ens_prediction(clfTypes, clfSelection, rand_or_optim_iteration_count)
    logger.info(f'Ending *** {clfTypes[0]} - {clfSelection} *** at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} .....')






# # Extract feature importances
# import pandas as pd
# fi = pd.DataFrame({'feature': list(X_train.columns), 'importance': fitted_model.feature_importances_}).sort_values('importance', ascending = False)
# print(fi.head())