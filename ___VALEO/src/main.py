from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

# NB: Initializing logger here allows "class loaders of application classes" to benefit from the global initialization
from mglearn import cm2
from mglearn.plot_2d_separator import plot_2d_separator
from valeo.domain.ValeoModeler import ValeoModeler
from valeo.domain.ValeoPredictor import ValeoPredictor
from valeo.infrastructure.LogManager import LogManager as lm
from valeo.domain import Preprocessor as pp
from valeo.infrastructure.XY_Loader import XY_Loader
from valeo.infrastructure.tools.DfUtil import DfUtil

logger = lm().logger(__name__)

from valeo.infrastructure import Const as C
from valeo.infrastructure.XY_metadata import XY_metadata as XY_metadata
# import valeo.infrastructure.XY_metadata as XY_metadata
# import valeo.infrastructure.XY_Loader as XY_Loader
# TODO : Paper for classfication Evaluation: https://www.researchgate.net/publication/327403649_Classification_assessment_methods_a_detailed_tutorial
# TODO : Confidence interval aka pvalues
# TODO : Select_kbest
# TODO : https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

def generate_y_ens(fitted_model : BaseEstimator,  clfTypes) :
    X_ens = DfUtil.read_csv([C.rootDataTest() , "testinputs.csv"])
    y_ens = fitted_model.predict( X_ens )
    DfUtil.write_y_csv(X_ens[C.PROC_TRACEINFO], y_ens, C.Binar_OP130_Resultat_Global_v, [C.rootDataTest() , f"testoutput_{clfTypes[0]}.csv"])

if __name__ == "__main__" :
    clfTypes = [C.BBC_GBC] # BRFC BBC_ADABoost BBC_GBC BBC_HGBC RUSBoost(ADABoost) /  RFC_SMOTEENN RFC_SMOTETOMEK RFC( BorderLineSmote, RandomUnderSample)
                        # BBC  GBC HGBC  #NotRetained
                        # SVC KNN / LR
    logger.info(f"Début ***{clfTypes[0]}*** .....")

    # 1 - Load the data
    mt_train = XY_metadata([C.rootDataTrain(), 'traininginputs.csv'], [C.rootDataTrain(), 'trainingoutput.csv'], [C.PROC_TRACEINFO], [C.PROC_TRACEINFO], C.Binar_OP130_Resultat_Global_v)
    xy_loader = XY_Loader();
    X_df, y_df = xy_loader.load_XY_df(mt_train, delete_XY_join_cols=False)

    # 2 - ValeoPredictor & ValeoModeler
    pred = ValeoPredictor()
    best_generalized_model = None

    # 2.a - Fit and predict on X_train, X_test
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=48, stratify=y_df)  # shuffle=True,
            # A SUPPRIMER (remplacer vers #3):
    # fitted_model = pred.fit_predict_and_plot(X_train, y_train, X_test, y_test, [ValeoModeler.BRFC])

    # 2.b - Fit using CV
    fitted_model = pred.fit_cv_best_score(X_df, y_df, clfTypes, n_splits=8)  #XGBC  BRFC LRC  criterion:string

    # # Extract feature importances
    # import pandas as pd
    # fi = pd.DataFrame({'feature': list(X_train.columns), 'importance': fitted_model.feature_importances_}).sort_values('importance', ascending = False)
    # print(fi.head())

    # 2.c - Fit using GridSearchCV or RandomizedSearchCV
    # fitted_model = pred.fit_cv_grid_or_random_search(X_train, y_train, [C.BRFC], n_random_iter=None, n_splits=8)
    # fitted_model = pred.fit_cv_grid_or_random_search(X_train, y_train, [C.LRC], n_random_iter=None, n_splits=8)
    # fitted_model = pred.fit_cv_grid_or_random_search(X_train, y_train, [C.GBC], n_random_iter=None, n_splits=8)

    # 3 - Predit and Plot For ALL using the TestSet
    pred.predict_and_plot(fitted_model, X_test, y_test, clfTypes)

                                        # 2.e - Fit on the tuned parameters - ll faut SUPPRIMER PEUT ETRE car NOT USED ?
                                        # fitted_model =  pred.fit(X_df, y_df, [ValeoModeler.BRFC])

    # 4 - Fit on all available DATA and THEN Test using ENS dataset. https://machinelearningmastery.com/train-final-machine-learning-model/
    # generate_y_ens(fitted_model)
    generate_y_ens(pred.fit(X_df, y_df, clfTypes),  clfTypes)







                                        # if (best_generalized_model != None) and ( best_generalized_model != fitted_model) :
                                        #     generate_y_ens(best_generalized_model)

    # X_ens = DfUtil.read_csv([C.rootDataTest() , "testinputs.csv"])
    # y_ens = fitted_model.predict( X_ens )
    # DfUtil.write_y_csv(X_ens[C.PROC_TRACEINFO], y_ens, C.Binar_OP130_Resultat_Global_v, [C.rootDataTest() , "testoutput.csv"])

    # logger.debug(f"results.isna(): {results.isna()}" )   # 8001 dont 3641 non-null
    # logger.debug(f"len(results): {len(results[results.isna()])}" )
