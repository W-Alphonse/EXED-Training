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
if __name__ == "__main__" :
    logger.info("Début .....")

    # 1 - Load the data
    mt_train = XY_metadata([C.rootDataTrain(), 'traininginputs.csv'], [C.rootDataTrain(), 'trainingoutput.csv'], [C.PROC_TRACEINFO], [C.PROC_TRACEINFO], C.Binar_OP130_Resultat_Global_v)
    xy_loader = XY_Loader();
    X_df, y_df = xy_loader.load_XY_df(mt_train, delete_XY_join_cols=False)

    # 2 - ValeoPredictor & ValeoModeler
    pred = ValeoPredictor()

    # 2.a - Fit and predict on X_train, X_test
    # X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.3, random_state=48, stratify=y_df)  # shuffle=True,
    # fitted_model = pred.fit_predict_and_plot(X_train, y_train, X_test, y_test, [ValeoModeler.BRFC])

    # import matplotlib.pyplot as plt
    # from mglearn import plot_2d_separator, cm2
    # import mglearn as mg
    # plot_2d_separator(fitted_model[-1], X_df.values, fill=True)
    # _ = plt.scatter( X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm2, s=60, alpha=.7, edgecolor='k')


    # 2.b - Fit using CV
    # fitted_model = pred.fit_cv_best_score(X_df, y_df, [ValeoModeler.BRFC], n_splits=8)  #XGBC  BRFC LRC  criterion:string
    # pred.predict_and_plot(fitted_model, X_test, y_test)  # **cette instruction doit etre supprimer**

    # 2.c - Fit using GridSearchCV
    # pred.fit_cv_grid_search(X_df, y_df, [ValeoModeler.BRFC])

    # 2.d - Fir using RandomizedSearchCV
    pred.fit_cv_randomized_search(X_df, y_df, [ValeoModeler.BRFC])

    # 2.e - Fit on the tuned parameters
    # fitted_model =  pred.fit(X_df, y_df, [ValeoModeler.BRFC])

    #3 - Test suning ENS data
    # X_ens = DfUtil.read_csv([C.rootDataTest() , "testinputs.csv"])
    # y_ens = fitted_model.predict( X_ens )
    # DfUtil.write_y_csv(X_ens[C.PROC_TRACEINFO], y_ens, C.Binar_OP130_Resultat_Global_v, [C.rootDataTest() , "testoutput.csv"])

    # logger.debug(f"results.isna(): {results.isna()}" )   # 8001 dont 3641 non-null
    # logger.debug(f"len(results): {len(results[results.isna()])}" )
