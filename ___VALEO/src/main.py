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

    import eli5
    cat_list =  ['proc_month_1', 'proc_month_2', 'proc_month_3', 'proc_month_4', 'proc_month_5', 'proc_month_6', 'proc_week_1', 'proc_week_2', 'proc_week_3', 'proc_week_4', 'proc_week_5', 'proc_week_6', 'proc_week_7', 'proc_week_8', 'proc_week_9', 'proc_week_10', 'proc_week_11', 'proc_week_12', 'proc_week_13', 'proc_week_14', 'proc_week_15', 'proc_week_16', 'proc_week_17', 'proc_week_18', 'proc_week_19', 'proc_week_20', 'proc_week_21', 'proc_weekday_1', 'proc_weekday_2', 'proc_weekday_3', 'proc_weekday_4', 'proc_weekday_5', 'proc_weekday_6', 'proc_weekday_7']
    onehot_columns = list(fitted_model.named_steps['preprocessor'].named_transformers_['cat_transformers_pipeline'].named_steps['hotencoder_transformer'].get_feature_names())  # input_features=cat_list
    numeric_features_list = list(DfUtil.numerical_cols(X_train.drop(C.UNRETAINED_FEATURES, axis=1)))
    numeric_features_list.extend(onehot_columns)
    print(eli5.explain_weights(fitted_model.named_steps['classifier'], top=50, feature_names=numeric_features_list))



# In case you are still looking for the answer. You just need to replace the line of code from:
#
# explanation_pred = eli5.formatters.as_dataframe.explain_prediction_df(estimator=my_model,
#                                                                       doc=X_test[0])
# to:
#
# explanation_pred = eli5.explain_prediction_df(estimator=my_model, doc=X_test[0])

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
    clfTypes = [C.LRC_SMOTEEN]
    clfSelection =  C.simple_train_test  # 1.b - clfSelection : Union[C.simple_train_test, C.cross_validation, C.grid_cv, C.rand_cv, C.optim_cv, C.view_hyp]
    rand_or_optim_iteration_count = 200  # 1.c - Number of iteration while performing RandomizedSearchCV or BayesSearchCV. It's useless for other model seelction

    # 2 - Perform the prediction
    logger.info(f'Starting *** {clfTypes[0]} - {clfSelection} *** at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} .....')
    generate_ens_prediction(clfTypes, clfSelection, rand_or_optim_iteration_count)
    logger.info(f'Ending *** {clfTypes[0]} - {clfSelection} *** at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} .....')






# # Extract feature importances
# import pandas as pd
# fi = pd.DataFrame({'feature': list(X_train.columns), 'importance': fitted_model.feature_importances_}).sort_values('importance', ascending = False)
# print(fi.head())