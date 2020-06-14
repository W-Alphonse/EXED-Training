from typing import Union

import pandas as pd
import numpy as np

from imblearn.metrics._classification import classification_report_imbalanced
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, auc, roc_auc_score, confusion_matrix, classification_report, \
    precision_recall_curve, precision_recall_fscore_support, roc_curve, plot_precision_recall_curve, \
    average_precision_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV

from valeo.domain.MetricPlotter   import MetricPlotter
from valeo.domain.ValeoModeler    import ValeoModeler
from valeo.domain.ClassifierParam import ClassifierParam
from valeo.infrastructure.tools.DfUtil import DfUtil
from valeo.infrastructure.LogManager import LogManager
from valeo.infrastructure import Const as C

class ValeoPredictor :
    logger = None

    def __init__(self):
        ValeoPredictor.logger = LogManager.logger(__name__)
        self.modeler = ValeoModeler()
        self.metricPlt = MetricPlotter()
        self.param = ClassifierParam()

    ''' ==========================================
        1/ Train / Test split : Fit and then Plot
        ==========================================
    '''
    def fit_predict_and_plot(self, X_train:pd.DataFrame, y_train:pd.DataFrame,  X_test:pd.DataFrame, y_test:pd.DataFrame, clfTypes:[str]) -> BaseEstimator:
        model = self.fit(X_train, y_train, clfTypes)
        self.predict_and_plot(model, X_test, y_test, clfTypes)
        return model

    def fit(self, X_train:pd.DataFrame, y_train:pd.DataFrame, clfTypes:[str]) -> BaseEstimator:
        model = self.modeler.build_predictor_pipeline(X_train, clfTypes)
        model.fit(X_train, y_train)
        return model

    def view_model_params_keys(self, X_df:pd.DataFrame, clfTypes:[str]):
        self.modeler.view_model_params_keys( X_df, clfTypes)
        return None

    ''' =================================
        2/ Fit with Cross Validation
        =================================
        NB :
        a - roc-auc-avo + roc-auc-ovr : 
            https://stackoverflow.com/questions/59453363/what-is-the-difference-of-roc-auc-values-in-sklearn
            roc_auc is the only one suitable for binary classification. The weighted, ovr and ovo are use for multi-class problems
            
        b - Micro-Average + Macro-Average (for Precision / Recall / F1) :
            http://rushdishams.blogspot.com/2011/08/micro-and-macro-average-of-precision.html
            https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin
            Ex: Micro-P = (TP1 + TP2) / ( TP1 + FP1 + TP2 + F2)
                Macro-P = (P1 + P2) / 2 
            Suitability:
            . Macro-average method can be used when you want to know how the system performs overall across the sets of data
            . Micro-average method can be a useful measure when your dataset varies in size.
            
        c - How can we report 'confusion matrix' while using 'cross_validate' ?
            https://stackoverflow.com/questions/40057049/using-confusion-matrix-as-scoring-metric-in-cross-validation-in-scikit-learn 
            c1. Either use 'cross_val_predict' and deduce confusion-matrix: 
                y_pred = cross_val_predict(clf, x, y, cv=10)
                conf_mat = confusion_matrix(y_test, y_pred)
                BUT BEWARE: Passing these predictions into an evaluation metric may not be a valid way to measure generalization performance. 
                            Results can differ from cross_validate and cross_val_score unless all tests sets have equal size and the metric decomposes over samples.
            c2. If you want to obtain confusion matrices for multiple evaluation runs (such as cross validation) you have to do this by hand:
                conf_matrix_list_of_arrays = []
                kf = cross_validation.KFold(len(y), n_folds=5)
                for train_index, test_index in kf:
                   X_train, X_test = X[train_index], X[test_index]   # Panda-Column index 'train_index' are of type 'numpy array'
                   y_train, y_test = y[train_index], y[test_index]
                   # 
                   model.fit(X_train, y_train)
                   conf_matrix = confusion_matrix(y_test, model.predict(X_test))
                   conf_matrix_list_of_arrays .append(conf_matrix)
                   
                On the end you can calculate your mean of list of numpy arrays (confusion matrices) with:
                mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)   
        ================================================================================================================                            
    '''
    def fit_cv_best_score(self, X:pd.DataFrame, y:pd.DataFrame, clfTypes:[str], n_splits=8, classifier_params=None) -> BaseEstimator :
        fitted_estimators, cv_results = self.fit_cv(X, y, clfTypes, n_splits=n_splits, classifier_params=classifier_params)
        ValeoPredictor.logger.info(f'- np.argmax(cv_results[test_roc_auc]:{np.argmax(cv_results["test_roc_auc"])} => test_roc_auc : {cv_results["test_roc_auc"][np.argmax(cv_results["test_roc_auc"])]}')
        best_model = cv_results["estimator"][np.argmax(cv_results["test_roc_auc"])]
        return best_model

    def fit_cv(self, X:pd.DataFrame, y:pd.DataFrame, clfTypes:[str], n_splits=8, classifier_params = None) -> ([BaseEstimator], dict):
        ValeoPredictor.logger.info(f'Cross validation : {n_splits} folds')
        model = self.modeler.build_predictor_pipeline(X, clfTypes)
        if classifier_params != None :
            model = classifier_params
        CV = StratifiedKFold(n_splits=n_splits, random_state=48)

        # The cross_validate function differs from cross_val_score in two ways:
        # It allows specifying multiple metrics for evaluation + It returns a dict containing fit-times, score-times ...
        # cv_results =  cross_validate(model, X, y, cv=CV, scoring=('f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'recall', 'precision', 'average_precision', 'roc_auc'), return_train_score=True, return_estimator=True)
        # cv_results =  cross_validate(model, X, y, cv=CV, scoring=('f1', 'recall', 'precision', 'roc_auc'), return_train_score=True, return_estimator=True)
        cv_results =  cross_validate(model, X, y, cv=CV, scoring=('roc_auc', 'recall', 'precision', 'f1'), return_train_score=True, return_estimator=True)
        return self.print_cv_result_and_extract_fitted_estimators(cv_results), cv_results

    def print_cv_result_and_extract_fitted_estimators(self, cv_results :dict, html_fmt=True) -> [BaseEstimator]:
        fitted_estimators = []
        mesures = []
        d = { 0: 'Temps consommé', 1: 'ROC_AUC', 2:'Recall', 3:'Precision', 4:'F1' }
        #
        for key in list(cv_results.keys()) : # cv_results.keys() is filled according to the request order: ('roc_auc', 'recall', 'precision', 'f1')
            if str(key) !=  "estimator" :
                # ValeoPredictor.logger.debug(f"{key} : min/mean/max/std -> {cv_results[key].min()} / {cv_results[key].mean()} / {cv_results[key].max()}  / {cv_results[key].std()}")
                ValeoPredictor.logger.debug(f"- {key} moyen :{self.fmt(cv_results[key].mean())}")
                if key in ['test_roc_auc', 'train_roc_auc'] :
                    ValeoPredictor.logger.debug(f"- {key} folds :{self.fmt(cv_results[key])}")
                mesures.append(cv_results[key].mean())
            fitted_estimators.append(cv_results[key])
        if html_fmt:
            ValeoPredictor.logger.debug('<table> <tr> <th>Moyenne par folds de CV</th> <th>Validation Set</th> <th>Train Set</th> </tr>')
            for i, mes in enumerate(mesures) :
                if i == 0 :
                    train_time = mes
                elif i == 1 :
                    ValeoPredictor.logger.debug((f'\t<tr> <td>{d[i//2]}</td> <td>{self.fmt(mes)}</td> <td>{self.fmt(train_time)}</td></tr>'))
                elif i%2 == 0 :
                    s1 = f'\t<tr> <td>{d[i//2]}</td> <td>{self.fmt(mes)}</td>'
                else :
                    ValeoPredictor.logger.debug((f'{s1} <td>{self.fmt(mes)}</td></tr>'))
            arg_max_test = np.argmax(cv_results["test_roc_auc"])
            ValeoPredictor.logger.info(f'\t<tr> <td>Best ROC_AUC on Validation Set (fold {arg_max_test}) </td>'
                                       f' <td>{self.fmt(cv_results["test_roc_auc"][arg_max_test])} </td> <td>{self.fmt(cv_results["train_roc_auc"][arg_max_test])}</td></tr>')
            ValeoPredictor.logger.debug('</table>')
        return fitted_estimators

    '''
        - Print metrics
        - Print report
        - Plot ROC : TP vs FP
        - Plot AUC : Precison vs Recall 
    '''
    def predict_and_plot(self, fitted_model: BaseEstimator, X_test:pd.DataFrame, y_test:pd.DataFrame, clfTypes:[str], fmt_html=True):
        y_pred = fitted_model.predict(X_test)
        #
        print(f"- ROC_AUC: {self.fmt(roc_auc_score(y_test, y_pred))}")
        print(f"- Recall : {self.fmt(recall_score(y_test, y_pred))}")
        print(f"- Precision : {self.fmt(precision_score(y_test, y_pred))}")
        print(f"- F1 : {self.fmt(f1_score(y_test, y_pred))}")
        m = confusion_matrix(y_test, y_pred)
        print(f"- Matrice de confusion:\n{confusion_matrix(y_test, y_pred)}")
        if fmt_html :
            ValeoPredictor.logger.debug('<table><tr><th>Valeur</th><th>Mesure</th> </tr>')
            ValeoPredictor.logger.debug((f'\t<tr><td>ROC_AUC</td><td>{self.fmt(roc_auc_score(y_test, y_pred))}</td></tr>'))
            ValeoPredictor.logger.debug((f'\t<tr><td>Recall</td><td>{self.fmt(recall_score(y_test, y_pred))}</td></tr>'))
            ValeoPredictor.logger.debug((f'\t<tr><td>Precision</td><td>{self.fmt(precision_score(y_test, y_pred))}</td></tr>'))
            ValeoPredictor.logger.debug((f'\t<tr><td>F1</td><td>{self.fmt(f1_score(y_test, y_pred))}</td></tr>'))
            ValeoPredictor.logger.debug((f'\t<tr><td>Matrice de confusion</td><td>{confusion_matrix(y_test, y_pred)[0,:]}</td></tr>'))
            ValeoPredictor.logger.debug((f'\t<tr><td></td><td>{confusion_matrix(y_test, y_pred)[1,:]}</td></tr>'))
            ValeoPredictor.logger.debug('</table>')
        # print("*** Predict and Plot on Test DataSet ***")
        # print(f"- Model score: {fitted_model.score(X_test, y_test)}")
        # print(f"- Accuracy score: {accuracy_score(y_test, y_pred)}")
        # print(f"- Balanced accuracy score: {balanced_accuracy_score(y_test, y_pred)} / The balanced accuracy to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.")
        # print(f"- auc : {auc(y_test, y_pred)}")  # ValueError: x is neither increasing nor decreasing : [0 0 0 ... 0 0 0]
        # print(f"- {m[0]}/{m[1]} - P:{precision_score(y_test, y_pred):0.4f} - R:{recall_score(y_test, y_pred):0.4f} - roc_auc:{roc_auc_score(y_test, y_pred):0.4f} - f1:{f1_score(y_test, y_pred):0.4f}")
        # print(f"- Average_precision_score: {average_precision_score(y_test, y_pred)}")
        # print(f"- classification_report_imbalanced:\n{classification_report_imbalanced(y_test, y_pred)}")
        # print(f"- classification_report:\n{classification_report(y_test, y_pred)}")
        # print(f"- precision_recall_curve: {precision_recall_curve(y_test, y_pred)}")
        # print(f"- precision_recall_fscore_support: {precision_recall_fscore_support(y_test, y_pred)}")
        # print(f"- roc_curve: {roc_curve(y_test, y_pred)}")
        #
        self.metricPlt.plot_roc(y_test, y_pred, clfTypes)
        self.metricPlt.plot_precision_recall(y_test, y_pred, clfTypes)


        # model = self.modeler.build_predictor_pipeline(X, clfTypes)
        # CV = StratifiedKFold(n_splits=n_splits)
        # space  = [Integer(100, 300, name='n_estimators'),
        #           Integer(5, 20, name='max_depth'),
        #           skopt.space.Categorical( ['sqrt', 'log2'], name='max_features'),
        #           Integer(5, 20, name='min_samples_split'),
        #           Integer(9, 15, name='min_samples_leaf'),
        #           skopt.space.Categorical( [True, False], name='oob_score'),
        #           skopt.space.Categorical( ['entropy', 'gini'], name='criterion'),
        #           Real(0.15, 0.25, name='ampling_strategy', prior='uniform')]
        # import valeo.domain.Optimizer as opt
        # opt.initialize (self, X, y, space, model, CV)
        # from skopt import gp_minimize
        # res_gp = gp_minimize(opt.initialize.objective, space, n_calls=50, random_state=0)
        # "Best score=%.4f" % res_gp.fun

    ''' =================================================
        3/ fit_cv_grid_or_random_or_opt_search
           In case of Grid Search then n_iter is useless 
        =================================================
    '''
    def fit_cv_grid_or_random_or_opt_search(self, X:pd.DataFrame, y:pd.DataFrame, clfTypes:[str], cv_type:Union[C.grid_cv, C.rand_cv, C.optim_cv], n_iter=None, n_splits=8) -> BaseEstimator:
        model = self.modeler.build_predictor_pipeline(X, clfTypes)
        CV = StratifiedKFold(n_splits=n_splits) #  andom_state=48, shuffle=True

        #  NB:
        #  1 - refit=True => Refit an estimator using the best found parameters on the whole dataset
        #  2 - For multiple metric evaluation, 'scoring' needs to be an str denoting the scorer
        #      that would be used to find the best parameters for refitting the estimator at the end.
        #  3 - The refitted estimator is made available at the best_estimator_ attribute and permits using predict directly on this SearchCV instance.

        search = GridSearchCV(model, param_grid=self.param.grid_param(clfTypes[0]), scoring='roc_auc', n_jobs=-1, refit=True, cv=CV, verbose=0, return_train_score=True) if cv_type == C.grid_cv else \
                 RandomizedSearchCV(model, param_distributions=self.param.distrib_param(clfTypes[0]), n_iter=n_iter, scoring='roc_auc', n_jobs=-1, refit=True, cv=CV, verbose=0, return_train_score=True)  if cv_type == C.rand_cv else \
                 BayesSearchCV(model, search_spaces= self.param.optimize_param(clfTypes[0]), refit=True, scoring='roc_auc', cv=CV, n_iter=n_iter, random_state=48, verbose=0, return_train_score=True)
        search.fit(X, y)
        print(search)

        # 1 - Write down the SearchCV result into a CSV file sorting by 'rank_test_score' asc
        #     and append 'scores' and 'params' to the search_history
        df_cv_results = pd.DataFrame(search.cv_results_)
        DfUtil.write_df_csv( df_cv_results, [C.rootReports(), f'{cv_type}_search_cv-notsorted-{clfTypes[0]}.csv'] )
        DfUtil.write_df_csv( df_cv_results.sort_values(by='rank_test_score', ascending=True), [C.rootReports(), f'{cv_type}_search_cv-{clfTypes[0]}.csv'] )
        DfUtil.write_cv_search_history_result(clfTypes + [cv_type], df_cv_results, search)

        # search attributes: best_score_,  best_params_ (short), best_estimator_ (long), best_index_ /*c'est lindex du meilleur rang*/ ; sklearn.metrics.SCORERS.keys()
        has_train_score = True if 'mean_train_score' in df_cv_results.columns.tolist() else False
        if has_train_score :
            ValeoPredictor.logger.info(f"- Mean best score(Test): {'%.4f' % search.best_score_} (mean Train {'%.4f' %  df_cv_results.iloc[search.best_index_] ['mean_train_score']}) - Best Params: {search.best_params_} - Best Estimator: {search.best_estimator_}")
        else :
            ValeoPredictor.logger.info(f"- Mean best score(Test): {'%.4f' % search.best_score_} - Best Params: {search.best_params_} - Best Estimator: {search.best_estimator_}")

        # 2 - Check whether there is a difference between the best_classifier score (the classifier whose rank is equal to 1)
        #     and the best_classifier that can generalize (the classifier whose test_score is the highest)
        # bg_dict = DfUtil.cv_best_generalized_score_and_param(df_cv_results)
        # 3 - (bg: best generalized) - bg_score_test_set, bg_score_train_set, bg_rank, bg_score_difference_with_1st, bg_params
        # ValeoPredictor.logger.info(f"- Best mean score(Test, rank {'%d' %  search.best_index_} | {'%d' %  bg_dict[C.bg_rank]}): {'%.4f' % bg_dict[C.bg_score_test_set]} "
        #                            f"(mean Train {'%.4f' %  bg_dict[C.bg_score_train_set]}) - "
        #                            f"Train-Test {'%.4f' % bg_dict[C.bg_score_diff]} - "
        #                            f"Best Generalized Params: {bg_dict[C.bg_params]}")
        return search.best_estimator_

    def fmt(self, float_to_format:float, format=4 ) -> str:
        f_format = '%.' + str(format) + 'f'
        if isinstance(float_to_format, float) :
            return f"{f_format % float_to_format}"
        else :
            return [ f"{f_format % f}" for f in float_to_format]


''' TODO :
-> Faire Ressortir l'importance et la contribution de chaque feature:
   https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
'''