# from random import randint
from datetime import datetime

import sklearn
from scipy.stats import randint

from imblearn.metrics._classification import classification_report_imbalanced
# https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.pipeline
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, auc, roc_auc_score, confusion_matrix, classification_report, \
    precision_recall_curve, precision_recall_fscore_support, roc_curve, plot_precision_recall_curve, \
    average_precision_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score
# from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV, RandomizedSearchCV


import pandas as pd
import numpy as np

from valeo.domain.MetricPlotter   import MetricPlotter
from valeo.domain.ValeoModeler    import ValeoModeler
from valeo.domain.ClassifierParam import ClassifierParam
from valeo.infrastructure.tools.DfUtil import DfUtil
from valeo.infrastructure.LogManager import LogManager
from valeo.infrastructure import Const as C

import xgboost as xgb


class ValeoPredictor :
    logger = None

    def __init__(self):
        ValeoPredictor.logger = LogManager.logger(__name__)
        self.modeler = ValeoModeler()
        self.metricPlt = MetricPlotter()
        self.param = ClassifierParam()

    # Cette fct doit etre supprimer car obsolete
    # def prepare_X_for_test(self, X_df: pd.DataFrame, add_flds_to_drop : list) -> pd.DataFrame:
    #     return self.modeler.prepare_X_for_test(X_df, add_flds_to_drop)

    ''' ==========================================
        1/ Train / Test split : Fit and then Plot
        ==========================================
    '''
    def fit_predict_and_plot(self, X_train:pd.DataFrame, y_train:pd.DataFrame,  X_test:pd.DataFrame, y_test:pd.DataFrame, clfTypes:[str]) -> BaseEstimator:
        model = self.fit(X_train, y_train, clfTypes)
        self.predict_and_plot(model, X_test, y_test, clfTypes)
        self.print_model_params_keys(model)
        return model

    def fit(self, X_train:pd.DataFrame, y_train:pd.DataFrame, clfTypes:[str]) -> BaseEstimator:
        model = self.modeler.build_predictor_pipeline(X_train, clfTypes)
        model.fit(X_train, y_train)
        return model

    def print_model_params_keys(self, model:BaseEstimator):
        pass
    #     for param in model.get_params().keys():
    #         ValeoPredictor.logger.info(param)

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
        CV = StratifiedKFold(n_splits=n_splits) #   n_splits    , random_state=48, shuffle=True

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
        for key in list(cv_results.keys()) : # it seems that cv_results.keys() was filled according to the request order: ('roc_auc', 'recall', 'precision', 'f1')
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
            ValeoPredictor.logger.debug('<table><tr><th colspan="2">Résultat de la prédiction sur Test Set</th> </tr>')
            ValeoPredictor.logger.debug((f'\t<tr><td>ROC_AUC</td><td>{self.fmt(roc_auc_score(y_test, y_pred))}</td></tr>'))
            ValeoPredictor.logger.debug((f'\t<tr><td>Recall</td><td>{self.fmt(recall_score(y_test, y_pred))}</td></tr>'))
            ValeoPredictor.logger.debug((f'\t<tr><td>Precision</td><td>{self.fmt(precision_score(y_test, y_pred))}</td></tr>'))
            ValeoPredictor.logger.debug((f'\t<tr><td>F1</td><td>{self.fmt(f1_score(y_test, y_pred))}</td></tr>'))
            ValeoPredictor.logger.debug((f'\t<tr><td rowspan="2">Matrice de confusion</td><td>{confusion_matrix(y_test, y_pred)[0,:]}</td></tr>'))
            ValeoPredictor.logger.debug((f'\t<tr><td>{confusion_matrix(y_test, y_pred)[1,:]}</td></tr>'))
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


    ''' ========================================
        3/ fit_cv_grid_search
        ========================================
    '''
    def fit_cv_grid_search(self, X:pd.DataFrame, y:pd.DataFrame, clfTypes:[str] , n_splits=8) -> ([BaseEstimator], dict): # (estimator, cv_results)
        # HGBC
        # param_grid = {
        #     'classifier__n_estimators': [3, 5, 10, 20, 50],
        #     'classifier__base_estimator__l2_regularization': [5, 50, 100, 50],
        #     'classifier__base_estimator__max_iter' : [100],
        #     'classifier__base_estimator__max_depth' : [10,50,10]
        # }
        # for BRFC read this link: https://machinelearningmastery.com/bagging-and-random-forest-for-imbalanced-classification/
        # BRFC : BalancedRandomForestClassifier(n_estimators = 300 , max_depth=20, random_state=0)
        # param_grid_0 = {                                                #param_grid_0
        #     'classifier__n_estimators': [250,300],    # 300
        #     'classifier__max_depth': [15,20,25],      # 20
        #     'classifier__max_features' : ['auto',13], # auto C'est le nombre de features selectionnées
        #     'classifier__min_samples_split' : [5, 8], # 8
        #     'classifier__min_samples_leaf' : [3, 5],  # 5
        #     'classifier__oob_score': [True, False],   # False
        #     'classifier__class_weight' : ['balanced', None] # None
        # }
        # param_grid_1 = {                                                  #param_grid=1  better than param_grid_0
        #     'classifier__n_estimators': [250,300],     # 300
        #     'classifier__max_depth': [20],             # 20
        #     'classifier__max_features' : ['auto'],     # auto C'est le nombre de features selectionnées
        #     'classifier__min_samples_split' : [8, 12], # 8 ou 12
        #     'classifier__min_samples_leaf' : [5, 10],  # 10
        #     'classifier__oob_score': [False],          # False
        #     'classifier__class_weight' : [None]        # None
        # } NB: test_roc_auc  : [0.65268942 0.71165229 0.65813965 0.73303875 0.74815986 0.69392817 0.70467358 0.69323889]  / Split 8
        #       train_roc_auc : [0.78869075 0.78907625 0.78553331 0.78299062 0.78295546 0.78258473  0.78412007 0.78626762]
        # param_grid_2 = {                                                  #param_grid=2   better than param_grid_1
        #     'classifier__n_estimators': [250, 300, 350],   # 300
        #     'classifier__max_depth': [20],                 # 20
        #     'classifier__max_features' : ['auto'],         # auto C'est le nombre de features selectionnées
        #     'classifier__min_samples_split' : [8, 12],     # 8 ou 12
        #     'classifier__min_samples_leaf' : [10, 15],     # 15
        #     'classifier__oob_score': [False],  # False
        #     'classifier__class_weight' : [None]            # None
        # } # NB: test_roc_auc  : [0.65268942 0.71165229 0.65813965 0.73303875 0.74815986 0.69392817 0.70467358 0.69323889]  / Split 8
        #       train_roc_auc : [0.78869075 0.78907625 0.78553331 0.78299062 0.78295546 0.78258473  0.78412007 0.78626762]
        #       test_roc_auc  : [0.65268942 0.71165229 0.65813965 0.73303875 0.74815986 0.69392817  0.70467358 0.69323889]  / Split 12
        #       train_roc_auc : [0.78869075 0.78907625 0.78553331 0.78299062 0.78295546 0.78258473  0.78412007 0.78626762]
        # param_grid_3 = {                                  #param_grid_3
        #     'classifier__n_estimators': [300],             # 300
        #     'classifier__max_depth': [20],                 # 20
        #     'classifier__max_features' : ['auto'],         # auto C'est le nombre de features selectionnées
        #     'classifier__min_samples_split' : [8, 12],     # 8 ou 12. Ils sont tjrs execau
        #     'classifier__min_samples_leaf' : [15],         # 15
        #     'classifier__oob_score': [False],              # False
        #     'classifier__class_weight' : [None],           # None
        #     'classifier__sampling_strategy' : [0.1 , 0.25, 0.5]  # 0.1 le meilleur ds la liste
        # }
        param_grid = {                                                  #param_grid_4
            'classifier__n_estimators': [300],             # 300
            'classifier__max_depth': [20],                 # 20
            'classifier__max_features' : ['auto'],         # auto C'est le nombre de features selectionnées
            'classifier__min_samples_split' : [8, 12],     # 8 ou 12. Ils sont tjrs execau
            'classifier__min_samples_leaf' : [15],         # 15
            'classifier__oob_score': [False],              # False
            'classifier__class_weight' : [None],           # None
            'classifier__criterion' : ['entropy', 'gini'],
            'classifier__sampling_strategy' : [ 'auto']  # 0.1 better than 'auto' Cependant l'overfitting est plus petit avec 'auto'. NB: # 0.1, 0.15 ou 0.2 sont tjrs execau
        }

        # cls.BRFC : BalancedRandomForestClassifier(n_estimators = 61 , max_depth=8, min_samples_split=8, min_samples_leaf=9,  sampling_strategy=0.15, random_state=0, criterion='gini') , # sampling_strategy=0.5),
        # cls.BRFC : BalancedRandomForestClassifier(n_estimators = 102 , max_depth=6, min_samples_split=18, min_samples_leaf=13,  sampling_strategy=0.15, random_state=0, criterion='gini') , # sampling_strategy=0.5),
        param_grid = {                                                  #param_grid_4
            'classifier__n_estimators': [61,102, 300],             # 300
            'classifier__max_depth': [6,8, 15, 20],                 # 20
            'classifier__max_features' : ['auto'],         # auto C'est le nombre de features selectionnées
            'classifier__min_samples_split' : [8, 12, 18],     # 8 ou 12. Ils sont tjrs execau
            'classifier__min_samples_leaf' : [9,13, 15],         # 15
            # 'classifier__oob_score': [False],              # False
            # 'classifier__class_weight' : [None],           # None
            # 'classifier__criterion' : ['entropy', 'gini'],
            'classifier__sampling_strategy' : [ 0.15, 'auto']  # 0.1 better than 'auto' Cependant l'overfitting est plus petit avec 'auto'. NB: # 0.1, 0.15 ou 0.2 sont tjrs execau
        }

        model = self.modeler.build_predictor_pipeline(X, clfTypes)
        CV = StratifiedKFold(n_splits=n_splits, random_state=48)   # shuffle=True
        grid = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, cv=CV)
        grid.fit(X, y)
        print(f"Best Estimator: {grid.best_estimator_}")
        df_results = pd.DataFrame(grid.cv_results_)
        # columns_to_keep = ['param_clf__max_depth', 'param_clf__n_estimators', 'mean_test_score', 'std_test_score',]
        # df_results = df_results[columns_to_keep]
        DfUtil.write_df_csv( df_results.sort_values(by='mean_test_score', ascending=False), C.ts_pathanme([C.rootReports(), f'grid_search_cv-{clfTypes[0]}.csv']) )


    ''' ========================================
        4/ fit_cv_randomized_search
        if n_iter == 0 or None => Grid Search Else RandomSearch 
        ========================================
    '''
    def fit_cv_grid_or_random_search(self, X:pd.DataFrame, y:pd.DataFrame, clfTypes:[str], n_random_iter=None, n_splits=8) -> BaseEstimator:
        is_grid = (n_random_iter == None) or (n_random_iter == 0)
        grid_or_random = "grid" if is_grid else "random"
        #
        model = self.modeler.build_predictor_pipeline(X, clfTypes)
        CV = StratifiedKFold(n_splits=n_splits) #  andom_state=48, shuffle=True

        #  refit=True => Refit an estimator using the best found parameters on the whole dataset
        #  NB: For multiple metric evaluation, this needs to be a str denoting the scorer that would be used to find the best parameters
        #      for refitting the estimator at the end.
        # The refitted estimator is made available at the best_estimator_ attribute and permits using predict directly on this RandomizedSearchCV instance.
        search = GridSearchCV(model, param_grid=self.param.grid_param(clfTypes[0]), scoring='roc_auc', n_jobs=-1, refit=True, cv=CV, verbose=0, return_train_score=True) \
                 if is_grid else  RandomizedSearchCV(model, param_distributions=self.param.distrib_param(clfTypes[0]), n_iter=n_random_iter,
                                                     scoring='roc_auc', n_jobs=-1, refit=True, cv=CV, verbose=0, return_train_score=True)
        search.fit(X, y)
        # 1 - Write down the SearchCV result
        df_cv_results = pd.DataFrame(search.cv_results_)
        DfUtil.write_df_csv( df_cv_results.sort_values(by='rank_test_score', ascending=True), [C.rootReports(), f'{grid_or_random}_search_cv-{clfTypes[0]}.csv'] )
        DfUtil.write_cv_search_result( clfTypes + [grid_or_random] , df_cv_results, search)

        # 2 - Check whether there is a difference between the best_classifier (rank=1) and the best_classifier that can generalize
        # randomized : best_score_,  best_params_ (short), best_estimator_ (long), best_index_ ; sklearn.metrics.SCORERS.keys()
        ValeoPredictor.logger.info(f"- Best Score: {'%.4f' % search.best_score_} (Train {'%.4f' %  df_cv_results.iloc[search.best_index_] ['mean_train_score']}) / Best Params: {search.best_params_}")
        bg_dict = DfUtil.cv_best_generalized_score_and_param(df_cv_results)
        ValeoPredictor.logger.info(f"- Generalized Score: {'%.4f' % bg_dict[C.bg_score_test_set]} (Train {'%.4f' %  bg_dict[C.bg_score_train_set]}, rank {'%.4f' %  bg_dict[C.bg_rank]}) / Best Generalized Params: {bg_dict[C.bg_params]}")

        # return  self.fit_cv_best_score(X, y, clfTypes, n_splits=8, classifier_params = search.best_estimator_)
        return search.best_estimator_
        # return search

    def fmt(self, float_to_format:float, format=4 ) -> str:
        f_format = '%.' + str(format) + 'f'
        return f"{f_format % float_to_format}"



        # HGBC
        # param_grid = {
        #     'classifier__n_estimators': [3, 5, 10, 20, 50],
        #     'classifier__base_estimator__l2_regularization': [5, 50, 100, 50],
        #     'classifier__base_estimator__max_iter' : [100],
        #     'classifier__base_estimator__max_depth' : [10,50,10]
        # }

        # random_search_cv-BRFC_2020_05_12-14.18.33_2020_05_12-14.18.33.csv
        # param_distributions = { # n_iter=20
        #     'classifier__n_estimators': randint(10, 300),         # 38
        #     'classifier__max_depth': randint(5, 20),              # 7
        #     'classifier__min_samples_split' : randint(5, 20),     # 13
        #     'classifier__min_samples_leaf' : randint(5, 20),      # 15
        #     'classifier__criterion' : ['entropy', 'gini'],        # gin
        #     # 'classifier__sampling_strategy' : [ 0.1, 0.15, 0.2, 'auto']
        #     'classifier__sampling_strategy' : [ 0.3, 0.25, 0.2, 'auto']  # 0.2
        # }
        #
        # #
        # param_distributions = { # n_iter=10
        #     'classifier__n_estimators': randint(30, 50),         # 38
        #     'classifier__max_depth': randint(5, 15),              # 7
        #     'classifier__min_samples_split' : randint(10, 20),     # 13
        #     'classifier__min_samples_leaf' : randint(10, 20),      # 15
        #     'classifier__sampling_strategy' : [ 0.1, 0.15, 0.2]  # 0.2
        # }
        #
        # param_distributions = { # n_iter=10
        #     'classifier__n_estimators': randint(30, 300),         # 38
        #     'classifier__max_depth': randint(5, 25),              # 7
        #     # 'classifier__min_samples_split' : randint(10, 20),     # 13
        #     'classifier__min_samples_leaf' : randint(5, 15),      # 15
        #     'classifier__sampling_strategy' : [ 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]  # 0.2
        # }

        # return fitted_model  , None

# https://medium.com/towards-artificial-intelligence/application-of-synthetic-minority-over-sampling-technique-smote-for-imbalanced-data-sets-509ab55cfdaf
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.model_selection import GridSearchCV
# parameters = {'n_estimators':[100,150,200,250,300,350,400,450,500],
#               'max_depth':[3,4,5]}
# clf= GradientBoostingClassifier()
# grid_search = GridSearchCV(param_grid = parameters, estimator = clf,
#                            verbose = 3)
# grid_search_2 = grid_search.fit(X_train,y_train)

# GOOGLE ON: classifier over sampled imbalanced dataset
# https://sci2s.ugr.es/imbalanced  : Tres interessant****
# https://www.datacamp.com/community/tutorials/diving-deep-imbalanced-data  : Tres interessant****
#---
# https://journalofbigdata.springeropen.com/articles/10.1186/s40537-017-0108-1  : Tres interessant****
# xperimental evaluation
# The projected technique works on binary-class/multi-class imbalanced Big Data sets in the organization to recommended LVH. Four basic classifiers viz. Random Forest (-P 100-I 100-num-slots 1-K 0-M 1.0-V 0.001-S 1), Naïve Bayes, AdaBoostM1 (-P 100-S 1-I 10-W weka.classifiers.trees.DecisionStump) and MultiLayer Perceptron (-L 0.3-M 0.2-N 500-V 0-S 0-E 20-H a) are applied to over_sampled data sets using dissimilar values of cross-validation and KNN. Lastly the results, based on the F-measure and AUC values are used to compare between benchmarking (SMOTE/Borderline-SMOTE/ADASYN/SPIDER2/SMOTEBoost/MWMOTE) and planned technique (UCPMOT). Tables 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 and 13 describe the results in detail. The analysis of results validates the superiority of UCPMOT for enhancing the classification.

# GOOGLE ON: scikit learn imbalanced dataset resampling type cross validation shuffle
# https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets
# https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/
# CV = ShuffleSplit(n_splits=10, test_size=0.25, random_state=48)
# https://www.alfredo.motta.name/cross-validation-done-wrong/

# https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
#     https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

# http://www.cs.nthu.edu.tw/~shwu/courses/ml/labs/08_CV_Ensembling/08_CV_Ensembling.html
# https://github.com/arrayslayer/ML-Project

# https://www.kaggle.com/shiqbal/first-data-exploration/notebook  applied on Porto Seguro's


''' TODO :

-> Faire Ressortir l'importance et la contribution de chaque feature:
   https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html

-> Essayer ces scénarios de modélisation:
   scen1: Oversampling + LogisticR
   scen2: BaggingClassifier + Histo
   
'''