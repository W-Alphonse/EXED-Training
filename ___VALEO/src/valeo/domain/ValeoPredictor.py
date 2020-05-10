
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

from valeo.domain.MetricPlotter import MetricPlotter
from valeo.domain.ValeoModeler import ValeoModeler
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

    # Cette fct doit etre supprimer car obsolete
    # def prepare_X_for_test(self, X_df: pd.DataFrame, add_flds_to_drop : list) -> pd.DataFrame:
    #     return self.modeler.prepare_X_for_test(X_df, add_flds_to_drop)

    ''' ========================================
        1/ fit_cv_grid_search
        ========================================
    '''
    def fit_cv_grid_search(self, X:pd.DataFrame, y:pd.DataFrame, clfTypes:[str] , n_splits=5) -> ([BaseEstimator], dict): # (estimator, cv_results)
        model = self.modeler.build_predictor_pipeline(X, clfTypes) # sampler_type)
        CV = StratifiedKFold(n_splits=n_splits, random_state=48) # , random_state=48, shuffle=True
        # HGBC
        # param_grid = {
        #     'classifier__n_estimators': [3, 5, 10, 20, 50],
        #     'classifier__base_estimator__l2_regularization': [5, 50, 100, 50],
        #     'classifier__base_estimator__max_iter' : [100],
        #     'classifier__base_estimator__max_depth' : [10,50,10]
        # }
        # BRFC
        param_grid = {
            'classifier__n_estimators': [250,300],
            'classifier__max_depth': [15,20,25],
            'classifier__max_features' : ['auto',13]
        }

        grid = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, cv=CV) # if is_grid else
        grid.fit(X, y)
        print(f"Best Estimator: {grid.best_estimator_}")
        df_results = pd.DataFrame(grid.cv_results_)
                    # columns_to_keep = ['param_clf__max_depth', 'param_clf__n_estimators', 'mean_test_score', 'std_test_score',]
                    # df_results = df_results[columns_to_keep]
        DfUtil.write_df_csv( df_results.sort_values(by='mean_test_score', ascending=False), C.ts_pathanme([C.rootReports(), 'grid_search_cv.csv']) )


    ''' ========================================
        4/ fit_cv_randomized_search
        ========================================
    '''
    def fit_cv_randomized_search(self, X:pd.DataFrame, y:pd.DataFrame, clfTypes:[str] , n_splits=5) -> ([BaseEstimator], dict): # (estimator, cv_results)
        model = self.modeler.build_predictor_pipeline(X, clfTypes) # sampler_type)
        CV = StratifiedKFold(n_splits=n_splits) # , random_state=48, shuffle=True
        # HGBC
        # param_grid = {
        #     'classifier__n_estimators': [3, 5, 10, 20, 50],
        #     'classifier__base_estimator__l2_regularization': [5, 50, 100, 50],
        #     'classifier__base_estimator__max_iter' : [100],
        #     'classifier__base_estimator__max_depth' : [10,50,10]
        # }

        grid = RandomizedSearchCV(model, param_distributions=param_grid, n_jobs=-1, cv=CV) # if is_grid else
        grid.fit(X, y)
        df_results = pd.DataFrame(grid.cv_results_)
        DfUtil.write_df_csv( df_results.sort_values(by='mean_test_score', ascending=False), C.ts_pathanme([C.rootReports(), 'grid_search_csv']) )

    def print_model_params_keys(self, model:BaseEstimator):
        for param in model.get_params().keys():
            print(param)


    ''' ========================================
        1/ Train / Test split : Fit and then Plot
        ========================================
    '''
    # 1 - Fit without any Cross Validation
    def fit_predict_and_plot(self, X_train:pd.DataFrame, y_train:pd.DataFrame,  X_test:pd.DataFrame, y_test:pd.DataFrame, clfTypes:[str]) -> BaseEstimator:
        # model = self.fit(X_train, y_train, clfTypes)
        model = self.modeler.build_predictor_pipeline(X_train, clfTypes)
        model.fit(X_train, y_train)
        self.predict_and_plot(model, X_test, y_test)
        return model
    # def fit(self, X_train:pd.DataFrame, y_train:pd.DataFrame, clfTypes:[str]) -> BaseEstimator:
    #     model = self.modeler.build_predictor_pipeline(X_train, clfTypes)
    #     return model.fit(X_train, y_train)


    ''' ================================================================================================================
        2/ Fit with Cross Validation
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
    def fit_cv_best_score(self, X:pd.DataFrame, y:pd.DataFrame, clfTypes:[str], n_splits=10) -> BaseEstimator :
        fitted_estimators, cv_results = self.fit_cv(X, y, clfTypes, n_splits=n_splits)
        # print(f'- np.argmax(cv_results[test_roc_auc]:{np.argmax(cv_results["test_roc_auc"])} => test_roc_auc : {cv_results["test_roc_auc"][np.argmax(cv_results["test_roc_auc"])]}')
        ValeoPredictor.logger.info(f'- np.argmax(cv_results[test_roc_auc]:{np.argmax(cv_results["test_roc_auc"])} => test_roc_auc : {cv_results["test_roc_auc"][np.argmax(cv_results["test_roc_auc"])]}')
        best_model = cv_results["estimator"][np.argmax(cv_results["test_roc_auc"])]
        return best_model

    def fit_cv(self, X:pd.DataFrame, y:pd.DataFrame, clfTypes:[str], n_splits=8) -> ([BaseEstimator], dict):
        ValeoPredictor.logger.info(f'Cross validation : {n_splits} folds')
        model = self.modeler.build_predictor_pipeline(X, clfTypes)
        CV = StratifiedKFold(n_splits=n_splits) # , random_state=48, shuffle=True
        cv_results =  cross_validate(model, X, y, cv=CV, scoring=('f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'recall', 'precision', 'average_precision', 'roc_auc'), return_train_score=True, return_estimator=True)
        fitted_estimators = []
        for key in cv_results.keys() :
            if str(key) !=  "estimator" :
                # print(f"{key} : {cv_results[key]}")
                ValeoPredictor.logger.debug(f"{key} : min/mean/max/std -> {cv_results[key].min()} / {cv_results[key].mean()} / {cv_results[key].max()}  / {cv_results[key].std()}")
            fitted_estimators.append(cv_results[key])
        return fitted_estimators, cv_results

    '''
        - Print metrics
        - Print report
        - Plot ROC : TP vs FP
        - Plot AUC : Precison vs Recall 
    '''
    def predict_and_plot(self, fitted_model: BaseEstimator, X_test:pd.DataFrame, y_test:pd.DataFrame):
        y_pred = fitted_model.predict(X_test)
        #
        print(f"- Model score: {fitted_model.score(X_test, y_test)}")
        print(f"- Accuracy score: {accuracy_score(y_test, y_pred)}")
        print(f"- Balanced accuracy score: {balanced_accuracy_score(y_test, y_pred)} / The balanced accuracy to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.")
        # print(f"- auc : {auc(y_test, y_pred)}")  # ValueError: x is neither increasing nor decreasing : [0 0 0 ... 0 0 0]
        print(f"- Average_precision_score: {average_precision_score(y_test, y_pred)}")
        print(f"- Precision_score: {precision_score(y_test, y_pred)}")
        print(f"- Recall score: {recall_score(y_test, y_pred)}")
        print(f"- Roc_auc_score: {roc_auc_score(y_test, y_pred)}")
        print(f"- F1 score: {f1_score(y_test, y_pred)}")
        m = confusion_matrix(y_test, y_pred)
        print(f"- {m[0]}/{m[1]} - P:{precision_score(y_test, y_pred):0.4f} - R:{recall_score(y_test, y_pred):0.4f} - roc_auc:{roc_auc_score(y_test, y_pred):0.4f} - f1:{f1_score(y_test, y_pred):0.4f}")
        print(f"- {confusion_matrix(y_test, y_pred)}")
        print(f"- classification_report_imbalanced:\n{classification_report_imbalanced(y_test, y_pred)}")
        print(f"- classification_report:\n{classification_report(y_test, y_pred)}")
        print(f"- precision_recall_curve: {precision_recall_curve(y_test, y_pred)}")
        print(f"- precision_recall_fscore_support: {precision_recall_fscore_support(y_test, y_pred)}")
        print(f"- roc_curve: {roc_curve(y_test, y_pred)}")
        #
        self.metricPlt.plot_roc(y_test, y_pred)
        self.metricPlt.plot_precision_recall(y_test, y_pred)
        # self.plot_roc(y_test, y_pred)
        # self.plot_precision_recall(y_test, y_pred)



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