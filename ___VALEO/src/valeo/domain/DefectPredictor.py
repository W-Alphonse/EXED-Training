import os

from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, SVMSMOTE, KMeansSMOTE, BorderlineSMOTE, ADASYN
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.ensemble import BalancedBaggingClassifier, RUSBoostClassifier, BalancedRandomForestClassifier
from imblearn.metrics._classification import classification_report_imbalanced
# https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.pipeline
from sklearn.base import BaseEstimator

from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.impute._iterative import IterativeImputer
# from sklearn.experimental import enable_iterative_imputer   # explicitly require this experimental feature
# from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.metrics import f1_score, auc, roc_auc_score, confusion_matrix, classification_report, \
    precision_recall_curve, precision_recall_fscore_support, roc_curve, plot_precision_recall_curve, \
    average_precision_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
# from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler, label_binarize, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict, ShuffleSplit, StratifiedKFold, \
    GridSearchCV
from sklearn.preprocessing import Normalizer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from valeo.infrastructure.SimpleImputer import SimpleImputer
# from valeo.infrastructure.StandardScaler import StandardScaler
from valeo.domain.MetricPlotter import MetricPlotter
from valeo.infrastructure.tools.DebugPipeline import DebugPipeline
from valeo.infrastructure.LogManager import LogManager
from valeo.infrastructure import Const as C

import xgboost as xgb


class DefectPredictor :
    logger = None

    def __init__(self):
        DefectPredictor.logger = LogManager.logger(__name__)
        self.metricPlt = MetricPlotter()

    # https://github.com/scikit-learn-contrib/imbalanced-learn/tree/master/examples
    # https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7
    # https://towardsdatascience.com/custom-transformers-and-ml-data-pipelines-with-python-20ea2a7adb65
    # https://jorisvandenbossche.github.io/blog/2018/05/28/scikit-learn-columntransformer/
    def build_transformers_pipeline(self, features_dtypes:pd.Series) -> ColumnTransformer:
        rand_state = 48
        numerical_features = (features_dtypes == 'int64') | (features_dtypes == 'float64')
        # categorical_features = ~numerical_features
        nan_imputer    = SimpleImputer(strategy='mean', missing_values=np.nan, verbose=False)
        # nan_imputer    = IterativeImputer(estimator=BayesianRidge(), missing_values=np.nan,  max_iter=10, initial_strategy = 'median',  add_indicator=False, random_state=rand_state)
        zeroes_imputer = IterativeImputer(estimator=BayesianRidge(), missing_values=0,  max_iter=10, initial_strategy = 'median',  add_indicator=False, random_state=rand_state)
        scaler         =  Normalizer()  # RobustScaler() #StandardScaler() # RobustScaler(with_centering=True, with_scaling=False)  # MinMaxScaler()
        #
        dbg = DebugPipeline()
        num_transformers_pipeline = Pipeline([#('dbg_0', dbg),
                                              ('nan_imputer', nan_imputer), # ('dbg_1', dbg),
                                              ('zeroes_imputer', zeroes_imputer), # ('dbg_2', dbg),
                                              ('scaler', scaler), # ('dbg_3', dbg)
                                            ])
        return ColumnTransformer([('transformers_pipeline',num_transformers_pipeline, numerical_features)], remainder='passthrough')

# classifiers = [
#     KNeighborsClassifier(3),
#     SVC(kernel="rbf", C=0.025, probability=True),
#     NuSVC(probability=True),
#     DecisionTreeClassifier(),
#     RandomForestClassifier(),
#     AdaBoostClassifier(),
#     GradientBoostingClassifier()
# ]
    def build_predictor_pipeline(self, features_dtypes:pd.Series, sampler_type: str) -> Pipeline:
        # HGBR = HistGradientBoostingClassifier(max_iter = 8 , max_depth=8,learning_rate=0.35, l2_regularization=500)
        HGBR = HistGradientBoostingClassifier(max_iter = 100 , max_depth=10,learning_rate=0.10, l2_regularization=5)
        # bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(), sampling_strategy='auto', replacement=False, random_state=48)
        bbc = BalancedBaggingClassifier(base_estimator=HGBR,  n_estimators=50, sampling_strategy='auto', replacement=False, random_state=48)
        # bbc = BalancedBaggingClassifier(base_estimator=HGBR,  sampling_strategy=1.0, replacement=False, random_state=48)
        rusboost = RUSBoostClassifier(n_estimators = 8 , algorithm='SAMME.R', random_state=42)
        BRFC = BalancedRandomForestClassifier(n_estimators = 50 , max_depth=20)
        dbg = DebugPipeline()
        pl= Pipeline([('preprocessor', self.build_transformers_pipeline(features_dtypes)) ,
                                        # ('imbalancer_resampler', self.build_resampler(sampler_type,sampling_strategy='not majority')),  ('dbg_1', dbg),
                        # ('classification', DecisionTreeClassifier())  # so bad
                        #  ('classification', GradientBoostingClassifier())
                        #                 ('classification', LogisticRegression(max_iter=500))  # Best for Recall 1
                        #  ('classification', GaussianNB())  # 0.5881085402220386
                        #  ('classification', ComplementNB())  # 0.523696690978335
                        #  ('classification', MultinomialNB())  # 0.523696690978335
                        # ('classification', KNeighborsClassifier(3))
                                                        ('classifier', bbc) # ENS(0.61) without explicit overSampling / test_roc_auc : [0.6719306  0.58851217 0.58250362 0.6094371  0.55757417]
                      # ('classifier', BRFC)
                      #  ('classifier', xgb.XGBClassifier())
                        #   ('classifier',SVC())
                        #  ('classifier',RandomForestClassifier(n_estimators=10, max_depth=10, max_features=10, n_jobs=4))
                        # ('classifier',rusboost)
                       ])
        for i, s in enumerate(pl.steps) :
            # Ex: 0 -> ('preprocessor', ColumnTransformer( ... +  1 -> ('classifier', BalancedBaggingClassifier(base_.....
            print(f"{i} -> {s[0]} / {str(s[1])[:70]}")
        return pl

    def fit_grid_search_cv(self, X:pd.DataFrame, y:pd.DataFrame, sampler_type:str, n_splits=5) -> ([BaseEstimator], dict): # (estimator, cv_results)
        model = self.build_predictor_pipeline(X.dtypes, sampler_type)
        CV = StratifiedKFold(n_splits=n_splits) # , random_state=48, shuffle=True
        param_grid = {
            'classifier__base_estimator__l2_regularization': [5, 50, 100, 50],
            'classifier__n_estimators': [3, 5, 10, 20, 50],
            'classifier__base_estimator__max_iter' : [100],
            'classifier__base_estimator__max_depth' : [10,50,10]
        }
        grid = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, cv=CV)
        grid.fit(X, y)
        df_results = pd.DataFrame(grid.cv_results_)
        # columns_to_keep = [
        #     'param_clf__max_depth',
        #     'param_clf__n_estimators',
        #     'mean_test_score',
        #     'std_test_score',
        # ]
        # df_results = df_results[columns_to_keep]
        print(df_results.sort_values(by='mean_test_score', ascending=False))
        # df_results.sort_values(by='mean_test_score', ascending=False)
        df_results.sort_values(by='mean_test_score', ascending=False).to_csv(os.path.join(C.rootProject(), 'log','grid_search_csv'), index = False)
# rootReports

    def print_model_params_keys(self, model:BaseEstimator):
        for param in model.get_params().keys():
            print(param)

    # 1 - Fit without any Cross Validation
    def fit_and_plot(self, X_train:pd.DataFrame, y_train:pd.DataFrame,  X_test:pd.DataFrame, y_test:pd.DataFrame, sampler_type:str) -> BaseEstimator:
        fitted_model = self.fit(X_train, y_train, sampler_type)
        # print(f"Type:{type(fitted_model)} - {fitted_model.get_params()}")
        # self.print_model_params_keys(fitted_model)
        self.predict_and_plot(fitted_model, X_test, y_test)
        return fitted_model

    def fit(self, X_train:pd.DataFrame, y_train:pd.DataFrame, sampler_type:str) -> BaseEstimator:
        model = self.build_predictor_pipeline(X_train.dtypes, sampler_type)
        return model.fit(X_train, y_train)

    ''' 2 - Fit with Cross Validation
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
    '''
    def fit_cv(self, X:pd.DataFrame, y:pd.DataFrame, sampler_type:str, n_splits=5) -> ([BaseEstimator], dict): # (estimator, cv_results)
        model = self.build_predictor_pipeline(X.dtypes, sampler_type)
        CV = StratifiedKFold(n_splits=n_splits) # , random_state=48, shuffle=True
        cv_results =  cross_validate(model, X, y, cv=CV, scoring=('f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'recall', 'precision', 'average_precision', 'roc_auc'), return_train_score=True, return_estimator=True)
        fitted_estimators = []
        for key in cv_results.keys() :
            if str(key) !=  "estimator" :
                print(f"{key} : {cv_results[key]}")
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
        print(f"- Roc_auc_score: {roc_auc_score(y_test, y_pred)}")
        # print(f"- auc : {auc(y_test, y_pred)}")  # ValueError: x is neither increasing nor decreasing : [0 0 0 ... 0 0 0]
        print(f"- Average_precision_score: {average_precision_score(y_test, y_pred)}")
        print(f"- Precision_score: {precision_score(y_test, y_pred)}")
        print(f"- Recall score: {recall_score(y_test, y_pred)}")
        print(f"- F1 score: {f1_score(y_test, y_pred)}")
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

    # def plot_roc(self, y_test, y_pred):
    #     # y_test = label_binarize(y_test.values, classes=[0, 1])  # y_test 'Series'
    #     # y_pred = label_binarize(y_pred, classes=[0, 1])         # y_pred  'numpy.ndarray'
    #     plt.figure()
    #     lw = 2
    #     roc = roc_curve(y_test, y_pred)
    #     plt.plot(roc[0], roc[1], color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % roc_auc_score(y_test, y_pred))
    #     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver operating characteristic')
    #     plt.legend(loc="lower right")
    #     plt.show()
    #
    # def plot_precision_recall(self, y_test, y_pred):
    #     average_precision = average_precision_score(y_test, y_pred)
    #     plt.figure()
    #     lw = 2
    #     pr = precision_recall_curve(y_test, y_pred)
    #     plt.plot(pr[0], pr[1], color='darkorange', lw=lw, label='Precision Recall curve (area = %0.4f)' % average_precision)
    #     plt.xlim([0.0, 1.05])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.title('Precision Recall curve')
    #     plt.legend(loc="upper right")
    #     plt.show()
        #
        # for i in range(0, len(pr[0]) ) :
        #     print(f"{i}: ({pr[0][i]},{pr[1][i]})")


    '''
    SMOTe is a technique based on nearest neighbors judged by Euclidean Distance between data points in feature space.
    random_over_sampling : The most naive strategy is to generate new samples by randomly sampling with replacement the current available samples.
    adasyn_over_sampling : Adaptive Synthetic: focuses on generating samples next to the original samples which are wrongly classified using a k-Nearest Neighbors classifier
    smote_over_sampling  : Synth Minority Oversampling Techn: will not make any distinction between easy and hard samples to be classified using the nearest neighbors rule
    ---    
    https://medium.com/towards-artificial-intelligence/application-of-synthetic-minority-over-sampling-technique-smote-for-imbalanced-data-sets-509ab55cfdaf
    https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html
    NB: 
    How to apply SMOTE : Shuffling and Splitting the Dataset into Training and Validation Sets and THEN applying SMOTe on the Training Dataset.
    '''
    def build_resampler(self, sampler_type: str, sampling_strategy='auto', k_neighbors=5) -> BaseOverSampler :
        rand_state = 48
        if sampler_type.lower() == C.random_over_sampler :
            return RandomOverSampler(sampling_strategy=sampling_strategy, random_state=rand_state)
        elif sampler_type.lower() == C.adasyn_over_sampling :
            return ADASYN(sampling_strategy=sampling_strategy, random_state=rand_state, n_neighbors=k_neighbors)
        elif sampler_type.lower() == C.smote_over_sampling :
            return SMOTE(sampling_strategy=sampling_strategy, random_state=rand_state, k_neighbors=k_neighbors)
        # elif sampler_type.lower() == C.smote_nc_over_sampling :      # SMOTE for dataset containing continuous and categorical features.
        #     return SMOTENC(sampling_strategy=sampling_strategy, random_state=rand_state, k_neighbors=k_neighbors)
        elif sampler_type.lower() == C.smote_svm_over_sampling :    # Use an SVM algorithm to detect sample to use for generating new synthetic samples
            return SVMSMOTE(sampling_strategy=sampling_strategy, random_state=rand_state, k_neighbors=k_neighbors, svm_estimator=SVC())
        elif sampler_type.lower() == C.smote_kmeans_over_sampling : # Apply a KMeans clustering before to over-sample using SMOTE
            return KMeansSMOTE(sampling_strategy=sampling_strategy, random_state=rand_state, k_neighbors=k_neighbors, kmeans_estimator=MiniBatchKMeans(n_clusters=2), cluster_balance_threshold=5)
        elif sampler_type.lower() == C.smote_bline_over_sampling :  # Borderline samples will be detected and used to generate new synthetic samples.
            return BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=rand_state, k_neighbors=k_neighbors, m_neighbors=3)
        else :
            raise ValueError(f"Unexpected argument [sampler_type:{sampler_type}] for method 'compute_sampler_preprocessor'")

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