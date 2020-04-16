from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, SVMSMOTE, KMeansSMOTE, BorderlineSMOTE, ADASYN
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.metrics._classification import classification_report_imbalanced
# https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.pipeline

from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.impute._iterative import IterativeImputer
# from sklearn.experimental import enable_iterative_imputer   # explicitly require this experimental feature
# from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.metrics import f1_score, auc, roc_auc_score, confusion_matrix, classification_report, \
    precision_recall_curve, precision_recall_fscore_support, roc_curve, plot_precision_recall_curve, \
    average_precision_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
# from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler, label_binarize, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
from sklearn.preprocessing import Normalizer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from valeo.infrastructure.SimpleImputer import SimpleImputer
# from valeo.infrastructure.StandardScaler import StandardScaler
from valeo.infrastructure.tools.DebugPipeline import DebugPipeline
from valeo.infrastructure.LogManager import LogManager
from valeo.infrastructure import Const as C

class DefectPredictor :
    logger = None

    def __init__(self):
        DefectPredictor.logger = LogManager.logger(__name__)

    # https://github.com/scikit-learn-contrib/imbalanced-learn/tree/master/examples
    # https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7
    # https://towardsdatascience.com/custom-transformers-and-ml-data-pipelines-with-python-20ea2a7adb65
    # https://jorisvandenbossche.github.io/blog/2018/05/28/scikit-learn-columntransformer/
    def build_transformers_pipeline(self, features_dtypes:pd.Series) -> ColumnTransformer:
        # numerical_features = (X_df.dtypes == 'int64') | (X_df.dtypes == 'float64')
        numerical_features = (features_dtypes == 'int64') | (features_dtypes == 'float64')
        rand_state = 48
        #categorical_features = ~numerical_features
        # scaled_cols = [C.OP070_V_1_angle_value, C.OP070_V_1_torque_value,
        #                C.OP070_V_2_angle_value, C.OP070_V_2_torque_value,
        #                C.OP090_StartLinePeakForce_value, C.OP090_SnapRingMidPointForce_val,
        #                C.OP090_SnapRingPeakForce_value,  C.OP090_SnapRingFinalStroke_value,
        #                C.OP100_Capuchon_insertion_mesure,
        #                C.OP110_Vissage_M8_angle_value, C.OP110_Vissage_M8_torque_value,
        #                C.OP120_Rodage_I_mesure_value,  C.OP120_Rodage_U_mesure_value]
        #
        # nan_imputer    = SimpleImputer(strategy='mean', missing_values=np.nan, verbose=False)
        # zeroes_imputer = SimpleImputer(strategy='mean', missing_values=0.0, verbose=False)
        nan_imputer    = IterativeImputer(estimator=BayesianRidge(), missing_values=np.nan,  max_iter=10, initial_strategy = 'median',  add_indicator=False, random_state=rand_state)
        zeroes_imputer = IterativeImputer(estimator=BayesianRidge(), missing_values=0,  max_iter=10, initial_strategy = 'median',  add_indicator=False, random_state=rand_state)

        scaler         =  Normalizer()  # RobustScaler() #StandardScaler() # RobustScaler(with_centering=True, with_scaling=False)  # MinMaxScaler()
        #
        dbg = DebugPipeline()
        num_transformers_pipeline = Pipeline([#('dbg_0', dbg),
                                              ('nan_imputer', nan_imputer),
                                              # ('dbg_1', dbg),
                                              ('zeroes_imputer', zeroes_imputer),
                                              # ('dbg_2', dbg),
                                              ('scaler', scaler),
                                              # ('dbg_3', dbg)
                                            ])
        return ColumnTransformer([('transformers_pipeline',num_transformers_pipeline, numerical_features)],
                                 remainder='passthrough')


    def build_predictor_pipeline(self, features_dtypes:pd.Series, sampler_type: str) -> Pipeline:
        #Create an object of the classifier.
        bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                        sampling_strategy='auto',
                                        replacement=False,
                                        random_state=0)

        dbg = DebugPipeline()
        return Pipeline([('preprocessor', self.build_transformers_pipeline(features_dtypes)) ,
                        ('imbalancer_resampler', self.build_resampler(sampler_type,sampling_strategy='minority')),
                         ('dbg_1', dbg),
                        # ('classification', DecisionTreeClassifier())  # so bad
                        #  ('classification', GradientBoostingClassifier())
                        #     ('classification', LogisticRegression())  # Best for Recall 1
                        #  ('classification', GaussianNB())  # 0.5881085402220386
                        #  ('classification', ComplementNB())  # 0.523696690978335
                        #  ('classification', MultinomialNB())  # 0.523696690978335
                         # ('classification', KNeighborsClassifier(3))
                        # ('classifier', bbc) # so bad
                        #   ('classifier',SVC())
                         ('classifier', RandomForestClassifier())
                       ])

# GOOGLE ON: classifier over sampled imbalanced dataset
# https://sci2s.ugr.es/imbalanced  : Tres interessant
# https://www.datacamp.com/community/tutorials/diving-deep-imbalanced-data  : Tres interessant
#---
# https://journalofbigdata.springeropen.com/articles/10.1186/s40537-017-0108-1
# xperimental evaluation
# The projected technique works on binary-class/multi-class imbalanced Big Data sets in the organization to recommended LVH. Four basic classifiers viz. Random Forest (-P 100-I 100-num-slots 1-K 0-M 1.0-V 0.001-S 1), Naïve Bayes, AdaBoostM1 (-P 100-S 1-I 10-W weka.classifiers.trees.DecisionStump) and MultiLayer Perceptron (-L 0.3-M 0.2-N 500-V 0-S 0-E 20-H a) are applied to over_sampled data sets using dissimilar values of cross-validation and KNN. Lastly the results, based on the F-measure and AUC values are used to compare between benchmarking (SMOTE/Borderline-SMOTE/ADASYN/SPIDER2/SMOTEBoost/MWMOTE) and planned technique (UCPMOT). Tables 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 and 13 describe the results in detail. The analysis of results validates the superiority of UCPMOT for enhancing the classification.

    def fit(self, X_train:pd.DataFrame, y_train:pd.DataFrame,
            X_test:pd.DataFrame, y_test:pd.DataFrame,
            sampler_type:str):
        model = self.build_predictor_pipeline(X_train.dtypes, sampler_type)
        model.fit(X_train, y_train)
        self.predict_and_plot(model, X_train, y_train, X_test, y_test)

    def predict_and_plot(self, model,  X_train:pd.DataFrame, y_train:pd.DataFrame,
               X_test:pd.DataFrame, y_test:pd.DataFrame):
        # print(f"Type:{type(model)} - {model}")
        y_pred = model.predict(X_test)
        #
        print(f"- Model score: {model.score(X_test, y_test)}")
        print(f"- F1 score:{f1_score(y_test, y_pred)}")
        print(f"- roc_auc_score:{roc_auc_score(y_test, y_pred)}")
        print(confusion_matrix(y_test, y_pred))
        print(f"->\t\t\t\tclassification_report_imbalanced:\n{classification_report_imbalanced(y_test, y_pred)}")
        print(f"->\t\t\t\tclassification_report:\n{classification_report(y_test, y_pred)}")
        print(f"->\t\t\t\tprecision_recall_curve:\n{precision_recall_curve(y_test, y_pred)}")
        print(f"->\t\t\t\tprecision_recall_fscore_support:\n{precision_recall_fscore_support(y_test, y_pred)}")
        print(f"->\t\t\t\troc_auc_score:\n{roc_auc_score(y_test, y_pred)}")
        print(f"->\t\t\t\troc_curve:\n{roc_curve(y_test, y_pred)}")
        # print(auc(y_test, y_pred))
        self.plot_roc(y_test, y_pred)
        self.plot_precision_recall(model, X_test, y_test, y_pred)

    def fit_cross_validate(self, X_train:pd.DataFrame, y_train:pd.DataFrame,
            X_test:pd.DataFrame, y_test:pd.DataFrame,
            sampler_type:str):
        model = self.build_predictor_pipeline(X_train.dtypes, sampler_type)
        cv_results = cross_validate(model, X_train, y_train, cv=10,
                                    scoring=('f1', 'f1_micro', 'f1_macro', 'f1_weighted',
                                             'recall', 'precision', 'average_precision',
                                             'roc_auc', 'roc_auc_ovr','roc_auc_ovo'),
                                    return_train_score=True, return_estimator=True)  #True
        for key in cv_results.keys() :
            print(f"{key} : {cv_results[key]}")

        for fitted_model in cv_results['estimator'] :
            self.predict_and_plot(fitted_model, X_train, y_train, X_test, y_test)

        # cross_val_predict(model, X_train, y_train, cv=10)


    def plot_roc(self, y_test, y_pred):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        y_test = label_binarize(y_test.values, classes=[0, 1])  # y_test 'Series'
        y_pred = label_binarize(y_pred, classes=[0, 1])         # y_pred  'numpy.ndarray'
        n_classes = y_test.shape[1]
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure()
        lw = 2
        plt.plot(fpr[0], tpr[0], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
        plt.plot([0, 1], [0, 1],  color='navy',
                 lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    def plot_precision_recall(self, model, X_test, y_test, y_pred):
        average_precision = average_precision_score(y_test, y_pred)
        print('->\t\t\t\tAverage precision-recall score: {0:0.2f}'.format(average_precision))
        disp = plot_precision_recall_curve(model, X_test, y_test)
        disp.ax_.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
        plt.show()


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