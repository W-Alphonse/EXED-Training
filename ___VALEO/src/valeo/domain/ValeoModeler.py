from imblearn.ensemble import BalancedBaggingClassifier, RUSBoostClassifier, BalancedRandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, SVMSMOTE, KMeansSMOTE, BorderlineSMOTE
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer

from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
# from sklearn.impute._iterative import IterativeImputer
from sklearn.experimental import enable_iterative_imputer   # explicitly require this experimental feature
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler, MinMaxScaler, label_binarize, StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

import pandas as pd
import numpy as np

from valeo.infrastructure.LogManager import LogManager
from valeo.infrastructure.tools.DebugPipeline import DebugPipeline
from valeo.infrastructure import Const as C

'''
https://github.com/scikit-learn-contrib/imbalanced-learn/tree/master/examples
https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7
https://towardsdatascience.com/custom-transformers-and-ml-data-pipelines-with-python-20ea2a7adb65
https://jorisvandenbossche.github.io/blog/2018/05/28/scikit-learn-columntransformer/
'''

class ValeoModeler :
    logger = None

    def __init__(self):
        logger = LogManager.logger(__name__)

    def build_transformers_pipeline(self, features_dtypes:pd.Series) -> ColumnTransformer:
        rand_state = 48
        numerical_features = (features_dtypes == 'int64') | (features_dtypes == 'float64')
        # categorical_features = ~numerical_features
        # nan_imputer    = SimpleImputer(strategy='mean', missing_values=np.nan, verbose=False)
        nan_imputer    = IterativeImputer(estimator=BayesianRidge(), missing_values=np.nan,  max_iter=10, initial_strategy = 'median',  add_indicator=False, random_state=rand_state)
        zeroes_imputer = IterativeImputer(estimator=BayesianRidge(), missing_values=0,  max_iter=10, initial_strategy = 'median',  add_indicator=False, random_state=rand_state)
        scaler         =  RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0))  # Normalizer()  # RobustScaler() #StandardScaler() # RobustScaler(with_centering=True, with_scaling=False)  # MinMaxScaler()
        # NB: When using log tranformer: Adopt this transformation -> log(-2) = -1×(log(abs(-2)+1))
        # dbg = DebugPipeline()
        num_transformers_pipeline = Pipeline([ #('dbg_0', dbg),
            ('nan_imputer', nan_imputer),       # ('dbg_1', dbg),
            ('zeroes_imputer', zeroes_imputer), # ('dbg_2', dbg),
            ('scaler', scaler),                 # ('dbg_3', dbg)
        ])
        return ColumnTransformer([('transformers_pipeline',num_transformers_pipeline, numerical_features)], remainder='passthrough')

                          # ENS(0.61) without explicit overSampling / test_roc_auc : [0.6719306  0.58851217 0.58250362 0.6094371  0.55757417]
    BBC  = "BBC"          # BalancedBaggingClassifier(base_estimator=HGBR,  sampling_strategy=1.0, replacement=False, random_state=48)
    HGBC = "HGBR"         # HistGradientBoostingClassifier(max_iter = 8 , max_depth=8,learning_rate=0.35, l2_regularization=500)

    BRFC = "BRFC"         # BalancedRandomForestClassifier(n_estimators = 50 , max_depth=20)
    RUSBoost = "RUSBoost" # RUSBoostClassifier(n_estimators = 8 , algorithm='SAMME.R', random_state=42)
    KNN = "KNN"           # KNeighborsClassifier(3),
    SVC = "SVC"           # SVC(kernel="rbf", C=0.025, probability=True)
    NuSVC = "NuSVC"       # NuSVC(probability=True),
    RFC = "RFC"           # RandomForestClassifier(n_estimators=10, max_depth=10, max_features=10, n_jobs=4))
    DTC = "DTC"           # DecisionTreeClassifier())  # so bad
    ADABoost = "ADABoost" # AdaBoostClassifier()
    GBC  = "GBC"          # GradientBoostingClassifier()
    LRC  = "LRC"          # LogisticRegression(max_iter=500))  # Best for Recall 1
    XGBC = "XGBC"         # xgb.XGBClassifier()
    #  ('classification', GaussianNB())  # 0.5881085402220386
    #  ('classification', ComplementNB())  # 0.523696690978335
    #  ('classification', MultinomialNB())  # 0.523696690978335
    Imbl_Resampler =  "Imbl_Resampler"  # ('imbalancer_resampler', self.build_resampler(sampler_type,sampling_strategy='not majority'))

    def build_predictor_pipeline(self, features_dtypes:pd.Series, clfTypes:[str]) -> Pipeline:
        cls = self.__class__
        clfs = {
            cls.HGBC : HistGradientBoostingClassifier(max_iter = 100 , max_depth=10,learning_rate=0.10, l2_regularization=5),
            cls.BBC  : BalancedBaggingClassifier(base_estimator=HistGradientBoostingClassifier(),  n_estimators=50, sampling_strategy='auto', replacement=False, random_state=48),

            # scale_pos_weight
            # ESTIM:100 depth:20 [6155 4108]/[41 51] - P:0.0123 - R:0.5543 - roc_auc:0.5770 - f1:0.0240 |
            #.ESTIM:300 depth:10 [6085 4178]/[37 55] - P:0.0130 - R:0.5978 - roc_auc:0.5954 - f1:0.0254
            # ESTIM:300 depth:15 [6306 3957]/[37 55] - P:0.0137 - R:0.5978 - roc_auc:0.6061 - f1:0.0268
            # ESTIM:300 depth:20 [6057 4206]/[33 59] - P:0.0138 - R:0.6413 - roc_auc:0.6157 - f1:0.0271   ***
            # ESTIM:300 depth:20 class_weight:{0:1, 1:100} [2860 7403]/[22 70] - P:0.0094 - R:0.7609 - roc_auc:0.5198 - f1:0.0185
            # [6127 4136]/[35 57] - P:0.0136 - R:0.6196 - roc_auc:0.6083 - f1:0.0266
            # [6184 4079]/[37 55] - P:0.0133 - R:0.5978 - roc_auc:0.6002 - f1:0.0260
            # [6121 4142]/[37 55] - P:0.0131 - R:0.5978 - roc_auc:0.5971 - f1:0.0256
            # ESTIM:300 depth:30 [6223 4040]/[36 56] - P:0.0137 - R:0.6087 - roc_auc:0.6075 - f1:0.0267
            # ESTIM:300 depth:40 [6243 4020]/[39 53] - P:0.0130 - R:0.5761 - roc_auc:0.5922 - f1:0.0255
            #.ESTIM:200 depth:10 [6236 4027]/[39 53] - P:0.0130 - R:0.5761 - roc_auc:0.5919 - f1:0.0254
            # ESTIM:200 depth:20 [6104 4159]/[34 58] - P:0.0138 - R:0.6304 - roc_auc:0.6126 - f1:0.0269
            # ESTIM:200 depth:40 [6227 4036]/[37 55] - P:0.0134 - R:0.5978 - roc_auc:0.6023 - f1:0.0263
            cls.BRFC : BalancedRandomForestClassifier(n_estimators = 300 , max_depth=20, random_state=0),

            cls.RUSBoost : RUSBoostClassifier(n_estimators = 8 , algorithm='SAMME.R', random_state=42),
            cls.XGBC :  xgb.
                XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                              colsample_bynode=1, colsample_bytree=1, gamma=0,
                              learning_rate=0.1, max_delta_step=0, max_depth=10, #max_depth=3,
                              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
                              nthread=None, objective='binary:logistic', random_state=0,
                              reg_alpha=0, reg_lambda=1, scale_pos_weight=100, seed=42,
                              silent=None, subsample=1, verbosity=1)
        }
        dbg = DebugPipeline()
        pl= Pipeline([('preprocessor', self.build_transformers_pipeline(features_dtypes)) ,
                       # ('imbalancer_resampler', self.build_resampler(C.smote_over_sampling,sampling_strategy='not majority')),  # ('dbg_1', dbg),
                      ('classifier', clfs[clfTypes[0]])  # ex: bbc : ENS(0.61) without explicit overSampling / test_roc_auc : [0.6719306  0.58851217 0.58250362 0.6094371  0.55757417]
                      ])
        for i, s in enumerate(pl.steps) :
            # Ex: 0 -> ('preprocessor', ColumnTransformer( ... +  1 -> ('classifier', BalancedBaggingClassifier(base_.....
            print(f"{i} -> {s[0]} / {str(s[1])[:70]}")
        return pl


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


# classifiers = [
#     KNeighborsClassifier(3),
#     SVC(kernel="rbf", C=0.025, probability=True),
#     NuSVC(probability=True),
#     DecisionTreeClassifier(),
#     RandomForestClassifier(),
#     AdaBoostClassifier(),
#     GradientBoostingClassifier()
# ]
# for classifier in classifiers:
#     pipe = Pipeline(steps=[('preprocessor', preprocessor),
#                            ('classifier', classifier)])
#     pipe.fit(X_train, y_train)
#     print(classifier)
#     print("model score: %.3f" % pipe.score(X_test, y_test))