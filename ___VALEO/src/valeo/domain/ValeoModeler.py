from category_encoders import OneHotEncoder
# from sklearn.preprocessing import OneHotEncoder
from imblearn.ensemble import BalancedBaggingClassifier, RUSBoostClassifier, BalancedRandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, SVMSMOTE, KMeansSMOTE, BorderlineSMOTE
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as pline
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer

from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
# from sklearn.impute._iterative import IterativeImputer
from sklearn.experimental import enable_iterative_imputer   # explicitly require this experimental feature
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.preprocessing import Normalizer, OrdinalEncoder
from sklearn.preprocessing import RobustScaler, MinMaxScaler, label_binarize, StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
import xgboost as xgb

import pandas as pd
import numpy as np

from valeo.domain import Preprocessor as pp
from valeo.infrastructure.LogManager import LogManager
from valeo.infrastructure.tools.DebugPipeline import DebugPipeline
from valeo.infrastructure import Const as C
from valeo.infrastructure.tools.DfUtil import DfUtil

'''
https://github.com/scikit-learn-contrib/imbalanced-learn/tree/master/examples
https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7
https://towardsdatascience.com/custom-transformers-and-ml-data-pipelines-with-python-20ea2a7adb65
https://jorisvandenbossche.github.io/blog/2018/05/28/scikit-learn-columntransformer/
'''

class ValeoModeler :
    logger = None

    def __init__(self):
        ValeoModeler.logger = LogManager.logger(__name__)

    # Cette fct doit etre supprimer car obsolete
    # def prepare_X_for_test(self, X_df: pd.DataFrame, add_flds_to_drop : list) -> pd.DataFrame:
    #     # date_proc = pp.ProcDateTransformer(X_df)
    #     # drop_features = pp.DropUnecessaryFeatures([C.PROC_TRACEINFO] if add_flds_to_drop == None else [C.PROC_TRACEINFO] + add_flds_to_drop )
    #     # X_df = date_proc.transform(X_df)
    #     # X_df = drop_features.transform(X_df)
    #
    #     dt_transf = pp.ProcDateTransformer()
    #     X_df = dt_transf.transform(X_df)
    #     print(f'X_df:{X_df.head()}')
    #     print(f'X_df:{X_df.columns}')
    #     return X_df

    # def _build_transformers_pipeline(self, features_dtypes:pd.Series) -> Pipeline:
    #     rand_state = 48
    #     # print(type(features_dtypes))
    #     numerical_features = (features_dtypes == 'int64') | (features_dtypes == 'float64')
    #     return pline([ #('dbg_0', dbg),
    #         ('nan_imputer', pp.NumericalImputer(IterativeImputer(estimator=BayesianRidge(), missing_values=np.nan,  max_iter=10, initial_strategy = 'median', add_indicator=True, random_state=rand_state)) ),   # ('dbg_1', dbg),
    #         ('zeroes_imputer', pp.NumericalImputer(IterativeImputer(estimator=BayesianRidge(), missing_values=0,  max_iter=10, initial_strategy = 'median',  add_indicator=True, random_state=rand_state)) ),     # ('dbg_2', dbg),
    #         ('scaler', pp.NumericalScaler(RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))) ),
    #         ('cat_proc_date', pp.ProcDateTransformer()),
    #         ('drop_unecessary_features', pp.DropUnecessaryFeatures([C.PROC_TRACEINFO]))  # Normalizer()  # RobustScaler() #StandardScaler() # RobustScaler(with_centering=True, with_scaling=False) )),     # ('dbg_3', dbg)
    #     ])


    # def _build_transformers_pipeline(self, features_dtypes:pd.Series) -> ColumnTransformer:
    def _build_transformers_pipeline(self, X_df: pd.DataFrame) -> ColumnTransformer:

        rand_state = 48
        # print(type(features_dtypes))
        # numerical_features = (features_dtypes == 'int64') | (features_dtypes == 'float64')
        numerical_features = DfUtil.numerical_cols(X_df)
        # categorical_features = ~numerical_features
        # nan_imputer    = SimpleImputer(strategy='median', missing_values=np.nan, verbose=False)
        nan_imputer    = IterativeImputer(estimator=BayesianRidge(), missing_values=np.nan,  max_iter=10, initial_strategy = 'median',  add_indicator=True, random_state=rand_state)
        zeroes_imputer = IterativeImputer(estimator=BayesianRidge(), missing_values=0,  max_iter=10, initial_strategy = 'median',  add_indicator=True, random_state=rand_state)
        scaler         =  RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0))  # Normalizer()  # RobustScaler() #StandardScaler() # RobustScaler(with_centering=True, with_scaling=False)  # MinMaxScaler()
        # OrdinalEncoder()
        # OneHotEncoder()
        # scaler  = Normalizer(norm='l1')
        # NB: When using log tranformer: Adopt this transformation -> log(-2) = -1×(log(abs(-2)+1))
        dbg = DebugPipeline()  # ('dbg_0', dbg),
        num_transformers_pipeline = Pipeline([ ('nan_imputer', nan_imputer),
                                               ('zeroes_imputer', zeroes_imputer),
                                               ('scaler', scaler)
                                               ])
        num_imputer_pipeline = Pipeline([ ('nan_imputer', nan_imputer), ('zeroes_imputer', zeroes_imputer)])



        return ColumnTransformer([
                                  ('num_transformers_pipeline',num_transformers_pipeline, numerical_features),
                                  # ('num_imputer_pipeline',num_imputer_pipeline, numerical_features),
                                  # ('dbg_0', dbg, [C.OP100_Capuchon_insertion_mesure]),
                                  # ('num_right_skewed_dist', pp.LogTransformer(True), [C.OP100_Capuchon_insertion_mesure]),
                                  # ('num_right_skewed_dist', pp.LogTransformer(True), [C.OP070_V_1_angle_value, C.OP070_V_2_angle_value, C.OP110_Vissage_M8_angle_value]),
                                  # ('num_right_skewed_dist', pp.LogTransformer(False), [C.OP110_Vissage_M8_angle_value]),
                                  # ('num_left_skewed_dist', pp.SqrtTransformer(), [C.OP090_SnapRingPeakForce_value]),
                                  # ('num_left_skewed_dist', pp.SqrtTransformer(), [C.OP090_SnapRingMidPointForce_val]),
                                  # ('num_scaler',nan_imputer, numerical_features),
                                  #
                                  ('cat_proc_date', pp.ProcDateTransformer(), [C.PROC_TRACEINFO]),
                                  # ('ht', OneHotEncoder(), [C.proc_weekday, C.proc_week, C.proc_month]),
                                  # ('cat_OP100', pp.OP100CapuchonInsertionMesureTransformer(), [C.OP100_Capuchon_insertion_mesure]),
                                  # ('OP120U', pp.BucketTransformer((C.OP120_Rodage_U_mesure_value,[-np.inf, 11.975, np.inf],[1,2])), [C.OP120_Rodage_U_mesure_value]),
                                  # ('V1_Value', pp.BucketTransformer((C.OP070_V_1_torque_value,[-np.inf, 6.5, np.inf],[1,2])), [C.OP070_V_1_torque_value]),
                                  # ('V2_Value', pp.BucketTransformer((C.OP070_V_2_torque_value,[-np.inf, 6.5, np.inf],[1,2])), [C.OP070_V_2_torque_value]),

                                  #
                                  ('drop_unecessary_features', pp.DropUnecessaryFeatures(), [C.OP120_Rodage_U_mesure_value]),
                                  # ('num_scaler',scaler, numerical_features),
                                  ], remainder='passthrough')


# -----Option-1
#     ('num_transformers_pipeline',num_transformers_pipeline, numerical_features), + quantile_range=(25.0, 75.0)) OR quantile_range=(5.0, 95.0)
#     ('cat_proc_date', pp.ProcDateTransformer(), [C.PROC_TRACEINFO])
#     le même score Avec-ou-Sans-Ceci: ('drop_unecessary_features', pp.DropUnecessaryFeatures(), [C.OP120_Rodage_U_mesure_value]),
#     - [7074 3189]/[32 60] - P:0.0185 - R:0.6522 - roc_auc:0.6707 - f1:0.0359
#     - [[7074 3189]
#       [  32   60]]
# -----Option-2 : Option-1  + Rajout de ('hotencoder_transformer', OneHotEncoder()),
#     - [7198 3065]/[32 60] - P:0.0192 - R:0.6522 - roc_auc:0.6768 - f1:0.0373
#     - [[7198 3065]
#        [  32   60]]


    def _build_transformers_pipeline_for_smote(self, X_df: pd.DataFrame) -> ColumnTransformer:
        rand_state = 48
        numerical_features = DfUtil.numerical_cols(X_df)
        nan_imputer    = IterativeImputer(estimator=BayesianRidge(), missing_values=np.nan,  max_iter=10, initial_strategy = 'median',  add_indicator=True, random_state=rand_state)
        zeroes_imputer = IterativeImputer(estimator=BayesianRidge(), missing_values=0,  max_iter=10, initial_strategy = 'median',  add_indicator=True, random_state=rand_state)
        scaler         =  RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0))  # Normalizer()  # RobustScaler() #StandardScaler() # RobustScaler(with_centering=True, with_scaling=False)  # MinMaxScaler()
        num_transformers_pipeline = Pipeline([ ('nan_imputer', nan_imputer),
                                               ('zeroes_imputer', zeroes_imputer),
                                               ('scaler', scaler)
                                               ])
        num_imputer_pipeline = Pipeline([ ('nan_imputer', nan_imputer), ('zeroes_imputer', zeroes_imputer)])



        return ColumnTransformer([
            # ('num_transformers_pipeline',num_transformers_pipeline, numerical_features),
            ('num_imputer_pipeline',num_imputer_pipeline, numerical_features),
            ('cat_proc_date', pp.ProcDateTransformer(), [C.PROC_TRACEINFO]),
            # ('ht',OneHotEncoder(), [C.proc_weekday, C.proc_week, C.proc_month]),
            # ('cat_OP100', pp.OP100CapuchonInsertionMesureTransformer(), [C.OP100_Capuchon_insertion_mesure]),
            # ('OP120U', pp.BucketTransformer((C.OP120_Rodage_U_mesure_value,[-np.inf, 11.975, np.inf],[1,2])), [C.OP120_Rodage_U_mesure_value]),
            # ('V1_Value', pp.BucketTransformer((C.OP070_V_1_torque_value,[-np.inf, 6.5, np.inf],[1,2])), [C.OP070_V_1_torque_value]),
            # ('V2_Value', pp.BucketTransformer((C.OP070_V_2_torque_value,[-np.inf, 6.5, np.inf],[1,2])), [C.OP070_V_2_torque_value]),
            #
            ('drop_unecessary_features', pp.DropUnecessaryFeatures(), [C.OP120_Rodage_U_mesure_value]),
            # ('smote', BorderlineSMOTE(sampling_strategy=0.1, m_neighbors=5)),
            # ('undersampler', RandomUnderSampler(sampling_strategy=0.5)),
            # ('num_scaler',scaler, numerical_features),
        ], remainder='passthrough')




    def build_transformers_pipeline(self, features_dtypes:pd.Series) -> ColumnTransformer:
        rand_state = 48
        print(type(features_dtypes))
        numerical_features = (features_dtypes == 'int64') | (features_dtypes == 'float64')
        # categorical_features = ~numerical_features
        # nan_imputer    = SimpleImputer(strategy='mean', missing_values=np.nan, verbose=False)
        nan_imputer    = IterativeImputer(estimator=BayesianRidge(), missing_values=np.nan,  max_iter=10, initial_strategy = 'median',  add_indicator=True, random_state=rand_state)
        zeroes_imputer = IterativeImputer(estimator=BayesianRidge(), missing_values=0,  max_iter=10, initial_strategy = 'median',  add_indicator=True, random_state=rand_state)
        scaler         =  RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))  # Normalizer()  # RobustScaler() #StandardScaler() # RobustScaler(with_centering=True, with_scaling=False)  # MinMaxScaler()
        # OrdinalEncoder()
        # OneHotEncoder()
        # scaler  = Normalizer(norm='l1')
        # NB: When using log tranformer: Adopt this transformation -> log(-2) = -1×(log(abs(-2)+1))
        # dbg = DebugPipeline()
        num_transformers_pipeline = Pipeline([ #('dbg_0', dbg),
            ('nan_imputer', nan_imputer),       # ('dbg_1', dbg),
            ('zeroes_imputer', zeroes_imputer), # ('dbg_2', dbg),
            ('scaler', scaler),                 # ('dbg_3', dbg)
        ])
        return ColumnTransformer([('transformers_pipeline',num_transformers_pipeline, numerical_features)], remainder='passthrough')

# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
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

    # def build_predictor_pipeline(self, features_dtypes:pd.Series, clfTypes:[str]) -> Pipeline:
    # def build_predictor_pipeline(self, columns_of_type_number: list, clfTypes:[str]) -> Pipeline:
    def build_predictor_pipeline(self, X_df: pd.DataFrame, clfTypes:[str]) -> Pipeline:
        cls = self.__class__
        clfs = {
            # 'classifier__loss' : ['deviance', 'exponential'],  #https://stackoverflow.com/questions/53533942/loss-parameter-explanation-for-sklearn-ensemble-gradientboostingclassifier
            # 'classifier__learning_rate'   : [0.07, 0.2, 0.5], #uniform(0.001, 0.5),
            # 'classifier__n_estimators' : [500, 300],
            # # 'classifier__l2_regularization' :  random.uniform(0.0, 0.5),
            # 'classifier__subsample' : [0.7, 0.8],
            # 'classifier__min_samples_split' :  [8, 12, 18],
            # 'classifier__max_depth'   : [10, 15, 20],
            # 'classifier__max_features' : ['sqrt', 'log2'],

            # local test roc = 0.6
            C.HGBC : HistGradientBoostingClassifier(max_iter = 100, max_depth=5,learning_rate=0.10, l2_regularization=1.5, scoring='roc_auc'),

            # Search_3 - Retained
            # https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
            C.GBC  : GradientBoostingClassifier(learning_rate= 0.05, n_estimators= 100, subsample= 0.7, max_depth= 10, max_features= 'log2', min_samples_split= 18, loss= 'exponential'),

# https://imbalanced-learn.readthedocs.io/en/stable/auto_examples/ensemble/plot_comparison_ensemble_classifier.html
# Ensembling classifiers have shown to improve classification performance compare to single learner. However, they will be affected by class imbalance.
# There is a benefit of balancing the training set before to learn learners. We are making the comparison with non-balanced ensemble methods.
# BalancedRandomForestClassfier vs RandomForestClassifier
# BaalncedBaggingClassifier vs Classifier
# EasyClassifier(AdaBoost) / RUSBoostClassifier(AdaBoost)
            # Cette config a obtenue le roc_auc le plus haut, score 0.6730(0.6721)  ENS. mais il avait beaucoup de overfitting
            # cls.BRFC : BalancedRandomForestClassifier(n_estimators = 300 , max_depth=20, random_state=0) , # sampling_strategy=0.5),
            #
            # Cette config a été selectionné en se basant sur GridSearchCV, mais elle a obtenue un score ENS de 0.65 seulement. Elle ameliore l overfitting par rapport au cas precedent
            # cls.BRFC : BalancedRandomForestClassifier(n_estimators = 300 , max_depth=20, min_samples_split=12, min_samples_leaf=15, random_state=0, criterion='gini') , # sampling_strategy=0.5),
            #
            # Cette config a été selectionné en se basant sur RandomizedSearchCV, mais elle genere lerreur:
            # UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples
            # cls.BRFC : BalancedRandomForestClassifier(n_estimators = 61 , max_depth=8, min_samples_split=8, min_samples_leaf=9,  sampling_strategy=0.15, random_state=0, criterion='gini') , # sampling_strategy=0.5),
            # cls.BRFC : BalancedRandomForestClassifier(n_estimators = 102 , max_depth=6, min_samples_split=18, min_samples_leaf=13,  sampling_strategy=0.15, random_state=0, criterion='gini') , # sampling_strategy=0.5),
            #
            # Cette config a obtenue 0.6536 ENS
            # C.BRFC : BalancedRandomForestClassifier(max_depth=15, min_samples_leaf=5, n_estimators=260, random_state=0) , # sampling_strategy=0.5),
            # - Best Score: 0.7124 (Train 0.8869) / Best Params: {'classifier__max_depth': 11, 'classifier__min_samples_leaf': 5, 'classifier__n_estimators': 113, 'classifier__sampling_strategy': 0.4}
            # - Generalized Score: 0.7086 (Train 0.8143, rank 11.0000) / Best Generalized Params: {'classifier__max_depth': 5, 'classifier__min_samples_leaf': 5, 'classifier__n_estimators': 260, 'classifier__sampling_strategy': 0.4}
            # ENS:  0.6676
            # C.BRFC : BalancedRandomForestClassifier(criterion= 'gini', max_depth= 10, max_features= 'log2', min_samples_split= 15, n_estimators= 300, oob_score= True, sampling_strategy= 'auto') ,
            # Best Score - SearchCV_02 -  Did not generalize well on local Test
            # C.BRFC : BalancedRandomForestClassifier(criterion= 'entropy', max_depth= 8, max_features= None, min_samples_split= 15, n_estimators= 152, oob_score= True, sampling_strategy= 0.15),
            #
            # Best Generalized - SearchCV_02 - 0.67207 ENS
            # A balanced random forest randomly under-samples each boostrap sample to balance it. sqrt
            C.BRFC : BalancedRandomForestClassifier(criterion= 'gini', max_depth= 10, max_features= 'log2', min_samples_split= 18, n_estimators= 300, oob_score= True, sampling_strategy= 'auto') ,

            # https://imbalanced-learn.readthedocs.io/en/stable/auto_examples/ensemble/plot_comparison_ensemble_classifier.html
            # This implementation of Bagging is similar to the scikit-learn implementation. It includes an additional step to balance the training set at fit time using a RandomUnderSampler
            C.BBC  : BalancedBaggingClassifier(n_estimators= 300, max_samples=0.9, max_features= 12,   oob_score= True, replacement=True , sampling_strategy= 'auto', n_jobs=-1),

            C.RFC : RandomForestClassifier(criterion= 'gini', max_depth= 8, max_features= 'log2', min_samples_split= 25, n_estimators=300,  oob_score= True, n_jobs=-1),

            # Best Score - SearchCV_02
            # https://medium.com/@venali/conventional-guide-to-supervised-learning-with-scikit-learn-logistic-regression-generalized-e9783c414588
            C.LRC  : LogisticRegression(C= 1000, fit_intercept= False, max_iter= 1000, penalty= 'l2', solver= 'saga'),
            # C.LRC  : LogisticRegression(C= 1000, fit_intercept= False, max_iter= 1000, penalty= 'l2', solver= 'lbfgs'),

            # https://medium.com/analytics-vidhya/hyperparameter-tuning-an-svm-a-demonstration-using-hyperparameter-tuning-cross-validation-on-96b05db54e5b
            # The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of thousands of sample
            # With the Radial Basis Function (RBF) kernel, two parameters must be considered: C and gamma / https://scikit-learn.org/stable/modules/svm.html#svm-kernels
            # A support vector machine constructs a hyper-plane or set of hyper-planes in a high or infinite dimensional space, which can be used for classification,
            # regression or other tasks. Intuitively, a good separation is achieved by the hyper-plane that has the largest distance to the nearest training data points
            # of any class (so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier.
            # C.SVC  : SVC(kernel="rbf",  gamma='scale', C=0.025, probability=True, random_state=42),
            # In problems where it is desired to give more importance to certain classes or certain individual samples, the parameters class_weight and sample_weight can be used.
            # C1=50 better(may be overfit) than C=25(still small overfit) better thant C=15(choui overfit) better than C1=10 better than C=1 better than C=0.025   / class_weight={1: 10}
            C.SVC  : SVC(kernel="rbf",  gamma='scale', C=10, probability=True, random_state=42) , #, class_weight={1: 10}) il se peut que le class_weight rajoute de l overfit,

            #  The higher the gamma value it tries to exactly fit the training data set
            # https://medium.com/@mohtedibf/in-depth-parameter-tuning-for-knn-4c0de485baf6
            # https://ashokharnal.wordpress.com/tag/kneighborsclassifier-explained/
            # https://ashokharnal.wordpress.com/2015/01/20/a-working-example-of-k-d-tree-formation-and-k-nearest-neighbor-algorithms/
            # n_neighbors=7 better than 5
            C.KNN : KNeighborsClassifier(n_neighbors=9, weights='uniform', n_jobs=-1, leaf_size=10), # esssayer entre n_neighbors=9 et 7

            # cls.RUSBoost : RUSBoostClassifier( n_estimators = 50 , algorithm='SAMME.R', random_state=42),
            C.RUSBoost : RUSBoostClassifier(base_estimator = AdaBoostClassifier(n_estimators=10), n_estimators = 50 , algorithm='SAMME.R', random_state=42),
            C.ADABoost : AdaBoostClassifier(n_estimators = 500, learning_rate= 0.05, algorithm='SAMME.R', random_state=42),

            cls.XGBC :  xgb.
                XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                              colsample_bynode=1, colsample_bytree=1, gamma=0,
                              learning_rate=0.1, max_delta_step=0, max_depth=10, #max_depth=3,
                              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
                              nthread=None, objective='binary:logistic', random_state=0,
                              reg_alpha=0, reg_lambda=1, scale_pos_weight=100, seed=42,
                              silent=None, subsample=1, verbosity=1)
        }
        rand_state = 48
        dbg = DebugPipeline()
        # X.select_dtypes('number').columns.to_list()
        # columns_of_type_number = (columns_of_type_number == 'int64') | (columns_of_type_number == 'float64')
        dt = ColumnTransformer([('delete', pp.DropUnecessaryFeatures(), [C.OP120_Rodage_U_mesure_value, C.OP100_Capuchon_insertion_mesure])] ,  remainder='passthrough')
        ct = ColumnTransformer([('cat_OP100', pp.OP100CapuchonInsertionMesureTransformer(), [C.OP100_Capuchon_insertion_mesure])] ,  remainder='passthrough')
        # ht = ColumnTransformer([('ht',OneHotEncoder(), [C.proc_weekday, C.proc_week, C.proc_month])], remainder='passthrough')
        feats = FeatureUnion([ ('self', self._build_transformers_pipeline(X_df)),
                                ('pp_delete',dt),
                               # ('ht',ht)
                              ])
        use_smote = clfTypes[0] in {C.LRC, C.GBC, C.HGBC, C.SVC, C.RFC, C.KNN, C.ADABoost}
        pl= Pipeline([ # ('preprocessor', self.build_transformers_pipeline(features_dtypes)) ,
                        # ('feats',feats),
                        ('preprocessor', self._build_transformers_pipeline(X_df) ) ,

                        # ('pp_delete',dt),
                        # ('pp_cat_OP100',ct),
                        # ('hotencoder_transformer', ht),
                        ('hotencoder_transformer', OneHotEncoder()),
                        # ('pca_transformer', PCA(n_components=0.9)),
                        ('smote', BorderlineSMOTE(sampling_strategy=0.1, m_neighbors=5) if use_smote else pp.EmtpyTransformer() ),
                        ('undersampler', RandomUnderSampler(sampling_strategy=0.5)  if use_smote else pp.EmtpyTransformer() ),
                       # ('preprocessor', self._build_transformers_pipeline(columns_of_type_number)) ,
                       ########################
                        # ('nan_imputer', pp.NumericalImputer(columns_of_type_number, IterativeImputer(estimator=BayesianRidge(), missing_values=np.nan,  max_iter=10, initial_strategy = 'median', add_indicator=True, random_state=rand_state)) ),   # ('dbg_1', dbg),
                        # ('zeroes_imputer', pp.NumericalImputer(columns_of_type_number, IterativeImputer(estimator=BayesianRidge(), missing_values=0,  max_iter=10, initial_strategy = 'median',  add_indicator=True, random_state=rand_state)) ),     # ('dbg_2', dbg),
                        # ('scaler', pp.NumericalScaler(columns_of_type_number, RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))) ),
                        # ('cat_proc_date', pp.ProcDateTransformer()),
                        # ('drop_unecessary_features', pp.DropUnecessaryFeatures([C.PROC_TRACEINFO])),
                       ########################
                       # ('imbalancer_resampler', self.build_resampler(C.smote_over_sampling,sampling_strategy='minority')),  # ('dbg_1', dbg),
                      ('classifier', clfs[clfTypes[0]])  # ex: bbc : ENS(0.61) without explicit overSampling / test_roc_auc : [0,.6719306  0.58851217 0.58250362 0.6094371  0.55757417]
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