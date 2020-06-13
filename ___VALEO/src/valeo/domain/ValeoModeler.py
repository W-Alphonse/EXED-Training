from category_encoders import OneHotEncoder
from imblearn.ensemble import BalancedBaggingClassifier, RUSBoostClassifier, BalancedRandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, SVMSMOTE, KMeansSMOTE, BorderlineSMOTE
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer

from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer   # explicitly require this experimental feature
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.preprocessing import Normalizer, OrdinalEncoder
from sklearn.preprocessing import RobustScaler, MinMaxScaler, label_binarize, StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

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

    def build_transformers_pipeline(self, X_df: pd.DataFrame) -> ColumnTransformer:
        rand_state = 48
        numerical_features = DfUtil.numerical_cols(X_df)
        nan_imputer    = IterativeImputer(estimator=BayesianRidge(), missing_values=np.nan,  max_iter=10, initial_strategy = 'median',  add_indicator=True, random_state=rand_state)
        zeroes_imputer = IterativeImputer(estimator=BayesianRidge(), missing_values=0,  max_iter=10, initial_strategy = 'median',  add_indicator=True, random_state=rand_state)
        scaler         =  RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0))  # Normalizer()  # RobustScaler() #StandardScaler() # RobustScaler(with_centering=True, with_scaling=False)  # MinMaxScaler()
        # NB: When using log tranformer: Adopt this transformation -> log(-2) = -1×(log(abs(-2)+1))
        num_transformers_pipeline = Pipeline([ ('nan_imputer', nan_imputer),
                                               ('zeroes_imputer', zeroes_imputer),
                                               ('scaler', scaler) ])
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



    def build_simple_transformers_pipeline(self, features_dtypes:pd.Series) -> ColumnTransformer:
        rand_state = 48
        print(type(features_dtypes))
        numerical_features = (features_dtypes == 'int64') | (features_dtypes == 'float64')
        # categorical_features = ~numerical_features
        # nan_imputer    = SimpleImputer(strategy='mean', missing_values=np.nan, verbose=False)
        nan_imputer    = IterativeImputer(estimator=BayesianRidge(), missing_values=np.nan,  max_iter=10, initial_strategy = 'median',  add_indicator=True, random_state=rand_state)
        zeroes_imputer = IterativeImputer(estimator=BayesianRidge(), missing_values=0,  max_iter=10, initial_strategy = 'median',  add_indicator=True, random_state=rand_state)
        scaler         =  RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))  # Normalizer()  # RobustScaler() #StandardScaler() # RobustScaler(with_centering=True, with_scaling=False)  # MinMaxScaler()
        # scaler  = Normalizer(norm='l1')
        # NB: When using log tranformer: Adopt this transformation -> log(-2) = -1×(log(abs(-2)+1))
        # dbg = DebugPipeline()
        num_transformers_pipeline = Pipeline([  #('dbg_0', dbg),
            ('nan_imputer', nan_imputer),       # ('dbg_1', dbg),
            ('zeroes_imputer', zeroes_imputer), # ('dbg_2', dbg),
            ('scaler', scaler),                 # ('dbg_3', dbg)
        ])
        return ColumnTransformer([('transformers_pipeline',num_transformers_pipeline, numerical_features)], remainder='passthrough')

    def build_predictor_pipeline(self, X_df: pd.DataFrame, clfTypes:[str]) -> Pipeline:
        clfs = {
            # The 1st one is the official one
            # C.BRFC : BalancedRandomForestClassifier(criterion= 'gini', max_depth= 10, max_features= 'log2', min_samples_split= 18, n_estimators= 300, oob_score= True, sampling_strategy= 'auto') ,

            # ***grid*** / rank_1 : roc_auc sur test_train is equal to '0.5164' - NB: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples.
            # C.BRFC : BalancedRandomForestClassifier(criterion= 'entropy', max_depth= 10, max_features= 'sqrt', min_samples_split= 18, n_estimators= 200, oob_score= True, sampling_strategy= 0.13) ,
            # grid / rank_7 related to min(trainScore - testScore) 1 : roc_auc sur test_train is equal to '0.7140' / ENS: O.65
            # C.BRFC : BalancedRandomForestClassifier(criterion= 'entropy', max_depth= 8, max_features= 'log2', min_samples_split= 18, n_estimators= 200, oob_score= False, sampling_strategy= 'auto') ,
            # ***random*** / rank_1 :roc_auc sur test_train is equal to '0.7548' / ENS:  0.6672(refit sur la totalité des donnée) vs 0.6691(use the best fit sur la totalité des données)
            # {'classifier__criterion': 'entropy', 'classifier__max_depth': 11, 'classifier__max_features': 'log2', 'classifier__min_samples_split': 16, 'classifier__n_estimators': 200, 'classifier__oob_score': False, 'classifier__sampling_strategy': 'auto'}
            # C.BRFC : BalancedRandomForestClassifier(criterion= 'entropy', max_depth= 11, max_features= 'log2', min_samples_split= 16, n_estimators= 200, oob_score= False, sampling_strategy='auto') ,
            # ***opt*** / rank1: roc_auc sur test_train is equal to '0.6770' / ENS: 0.5982
            # OrderedDict([('classifier__criterion', 'entropy'), ('classifier__max_depth', 10), ('classifier__max_features', 'log2'), ('classifier__min_samples_split', 25), ('classifier__n_estimators', 100), ('classifier__oob_score', False), ('classifier__sampling_strategy', 0.35)])
            # C.BRFC : BalancedRandomForestClassifier(criterion= 'entropy', max_depth= 10, max_features= 'log2', min_samples_split= 25, n_estimators= 100, oob_score= False, sampling_strategy= 0.35) ,

            C.BBC_ADABoost  : BalancedBaggingClassifier(base_estimator=AdaBoostClassifier(), n_estimators= 200, max_samples=0.7, max_features= 8,   oob_score= True, replacement=True , sampling_strategy= 'auto', n_jobs=-1),
            C.BBC_GBC : BalancedBaggingClassifier(base_estimator=GradientBoostingClassifier(learning_rate= 0.1,  max_depth= 10, max_features= 'log2', min_samples_split= 18),
                                                  n_estimators= 200, max_samples=0.7, max_features= 8,   oob_score= True, replacement=True , sampling_strategy= 'auto', n_jobs=-1),
            C.BBC_HGBC : BalancedBaggingClassifier(base_estimator=HistGradientBoostingClassifier(max_iter = 100, max_depth=5,learning_rate=0.10, l2_regularization=15, scoring='roc_auc'),
                                                   n_estimators= 200, max_samples=0.7, max_features= 8,   oob_score= True, replacement=True , sampling_strategy= 'auto', n_jobs=-1),
            C.RUSBoost_ADABoost : RUSBoostClassifier(base_estimator = AdaBoostClassifier(), n_estimators = 50, algorithm='SAMME.R', random_state=42),
            #
            C.RFC_SMOTEEN : RandomForestClassifier(criterion= 'gini', max_depth= 8, max_features= 'log2', min_samples_split= 25, n_estimators=100,  oob_score= True, n_jobs=-1),
            C.RFC_SMOTETOMEK : RandomForestClassifier(criterion= 'gini', max_depth= 8, max_features= 'log2', min_samples_split= 25, n_estimators=100,  oob_score= True, n_jobs=-1),
            C.RFC_BLINESMT_RUS : RandomForestClassifier(criterion='gini', max_depth= 8, max_features='log2', min_samples_split= 25, n_estimators=100, oob_score= True, n_jobs=-1),
            #
            # Resultat officiel occupant actuellment la premiere place
            C.LRC_SMOTEEN  : LogisticRegression(C= 1000, fit_intercept= False, max_iter= 1000, penalty='l2', solver='saga'),

            # ***grid*** / rank_1 : roc_auc sur test_train is equal to '0.6659' / ENS : 0.6545
            # C.LRC_SMOTEEN  : LogisticRegression(C= 10, fit_intercept= True, max_iter= 600, penalty='l2', solver='lbfgs'),
            # ***random*** / rank_1 : roc_auc sur test_train is equal to '0.' / ENS : 0.6545  / 0.6481 best split
            # {'classifier__solver': 'lbfgs', 'classifier__penalty': 'l2', 'classifier__max_iter': 600, 'classifier__fit_intercept': True, 'classifier__C': 100}
            # C.LRC_SMOTEEN  : LogisticRegression(C= 100, fit_intercept= True, max_iter= 600, penalty='l2', solver='lbfgs'),


            C.SVC_SMOTEEN  : SVC(kernel="rbf", gamma='scale', C=10, probability=True, random_state=42) , #, class_weight={1: 10}) il se peut que le class_weight rajoute de l overfit,
            C.KNN_SMOTEEN : KNeighborsClassifier(n_neighbors=7, weights='uniform'),
            C.GNB_SMOTENN: GaussianNB(),

            # SVM --> https://medium.com/analytics-vidhya/hyperparameter-tuning-an-svm-a-demonstration-using-hyperparameter-tuning-cross-validation-on-96b05db54e5b
            # The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of thousands of sample
            # With the Radial Basis Function (RBF) kernel, two parameters must be considered: C and gamma / https://scikit-learn.org/stable/modules/svm.html#svm-kernels
            # A support vector machine constructs a hyper-plane or set of hyper-planes in a high or infinite dimensional space, which can be used for classification,
            # regression or other tasks. Intuitively, a good separation is achieved by the hyper-plane that has the largest distance to the nearest training data points
            # of any class (so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier.
            # C.SVC  : SVC(kernel="rbf",  gamma='scale', C=0.025, probability=True, random_state=42),
            # In problems where it is desired to give more importance to certain classes or certain individual samples, the parameters class_weight and sample_weight can be used.
            # C1=50 better(may be overfit) than C=25(still small overfit) better thant C=15(choui overfit) better than C1=10 better than C=1 better than C=0.025   / class_weight={1: 10}

            # local test roc = 0.6
            C.HGBC : HistGradientBoostingClassifier(max_iter = 100, max_depth=5,learning_rate=0.10, l2_regularization=15, scoring='roc_auc'),

            # cls.XGBC :  xgb.
            #     XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
            #                   colsample_bynode=1, colsample_bytree=1, gamma=0,
            #                   learning_rate=0.1, max_delta_step=0, max_depth=10, #max_depth=3,
            #                   min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
            #                   nthread=None, objective='binary:logistic', random_state=0,
            #                   reg_alpha=0, reg_lambda=1, scale_pos_weight=100, seed=42,
            #                   silent=None, subsample=1, verbosity=1)
        }

        rand_state = 48
        dbg = DebugPipeline()
        # X.select_dtypes('number').columns.to_list()
        # columns_of_type_number = (columns_of_type_number == 'int64') | (columns_of_type_number == 'float64')
        dt = ColumnTransformer([('delete', pp.DropUnecessaryFeatures(), [C.OP120_Rodage_U_mesure_value, C.OP100_Capuchon_insertion_mesure])] ,  remainder='passthrough')
        ct = ColumnTransformer([('cat_OP100', pp.OP100CapuchonInsertionMesureTransformer(), [C.OP100_Capuchon_insertion_mesure])] ,  remainder='passthrough')
        # ht = ColumnTransformer([('ht',OneHotEncoder(), [C.proc_weekday, C.proc_week, C.proc_month])], remainder='passthrough')
        feats = FeatureUnion([ ('self', self.build_transformers_pipeline(X_df)),
                                ('pp_delete',dt),
                               # ('ht',ht)
                              ])
        # use_smote = clfTypes[0] in {C.LRC, C.GBC, C.HGBC, C.SVC, C.RFC, C.KNN, C.ADABoost}
        pl= Pipeline([ # ('preprocessor', self.build_transformers_pipeline(features_dtypes)) ,
                        # ('feats',feats),
                        ('preprocessor', self.build_transformers_pipeline(X_df) ) ,

                        # ('pp_delete',dt),
                        # ('pp_cat_OP100',ct),
                        # ('hotencoder_transformer', ht),
                        ('hotencoder_transformer', OneHotEncoder()),
                        # ('pca_transformer', PCA(n_components=0.9)),
                        # ('smote', SMOTE(sampling_strategy=0.1) if use_smote else pp.EmtpyTransformer() ),
                        # ('smote', BorderlineSMOTE(sampling_strategy=0.1, m_neighbors=5) if use_smote else pp.EmtpyTransformer() ),
                        # ('undersampler', RandomUnderSampler(sampling_strategy=0.5)  if use_smote else pp.EmtpyTransformer() ),
                        # ('undersampler', SMOTETomek(sampling_strategy='auto')  if use_smote else pp.EmtpyTransformer() ),
                        # ('undersampler', SMOTEENN(sampling_strategy='auto')  if use_smote else pp.EmtpyTransformer() ),
                        *self.compute_first_level_classifier(clfTypes) ,



            # SMOTETomek SMOTEENN
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
        # for i, s in enumerate(pl.steps) :
        #     print(f"{i} -> {s[0]} / {str(s[1])[:70]}")
        return pl

    def compute_first_level_classifier(self, clfTypes:[str]) -> [(str, BaseEstimator)]:
        if clfTypes[0] in {C.RFC_BLINESMT_RUS, C.HGBC} :
            return  [('over_sampler', BorderlineSMOTE(sampling_strategy=0.1, m_neighbors=5)),
                     ('under_sampler', RandomUnderSampler(sampling_strategy=0.5))]
        else :
            return [('combined_over_and_under_sampler',
                     SMOTEENN(sampling_strategy='auto')  if clfTypes[0] in {C.RFC_SMOTEEN, C.LRC_SMOTEEN, C.SVC_SMOTEEN, C.KNN_SMOTEEN, C.GNB_SMOTENN} else
                     SMOTETomek(sampling_strategy='auto')  if clfTypes[0] in { C.RFC_SMOTETOMEK} else
                     pp.EmtpyTransformer() )]
