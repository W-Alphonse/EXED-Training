from imblearn.ensemble import BalancedBaggingClassifier, RUSBoostClassifier, BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import Normalizer
import pandas as pd

from valeo.infrastructure.LogManager import LogManager
from valeo.infrastructure.tools.DebugPipeline import DebugPipeline


class ValeoModeler :
    logger = None

    def __init__(self):
        logger = LogManager.logger(__name__)

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

    def build_predictor_pipeline(self, features_dtypes:pd.Series, sampler_type: str, clfKeys:[str]) -> Pipeline:
        cls = self.__class__
        dict = {
            cls.HGBC : HistGradientBoostingClassifier(max_iter = 100 , max_depth=10,learning_rate=0.10, l2_regularization=5),

            cls.BBC  : BalancedBaggingClassifier(base_estimator='cls.HGBR',  n_estimators=50, sampling_strategy='auto', replacement=False, random_state=48),
            cls.BRFC : BalancedRandomForestClassifier(n_estimators = 50 , max_depth=20),
            cls.RUSBoost : RUSBoostClassifier(n_estimators = 8 , algorithm='SAMME.R', random_state=42)
        }
        dbg = DebugPipeline()
        pl= Pipeline([('preprocessor', self.build_transformers_pipeline(features_dtypes)) ,
                      # ('imbalancer_resampler', self.build_resampler(sampler_type,sampling_strategy='not majority')),  ('dbg_1', dbg),
                      ('classifier', dict[clfKeys[0]] ) # ex: bbc : ENS(0.61) without explicit overSampling / test_roc_auc : [0.6719306  0.58851217 0.58250362 0.6094371  0.55757417]
                      ])
        for i, s in enumerate(pl.steps) :
            # Ex: 0 -> ('preprocessor', ColumnTransformer( ... +  1 -> ('classifier', BalancedBaggingClassifier(base_.....
            print(f"{i} -> {s[0]} / {str(s[1])[:70]}")
        return pl