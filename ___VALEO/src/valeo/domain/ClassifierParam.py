from scipy.stats import randint, uniform
from skopt.space import Integer, Real

from valeo.infrastructure.LogManager import LogManager
from valeo.infrastructure import Const as C

class ClassifierParam :
    logger = LogManager.logger(__name__)

    def __init__(self):
        self.g_param = {}    # grid param used with GridSearchCV
        self.d_param = {}    # dist param used with RandomizedSearchParam
        self.o_param = {}    # space param used with BayesSearchCV and scikit-optimize
        self._initialize_param()

    def grid_param(self, clf_type:str) -> dict:
        return self.g_param[clf_type]

    def distrib_param(self, clf_type:str) -> dict:
        return self.d_param[clf_type]

    def optimize_param(self, clf_type:str) -> dict:
        return self.o_param[clf_type]

    # https://stackoverflow.com/questions/49036853/scipy-randint-vs-numpy-randint
    def _initialize_param(self):
        # ======================== Balanced RandomForest ========================
        self.g_param[C.BRFC] =  {
            'classifier__n_estimators': [200, 300],
            'classifier__max_depth': [8, 10],
            'classifier__max_features' : ['sqrt', 'log2'],
            'classifier__min_samples_split' : [12, 18],
            'classifier__oob_score': [True, False], # default:False => Whether to use out-of-bag samples to estimate the generalization accuracy
            'classifier__criterion' : ['entropy', 'gini'],
            'classifier__sampling_strategy' : [ 0.2, 0.3, 'auto']
         }
        self.d_param[C.BRFC]  =  {
            'classifier__n_estimators': randint(150,300),
            'classifier__max_depth': randint(8, 12),
            'classifier__max_features' : ['sqrt', 'log2'],
            'classifier__min_samples_split' : randint(12,25),
            'classifier__oob_score': [True, False], # default:False => Whether to use out-of-bag samples to estimate the generalization accuracy
            'classifier__criterion' : ['entropy', 'gini'],
            'classifier__sampling_strategy' : [ 0.2, 0.3, 'auto']
        }
        self.o_param[C.BRFC] =  {
            'classifier__n_estimators': Integer(100, 300),
            'classifier__max_depth': Integer(5, 10),
            'classifier__max_features' : ['sqrt', 'log2'],
            'classifier__min_samples_split' : Integer(12, 25),
            'classifier__oob_score': [True, False], # default:False => Whether to use out-of-bag samples to estimate the generalization accuracy
            'classifier__criterion' : ['entropy', 'gini'],
            'classifier__sampling_strategy' : Real(0.15, 0.35)
        }
        # ======================== LogisticRegression ========================
        self.g_param[C.LRC_SMOTEEN]  = {
            'classifier__penalty': ['l2'],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'classifier__fit_intercept' : [True, False],
            'classifier__solver' : ['lbfgs', 'saga'],
            'classifier__max_iter' : [1000]
        }
        self.d_param[C.LRC_SMOTEEN]  ={
                'classifier__penalty': ['l2'],
                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'classifier__fit_intercept' : [True, False],
                'classifier__solver' : ['lbfgs', 'saga'],
                'classifier__max_iter' : [1000]
        }
        self.o_param[C.LRC_SMOTEEN] =  {
            'classifier__penalty': ['l2'],
            'classifier__C': Real(0.0001, 1000),
            'classifier__fit_intercept' : [True, False],
            'classifier__solver' : ['lbfgs', 'saga'],
            'classifier__max_iter' : [1000]
         }
        # ======================== Balanced Bag Classifier + Gradient Boost ========================
        self.g_param[C.BBC_GBC] =  {
            'classifier__base_estimator__learning_rate': [0.05, 0.1],
            'classifier__base_estimator__n_estimators':  [35, 50],
            'classifier__base_estimator__max_depth' : [3,5],
            'classifier__base_estimator__max_features' : ['log2'],
            'classifier__base_estimator__min_samples_split' : [25, 30],
            'classifier__n_estimators' : [200, 250],
            'classifier__max_samples'  : [0.7, 0.8],
            'classifier__max_features' : [30,40],
            'classifier__oob_score' : [True]
        }
        self.d_param[C.BBC_GBC] =  {}
        self.o_param[C.BBC_GBC] =  {
            'classifier__base_estimator__learning_rate': Real(0.1, 10),
            'classifier__base_estimator__n_estimators': Integer(50,100),
            'classifier__base_estimator__max_depth' : Integer(5,10),
            'classifier__base_estimator__max_features' : ['sqrt', 'log2'],
            'classifier__base_estimator__min_samples_split' :Integer(15, 25),
            'classifier__n_estimators' : Integer(10,200),
            'classifier__max_samples'  : Real(0.5, 0.7),
            'classifier__max_features' : Integer(5,30),
            'classifier__oob_score' : [True, False]
        }


# https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
# https://stackoverflow.com/questions/53533942/loss-parameter-explanation-for-sklearn-ensemble-gradientboostingclassifier
#
# param_distributions: 'dict' or 'list of dicts'
#  1 - When it is 'dict' Then : Keys -> parameters names (str) / Values : lists of parameters to try OR Statistical Distributions to try
#     . Distributions must provide a rvs method for sampling (such as those from scipy.stats.distributions).
#     . If a list is given, it is sampled uniformly.
#  2 - When it is 'list of dicts' Then each 'dict' is sampled uniformly, and then each value in the dict is sampled as described above.
#
# param_distributions = {
#     'clf__n_estimators': randint(1, 100),
#     'clf__max_depth': randint(2, 15),
#     'clf__max_features': [1, 2, 3, 4, 5],
#     'clf__min_samples_split': [2, 3, 4, 5, 10, 30],
# }
