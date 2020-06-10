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
    #  {'classifier__max_depth': 10, 'classifier__min_samples_split': 12, 'classifier__n_estimators': 200, 'classifier__oob_score': True, 'classifier__sampling_strategy': 'auto'}
    def _initialize_param(self):

        self.o_param[C.BRFC] =  { #Search_02
            'classifier__n_estimators': Integer(100, 300),
            'classifier__max_depth': Integer(5, 20),
            'classifier__max_features' : ['sqrt', 'log2'],
            'classifier__min_samples_split' : Integer(5, 20),
            # 'classifier__min_samples_leaf' : [9,13, 15],
            'classifier__oob_score': [True, False], # default:False -> Whether to use out-of-bag samples to estimate the generalization accuracy
            # 'classifier__class_weight' : [None],
            'classifier__criterion' : ['entropy', 'gini'], # default: gini
            'classifier__sampling_strategy' : Real(0.15, 0.25)  # 0.1 better than 'auto' Cependant l'overfitting est plus petit avec 'auto'. NB: # 0.1, 0.15 ou 0.2 sont tjrs execau
         }

        # BalancedRandomForestClassifier(n_estimators = 300 , max_depth=20, random_state=0) , # sampling_strategy=0.5),
        self.g_param[C.BRFC] =  { #Search_02
            'classifier__n_estimators': [ 200],
            'classifier__max_depth': [10],
            'classifier__min_samples_split' : [12, 18],
            # 'classifier__min_samples_leaf' : [9,13, 15],
            'classifier__oob_score': [True, False], # default:False -> Whether to use out-of-bag samples to estimate the generalization accuracy
            # 'classifier__class_weight' : [None],
            'classifier__sampling_strategy' : ['auto']  # 0.1 better than 'auto' Cependant l'overfitting est plus petit avec 'auto'. NB: # 0.1, 0.15 ou 0.2 sont tjrs execau
        }

        # self.g_param[C.BRFC] =  { #Search_02
        #     'classifier__n_estimators': [100, 200, 300],
        #     'classifier__max_depth': [10, 15, 20],
        #     'classifier__max_features' : ['sqrt', 'log2'],
        #     'classifier__min_samples_split' : [8, 12, 18],
        #     # 'classifier__min_samples_leaf' : [9,13, 15],
        #     'classifier__oob_score': [True, False], # default:False -> Whether to use out-of-bag samples to estimate the generalization accuracy
        #     # 'classifier__class_weight' : [None],
        #     'classifier__criterion' : ['entropy', 'gini'], # default: gini
        #     'classifier__sampling_strategy' : [ 0.15, 0.2, 0.25, 'auto']  # 0.1 better than 'auto' Cependant l'overfitting est plus petit avec 'auto'. NB: # 0.1, 0.15 ou 0.2 sont tjrs execau
        #  }
        # self.d_param[C.BRFC]  =  { #Search_01
        #     'classifier__n_estimators': randint(50,500),
        #     'classifier__max_depth': randint(6, 20),
        #     'classifier__max_features' : ['sqrt', 'log2', None, 12, 15],
        #     'classifier__min_samples_split' : randint(5,18),
        #     # 'classifier__min_samples_leaf' : [9,13, 15],
        #     'classifier__oob_score': [True, False], # default:False -> Whether to use out-of-bag samples to estimate the generalization accuracy
        #     # 'classifier__class_weight' : [None],
        #     'classifier__criterion' : ['entropy', 'gini'],
        #     'classifier__sampling_strategy' : [ 0.15, 0.2, 0.25, 0.3, 'auto']
        # }

        # self.d_param[C.BRFC]  = { 'classifier__n_estimators': [260],
        #                           'classifier__max_depth': [15],
        #                           'classifier__min_samples_leaf' : [5]
        #  }
        # ========================   LogisticRegression(max_iter=500)
        self.g_param[C.LRC_SMOTEEN]  = { #Search_02
            'classifier__penalty': ['l2'],
            'classifier__C': [1, 10, 100, 1000],
            'classifier__fit_intercept' : [True, False],
            'classifier__solver' : ['newton-cg', 'lbfgs', 'sag', 'saga'],
            'classifier__max_iter' : [500]
            # 'classifier__l1_ratio' : uniform(0, 1)
        }
        self.d_param[C.LRC_SMOTEEN]  = [ { #Search_01
                'classifier__penalty': ['l2', 'elasticnet'],
                'classifier__C': [1, 10, 100, 1000],
                'classifier__fit_intercept' : [True, False],
                'classifier__solver' : ['saga'],
                'classifier__max_iter' : randint(300, 500),
                'classifier__l1_ratio' : uniform(0, 1)
            } ,
            {
                'classifier__penalty': ['l2'],
                'classifier__C': [1, 10, 100, 1000],
                'classifier__fit_intercept' : [True, False],
                'classifier__solver' : ['newton-cg', 'lbfgs', 'sag', 'saga'],
                'classifier__max_iter' : randint(100, 500)
            }
        ]
        # =======================  HGBC : HistGradientBoostingClassifier(max_iter = 100 , max_depth=10,learning_rate=0.10, l2_regularization=5),
        # self.g_param[C.HGBC]  = {
        # }
        # self.d_param[C.HGBC]  = {
        # }
# Build a machine-learning pipeline using a HistGradientBoostingClassifier and fine tune your model on the Titanic dataset using a RandomizedSearchCV.
#
# You may want to set the parameter distributions is the following manner:
#
# learning_rate with values ranging from 0.001 to 0.5 following a reciprocal distribution.
# l2_regularization with values ranging from 0.0 to 0.5 following a uniform distribution.
# max_leaf_nodes with integer values ranging from 5 to 30 following a uniform distribution.
# min_samples_leaf with integer values ranging from 5 to 30 following a uniform distribution.
        # =======================  GBC : GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=.2)
        # self.g_param[C.GBC]  = {  #Search_02 - Not Retaine
        #     # https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
        #     'classifier__loss' : ['deviance', 'exponential'],  #https://stackoverflow.com/questions/53533942/loss-parameter-explanation-for-sklearn-ensemble-gradientboostingclassifier
        #     'classifier__learning_rate'   : [0.05, 0.1, 0.2], #uniform(0.001, 0.5),
        #     'classifier__n_estimators' : [500, 300],
        #     # 'classifier__l2_regularization' :  random.uniform(0.0, 0.5),
        #     'classifier__subsample' : [0.7, 0.8],
        #     'classifier__min_samples_split' :  [8, 12, 18],
        #      'classifier__max_depth'   : [10, 15, 20],
        #     'classifier__max_features' : ['sqrt', 'log2'],
        # }

        # self.g_param[C.GBC]  = {  #Search_03 - Retained
        #     # https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
        #     'classifier__loss' : ['deviance', 'exponential'],  #https://stackoverflow.com/questions/53533942/loss-parameter-explanation-for-sklearn-ensemble-gradientboostingclassifier
        #     'classifier__learning_rate'   : [0.05, 0.1, 0.2],
        #     'classifier__n_estimators' : [100, 200,300],
        #     'classifier__min_samples_split' : [12, 18],
        #     'classifier__subsample' : [0.6, 0.7, 0.8, 0.9],
        #     'classifier__min_samples_split' :  [12, 18],
        #     'classifier__max_depth'   : [10, 15],
        #     'classifier__max_features' : ['log2'],
        # }
        # self.d_param[C.GBC]  = { # didn't try
        #     'classifier__learning_rate'   : uniform(0.001, 0.2),
        #     'classifier__n_estimators' : [60, 80],
        #     # 'classifier__l2_regularization' :  random.uniform(0.0, 0.5),
        #     'classifier__subsample' : [0.6,0.7,0.75,0.8,0.85,0.9],
        #     'classifier__min_samples_split' : range(5,18,2),
        #     'classifier__max_depth'   : range(5,16,2),
        #     'classifier__max_features' : ['log2'],
        # }

        # self.d_param[C.GBC]  = {  #Search_01 - Not Retained
        #     'classifier__loss' : ['deviance', 'exponential'],
        #     'classifier__learning_rate'   : uniform(0.001, 0.5),
        #     'classifier__n_estimators' : randint(100, 1000),
        #     # 'classifier__l2_regularization' :  random.uniform(0.0, 0.5),
        #     'classifier__subsample' : uniform(0, 1),
        #     'classifier__min_samples_split' : randint(5, 15),
        #     'classifier__max_depth'   : randint(6, 20),
        #     'classifier__max_features' : ['sqrt', 'log2', None],
        # }


# param_distributions: 'dict' or 'list of dicts'
#  1 - When it is 'dict' Then : Keys -> parameters names (str) / Values : lists of parameters to try OR Statistical Distributions to try
#     . Distributions must provide a rvs method for sampling (such as those from scipy.stats.distributions).
#     . If a list is given, it is sampled uniformly.
#  2 - When it is 'list of dicts' Then each 'dict' is sampled uniformly, and then each value in the dict is sampled as described above.

# param_distributions = {
#     'clf__n_estimators': randint(1, 100),
#     'clf__max_depth': randint(2, 15),
#     'clf__max_features': [1, 2, 3, 4, 5],
#     'clf__min_samples_split': [2, 3, 4, 5, 10, 30],
# }
