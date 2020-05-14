from scipy.stats import randint

from valeo.infrastructure.LogManager import LogManager
from valeo.infrastructure import Const as C

class ClassifierParam :
    logger = LogManager.logger(__name__)

    def __init__(self):
        self._initialize_param()
        # self.g_param = {}    # grid param used with GridSearchCV
        # self.d_param = {} # dist param used with RandomizedSearchParam

    def grid_param(self, clf_type:str) -> dict:
        return self.g_param[clf_type]

    def distrib_param(self, clf_type:str) -> dict:
        return self.d_param[clf_type]
    
    # https://stackoverflow.com/questions/49036853/scipy-randint-vs-numpy-randint
    def _initialize_param(self):
        self.g_param = {
            # BalancedRandomForestClassifier(n_estimators = 300 , max_depth=20, random_state=0) , # sampling_strategy=0.5),
            C.BRFC : { 'classifier__n_estimators': [60, 100, 200, 250, 300],
                       'classifier__max_depth': [6, 8, 15, 20],
                       'classifier__max_features' : ['sqrt', 'log2', None],
                       'classifier__min_samples_split' : [8, 12, 18],
                       # 'classifier__min_samples_leaf' : [9,13, 15],
                       'classifier__oob_score': [True, False], # default:False -> Whether to use out-of-bag samples to estimate the generalization accuracy
                       # 'classifier__class_weight' : [None],
                       'classifier__criterion' : ['entropy', 'gini'], # default: gini
                       'classifier__sampling_strategy' : [ 0.15, 0.2, 0.25, 0.3, 'auto']  # 0.1 better than 'auto' Cependant l'overfitting est plus petit avec 'auto'. NB: # 0.1, 0.15 ou 0.2 sont tjrs execau
                      }
        }
        self.d_param = {
            C.BRFC : { 'classifier__n_estimators': randint(50,500),
                       'classifier__max_depth': randint(6, 20),
                       'classifier__max_features' : ['sqrt', 'log2', None, 12, 15],
                       'classifier__min_samples_split' : randint(5,30),
                       # 'classifier__min_samples_leaf' : [9,13, 15],
                       'classifier__oob_score': [True, False], # default:False -> Whether to use out-of-bag samples to estimate the generalization accuracy
                       # 'classifier__class_weight' : [None],
                       'classifier__criterion' : ['entropy', 'gini'],
                       'classifier__sampling_strategy' : [ 0.15, 0.2, 0.25, 0.3, 'auto']
                       }
        }

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
