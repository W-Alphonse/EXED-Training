from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import f1_score, auc, roc_auc_score
from sklearn.tree import DecisionTreeClassifier


import pandas as pd
from sklearn.model_selection import train_test_split

from valeo.domain.ValeoPreprocessor import ValeoPreprocessor
from valeo.infrastructure import Const as C
from valeo.infrastructure.LogManager import LogManager


class ValeoPipeline:
    logger = None

    def __init__(self):
        self.preproc = ValeoPreprocessor()
        ValeoPipeline.logger = LogManager.logger(__name__)

    def pplSmote(self):
        Pipeline([('column_preprocessor', self.preproc.build_column_preprocessor()) ,
                  ('smote_resampler', self.preproc.build_resampler(C.smote_over_sampling))])

    # https://towardsdatascience.com/custom-transformers-and-ml-data-pipelines-with-python-20ea2a7adb65
    def execute(self, X_df:pd.DataFrame, y_df:pd.DataFrame, sampler_type: str):
        # setting up testing and training sets
        X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.25, random_state=48)

        #Create an object of the classifier.
        bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                        sampling_strategy='auto',
                                        replacement=False,
                                        random_state=0)

        p = Pipeline([('column_preprocessor', self.preproc.build_column_preprocessor()) ,
                      ('smote_resampler', self.preproc.build_resampler(sampler_type)),
                      ('classifier', bbc)
                      ])
        # p.fit_transform(X_train, y_train)
        p.fit(X_train, y_train)
        #
        y_predict = p.predict(X_test)
        x = f1_score(y_test, y_predict)
        y = 0 # auc(y_test, y_predict)
        z = 0 # roc_auc_score(y_test, y_predict)
        ValeoPipeline.logger.info(f"F1:{x} - auc:{y} - roc_auc:{z}")


# ---------------------------------
# Exemple Type : PipeLine entier
# ---------------------------------
# >>> pca = PCA()
# >>> smt = SMOTE(random_state=42)
# >>> knn = KNN()
# >>> pipeline = Pipeline([('smt', smt), ('pca', pca), ('knn', knn)])



# -----------------------
# Exemple_1
# -----------------------
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
#
# # 1 - Define Categorical pipe_line
# cat_col = ['sex', 'embarked', 'pclass']
# cat_pipeline = Pipeline(steps=[
#     ("constant-imputer", SimpleImputer(strategy='constant', fill_value='missing')),
#     ("ordinal-encoder", OrdinalEncoder()),
# ])
#
# # 2 - Define Numerical pipe_line
# num_cols = ['age', 'parch', 'fare']
# num_pipeline = SimpleImputer(
#     strategy="mean", add_indicator=True,
# )
#
# # 3 - Define Column Transformer
# preprocessor = ColumnTransformer(transformers=[
#     ("cat-preprocessor", cat_pipeline, cat_col),
#     ("num-preprocessor", num_pipeline, num_cols),
# ])
#
# model = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("clf", RandomForestClassifier(n_estimators=100))
# ])
#
# _ = model.fit(X_train, y_train)
#
# (model.named_steps["preprocessor"]
#  .named_transformers_["cat-preprocessor"]
#  .named_steps["ordinal-encoder"].categories_)



# -----------------------
# Exemple_2
# -----------------------
# define the pipelines
# cat_pipe = make_pipeline(
#     SimpleImputer(strategy='constant', fill_value='missing'),
#     OrdinalEncoder(categories=categories)
# )
# num_pipe = SimpleImputer(strategy='mean')
#
# preprocessing = ColumnTransformer(
#     [('cat_preprocessor', cat_pipe, cat_col),
#      ('num_preprocessor', num_pipe, num_cols)]
# )