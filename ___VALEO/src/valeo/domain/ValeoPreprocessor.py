from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, SVMSMOTE, KMeansSMOTE, BorderlineSMOTE, ADASYN
from imblearn.over_sampling.base import BaseOverSampler
# from sklearn.experimental import enable_iterative_imputer   # explicitly require this experimental feature
# from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer  # now you can import normally from sklearn.impute
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

from sklearn.compose import ColumnTransformer, make_column_transformer
from imblearn.pipeline import make_pipeline, Pipeline

import pandas as pd
import numpy as np

from valeo.infrastructure import Const as C
from valeo.infrastructure.LogManager import LogManager

from valeo.infrastructure.tools.DebugPipeline import DebugPipeline


class ValeoPreprocessor:
    logger = None

    def __init__(self):
        ValeoPreprocessor.logger = LogManager.logger(__name__)

    # def build_iterative_preprocessor(self)  -> ColumnTransformer:
    #     imp_cols = [C.OP100_Capuchon_insertion_mesure]
    #     it_imp = IterativeImputer(estimator=BayesianRidge(),  max_iter=10, initial_strategy = 'median', add_indicator=True, random_state=48)
    #     return ColumnTransformer([('iterative_imputer', it_imp, imp_cols)] )

    def build_column_preprocessor(self) -> Pipeline: # ColumnTransformer:
        rand_state = 48
        # 1 - IterativeImputer : models each feature with missing values as a function of other features, and uses that estimate for imputation
        # imputer_pipe = Pipeline( IterativeImputer(estimator=BayesianRidge(), missing_values=[np.nan, 0],  max_iter=10, initial_strategy = 'median') )

        # imputer_nan_values = IterativeImputer(estimator=BayesianRidge(), missing_values=np.nan,  max_iter=10, initial_strategy = 'median',  add_indicator=True, random_state=rand_state)
        # imputer_nan_values = IterativeImputer(estimator=BayesianRidge(), missing_values=np.nan,  max_iter=10, initial_strategy = 'median',  add_indicator=False, random_state=rand_state)
        imputed_nan_cols = [C.OP100_Capuchon_insertion_mesure]  # columns having too much missing values
        #
        # imputer_zeroes_values = IterativeImputer(estimator=BayesianRidge(), missing_values=0,  max_iter=10, initial_strategy = 'median',  add_indicator=False, random_state=rand_state)
        imputed_zeroes_cols = [C.OP100_Capuchon_insertion_mesure, C.OP090_StartLinePeakForce_value, C.OP090_SnapRingMidPointForce_val,  # columns equals to 0 for a few rows
                               C.OP090_SnapRingPeakForce_value, C.OP090_SnapRingFinalStroke_value]

        # 2 - Scale features using statistics that are robust to outliers.
        scaler      = MinMaxScaler() # StandardScaler() # RobustScaler(with_centering=True, with_scaling=False)
        scaled_cols = [C.OP070_V_1_angle_value, C.OP070_V_1_torque_value,
                       C.OP070_V_2_angle_value, C.OP070_V_2_torque_value,
                       C.OP090_StartLinePeakForce_value, C.OP090_SnapRingMidPointForce_val,
                       C.OP090_SnapRingPeakForce_value,  C.OP090_SnapRingFinalStroke_value,
                       C.OP100_Capuchon_insertion_mesure,
                       C.OP110_Vissage_M8_angle_value, C.OP110_Vissage_M8_torque_value,
                       C.OP120_Rodage_I_mesure_value,  C.OP120_Rodage_U_mesure_value]

        return  Pipeline([ ('dbg_0', DebugPipeline(), scaled_cols),
                                   ('imputer_preprocessor_nan', SimpleImputer(strategy='mean', missing_values=np.nan, verbose=True), imputed_nan_cols),
                                   ('dbg_1', DebugPipeline(), scaled_cols),
                                   ('imputer_preprocessor_zeroes', SimpleImputer(strategy='mean', missing_values=0.0, verbose=True), imputed_zeroes_cols),
                                   ('dbg_2', DebugPipeline(), scaled_cols),
                                   ('scaler_preprocessor', scaler, scaled_cols),
                                   ('dbg_3', DebugPipeline(), scaled_cols),
                                   # #
                                   # ('imputer_preprocessor_nan_bis', SimpleImputer(strategy='mean'), scaled_cols),
                                   # ('dbg_1_bis', DebugPline(),scaled_cols)
                                   ])
                         # , remainder='passthrough')


    def execute(self, X_train:pd.DataFrame):
        imputed_nan_cols = [C.OP100_Capuchon_insertion_mesure]
        imputed_zeroes_cols = [C.OP100_Capuchon_insertion_mesure, C.OP090_StartLinePeakForce_value, C.OP090_SnapRingMidPointForce_val,  # columns equals to 0 for a few rows
                               C.OP090_SnapRingPeakForce_value, C.OP090_SnapRingFinalStroke_value]
        scaled_cols = [C.OP070_V_1_angle_value, C.OP070_V_1_torque_value,
                       C.OP070_V_2_angle_value, C.OP070_V_2_torque_value,
                       C.OP090_StartLinePeakForce_value, C.OP090_SnapRingMidPointForce_val,
                       C.OP090_SnapRingPeakForce_value,  C.OP090_SnapRingFinalStroke_value,
                       C.OP100_Capuchon_insertion_mesure,
                       C.OP110_Vissage_M8_angle_value, C.OP110_Vissage_M8_torque_value,
                       C.OP120_Rodage_I_mesure_value,  C.OP120_Rodage_U_mesure_value]
        # DebugPipeline.counter += 10
        d = DebugPipeline()
        #
        d.fit_transform(X_train[scaled_cols])
        s = SimpleImputer(strategy='mean', missing_values=np.nan, verbose=True)
        X_train[imputed_nan_cols] = s.fit_transform(X_train[imputed_nan_cols])
        d.fit_transform(X_train[scaled_cols])
        #
        s = SimpleImputer(strategy='mean', missing_values=0.0, verbose=True)
        X_train[imputed_zeroes_cols] = s.fit_transform(X_train[imputed_zeroes_cols])
        d.fit_transform(X_train[scaled_cols])
        #
        scaler      = MinMaxScaler() # StandardScaler() # RobustScaler(with_centering=True, with_scaling=False)
        X_train[scaled_cols]  = scaler.fit_transform(X_train[scaled_cols])
        d.fit_transform(X_train[scaled_cols])



    '''
    def build_column_preprocessor(self) -> ColumnTransformer:
        rand_state = 48
        # 1 - IterativeImputer : models each feature with missing values as a function of other features, and uses that estimate for imputation
        imputer_pipe = make_pipeline( IterativeImputer(estimator=BayesianRidge(), missing_values=np.nan,  max_iter=10, initial_strategy = 'median',  add_indicator=True, random_state=rand_state) )
        # imputer_pipe = Pipeline( IterativeImputer(estimator=BayesianRidge(), missing_values=[np.nan, 0],  max_iter=10, initial_strategy = 'median') )
        imputed_cols = [C.OP100_Capuchon_insertion_mesure]  # columns having too much missing values
                        # C.OP090_StartLinePeakForce_value, C.OP090_SnapRingMidPointForce_val,  # columns equals to 0 for a few rows
                        # C.OP090_SnapRingPeakForce_value, C.OP090_SnapRingFinalStroke_value]

        # 2 - Scale features using statistics that are robust to outliers.
        scaler_pipe = make_pipeline(RobustScaler(with_centering=True, with_scaling=False))
        scaled_cols = [C.OP070_V_1_angle_value, C.OP070_V_1_torque_value,
                       C.OP070_V_2_angle_value, C.OP070_V_2_torque_value,
                       C.OP090_StartLinePeakForce_value, C.OP090_SnapRingMidPointForce_val,
                       C.OP090_SnapRingPeakForce_value,  C.OP090_SnapRingFinalStroke_value,
                       C.OP100_Capuchon_insertion_mesure,
                       C.OP110_Vissage_M8_angle_value, C.OP110_Vissage_M8_torque_value,
                       C.OP120_Rodage_I_mesure_value,  C.OP120_Rodage_U_mesure_value,]

        return  ColumnTransformer([('imputer_preprocessor', imputer_pipe, imputed_cols),
                                   ('scaler_preprocessor', scaler_pipe, scaled_cols)] )
    '''

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
            return SVMSMOTE(sampling_strategy=sampling_strategy, random_state=rand_state, k_neighbors=k_neighbors)
        elif sampler_type.lower() == C.smote_kmeans_over_sampling : # Apply a KMeans clustering before to over-sample using SMOTE
            return KMeansSMOTE(sampling_strategy=sampling_strategy, random_state=rand_state, k_neighbors=k_neighbors)
        elif sampler_type.lower() == C.smote_bline_over_sampling :  # Borderline samples will be detected and used to generate new synthetic samples.
            return BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=rand_state, k_neighbors=k_neighbors)
        else :
            raise ValueError(f"Unexpected argument [sampler_type:{sampler_type}] for method 'compute_sampler_preprocessor'")



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