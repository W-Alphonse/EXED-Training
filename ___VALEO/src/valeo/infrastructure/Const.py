# ENVIRONMENT keys used to refer to configuration files
ENV_KEY_CONFIG_FILE_PATHNAME = '__VALEO__APP_CONFIG_FILE_PATHNAME' # ex: SET __VALEO__APP_CONFIG_FILE_PATHNAME=...../valeo.yaml'
ENV_KEY_LOG_FILE_PATHNAME    = '__VALEO__APP_LOG_FILE_PATHNAME'    # ex: SET __VALEO__APP_LOG_FILE_PATHNAME=...../logging.yaml'
#
# Symbolic name of configuration files
APP_DEFAULT_CONFIG_FILE = 'valeo.yaml'
APP_DEFAULT_LOG_FILE    = 'logging.yaml'
#
# Valeo Dataset columns names
PROC_TRACEINFO         = 'PROC TRACEINFO'
OP070_V_1_angle_value  = 'OP070_V_1_angle_value'
OP070_V_1_torque_value = 'OP070_V_1_torque_value'
OP070_V_2_angle_value  = 'OP070_V_2_angle_value'
OP070_V_2_torque_value = 'OP070_V_2_torque_value'
OP090_StartLinePeakForce_value = 'OP090_StartLinePeakForce_value'
OP090_SnapRingMidPointForce_value = 'OP090_SnapRingMidPointForce_value'
OP090_SnapRingPeakForce_value     = 'OP090_SnapRingPeakForce_value'
OP090_SnapRingFinalStroke_value   = 'OP090_SnapRingFinalStroke_value'
OP100_Capuchon_insertion_mesure   = 'OP100_Capuchon_insertion_mesure'
OP110_Vissage_M8_angle_value  = 'OP110_Vissage_M8_angle_value'
OP110_Vissage_M8_torque_value = 'OP110_Vissage_M8_torque_value'
OP120_Rodage_I_mesure_value   = 'OP120_Rodage_I_mesure_value'
OP120_Rodage_U_mesure_value   = 'OP120_Rodage_U_mesure_value'
Binar_OP130_Resultat_Global_v = 'Binar OP130_Resultat_Global_v '
#
# Imbalanced resampling type

random_over_sampling = 'random_over_sampling' # The most naive strategy is to generate new samples by randomly sampling with replacement the current available samples.
adasyn_over_sampling = 'adasyn_over_sampling' # Adaptive Synthetic: focuses on generating samples next to the original samples which are wrongly classified using a k-Nearest Neighbors classifier
smote_over_sampling  = 'smote_over_sampling'  # Synth Minority Oversampling Techn: will not make any distinction between easy and hard samples to be classified using the nearest neighbors rule
smote_nc_over_sampling   = 'smote_nc_over_sampling'
smote_svm_over_sampling  = 'smote_svm_over_sampling'
smote_kmeans_over_sampling  = 'smote_kmeans_over_sampling'
smote_bline_over_sampling   = 'smote_bline_over_sampling'


import os

def rootProject() -> str :
    return  os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..')  # this_folder = D-Training.git/trunk/___VALEO/src/valeo/infrastructure

def rootSrc() -> str :
    return  os.path.join(rootProject(),  'src' )

def rootData() -> str :
    return  os.path.join(rootProject(),  'data' )

def rootImages() -> str :
    return  os.path.join(rootProject(),  'images' )

def rootResources() -> str :
    return  os.path.join(rootProject(), 'src', 'valeo', 'resources')