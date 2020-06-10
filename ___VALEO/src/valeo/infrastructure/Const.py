# 1 - ENVIRONMENT keys used to refer to configuration files
ENV_KEY_CONFIG_FILE_PATHNAME = '__VALEO__APP_CONFIG_FILE_PATHNAME' # ex: SET __VALEO__APP_CONFIG_FILE_PATHNAME=...../valeo.yaml'
ENV_KEY_LOG_FILE_PATHNAME    = '__VALEO__APP_LOG_FILE_PATHNAME'    # ex: SET __VALEO__APP_LOG_FILE_PATHNAME=...../logging.yaml'
#
# 2 - Symbolic name of configuration files
APP_DEFAULT_CONFIG_FILE = 'valeo.yaml'
APP_DEFAULT_LOG_FILE    = 'logging.yaml'
#
# 3 - Valeo Dataset columns names
PROC_TRACEINFO         = 'PROC_TRACEINFO'
OP070_V_1_angle_value  = 'OP070_V_1_angle_value'
OP070_V_1_torque_value = 'OP070_V_1_torque_value'
OP070_V_2_angle_value  = 'OP070_V_2_angle_value'
OP070_V_2_torque_value = 'OP070_V_2_torque_value'
OP090_StartLinePeakForce_value  = 'OP090_StartLinePeakForce_value'
OP090_SnapRingMidPointForce_val = 'OP090_SnapRingMidPointForce_val'
OP090_SnapRingPeakForce_value   = 'OP090_SnapRingPeakForce_value'
OP090_SnapRingFinalStroke_value = 'OP090_SnapRingFinalStroke_value'
OP100_Capuchon_insertion_mesure = 'OP100_Capuchon_insertion_mesure'
OP110_Vissage_M8_angle_value  = 'OP110_Vissage_M8_angle_value'
OP110_Vissage_M8_torque_value = 'OP110_Vissage_M8_torque_value'
OP120_Rodage_I_mesure_value   = 'OP120_Rodage_I_mesure_value'
OP120_Rodage_U_mesure_value   = 'OP120_Rodage_U_mesure_value'
Binar_OP130_Resultat_Global_v = 'Binar OP130_Resultat_Global_v'
# F_Manuf_day_count = 'F_Manuf_day_count'
proc_month   = 'proc_month'
proc_week    = 'proc_week'
proc_weekday = 'proc_weekday'
#
# 4 - Algorithm Classifiers keys
BRFC         = "BRFC"
BBC_ADABoost = "BBC_ADABoost"
BBC_GBC      = "BBC_GBC"
BBC_HGBC     = "BBC_HGBC"
RFC_SMOTEEN  = "RFC_SMOTEEN"
RFC_SMOTETOMEK    = "RFC_SMOTETOMEK"
RFC_BLINESMT_RUS  = "RFC_BLINESMT_RUS"
RUSBoost_ADABoost = "RUSBoost_ADABoost"
LRC_SMOTEEN  = "LRC_SMOTEEN"
KNN_SMOTEEN  = "KNN_SMOTEEN"
SVC_SMOTEEN  = "SVC_SMOTEEN"
GNB_SMOTENN  = "GNB_SMOTENN"
#
HGBC         = "HGBC" # (RFC, BorderLineSmote, RandomUnderSample): pas retenu - Split_Test : ROC_AUC: 0.6168
# BBC  = "BBC"
# NuSVC = "NuSVC"       # NuSVC(probability=True),
# GBC  = "GBC"          # GradientBoostingClassifier()
# XGBC = "XGBC"         # xgb.XGBClassifier()
#
bg_rank = "best_generalized_rank"
bg_params = "best_generalized_params"
bg_score_diff = "best_generalized_score_difference_with_1st"
bg_score_test_set = "best_generalized_score_test_set"
bg_score_train_set  = "best_generalized_score_train_set"
#
grid = "grid"
rand = "random"
opt  = "opt"


import os
from datetime import datetime
# timestamp : none / suffix / prefix
ts_none = 0
ts_sfix = 1
ts_pfix = 2

def rootProject() -> str :
    return  os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..')  # this_folder = D:/Training.git/trunk/___VALEO/src/valeo/infrastructure

def rootSrc() -> str :
    return  os.path.join(rootProject(),  'src' )

def rootData() -> str :
    return  os.path.join(rootProject(),  'data' )

def rootDataTrain() -> str :
    return  os.path.join(rootData(),  'train' )

def rootDataTest() -> str :
    return  os.path.join(rootData(),  'test' )

def rootImages() -> str :
    return  os.path.join(rootProject(),  'images' )

def rootReports() -> str :
    return  os.path.join(rootProject(),  'reports' )

def rootResources() -> str :
    return  os.path.join(rootProject(), 'src', 'valeo', 'resources')

def ts_pathanme(pathAsStrList : [], ts_type=ts_sfix) -> str:
    if not isinstance(pathAsStrList,list) :
        pathAsStrList = [pathAsStrList]
    fname_with_ext = os.path.splitext(pathAsStrList[-1])
    return os.path.join(pathAsStrList[0], '' if len(pathAsStrList) <= 2 else str(*pathAsStrList[1:-1] ),
              f"{fname_with_ext[0]}{datetime.now().strftime('_%Y_%m_%d-%H.%M.%S')}{fname_with_ext[1]}" if ts_type == ts_sfix else \
             (f"{datetime.now().strftime('%Y_%m_%d-%H.%M.%S_')}{pathAsStrList[-1]}" if ts_type == ts_pfix  else pathAsStrList[-1]) )
