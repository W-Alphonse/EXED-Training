

# APP_RESOURCE_PATH = '../resources/'
APP_DEFAULT_CONFIG_FILE = 'valeo.yaml'
APP_DEFAULT_LOG_FILE    = 'logging.yaml'

ENV_KEY_CONFIG_FILE_PATHNAME = '__VALEO__APP_CONFIG_FILE_PATHNAME' # ex: SET __VALEO__APP_CONFIG_FILE_PATHNAME=...../valeo.yaml'
ENV_KEY_LOG_FILE_PATHNAME    = '__VALEO__APP_LOG_FILE_PATHNAME'    # ex: SET __VALEO__APP_LOG_FILE_PATHNAME=...../logging.yaml'

import os


def rootProject() -> str :
    # return  os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../..')  # this_folder = D-Training.git/trunk/___VALEO/src/valeo/infrastructure
    return  os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..')  # this_folder = D-Training.git/trunk/___VALEO/src/valeo/infrastructure

def rootSrc() -> str :
    return  os.path.join(rootProject(),  'src' )

def rootData() -> str :
    return  os.path.join(rootProject(),  'data' )

def rootImages() -> str :
    return  os.path.join(rootProject(),  'images' )

def rootResources() -> str :
    return  os.path.join(rootProject(), 'src', 'valeo', 'resources')