import os

import valeo.infrastructure.Const as const
from valeo.infrastructure.tools.ConfigLoader import ConfigLoader

class AppConfigManager():

    def __init__(self):
        cl = AppConfigLoader()
        self.app_config = cl.load()

    def getValue(self, keys_path:[]) -> str :
        return self.app_config[keys_path[0]] if len(keys_path) ==  1 else self.getValue(self.app_config[keys_path[0]] , keys_path[1:])


class AppConfigLoader(ConfigLoader) :
    def load(self) -> dict:
        return super().load(os.path.join(const.rootResources(), const.APP_DEFAULT_CONFIG_FILE), const.ENV_KEY_CONFIG_FILE_PATHNAME)