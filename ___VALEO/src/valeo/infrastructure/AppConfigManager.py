import valeo.infrastructure.Const as const
from valeo.infrastructure.util.ConfigLoader import ConfigLoader

class AppConfigManager():

    def __init__(self):
        cl = AppConfigLoader()
        self.app_config = cl.load()

    def getValue(self, nested_dict:{}, keys:[]) -> str :
        return nested_dict[keys[0]] if len(keys) ==  1 else self.getValue(nested_dict[keys[0]] , keys[1:])



class AppConfigLoader(ConfigLoader) :

    def load(self) -> dict:
        return super().load(f'{const.APP_RESOURCE_PATH}{const.APP_DEFAULT_CONFIG_FILE}', const.ENV_KEY_CONFIG_FILE_PATHNAME)