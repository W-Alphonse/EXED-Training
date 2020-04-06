
import os
import logging

from valeo.infrastructure.util.YamlLoader import YamlLoader

class ConfigLoader(YamlLoader) :
    logger = None
    """
    Load an external or a package embedded configuration file.
    Check first if the environment variable {APP_CONFIG_PATHNAME}
    """

    def __init__(self):
        super().__init__()
        ConfigLoader.logger = logging.getLogger(__name__)

    def load(self, file_pathname:str, env_key_as_config_pathname:str) -> dict :
        try :
            path_as_key = os.getenv(env_key_as_config_pathname, None)
            return super().load(path_as_key if path_as_key else file_pathname )
        except Exception as ex :
            ConfigLoader.logger.exception(f'Error while loading file "{file_pathname}"')
            raise ex
            # self.logger.error(ex, exc_info=True)
        return None
