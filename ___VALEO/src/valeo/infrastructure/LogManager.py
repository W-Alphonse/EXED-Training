# https://docs.python.org/3/library/logging.html#logrecord-attributes + Useful Handlers
# https://docs.python-guide.org/writing/logging/
# https://github.com/Delgan/loguru
# https://kingspp.github.io/design/2017/11/06/the-head-and-tail-of-logging.html
# https://stackoverflow.com/questions/4690600/python-exception-message-capturing
import logging.config

from valeo.infrastructure.util.ConfigLoader import ConfigLoader
import valeo.infrastructure.Const as Const

class LogManager():

    def __init__(self):
        self.log_config = LogLoader().load()

    @classmethod
    def logger(cls,logname):
        return logging.getLogger(logname)


class LogLoader(ConfigLoader):
    """
    Load the logging configuration file
    """
    def load(self) -> dict:
        try :
            dict = super().load(f'{Const.APP_RESOURCE_PATH}{Const.APP_DEFAULT_LOG_FILE}', Const.ENV_KEY_LOG_FILE_PATHNAME)
            logging.config.dictConfig(dict)
            return dict
        except Exception as ex:
            logging.basicConfig(level=logging.INFO)
            logging.warning(f'Error while loading logging configuration file:\n' \
                            f'\t- APP_RESOURCE_PATH = {Const.APP_RESOURCE_PATH}\n' \
                            f'\t- APP_DEFAULT_LOG_FILE = {Const.APP_DEFAULT_LOG_FILE}\n' \
                            f'\t- ENV_KEY_LOG_FILE_PATHNAME = {Const.ENV_KEY_LOG_FILE_PATHNAME}')
            logging.exception(ex)
            return None
