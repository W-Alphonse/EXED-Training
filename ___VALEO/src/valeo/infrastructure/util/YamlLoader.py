
import os
import yaml
import logging

# from infrastructure.LogManager import LogManager

class YamlLoader :
    """
    Load a yaml  configuration file
    """
    logger = None

    def __init__(self):
        if YamlLoader.logger is None :
            YamlLoader.logger =  logging.getLogger(__name__)
            logging.basicConfig(level=logging.INFO)

    def load(self, file_pathname:str) -> dict :
        if os.path.exists(file_pathname):
            with open(file_pathname, 'rt') as f:
                try:
                    dict =  yaml.safe_load(f.read())
                    YamlLoader.logger.info(f'Loading file "{file_pathname}":\n\t{dict}')
                    return dict
                except Exception as ex:
                    YamlLoader.logger.exception(f'Error while loading file "{file_pathname}"')
        else:
            YamlLoader.logger.error(f'Error while loading file "{file_pathname}"')

        return None
