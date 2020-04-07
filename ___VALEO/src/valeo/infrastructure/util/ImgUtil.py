from valeo.infrastructure import Const
from valeo.infrastructure.LogManager import LogManager
import os
import matplotlib.pyplot as plt

class ImgUtil() :
    logger = None

    def __init__(self):
        ImgUtil.logger = LogManager.logger(__name__)

    # https://stackabuse.com/pythons-classmethod-and-staticmethod-explained/
    @classmethod
    def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
        path = os.path.join(Const.rootImages() , fig_id + "." + fig_extension)
        ImgUtil.logger.debug(f"Saving figure '{fig_id}'")
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)