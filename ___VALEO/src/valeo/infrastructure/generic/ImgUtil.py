from valeo.infrastructure import Const
from valeo.infrastructure.LogManager import LogManager
import os
import matplotlib.pyplot as plt

class ImgUtil() :
    # logger = None

    def __init__(self):
        self.logger = LogManager.logger(__name__)


    # https://stackabuse.com/pythons-classmethod-and-staticmethod-explained/
    # @classmethod
    # def save_fig(self, fig_id:str):
    def save_fig(self, fig_id:str , tight_layout=True, fig_extension="png", resolution=300):
        # tight_layout=True
        # fig_extension="png"
        # resolution=300
        #
        path = os.path.join(Const.rootImages() , fig_id + "." + fig_extension)
        self.logger.debug(f"Saving figure '{fig_id}'")
        if tight_layout:
            plt.tight_layout()
        # Save the current figure. pyplot.py / def gcf()
        plt.savefig(path, format=fig_extension, dpi=resolution)

    # @classmethod
    # def sssave_fig(cls, fig_id : str) :
    #     # path = os.path.join(Const.rootImages() , "xxx")
    #     path = "C:/EXED/Training/___VALEO/src/valeo/infrastructure/../../../images/attribute_histogram_plots"
    #     plt.tight_layout()
    #     plt.savefig(path, format="png", dpi=300)