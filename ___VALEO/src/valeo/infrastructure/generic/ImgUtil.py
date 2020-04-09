from valeo.infrastructure import Const
from valeo.infrastructure.LogManager import LogManager
import os
import matplotlib.pyplot as plt
import pandas as pd

class ImgUtil() :
    logger = LogManager.logger(__name__)

    # https://stackabuse.com/pythons-classmethod-and-staticmethod-explained/
    @classmethod
    def save_fig(cls, fig_id:str , tight_layout=True, fig_extension="png", resolution=300):
        path = os.path.join(Const.rootImages() , fig_id + "." + fig_extension)
        cls.logger.debug(f"Saving figure '{fig_id}'")
        if tight_layout:
            plt.tight_layout()
        # Save "the current figure plot" that is set by "df.hist(...))". @ReferTo: pyplot.py / def gcf()
        plt.savefig(path, format=fig_extension, dpi=resolution)

    @classmethod
    def save_df_as_hist(cls, df:pd.DataFrame, fig_id:str , bins=50, figsize=(20,15), tight_layout=True, fig_extension="png", resolution=300):
        cls.logger.debug(f"bins:{bins} - figsize:{figsize}")
        df.hist(bins=bins, figsize=figsize)
        cls.save_fig(fig_id=f"{fig_id}_histogram_{figsize[0]}x{figsize[1]}", tight_layout=tight_layout, fig_extension=fig_extension, resolution=resolution)