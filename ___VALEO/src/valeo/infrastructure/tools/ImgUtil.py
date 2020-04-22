from valeo.infrastructure import Const as C
from valeo.infrastructure.LogManager import LogManager
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ImgUtil() :
    logger = LogManager.logger(__name__)

    # https://stackabuse.com/pythons-classmethod-and-staticmethod-explained/
    @classmethod
    def save_fig(cls, fig_id:str , tight_layout=True, fig_extension="png", resolution=300, ts_type=C.ts_sfix):
        path = C.ts_pathanme([C.rootImages() , fig_id + "." + fig_extension], ts_type=ts_type)  # os.path.join(C.rootImages() , fig_id + "." + fig_extension)
        cls.logger.debug(f"Saving figure '{fig_id}'")
        if tight_layout:
            plt.tight_layout()
        # Save "the current figure plot" that is set by "df.hist(...))". @ReferTo: pyplot.py / def gcf()
        plt.savefig(path, format=fig_extension, dpi=resolution)

    @classmethod
    def save_df_hist_plot(cls, df:pd.DataFrame, fig_id:str , bins=50, figsize=(20,15), tight_layout=True, fig_extension="png", resolution=300, ts_type=C.ts_sfix):
        cls.logger.debug(f"Generating 'hist' plot: bins={bins} - figsize={figsize}")
        df.hist(bins=bins, figsize=figsize)
        cls.save_fig(fig_id=f"{fig_id}_histogram_{figsize[0]}x{figsize[1]}", tight_layout=tight_layout, fig_extension=fig_extension, resolution=resolution, ts_type=ts_type)

    @classmethod
    def save_df_scatter_matrix_plot(cls, df:pd.DataFrame, fig_id:str , figsize=(20,15), tight_layout=True, fig_extension="png", resolution=300, ts_type=C.ts_sfix):
        cls.logger.debug(f"Generating 'scatter matrix' plot: figsize:{figsize}")
        pd.plotting.scatter_matrix(df, figsize=figsize)
        cls.save_fig(fig_id=f"{fig_id}_scatter_matrix_{figsize[0]}x{figsize[1]}", tight_layout=tight_layout, fig_extension=fig_extension, resolution=resolution, ts_type=ts_type)

    @classmethod
    def save_df_heatmap_plot(cls, df:pd.DataFrame, fig_id:str , figsize=(20,20), cmap='RdBu', annot=True , annot_kws={'size':15}, fig_extension="png", resolution=300, ts_type=C.ts_sfix):
        cls.logger.debug(f"Generating 'heatmap' plot: figsize:{figsize}")
        fig, ax = plt.subplots(figsize=figsize)
        sns.set(font_scale=1.1)
        sns.heatmap(df, cmap=cmap, annot=annot , annot_kws=annot_kws, ax=ax)
        ax.set_title(fig_id, fontsize=28)
        cls.save_fig(fig_id=f"{fig_id.replace(' ','_')}_heatmap_{figsize[0]}x{figsize[1]}", tight_layout=True, fig_extension=fig_extension, resolution=resolution, ts_type=ts_type)

    @classmethod
    def save_df_violin_plot(cls, df:pd.DataFrame, fig_id:str, grid_elmt_x:int, figsize=(20,20), fig_extension="png", resolution=300, ts_type=C.ts_sfix):
        cls.logger.debug(f"Generating 'violin' plot: figsize:{figsize}")
        grid_elmt_y = len(df.columns) // grid_elmt_x if (len(df.columns) % grid_elmt_x) == 0 else (len(df.columns) // grid_elmt_x) + 1
        #
        fig, axs = plt.subplots(grid_elmt_y, grid_elmt_x, figsize=figsize)
        for i, col in enumerate(sorted(df.columns)) :
            sns.violinplot(x=df[col], linewidth=1, ax=axs[i//grid_elmt_x, i%grid_elmt_x])
            # sns.stripplot( x=X_data_transformed[col], color="orange", jitter=0.2, linewidth=1, ax=axs[i//3,i%3])
            sns.boxplot( x=df[col], linewidth=1, ax=axs[i//grid_elmt_x, i%grid_elmt_x], saturation=0 )
        # axs.set_title(fig_id, fontsize=28)
        cls.save_fig(fig_id=f"{fig_id.replace(' ','_')}_violin_{figsize[0]}x{figsize[1]}", tight_layout=True, fig_extension=fig_extension, resolution=resolution, ts_type=ts_type)




# NB: Generic way to plot whathever :
# fig, ax = plt.subplots(figsize=(20,20))
# sns.heatmap(corr_matrix, cmap='RdBu', annot=True , annot_kws={'size':15}, ax=ax)
# ax.set_title("Valeo starter production correlation measures", fontsize=14)
# plt.show()

