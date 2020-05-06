from valeo.infrastructure import Const as C
from valeo.infrastructure.LogManager import LogManager
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from valeo.infrastructure.tools.DfUtil import DfUtil


class ImgUtil() :
    logger = LogManager.logger(__name__)

    # https://stackabuse.com/pythons-classmethod-and-staticmethod-explained/
    @classmethod
    def save_fig(cls, fig_id:str , tight_layout=True, fig_extension="png", resolution=300, ts_type=C.ts_sfix):
        path = C.ts_pathanme([C.rootImages() , fig_id + "." + fig_extension], ts_type=ts_type)
        # cls.logger.debug(f"Saving figure '{fig_id}'")
        if tight_layout:
            plt.tight_layout()
        # Save "the current figure plot" that is set by "df.hist(...))". @ReferTo: pyplot.py / def gcf()
        plt.savefig(path, format=fig_extension, dpi=resolution)

    @classmethod
    def save_df_bar_plot(cls, df:pd.DataFrame, fig_id:str, ncols:int, figsize=(20,15), tight_layout=True,
                          fig_extension="png", resolution=300, ts_type=C.ts_sfix):
        cls.logger.debug(f"Generating 'bar' plot: figsize={figsize}")
        cat_cols = DfUtil.categorical_cols(df)
        nrows = len(cat_cols) // ncols if (len(cat_cols) % ncols) == 0 else (len(cat_cols) // ncols) + 1
        #
        # --Case where each 'plot' is on a different 'figure'--
        # fig, axs = plt.subplots(figsize=figsize)
        # for i, col in enumerate(sorted(cat_cols)) :
        #     df_bar = df.groupby([col]).size()
        #     df_bar.plot(kind='bar', figsize=figsize)
        #     cls.save_fig(fig_id=f"{fig_id}_{col}_bar_{figsize[0]}x{figsize[1]}", tight_layout=tight_layout, fig_extension=fig_extension, resolution=resolution, ts_type=ts_type)
        #
        # --Case where all the 'plots' are on the same 'figure'--
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

        # df_bar.plot.bar(x=col, y = df.groupby([col]).size())
        for i, col in enumerate(sorted(cat_cols)) :
            df_bar = df.groupby([col]).size()
            df.plot.bar(x=df[col], y = df.groupby([col]).size()) #  ax=axs[i//ncols, i%ncols])
            # df.groupby(DfUtil.categorical_cols(df)).size().plot.bar( figsize=figsize)
            # df.groupby(DfUtil.categorical_cols(df)).size().plot.bar( figsize=figsize)
        cls.save_fig(fig_id=f"{fig_id}_bar_{figsize[0]}x{figsize[1]}", tight_layout=tight_layout, fig_extension=fig_extension, resolution=resolution, ts_type=ts_type)


    @classmethod
    def save_df_hist_plot(cls, df:pd.DataFrame, fig_id:str , bins=50, figsize=(20,15), tight_layout=True,
                          fig_extension="png", resolution=300, ts_type=C.ts_sfix):
        cls.logger.debug(f"Generating 'hist' plot: bins={bins} - figsize={figsize}")
        df[DfUtil.numerical_cols(df)].hist(bins=bins, figsize=figsize)
        cls.save_fig(fig_id=f"{fig_id}_histogram_{figsize[0]}x{figsize[1]}", tight_layout=tight_layout, fig_extension=fig_extension, resolution=resolution, ts_type=ts_type)

    @classmethod
    def save_df_XY_hist_plot(cls, df_XY:pd.DataFrame, fig_id:str, bins=50, figsize=(5, 5), y_target_name=None, tight_layout=True,
                             fig_extension="png", resolution=300, ts_type=C.ts_sfix):
        cls.logger.debug(f"Generating 'XY_hist' plot: bins={bins} - figsize={figsize}")
        df_X = df_XY.drop(columns=y_target_name, axis=1)
        y    = df_XY[y_target_name]
        fig, ax = plt.subplots(figsize=figsize)
        for i, col in enumerate(sorted(df_X.columns)) :
            for clazz in y.unique() :
                df_X[y==clazz][col].plot.hist(bins=bins, figsize=figsize, alpha=0.3, label=f'Class #{int(clazz)}')
            plt.legend()
            plt.xlabel(col)
            ImgUtil.save_fig(fig_id=f"{fig_id}_{col}_histogram_{figsize[0]}x{figsize[1]}", tight_layout=tight_layout, fig_extension=fig_extension, resolution=resolution, ts_type=ts_type)
            ax.clear()
    # NB:
    # df.hist: => Plot 1 Histo per dfColumn
    # df.plot.hist: => Plot all df-referenced-Columns on same Histo

    @classmethod
    def save_df_scatter_matrix_plot(cls, df:pd.DataFrame, fig_id:str , figsize=(20,15), cfield=None, tight_layout=True,
                                    fig_extension="png", resolution=300, ts_type=C.ts_sfix):
        cls.logger.debug(f"Generating 'scatter matrix' plot: figsize:{figsize}")
        if cfield == None :
            pd.plotting.scatter_matrix(df, figsize=figsize)
        else :
            pd.plotting.scatter_matrix(df, figsize=figsize,  alpha=0.3, c=df[cfield].values, cmap='RdBu')
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
    def save_df_violin_plot(cls, df:pd.DataFrame, fig_id:str, ncols:int, figsize=(20,20), fig_extension="png", resolution=300, ts_type=C.ts_sfix):
        cls.logger.debug(f"Generating 'violin' plot: figsize:{figsize}")
        nrows = len(df.columns) // ncols if (len(df.columns) % ncols) == 0 else (len(df.columns) // ncols) + 1
        #
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        for i, col in enumerate(sorted(df.columns)) :
            sns.violinplot(x=df[col], linewidth=1, ax=axs[i//ncols, i%ncols])
            # sns.stripplot( x=df[col], color="orange", jitter=0.2, linewidth=1, ax=axs[i//3,i%3])
            sns.boxplot( x=df[col], linewidth=1, ax=axs[i//ncols, i%ncols], saturation=0 )
        # axs.set_title(fig_id, fontsize=28)
        cls.save_fig(fig_id=f"{fig_id.replace(' ','_')}_violin_{figsize[0]}x{figsize[1]}", tight_layout=True, fig_extension=fig_extension, resolution=resolution, ts_type=ts_type)


    def save_df_XY_violin_plot(df_XY:pd.DataFrame, y_target_name:str, fig_id:str, ncols:int, figsize=(20,20), fig_extension="png", resolution=300, ts_type=C.ts_sfix):
        df = df_XY.drop(columns=y_target_name, axis=1)
        nrows = len(df.columns) // ncols if (len(df.columns) % ncols) == 0 else (len(df.columns) // ncols) + 1
        #
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        for i, col in enumerate(sorted(df.columns)) :
            sns.violinplot(x=y_target_name, y=col, data=df_XY, linewidth=1, ax=axs[i//ncols, i%ncols])
            sns.boxplot   (x=y_target_name, y=col, data=df_XY, linewidth=1, ax=axs[i//ncols, i%ncols] )
        ImgUtil.save_fig(fig_id=f"{fig_id.replace(' ','_')}_violin_{figsize[0]}x{figsize[1]}", tight_layout=True, fig_extension=fig_extension, resolution=resolution, ts_type=ts_type)


    @classmethod
    # SWARM PLOT did not work correctly
    def save_df_swarm_plot(cls, df:pd.DataFrame, fig_id:str, ncols:int, figsize=(20,20), cfield=None, fig_extension="png", resolution=300, ts_type=C.ts_sfix):
        cls.logger.debug(f"Generating 'swarm' plot: figsize:{figsize}")
        nrows = len(df.columns) // ncols if (len(df.columns) % ncols) == 0 else (len(df.columns) // ncols) + 1
        #
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        for i, col in enumerate(sorted(df.columns)) :
            sns.swarmplot(x=df[col], linewidth=1, ax=axs[i//ncols, i%ncols], hue=df[cfield].values)
        cls.save_fig(fig_id=f"{fig_id.replace(' ','_')}_swarm_{figsize[0]}x{figsize[1]}", tight_layout=True, fig_extension=fig_extension, resolution=resolution, ts_type=ts_type)

    # def save_df_swarm_plot(df_XY:pd.DataFrame, fig_id:str, figsize=(5,5), y_target_name=None, fig_extension="png", resolution=300, ts_type=Const.ts_sfix):
    #     df_X = df_XY.drop(columns=y_target_name, axis=1)
    #     y    = df_XY[y_target_name]
    #     fig, ax = plt.subplots(figsize=figsize)
    #     for i, col in enumerate(sorted(df_X.columns)) :
    #         for clazz in y.unique() :
    #             sns.swarmplot(x=col, hue=y_target_name, data=df_XY[y==clazz])
    #             # sns.swarmplot(x='data', y='feature',  hue='label', data=df)
    #         plt.legend()
    #         plt.xlabel(col)
    #         ImgUtil.save_fig(fig_id=f"{fig_id}_{col}_swarm_{figsize[0]}x{figsize[1]}", tight_layout=tight_layout, fig_extension=fig_extension, resolution=resolution, ts_type=ts_type)
    #         ax.clear()


# NB: Generic way to plot whathever :
# fig, ax = plt.subplots(figsize=(20,20))
# sns.heatmap(corr_matrix, cmap='RdBu', annot=True , annot_kws={'size':15}, ax=ax)
# ax.set_title("Valeo starter production correlation measures", fontsize=14)
# plt.show()

