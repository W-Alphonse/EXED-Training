from valeo.infrastructure import Const as C
from valeo.infrastructure.LogManager import LogManager
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from valeo.infrastructure.tools.DfUtil import DfUtil

# http://jonathansoma.com/lede/algorithms-2017/classes/fuzziness-matplotlib/how-pandas-uses-matplotlib-plus-figures-axes-and-subplots/
'''
NB:
1 - 'fig' means 'figure', and it’s your entire graphic.
2 - The graph is not the 'figure': The figure is the part around the graph.
    The graph sits on top of the 'figure'. 
    The graph is what’s called a 'subplot' or 'axis'. Or, technically, an 'AxesSubplot'.
    
3 - a - How to select explicitly a 'subplot/axis' or how to let pandas/matplotlib do it ?  
        Ex: df.groupby('country').plot(x='year', y='unemployment', ax=axis, legend=False)
    b - When we pass ax=axis to our plot => It means like we’re saying “hey, we already have a graph made up! 
        => Use it instead” and then pandas/matplotlib does, 
       instead of letting pandas/matplotlib using a brand-new image for each.  
    c - 'ax=axis' allow to link a pandas/matplotlib graph to 'subplot/axis' on top of the 'figure'    
'''
class ImgUtil() :
    logger = LogManager.logger(__name__)

    # https://stackabuse.com/pythons-classmethod-and-staticmethod-explained/
    @classmethod
    def save_fig(cls, fig_id:str , tight_layout=True, fig_extension="png", resolution=300, ts_type=C.ts_sfix):
        path = C.ts_pathanme([C.rootImages() , fig_id + "." + fig_extension], ts_type=ts_type)
        # cls.logger.debug(f"Saving figure '{fig_id}'")
        #-- https://matplotlib.org/3.2.1/tutorials/intermediate/tight_layout_guide.html
        if tight_layout:
            plt.tight_layout() # tight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area
        #-- Save "the current figure plot" that is set by "df.hist(...))". @ReferTo: pyplot.py / def gcf()
        plt.savefig(path, format=fig_extension, dpi=resolution)

    @classmethod
    def save_df_bar_plot(cls, df:pd.DataFrame, fig_id:str, ncols:int, figsize=(20,15), use_same_figure=True, tight_layout=True,
                          fig_extension="png", resolution=300, ts_type=C.ts_sfix):
        cls.logger.debug(f"Generating 'bar' plot: figsize={figsize}")
        cat_cols = DfUtil.categorical_cols(df)
        nrows = len(cat_cols) // ncols if (len(cat_cols) % ncols) == 0 else (len(cat_cols) // ncols) + 1
        #
        if use_same_figure :  #--Case where all the 'plots' are on the same 'figure'--
            fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
            for i, col in enumerate(sorted(cat_cols)) :
                df_bar = df.groupby([col]).size()
                df_bar.plot(kind='bar', x = df[col], rot=0, figsize=figsize, ax= axs[i])
            ImgUtil.save_fig(fig_id=f"{fig_id}_{col}_bar_{figsize[0]}x{figsize[1]}", tight_layout=tight_layout, fig_extension=fig_extension, resolution=resolution, ts_type=ts_type)
        else :               #--Case where each 'plot' is on a different 'figure'--
            for i, col in enumerate(sorted(cat_cols)) :
                df_bar = df.groupby([col]).size()
                df_bar.plot.bar(x = df[col], rot=0, figsize=figsize)  # same-as:  # df_bar.plot(kind='bar', x = df[col], rot=0, figsize=figsize)
                cls.save_fig(fig_id=f"{fig_id}_{col}_bar_{figsize[0]}x{figsize[1]}", tight_layout=tight_layout, fig_extension=fig_extension, resolution=resolution, ts_type=ts_type)


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

