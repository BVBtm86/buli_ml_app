import numpy as np
import pandas as pd
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage
from python_scripts.algo_page.algo_scripts.unsupervised_algo.utilities_unsupervised import colors_plot


def hierarchical_dendogram(data, filter_labels, variables, var_filter, code_filter,
                           dist_calc, link_method, dist_metric, plot_title):
    # ##### Label Map
    team_map = dict(zip(filter_labels['Code'].values, filter_labels['Label'].values))

    # ##### Filter Data
    final_features = variables.copy()
    final_features.append('Team')
    filter_df = data.loc[(data[var_filter] == code_filter), final_features].reset_index(drop=True)
    filter_df['Team'] = filter_df['Team'].map(team_map)

    # Grouping Data
    x = filter_df.groupby('Team')[variables].agg(dist_calc.lower())

    # ##### Create Dendogram
    dist_metric = dist_metric.replace("Manhattan", "cityblock")
    color_plot = np.mean(pd.DataFrame(linkage(np.array(x), method=str(link_method).lower(),
                                              metric=str(dist_metric).lower())).iloc[:, 2])
    dendogram_figure = ff.create_dendrogram(
        x,
        labels=x.index,
        orientation='left',
        linkagefun=lambda value: linkage(np.array(x),
                                         method=str(link_method).lower(),
                                         metric=str(dist_metric).lower()),
        color_threshold=color_plot,
        colorscale=colors_plot)

    dendogram_figure.update_layout(title=f"<b>{plot_title} Dendogram</b>",
                                   plot_bgcolor='rgba(0,0,0,0)',
                                   font=dict(size=12),
                                   height=600)
    return dendogram_figure
