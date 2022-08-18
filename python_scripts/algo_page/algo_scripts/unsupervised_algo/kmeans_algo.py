import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly import subplots
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from python_scripts.algo_page.algo_scripts.unsupervised_algo.utilities_unsupervised import colors_plot, sig_markers


def kmeans_eda(data, variables, var_filter, code_filter, progress):
    # ##### Filter Data
    filter_df = data.loc[(data[var_filter] == code_filter), variables].reset_index(drop=True)

    # ##### SSE and Silhouette Coefficients
    sse_coef = []
    silhouette_coef = []
    calinski_coef = []
    davies_coef = []
    no_of_runs = 19
    current_run = 0
    for no_cluster in range(2, 21):
        km_model = KMeans(n_clusters=no_cluster, init="random", n_init=10, random_state=1909)
        km_model.fit(filter_df)
        sse_coef.append(km_model.inertia_)
        silhouette_coef.append(silhouette_score(filter_df, km_model.labels_))
        calinski_coef.append(calinski_harabasz_score(filter_df, km_model.labels_))
        davies_coef.append(davies_bouldin_score(filter_df, km_model.labels_))
        current_run += 1
        progress.progress((current_run / no_of_runs))

    kmeans_scores = pd.DataFrame([sse_coef, silhouette_coef, calinski_coef, davies_coef])
    kmeans_scores.columns = [f"{i} Segments" for i in range(2, 21)]
    kmeans_scores.index = ['SSE', 'Silhouette Score', 'Calinski Harabaz Index', 'Davies Bouldin index']

    # ##### K-Means EDA Plot
    km_eda_plots = make_subplots(rows=2,
                                 cols=2,
                                 subplot_titles=("SSE Plot", "Silhouette Plot",
                                                 "Calinski Harabaz Plot", "Davies Bouldin Plot"))
    km_eda_plots.add_trace(go.Scatter(x=kmeans_scores.columns,
                                      y=kmeans_scores.iloc[0, :]),
                           row=1, col=1)
    km_eda_plots.add_trace(go.Scatter(x=kmeans_scores.columns,
                                      y=kmeans_scores.iloc[1, :]),
                           row=1, col=2)
    km_eda_plots.add_trace(go.Scatter(x=kmeans_scores.columns,
                                      y=kmeans_scores.iloc[2, :]),
                           row=2, col=1)
    km_eda_plots.add_trace(go.Scatter(x=kmeans_scores.columns,
                                      y=kmeans_scores.iloc[3, :]),
                           row=2, col=2)
    km_eda_plots.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                               showlegend=False,
                               height=700,
                               title_text="K-Means Segment Solution Metrics")
    km_eda_plots.update_traces(line_color='#c3110f')
    km_eda_plots.update_xaxes(showgrid=False)

    # ##### Ranking Scores
    prepare_rank_df = kmeans_scores.iloc[1:, :].T
    rank_df = prepare_rank_df.copy()
    rank_df[['Silhouette Score', 'Calinski Harabaz Index']] = \
        rank_df[['Silhouette Score', 'Calinski Harabaz Index']].rank(axis=0, ascending=False)
    rank_df[['Davies Bouldin index']] = \
        rank_df[['Davies Bouldin index']].rank(axis=0)
    rank_df['Final Rank'] = np.mean(rank_df, axis=1)
    rank_df = rank_df.iloc[:10, :]
    best_segment = int(rank_df[['Final Rank']].idxmin().values[0].replace(" Segments", ""))

    if int(best_segment) == 2:
        try_out_segments = [2, 3, 4]
    elif best_segment == 10:
        try_out_segments = [8, 9, 10]
    else:
        try_out_segments = [best_segment - 1, best_segment, best_segment + 1]

    return km_eda_plots, kmeans_scores, try_out_segments


def kmeans_final(data, data_stats, data_filter_map, variables, var_filter, code_filter, no_clusters,
                 feature_x, feature_y, data_format, plot_title):
    if data_format == 'Original Data':
        # ##### Filter Data
        kmeans_df = data.loc[(data[var_filter] == code_filter), variables].reset_index(drop=True)
        final_df = data_stats.loc[(data_stats[var_filter] == code_filter), variables].reset_index(drop=True)
    else:
        kmeans_df = data.loc[(data[var_filter] == code_filter), variables].reset_index(drop=True)
        final_df = data.loc[(data[var_filter] == code_filter), variables].reset_index(drop=True)

    # ##### Run KMeans
    km_model = KMeans(n_clusters=no_clusters, init="random", n_init=10, random_state=1909)
    cluster_labels = km_model.fit_predict(kmeans_df) + 1
    kmeans_df['Segment'] = cluster_labels
    final_df['Segment'] = cluster_labels

    # ##### Silhouette Score
    silhouette_avg = silhouette_score(kmeans_df, cluster_labels)
    calinski_avg = calinski_harabasz_score(kmeans_df, cluster_labels)
    davies_avg = davies_bouldin_score(kmeans_df, cluster_labels)
    sample_silhouette_values = silhouette_samples(kmeans_df, cluster_labels)
    min_values = np.min(sample_silhouette_values)

    # ##### Silhouette Analysis
    if len(variables) > 1:
        if data_format == "Original Data":
            fig_silhouette = subplots.make_subplots(rows=2, cols=1,
                                                    print_grid=False,
                                                    subplot_titles=(f'The silhouette plot for the various clusters.',
                                                                    f'The visualization of the clustered data.'))
        else:
            fig_silhouette = subplots.make_subplots(rows=1, cols=2,
                                                    print_grid=False,
                                                    subplot_titles=(f'The silhouette plot for the various clusters.',
                                                                    f'The visualization of the clustered data.'))
    else:
        fig_silhouette = subplots.make_subplots(rows=1, cols=1,
                                                print_grid=False,
                                                subplot_titles=f'The silhouette plot for the various clusters.')

    fig_silhouette['layout']['xaxis1'].update(title=f'Silhouette coefficient',
                                              range=[min_values, 1],
                                              showgrid=True)
    fig_silhouette['layout']['yaxis1'].update(title='Cluster',
                                              showticklabels=False,
                                              range=[0, len(kmeans_df) + (no_clusters + 1) * 10],
                                              showgrid=False)
    y_lower = 10
    for i in range(1, no_clusters + 1):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        filled_area = go.Scatter(y=np.arange(y_lower, y_upper),
                                 x=ith_cluster_silhouette_values,
                                 mode='lines',
                                 showlegend=False,
                                 line=dict(width=0.5,
                                           color=colors_plot[i - 1]),
                                 fill='tozerox')
        fig_silhouette.add_trace(filled_area, 1, 1)
        y_lower = y_upper + 10

    if len(variables) > 1:
        clusters = go.Scatter(x=kmeans_df[feature_x],
                              y=kmeans_df[feature_y],
                              text=kmeans_df['Segment'].apply(lambda x: f'Segment {x}'),
                              marker_color=kmeans_df['Segment'],
                              showlegend=False,
                              mode='markers',
                              marker=dict(colorscale=colors_plot[:no_clusters],
                                          size=3,
                                          opacity=0.9),
                              hoverinfo=['x', 'y', 'text'])

        if data_format == "Original Data":
            fig_silhouette.add_trace(clusters, 2, 1)
        else:
            fig_silhouette.add_trace(clusters, 1, 2)

        centers_ = km_model.cluster_centers_
        centers_ = pd.DataFrame(centers_, columns=variables)
        centers = go.Scatter(x=centers_.loc[:, feature_x],
                             y=centers_.loc[:, feature_y],
                             text=[f'Segment {i + 1}' for i in range(no_clusters)],
                             showlegend=False,
                             mode='markers',
                             marker=dict(size=14,
                                         color=colors_plot[:no_clusters],
                                         line=dict(color='black',
                                                   width=0.5)))
        if data_format == "Original Data":
            fig_silhouette.add_trace(centers, 2, 1)
        else:
            fig_silhouette.add_trace(centers, 1, 2)

        fig_silhouette['layout']['xaxis2'].update(title=f'{feature_x}',
                                                  zeroline=False,
                                                  showgrid=False)
        fig_silhouette['layout']['yaxis2'].update(title=f'{feature_y}',
                                                  zeroline=False,
                                                  showgrid=False)

    if data_format == "Original Data":
        height_plot = 820 + (no_clusters * 40)
    else:
        height_plot = 440 + (no_clusters * 40)
    fig_silhouette.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                                 height=height_plot)
    fig_silhouette['layout'].update(title="Silhouette analysis for KMeans clustering on sample data "
                                          "with n clusters = %d" % no_clusters)

    # ##### Silhouette Results
    silhouette_df = pd.DataFrame(kmeans_df['Segment'].value_counts().sort_index())
    silhouette_df.columns = ['Size']

    size = pd.DataFrame(kmeans_df['Segment'].value_counts(normalize=True).sort_index())
    size.columns = ['Size %']

    kmeans_df['Silhouette_Score'] = sample_silhouette_values
    score = pd.DataFrame(kmeans_df.groupby('Segment')['Silhouette_Score'].mean().sort_index())
    score.columns = ['Silhouette']
    silhouette_df = pd.concat([silhouette_df, size], axis=1)
    silhouette_df = pd.concat([silhouette_df, score], axis=1)
    cluster_scores = [silhouette_avg, calinski_avg, davies_avg]

    # ##### Final KMeans Data
    final_kmeans_results = final_df.groupby('Segment')[variables].mean().T
    final_kmeans_results.columns = [f"Segment {i}" for i in range(1, no_clusters + 1)]
    final_kmeans_results = np.round(final_kmeans_results, 3)
    kmeans_sig_tab = kmeans_sig(data=final_df,
                                kmeans_results=final_kmeans_results,
                                stats=variables,
                                no_clusters=no_clusters)
    # #### Final Sig Results
    final_kmeans_results.reset_index(inplace=True)
    final_kmeans_results['Type'] = 'Score'
    kmeans_sig_tab.set_index(final_kmeans_results['index'].values, inplace=True)
    kmeans_sig_tab.reset_index(inplace=True)
    kmeans_sig_tab['Type'] = 'Sig'
    final_sig_tab = pd.concat([final_kmeans_results, kmeans_sig_tab], axis=0)
    final_sig_tab.sort_index(inplace=True)
    final_sig_tab.set_index(['index', 'Type'], inplace=True)

    final_sig_tab.columns = [f"Segment {sig_markers[i - 1]}" for i in range(1, no_clusters + 1)]

    # ##### Filter Results
    data_filter = data.loc[(data[var_filter] == code_filter)].reset_index(drop=True)
    data_filter = data_filter.drop(columns=variables)
    data_filter = data_filter.drop(columns=['Match Day', 'Opponent'])
    data_filter['Segment'] = cluster_labels
    filter_results_df, filter_sig_tab = kmeans_filters(data=data_filter,
                                                       data_map=data_filter_map,
                                                       no_clusters=no_clusters,
                                                       filter_feature=var_filter)

    # ##### Map Data with labels
    plot_label_df = data.loc[(data[var_filter] == code_filter),
                             ['Team', 'Opponent', 'Match Day', 'Season',
                              'Season Stage', 'Venue', 'Result']].reset_index(drop=True)
    team_map = dict(zip(data_filter_map[data_filter_map['Statistics'] == 'Team']['Code'].values,
                        data_filter_map[data_filter_map['Statistics'] == 'Team']['Label'].values))
    season_map = dict(zip(data_filter_map[data_filter_map['Statistics'] == 'Season']['Code'].values,
                          data_filter_map[data_filter_map['Statistics'] == 'Season']['Label'].values))
    stage_map = dict(zip(data_filter_map[data_filter_map['Statistics'] == 'Season Stage']['Code'].values,
                         data_filter_map[data_filter_map['Statistics'] == 'Season Stage']['Label'].values))
    venue_map = dict(zip(data_filter_map[data_filter_map['Statistics'] == 'Venue']['Code'].values,
                         data_filter_map[data_filter_map['Statistics'] == 'Venue']['Label'].values))
    result_map = dict(zip(data_filter_map[data_filter_map['Statistics'] == 'Result']['Code'].values,
                          data_filter_map[data_filter_map['Statistics'] == 'Result']['Label'].values))

    plot_label_df['Team'] = plot_label_df['Team'].map(team_map)
    plot_label_df['Opponent'] = plot_label_df['Opponent'].map(team_map)
    plot_label_df['Season'] = plot_label_df['Season'].map(season_map)
    plot_label_df['Season Stage'] = plot_label_df['Season Stage'].map(stage_map)
    plot_label_df['Venue'] = plot_label_df['Venue'].map(venue_map)
    plot_label_df['Result'] = plot_label_df['Result'].map(result_map)

    # ##### Final Kmeans Plot
    final_km_df = final_df.copy()
    final_km_df = pd.merge(left=final_km_df, right=plot_label_df, left_index=True, right_index=True)
    final_km_df['Segment'] = final_km_df['Segment'].apply(lambda x: 'Segment ' + str(x))
    fig_kmeans = px.scatter(final_km_df,
                            x=feature_x,
                            y=feature_y,
                            color="Segment",
                            color_discrete_map=dict(zip(['Segment ' + str(i) for i in range(1, no_clusters + 1)],
                                                        colors_plot[:no_clusters])),
                            title=f"{plot_title} <b>{feature_x}</b> vs <b>{feature_y}</b> - <b>{no_clusters}</b> "
                                  f"Cluster Solution",
                            hover_data=['Team', 'Opponent', 'Season', 'Season Stage', 'Venue', 'Result', 'Match Day'])
    fig_kmeans.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                             height=600)
    fig_kmeans.update_xaxes(title_text=feature_x,
                            showgrid=False)
    fig_kmeans.update_yaxes(title_text=feature_y,
                            showgrid=False)

    # ##### Final KM Data
    if data_format == 'Original Data':
        km_final_df = data_stats.copy()
    else:
        km_final_df = data.copy()

    km_final_df = km_final_df.loc[(km_final_df[var_filter] == code_filter)].reset_index(drop=True)
    km_final_df.drop(columns=['Total'], inplace=True)
    km_final_df['Segment'] = cluster_labels
    filter_stat = [col for col in km_final_df.columns.to_list() if col not in variables]

    for col in filter_stat:
        if col == 'Segment':
            km_final_df[col] = km_final_df[col].apply(lambda x: f"Segment {x}")
        else:
            codes = data_filter_map[data_filter_map['Statistics'] == col]['Code'].values
            labels = data_filter_map[data_filter_map['Statistics'] == col]['Label'].values
            km_final_df[col] = km_final_df[col].map(dict(zip(codes, labels)))

    # #### Tabl Preparation
    final_sig_tab.reset_index(inplace=True)
    final_sig_tab['Feature'] = final_sig_tab['index'] + ": " + final_sig_tab['Type']
    final_sig_tab.drop(columns=['index', 'Type'], inplace=True)
    final_sig_tab = final_sig_tab.astype(str)
    final_sig_tab.set_index('Feature', inplace=True)

    return fig_silhouette, silhouette_df, cluster_scores, final_sig_tab, fig_kmeans, filter_results_df, \
        filter_sig_tab, km_final_df


def kmeans_sig(data, kmeans_results, stats, no_clusters):
    # ##### Significance Testing
    kmeans_sig_results = []
    for col in stats:
        col_sig = []
        for i in range(1, no_clusters + 1):
            iter_sig = ""
            for j in range(1, no_clusters + 1):
                if i != j:
                    sample_1 = data[data['Segment'] == i][col]
                    sample_2 = data[data['Segment'] == j][col]
                    p_value = ttest_ind(sample_1, sample_2)[1]
                    if (p_value <= 0.05) and \
                            (kmeans_results[kmeans_results.index == col][f'Segment {i}'].values[0] >
                             kmeans_results[kmeans_results.index == col][f'Segment {j}'].values[0]):
                        iter_sig += sig_markers[j - 1]
            col_sig.append(iter_sig)
        kmeans_sig_results.append(col_sig)

    kmeans_tab = pd.DataFrame(kmeans_sig_results)
    kmeans_tab.columns = [f"Segment {i}" for i in range(1, no_clusters + 1)]

    return kmeans_tab


def kmeans_filters(data, data_map, no_clusters, filter_feature):
    # #### Select Filter Features
    filter_columns = [col for col in data.columns.to_list() if
                      col != 'Total' and col != 'Segment' and col != filter_feature]
    # ##### Add Filter Tab
    group_final_df = pd.DataFrame()
    group_counts_df = pd.DataFrame()
    for filter_var in filter_columns:

        # ##### Create Group By Tabel
        df_group = pd.DataFrame(data.groupby(['Segment'])[filter_var].value_counts(normalize=True))
        df_group.columns = ['%']
        df_group = df_group.reset_index()

        # ##### Create Code and Values Dictionary
        codes = list(data_map[data_map['Statistics'] == filter_var]['Code'].values)
        values = [f'{filter_var}: ' + _ for _ in list(data_map[data_map['Statistics'] == filter_var]['Label'].values)]
        recode_values = dict(zip(codes, values))
        df_group[filter_var] = df_group[filter_var].map(recode_values)

        # #### Final Pivot filter Tab
        final_group_df = df_group.pivot(index=filter_var, columns='Segment', values='%')
        group_final_df = pd.concat([group_final_df, final_group_df])

        # ##### Sig Testing
        group_counts = pd.DataFrame(data.groupby(['Segment'])[filter_var].value_counts(normalize=False))
        group_counts.columns = ['Counts']
        group_counts = group_counts.reset_index()
        final_group_counts = group_counts.pivot(index=filter_var, columns='Segment', values='Counts')
        group_counts_df = pd.concat([group_counts_df, final_group_counts])

        seg_count = list(data['Segment'].value_counts().sort_index().values)
        final_sig = []
        for idx in range(group_final_df.shape[0]):
            col_sig = []
            for i in range(len(seg_count)):
                iter_sig = ""
                for j in range(len(seg_count)):
                    if i != j:
                        sample_1 = seg_count[i]
                        sample_2 = seg_count[j]
                        success_1 = group_counts_df.iloc[idx, i]
                        success_2 = group_counts_df.iloc[idx, j]
                        if success_1 > 0 and success_2 > 0:
                            _, p_value = proportions_ztest(count=[success_1, success_2],
                                                           nobs=[sample_1, sample_2],
                                                           alternative='two-sided')
                            if (p_value <= 0.05) and (group_final_df.iloc[idx, i] >
                                                      group_final_df.iloc[idx, j]):
                                iter_sig += sig_markers[j]
                col_sig.append(iter_sig)
            final_sig.append(col_sig)

    # ##### Final Tabs
    group_final_df = group_final_df.T.reset_index(drop=True)
    group_final_df['Segment'] = [f'Segment {i}' for i in range(1, no_clusters + 1)]
    group_final_df.set_index('Segment', inplace=True)
    group_final_df = group_final_df.T
    group_final_df.columns = [f"Segment {sig_markers[i - 1]}" for i in range(1, no_clusters + 1)]

    final_sig_tab = pd.DataFrame(final_sig)
    final_sig_tab.columns = [f"Segment {sig_markers[i - 1]}" for i in range(1, no_clusters + 1)]
    final_sig_tab.set_index(group_final_df.index, inplace=True)

    return group_final_df, final_sig_tab
