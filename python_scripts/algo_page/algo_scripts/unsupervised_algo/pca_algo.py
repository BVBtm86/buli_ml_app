from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from python_scripts.algo_page.algo_scripts.unsupervised_algo.utilities_unsupervised import colors_plot
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px


def pca_eda(data, variables, var_filter, code_filter, plot_title):
    # ##### Filter Data
    final_df = data.loc[(data[var_filter] == code_filter), variables].reset_index(drop=True)
    # final_df.to_csv("test.csv")
    # ##### Factor Analysis Statistics
    chi_square_factor, p_value_factor = calculate_bartlett_sphericity(final_df)
    _, kmo_model_factor = calculate_kmo(final_df)
    pca_stats = np.round(pd.DataFrame(['#', kmo_model_factor, chi_square_factor, p_value_factor]).T, 3)
    pca_stats.columns = ["Index", "KMO Sampling Adequacy", "Bartlett's Chi-Square", "Bartlett's p-value"]
    pca_stats.set_index("Index", inplace=True)

    # ##### Principal Component Analysis
    eda_pca = PCA(n_components=len(variables))
    eda_pca.fit(final_df)

    # # ##### Create Table
    pca_eda_tab = pd.DataFrame(eda_pca.explained_variance_,
                               columns=['Eigenvalue'],
                               index=['Principal Component ' + str(i) for i in range(1, final_df.shape[1] + 1)])

    pca_eda_tab['% of Variance'] = eda_pca.explained_variance_ratio_
    pca_eda_tab['Cumulative %'] = pca_eda_tab['% of Variance'].cumsum()

    optimum_factors = sum(pca_eda_tab['Eigenvalue'] >= 1)

    # ##### Elbow Plot
    fig_elbow = px.line(pca_eda_tab,
                        y="Eigenvalue",
                        title="Elbow Plot Method",
                        height=700)
    fig_elbow.add_hline(y=1, line_width=3, line_dash="dash",
                        line_color="#1e1e1e", opacity=0.5, )
    fig_elbow.update_layout(plot_bgcolor='#ffffff',
                            xaxis_title='Factors')
    fig_elbow.update_traces(line_color='#c3110f')
    fig_elbow.update_xaxes(showgrid=False, zeroline=False)

    # ##### Corr Plot
    corr_final_df = pd.DataFrame(final_df, columns=variables).corr()
    for i in range(len(corr_final_df)):
        corr_final_df.iloc[i, i] = np.nan

    pca_corr_plot = px.imshow(np.round(corr_final_df, 3),
                              labels=dict(color="Correlation"),
                              color_continuous_scale='RdGy',
                              range_color=[-1, 1],
                              aspect="auto",
                              text_auto=True)

    pca_corr_plot.update_layout(title=f"<b>{plot_title} Correlation Heatmap</b>",
                                font=dict(size=14),
                                plot_bgcolor='#ffffff')
    return pca_stats, pca_eda_tab, optimum_factors, fig_elbow, pca_corr_plot


def optimum_pca(data, variables, no_components, var_filter, code_filter):
    # ##### Filter Data
    final_df = data.loc[(data[var_filter] == code_filter), variables].reset_index(drop=True)
    df_filter = data.loc[(data[var_filter] == code_filter)].reset_index(drop=True)

    # ##### Principal Component Analysis
    pca_final = PCA(n_components=no_components)
    pca_final.fit(final_df)
    pca_scores = pd.DataFrame(pca_final.transform(final_df), columns=[f"PCA {i}" for i in range(1, no_components + 1)])
    cum_var = np.sum(pca_final.explained_variance_ratio_[:no_components])

    # ##### Final Correlation Analysis
    pca_final_df = pd.merge(left=final_df, right=pca_scores, left_index=True, right_index=True)
    final_cor_df = pca_final_df.corr().iloc[:-no_components, -no_components:]
    final_cor_df = final_cor_df.sort_values(by='PCA 1', ascending=False)

    # ##### Data for plot
    df_plot = pd.merge(df_filter, pca_scores, left_index=True, right_index=True)

    return final_cor_df, df_plot, cum_var


def pca_plot(data, filter_labels, x_feature, y_feature, pca_representation, game_filter):
    # ##### Data Plot
    data_plot = data.copy()
    if 'Team: ' not in pca_representation:
        label_transform = dict(zip(filter_labels[filter_labels['Statistics'] == pca_representation]['Code'],
                               filter_labels[filter_labels['Statistics'] == pca_representation]['Label']))
        data_plot[pca_representation] = data_plot[pca_representation].map(label_transform)
        final_map = \
            dict(zip(filter_labels[filter_labels['Statistics'] == pca_representation]['Label'].values,
                     colors_plot[:len(filter_labels[filter_labels['Statistics'] == pca_representation]
                                      ['Label'].values)]))
    else:
        label_transform = dict(zip(filter_labels[filter_labels['Option'] == pca_representation]['Code'],
                               filter_labels[filter_labels['Option'] == pca_representation]['Label']))
        data_plot[pca_representation] = data_plot['Team'].map(label_transform)
        final_map = dict(zip(data_plot[pca_representation].dropna().unique(), [colors_plot[0]]))

    # ##### Label for Plot
    pca_fig = px.scatter(data_plot,
                         x=x_feature,
                         y=y_feature,
                         color=pca_representation,
                         color_discrete_map=final_map,
                         title=f"<b>{x_feature}</b> vs <b>{y_feature}</b> Relationship for <b>{game_filter}</b> "
                               f"Season Games")
    pca_fig.update_layout({
        "plot_bgcolor": "rgba(0, 0, 0, 0)"},
        height=600)
    pca_fig.update_xaxes(title_text=x_feature,
                         showgrid=False)
    pca_fig.update_yaxes(title_text=y_feature,
                         showgrid=False)

    return pca_fig
