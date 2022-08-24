import streamlit as st
from python_scripts.algo_page.algo_scripts.unsupervised_algo.utilities_unsupervised import cluster_pca, \
    linkage_info, metric_info, plot_downloader, data_download
from python_scripts.algo_page.algo_scripts.unsupervised_algo.hierarchical_algo import hierarchical_dendogram
from python_scripts.algo_page.algo_scripts.unsupervised_algo.kmeans_algo import kmeans_eda, kmeans_final
from PIL import Image


def cluster_application(cluster_algo, data_app, data_raw, data_map,
                        all_features, app_filter, feature_filter, feature_code, data_file):
    config = {'displayModeBar': False}

    if cluster_algo == "Hierarchical Clustering":
        st.subheader("Hierarchical Clustering")
        data_type = st.sidebar.selectbox(label="Data to use",
                                         options=['Original Data', 'PCA Data'])
        feature_col, cluster_col = st.columns([3, 9])
        with feature_col:
            st.markdown("<b>Game Stats</b>", unsafe_allow_html=True)
            if data_type == "Original Data":
                final_hierarchical_df = data_app.copy()
                if data_file == "Top Statistics":
                    analysis_stats = [col for col in all_features if st.checkbox(col, True)]
                else:
                    with st.expander(""):
                        analysis_stats = [col for col in all_features if st.checkbox(col, True)]
                run_cluster = True
            else:
                cluster_sample = data_app[(data_app[feature_filter] == feature_code)].shape[0]
                if data_file == "All Statistics" and cluster_sample <= 60:
                    run_cluster = False
                else:
                    run_cluster = True
                if run_cluster:
                    final_hierarchical_df, pca_features = cluster_pca(data=data_app,
                                                                      variables=all_features,
                                                                      var_filter=feature_filter,
                                                                      code_filter=feature_code)
                    analysis_stats = [col for col in pca_features if st.checkbox(col, True)]
                else:
                    analysis_stats = []
        # ##### Metric Calculation
        distance_calc = st.sidebar.selectbox(label="Metric Calculations",
                                             options=['Mean', "Median", "Min", "Max", "Std"])

        st.sidebar.header("Hyperparameters")
        with st.sidebar.expander("Tune Hyperparameters"):
            # ##### Linkage Method
            linkage_types = ["Ward", "Single", "Complete", "Average"]
            linkage_method = st.selectbox(label="Linkage Method",
                                          options=linkage_types)
            # ##### Distance Metric
            distance_type = ["Euclidean", "Manhattan", "Minkowski"]
            if linkage_method != "Ward":
                distance_method = st.selectbox(label="Distance Metric",
                                               options=distance_type)
            else:
                distance_method = "Euclidean"

        st.sidebar.markdown("")
        if run_cluster:
            if len(analysis_stats) > 0:
                # ##### Dendogram Plot
                dendogram_plot = hierarchical_dendogram(data=final_hierarchical_df,
                                                        filter_labels=data_map,
                                                        variables=analysis_stats,
                                                        var_filter=feature_filter,
                                                        code_filter=feature_code,
                                                        dist_calc=distance_calc,
                                                        link_method=linkage_method,
                                                        dist_metric=distance_method,
                                                        plot_title=app_filter)

                with cluster_col:
                    st.plotly_chart(dendogram_plot,
                                    config=config,
                                    use_container_width=True)
                    # ##### Display Info
                    st.markdown(f"<b><font color=#c3110f>{linkage_method} Linkage Method</font></b>: "
                                f"<i>{linkage_info[linkage_types.index(linkage_method)]}</i>.",
                                unsafe_allow_html=True)
                    st.markdown(f"<b><font color=#c3110f>{distance_method} Distance Metric</font></b>: "
                                f"<i>{metric_info[distance_type.index(distance_method)]}</i>.",
                                unsafe_allow_html=True)

                with feature_col:
                    download_dendogram = plot_downloader(dendogram_plot)
                    st.download_button(
                        label='📥 Download Dendogram',
                        data=download_dendogram,
                        file_name=f"{app_filter.replace('_', '').replace(': ', '_')}_Dendogram.html",
                        mime='text/html')
            else:
                with cluster_col:
                    st.info("You need at least 1 game stat to run the algorithm!")
        else:
            with cluster_col:
                st.info("Not enough Games to run the analysis!")

    elif cluster_algo == "K-Means Clustering":
        st.subheader("K-Means Clustering")
        data_type = st.sidebar.selectbox(label="Data to use",
                                         options=['Original Data', 'PCA Data'])
        feature_col, cluster_col = st.columns([3, 9])
        with feature_col:
            st.markdown("<b>Game Stats</b>", unsafe_allow_html=True)
            if data_type == "Original Data":
                final_kmeans_df = data_app.copy()
                if data_file == "Top Statistics":
                    analysis_stats = [col for col in all_features if st.checkbox(col, True)]
                else:
                    with st.expander(""):
                        analysis_stats = [col for col in all_features if st.checkbox(col, True)]
                run_cluster = True
            else:
                cluster_sample = data_app[(data_app[feature_filter] == feature_code)].shape[0]
                if data_file == "All Statistics" and cluster_sample <= 60:
                    run_cluster = False
                else:
                    run_cluster = True
                if run_cluster:
                    final_kmeans_df, pca_features = cluster_pca(data=data_app,
                                                                variables=all_features,
                                                                var_filter=feature_filter,
                                                                code_filter=feature_code)
                    analysis_stats = [col for col in pca_features if st.checkbox(col, True)]
                else:
                    analysis_stats = []

        analysis_stage = st.sidebar.selectbox("Analysis Stage", ["Data Exploratory", "Final Analysis"])
        if run_cluster:
            if analysis_stage == 'Data Exploratory':
                with feature_col:
                    if 'Team' in app_filter:
                        team_name = app_filter.split(": ")[1]
                        team_logo = Image.open(f'images/{team_name}.png')
                        st.image(team_logo, width=100)
                    else:
                        team_name = "Bundesliga"
                        team_logo = Image.open(f'images/{team_name}.png')
                        st.image(team_logo, width=100)

                if len(analysis_stats) > 0:
                    with cluster_col:
                        with st.spinner("Running KMeans Analysis (this may take a couple of minutes) ....."):
                            progress_bar = st.progress(0)
                            km_metrics_plot, kmeans_metric, info_segments = \
                                kmeans_eda(data=final_kmeans_df,
                                           variables=analysis_stats,
                                           var_filter=feature_filter,
                                           code_filter=feature_code,
                                           progress=progress_bar)

                            progress_bar.empty()
                            st.plotly_chart(km_metrics_plot,
                                            config=config,
                                            use_container_width=True)

                    st.markdown("Segment Solutions Metrics")
                    st.dataframe(data=kmeans_metric.style.format(formatter="{:.3f}").apply(
                        lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                                   for i in range(len(x))], axis=0).apply(
                        lambda x: ['background-color: #ffffff' if i % 2 == 0 else 'background-color: #e5e5e6'
                                   for i in range(len(x))], axis=0))
                    st.markdown(f"Based on the Exploratory Analysis, we recommend to start with <b><font color=#c3110f>"
                                f"{info_segments[0]}</font></b>, <b><font color=#c3110f>"
                                f"{info_segments[1]}</font></b> and <b><font color=#c3110f>"
                                f"{info_segments[2]}</font></b> Segments Solution", unsafe_allow_html=True)
                else:
                    with cluster_col:
                        st.info("You need at least 1 feature to run the algorithm!")
            elif analysis_stage == 'Final Analysis':
                no_clusters = st.sidebar.slider(label="Final No Segments",
                                                min_value=2,
                                                max_value=10,
                                                value=3)
                if len(analysis_stats) > 1:
                    x_vars = [var for var in analysis_stats]
                    feature_plot_x = st.sidebar.selectbox("Feature X", x_vars)
                    y_vars = [var for var in analysis_stats if var != feature_plot_x]
                    feature_plot_y = st.sidebar.selectbox("Feature Y", y_vars)
                else:
                    feature_plot_x = None
                    feature_plot_y = None
                st.sidebar.markdown("")

                if len(analysis_stats) > 0:
                    plot_silhouette, silhouette_df, metrics_avg, kmeans_sig, kmeans_plot, \
                        kmeans_filter, km_filter_sig, km_final_df = \
                        kmeans_final(data=final_kmeans_df,
                                     data_stats=data_raw,
                                     data_filter_map=data_map,
                                     variables=analysis_stats,
                                     var_filter=feature_filter,
                                     code_filter=feature_code,
                                     no_clusters=no_clusters,
                                     feature_x=feature_plot_x,
                                     feature_y=feature_plot_y,
                                     data_format=data_type,
                                     plot_title=app_filter)

                    with cluster_col:
                        st.plotly_chart(plot_silhouette,
                                        config=config,
                                        use_container_width=True)

                    with feature_col:
                        st.markdown(f"Average Silhouette Score: <b><font color=#c3110f>{metrics_avg[0]:.3f}"
                                    f"</font></b>", unsafe_allow_html=True)
                        st.markdown(f"Calinski Harabaz index : <b><font color=#c3110f>{metrics_avg[1]:.3f}"
                                    f"</font></b>", unsafe_allow_html=True)
                        st.markdown(f"Davies Bouldin Index: <b><font color=#c3110f>{metrics_avg[2]:.3f}"
                                    f"</font></b>", unsafe_allow_html=True)
                        st.table(data=silhouette_df.style.format(subset=['Size %'], formatter="{:.2%}").format(
                            subset=['Silhouette'], formatter="{:.4f}").apply(
                            lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                                       for i in range(len(x))], axis=0).apply(
                            lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                                       for i in range(len(x))], axis=0).set_table_styles(
                            [{'selector': 'th',
                              'props': [('background-color', '#c3110f'), ('color', '#ffffff')]}]))

                    st.subheader("Optimum K-Means Solution")
                    st.dataframe(data=kmeans_sig.T.style.apply(
                        lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                                   for i in range(len(x))], axis=0).apply(
                        lambda x: ['background-color: #ffffff' if i % 2 == 0 else 'background-color: #e5e5e6'
                                   for i in range(len(x))], axis=0))

                    st.markdown(f"<b>Note</b>: If a cell contains the <b>No</b> of another column cell, the mean of that "
                                f"cell is <b><font color=#c3110f>Statistically Higher</font></b> then the mean of "
                                f"that specific column cell", unsafe_allow_html=True)
                    tab_col, kmeans_col = st.columns([2, 8])
                    with tab_col:
                        df_kmeans = data_download(km_final_df, app_filter.replace(': ', '_'))
                        st.download_button(label='📥 Download K-Means Data',
                                           data=df_kmeans,
                                           file_name=f"{app_filter.replace(': ', '_')}_K-Means Results.xlsx")
                        download_km_plot = plot_downloader(kmeans_plot)
                        st.download_button(
                            label=f'📥 Download K-Means Plot',
                            data=download_km_plot,
                            file_name=f"{app_filter.replace('_', '').replace(': ', '_')}_"
                                      f"{feature_plot_x} vs {feature_plot_y} KM Plot.html",
                            mime='text/html')

                        if 'Team' in app_filter:
                            team_name = app_filter.split(": ")[1]
                            team_logo = Image.open(f'images/{team_name}.png')
                            st.image(team_logo, width=100)
                        else:
                            team_name = "Bundesliga"
                            team_logo = Image.open(f'images/{team_name}.png')
                            st.image(team_logo, width=100)

                    with kmeans_col:
                        st.plotly_chart(kmeans_plot,
                                        config=config,
                                        use_container_width=True)

                    st.markdown(f"<h5>K-Means Solution by Filter Stats</h5>", unsafe_allow_html=True)
                    filter_col, sig_col_results = st.columns([5, 5])
                    with filter_col:
                        st.markdown("<b>Filter Results</b>", unsafe_allow_html=True)
                        st.table(data=kmeans_filter.style.format(formatter="{:.2%}").apply(
                            lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                                       for i in range(len(x))], axis=0).apply(
                            lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                                       for i in range(len(x))], axis=0).set_table_styles(
                            [{'selector': 'th',
                              'props': [('background-color', '#c3110f'), ('color', '#ffffff')]}]))

                    with sig_col_results:
                        st.markdown("<b>Filter Significance Testing</b>", unsafe_allow_html=True)
                        st.table(data=km_filter_sig.style.apply(
                            lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                                       for i in range(len(x))], axis=0).apply(
                            lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                                       for i in range(len(x))], axis=0).set_table_styles(
                            [{'selector': 'th',
                              'props': [('background-color', '#c3110f'), ('color', '#ffffff')]}]))
                        st.markdown(f"<b>None</b>: If a column contains the <b>No</b> of another column, the % of that "
                                    f"column is <b><font color=#c3110f>Statistically Higher</font></b> then the % of "
                                    f"that specific column", unsafe_allow_html=True)
        else:
            st.info("Not enough Games to run the analysis!")
