import streamlit as st
from python_scripts.algo_page.algo_scripts.unsupervised_algo.pca_algo import pca_eda, optimum_pca, pca_plot
from python_scripts.algo_page.algo_scripts.unsupervised_algo.utilities_unsupervised import data_download, \
    plot_downloader
from PIL import Image
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def pca_application(data_app, data_map, all_features, main_filter, app_filter, feature_filter, feature_code, data_file):

    config = {'displayModeBar': False}
    feature_col, _, pca_col = st.columns([2, 0.5, 7])
    with feature_col:
        pca_analysis = st.sidebar.selectbox("Show Analysis", ["Table", "Elbow Plot", "Correlation Plot"])
        st.markdown("<b>Game Stats</b>", unsafe_allow_html=True)
        if data_file == "Top Statistics":
            analysis_stats = [col for col in all_features if st.checkbox(col, True)]
        else:
            with st.expander(""):
                analysis_stats = [col for col in all_features if st.checkbox(col, True)]

    if len(analysis_stats) > 1:
        with feature_col:
            pca_stats, pca_analysis_tab, pca_optimum, fig_elbow, fig_corr = \
                pca_eda(data=data_app,
                        variables=analysis_stats,
                        var_filter=feature_filter,
                        code_filter=feature_code,
                        plot_title=app_filter)
        if pca_analysis == 'Table':
            with pca_col:
                st.markdown("Principal Component Analysis Adequacy Statistics")
                st.table(data=pca_stats.style.format(formatter="{:.3f}").apply(
                    lambda x: ['background: #aeaec5' if i % 2 == 1 else 'background: #ffffff'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#c3110f'), ('color', '#ffffff')]}]))
                st.markdown("Principal Component Analysis Table")
                st.table(data=pca_analysis_tab.style.format(subset=['% of Variance', 'Cumulative %'],
                                                            formatter="{:.2%}").format(
                    subset=['Eigenvalue'],
                    formatter="{:.3f}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#c3110f'), ('color', '#ffffff')]}]))

        elif pca_analysis == 'Elbow Plot':
            with pca_col:
                st.plotly_chart(fig_elbow,
                                config=config,
                                use_container_width=True)
        else:
            with pca_col:
                st.plotly_chart(fig_corr,
                                config=config,
                                use_container_width=True)
        with feature_col:
            st.markdown(f"The recommended no of final <b><font color=#c3110f>Principal Components</font></b> is "
                        f"<b><font color=#c3110f>{pca_optimum}</font></b>", unsafe_allow_html=True)

        st.subheader("PCA Optimum Solution")
        st.sidebar.header("Results")
        no_final_components = st.sidebar.selectbox(label="Final No of Components",
                                                   options=[i + 1 for i in range(len(analysis_stats))],
                                                   index=pca_optimum - 1)
        optimum_col, plot_col = st.columns([5, 6])
        pca_corr, final_pca_data, final_variance = optimum_pca(data=data_app,
                                                               variables=analysis_stats,
                                                               no_components=no_final_components,
                                                               var_filter=feature_filter,
                                                               code_filter=feature_code)

        # ##### PCA Eigenvalues
        with optimum_col:
            st.table(pca_corr.style.background_gradient(cmap='RdGy', vmin=-1, vmax=1).set_table_styles(
                [{'selector': 'th',
                  'props': [('background-color', '#c3110f'), ('color', '#ffffff')]
                  }]).format(formatter="{:.3f}"))

            st.markdown(f"If we keep <b><font color=#c3110f>{no_final_components}</font></b> Components we keep "
                        f"<b><font color=#c3110f>{final_variance:.2%}</font></b> of the original variance",
                        unsafe_allow_html=True)

        # ##### PCA Feature Selection
        pca_x = st.sidebar.selectbox(label="Select PCA Feature for x axis:",
                                     options=[f"PCA {i}" for i in range(1, no_final_components + 1)])
        pca_y_options = [f"PCA {i}" for i in range(1, no_final_components + 1) if f"PCA {i}" != pca_x]
        pca_y = st.sidebar.selectbox(label="Select PCA Feature for y axis:",
                                     options=pca_y_options)
        available_features = [var for var in data_map["Statistics"].unique() if var != 'Team' and var != main_filter]
        pca_feature = st.sidebar.selectbox("PCA Representation",
                                           options=available_features)
        st.sidebar.markdown("")
        # ##### PCA plot
        plot_pca = pca_plot(data=final_pca_data,
                            filter_labels=data_map,
                            x_feature=pca_x,
                            y_feature=pca_y,
                            pca_representation=pca_feature,
                            game_filter=app_filter)

        with plot_col:
            st.plotly_chart(plot_pca, config=config, use_container_width=True)

        _, download_col_1, download_col_2, download_col_3, logo_col = st.columns([1, 3, 3, 3, 1])
        with logo_col:
            if 'Team' in app_filter:
                team_name = app_filter.split(": ")[1]
                team_logo = Image.open(f'images/{team_name}.png')
                st.image(team_logo, width=75)
            else:
                team_name = "Bundesliga"
                team_logo = Image.open(f'images/{team_name}.png')
                st.image(team_logo, width=75)

        with download_col_1:
            df_loadings = data_download(pca_corr, app_filter.replace(': ', '_'))
            st.download_button(label='📥 Download PCA Loadings',
                               data=df_loadings,
                               file_name=f"{app_filter.replace(': ', '_')}_PCA Loadings.xlsx")

        with download_col_2:
            df_score = data_download(final_pca_data, app_filter.replace(': ', '_'))
            st.download_button(label='📥 Download PCA Scores',
                               data=df_score,
                               file_name=f"{app_filter.replace(': ', '_')}_PCA Scores.xlsx")

        with download_col_3:
            download_pca_plot = plot_downloader(plot_pca)
            st.download_button(
                label=f'📥 Download {pca_x} vs {pca_y} Plot',
                data=download_pca_plot,
                file_name=f"{app_filter.replace('_', '').replace(': ', '_')}_{pca_x} vs {pca_y} Plot.html",
                mime='text/html')
    else:
        with pca_col:
            st.info("You need at least 2 game stats to run the algorithm!")
