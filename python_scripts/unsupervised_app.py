import streamlit as st
import math
from streamlit_option_menu import option_menu
from python_scripts.algo_page.pca_page import pca_application
from python_scripts.algo_page.cluster_page import cluster_application
from python_scripts.algo_page.algo_scripts.utilities import filter_stats_query, load_data_unsupervised


# ##### Main Application
def unsupervised_application(data_file):
    # ##### App Name
    title_col, image_col, _ = st.columns([7, 1, 1.5])
    with title_col:
        st.subheader("Unsupervised Learning")

    st.markdown(
        "<b><font color=#c3110f>Unsupervised Learning</font></b> refers to the use of machine learning "
        "algorithms to identify patterns in data sets containing data points that are neither classified nor "
        "labeled. The algorithms are thus allowed to classify, label and/or group the data points contained "
        "within the data sets without having any external guidance in performing that task.",
        unsafe_allow_html=True)

    # ##### Load Data
    df_filter, filter_map = filter_stats_query()
    df_unsupervised, df_raw, all_stats = load_data_unsupervised(data_file=data_file,
                                                                data_filter=df_filter)

    # ##### Analysis Options
    st.sidebar.header("Analysis Options")

    # ##### Algo Options
    unsupervised_menu = ["Principal Component Analysis", "Cluster Analysis"]
    unsupervised_algo = option_menu(menu_title=None,
                                    options=unsupervised_menu,
                                    icons=["collection", "signpost-split"],
                                    menu_icon="cast",
                                    orientation="horizontal",
                                    styles={
                                        "container": {"width": "100%!important", "background-color": "#e5e5e6"},
                                        "nav-link": {"--hover-color": "#ffffff"},
                                    })

    if unsupervised_algo == "Principal Component Analysis":
        st.markdown(f"<b><font color=#c3110f>Principal Component Analysis</font></b> or PCA, is a "
                    f"dimensionality-reduction method that is often used to reduce the dimensionality of large data "
                    f"sets, by transforming a large set of variables into a smaller one that still contains most of the"
                    f" information in the large set.", unsafe_allow_html=True)

        # ##### Sample Group
        main_stats = list(filter_map['Statistics'].unique())
        filter_options = [filter_name for filter_name in main_stats]
        filter_main_stat = st.sidebar.selectbox('Stat Filter', filter_options)

        if filter_main_stat != 'Total':
            type_stats = filter_map[filter_map['Statistics'] == filter_main_stat]['Label'].values
            filter_final_stat = st.sidebar.selectbox('Game Filter', type_stats)
            filter_var, filter_code = filter_map[
                filter_map['Label'] == filter_final_stat][['Statistics', 'Code']].values[0]
            filter_stat = f"{filter_main_stat}: {filter_final_stat}"
        else:
            filter_var, filter_code = 'Total', 1
            filter_stat = f"{filter_main_stat}"

        # ##### Sample Size
        df_size = df_unsupervised[(df_unsupervised[filter_var] == filter_code)].shape[0]
        if filter_main_stat != "Team" and filter_main_stat != "Result":
            app_games = math.ceil(df_size / 2)
        else:
            app_games = math.ceil(df_size)
        st.sidebar.markdown(f'<b>No of Games</b>: <b><font color=#c3110f>{app_games}</font></b>',
                            unsafe_allow_html=True)
        pca_application(data_app=df_unsupervised,
                        data_map=filter_map,
                        all_features=all_stats,
                        main_filter=filter_main_stat,
                        app_filter=filter_stat,
                        feature_filter=filter_var,
                        feature_code=filter_code,
                        data_file=data_file)

    elif unsupervised_algo == "Cluster Analysis":
        st.markdown("<b><font color=#c3110f>Cluster Analysis</font></b> or clustering is the task of grouping a set of"
                    " objects in such a way that objects in the same group (called a cluster) are more similar (in some"
                    " sense) to each other than to those in other groups (clusters).", unsafe_allow_html=True)

        # ##### Cluster Algorithm
        cluster_option = st.sidebar.selectbox(label="Cluster Algorithm",
                                              options=['Hierarchical Clustering', 'K-Means Clustering'])

        if cluster_option == "K-Means Clustering":
            # ##### Sample Group
            main_stats = list(filter_map['Statistics'].unique())
            filter_options = [filter_name for filter_name in main_stats]
            filter_main_stat = st.sidebar.selectbox('Stat Filter', filter_options)

            if filter_main_stat != 'Total':
                type_stats = filter_map[filter_map['Statistics'] == filter_main_stat]['Label'].values
                filter_final_stat = st.sidebar.selectbox('Game Filter', type_stats)
                filter_var, filter_code = filter_map[
                    filter_map['Label'] == filter_final_stat][['Statistics', 'Code']].values[0]
                filter_stat = f"{filter_main_stat}: {filter_final_stat}"
            else:
                filter_var, filter_code = 'Total', 1
                filter_stat = f"{filter_main_stat}"

        else:
            # ##### Sample Group
            main_stats = list(filter_map['Statistics'].unique())
            main_stats.remove('Team')
            filter_options = [filter_name for filter_name in main_stats]
            filter_main_stat = st.sidebar.selectbox('Stat Filter', filter_options)

            if filter_main_stat != 'Total':
                type_stats = filter_map[filter_map['Statistics'] == filter_main_stat]['Label'].values
                filter_final_stat = st.sidebar.selectbox('Game Filter', type_stats)
                filter_var, filter_code = \
                    filter_map[filter_map['Label'] == filter_final_stat][['Statistics', 'Code']].values[0]
                filter_stat = f"{filter_main_stat}: {filter_final_stat}"
            else:
                filter_var, filter_code = 'Total', 1
                filter_stat = f"{filter_main_stat}"

        # ##### Sample Size
        df_size = df_unsupervised[(df_unsupervised[filter_var] == filter_code)].shape[0]
        if filter_main_stat != "Team" and filter_main_stat != "Result":
            app_games = math.ceil(df_size / 2)
        else:
            app_games = math.ceil(df_size)
        st.sidebar.markdown(f'<b>No of Games</b>: <b><font color=#c3110f>{app_games}</font></b>',
                            unsafe_allow_html=True)

        cluster_application(cluster_algo=cluster_option,
                            data_app=df_unsupervised,
                            data_raw=df_raw,
                            data_map=filter_map,
                            all_features=all_stats,
                            app_filter=filter_stat,
                            feature_filter=filter_var,
                            feature_code=filter_code,
                            data_file=data_file)
