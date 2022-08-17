import streamlit as st
from streamlit_option_menu import option_menu
from python_scripts.algo_page.algo_scripts.supervised_algo.utilities_supervised import load_data_supervised, \
    home_away_data, supervised_pca
from python_scripts.algo_page.classification_page import classification_application
from python_scripts.algo_page.regression_page import regression_application


def supervised_application():
    # ##### App Name
    title_col, image_col, _ = st.columns([7, 1, 1.5])
    with title_col:
        st.subheader("Supervised Learning")

    st.markdown(
        '<b><font color=#c3110f>Supervised Learning</font></b> is the machine learning task of learning a function that'
        ' maps an input to an output based on example input-output pairs. It infers a function from labeled training '
        'data consisting of a set of training examples.', unsafe_allow_html=True)

    supervised_menu = ["Classification Analysis", "Regression Analysis"]
    supervised_algo = option_menu(menu_title=None,
                                  options=supervised_menu,
                                  icons=["ui-checks-grid", "graph-up"],
                                  menu_icon="cast",
                                  orientation="horizontal",
                                  styles={
                                      "container": {"width": "100%!important", "background-color": "#e5e5e6"},
                                      "nav-link": {"--hover-color": "#ffffff"},
                                  })
    # ##### Load Data
    df_supervised, filter_map, all_stats = load_data_supervised()

    # ##### Analysis Options
    st.sidebar.header("Analysis Options")

    # ##### Classification Analysis
    if supervised_algo == "Classification Analysis":

        # ##### Final Classification Data
        df_raw = df_supervised.copy()
        df_raw.drop(columns=['xG'], inplace=True)
        classification_features_raw = all_stats.copy()
        classification_features_raw.remove('xG')

        # ##### Analysis Type
        prediction_type = st.sidebar.selectbox(label="Prediction by",
                                               options=['Game Result', 'Home Team', 'Away Team'])

        if prediction_type == "Home Team":
            df_classification_type = home_away_data(data=df_raw,
                                                    features=classification_features_raw,
                                                    type_game=prediction_type)
        elif prediction_type == "Away Team":
            df_classification_type = home_away_data(data=df_raw,
                                                    features=classification_features_raw,
                                                    type_game=prediction_type)
        else:
            df_classification_type = df_raw.copy()

        # ##### Sample Group
        main_stats = list(filter_map['Statistics'].unique())

        if prediction_type == "All Games":
            main_stats.remove("Result")
            main_stats.remove("Team")
        else:
            main_stats.remove("Venue")
            main_stats.remove("Result")
            main_stats.remove("Team")

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
        df_size = df_classification_type[(df_classification_type[filter_var] == filter_code)].shape[0]
        st.sidebar.markdown(f'<b>No of Games</b>: <b><font color=#c3110f>{df_size}</font></b>', unsafe_allow_html=True)

        if prediction_type == "All Games":
            st.markdown("<b><font color=#c3110f>Classification</font></b> Analysis: predict the <b>"
                        "<font color=#c3110f>Game Result</font></b> based on a teams Game Stats.",
                        unsafe_allow_html=True)
        else:
            st.markdown("<b><font color=#c3110f>Classification</font></b> predict the <b>"
                        "<font color=#c3110f>Game Result</font></b> based on the Game Stats difference between the "
                        "<b><font color=#c3110f>Home</font></b> and <b><font color=#c3110f>Away</font></b> team.",
                        unsafe_allow_html=True)

        # ##### Prediction Stat
        prediction_stat = "Result"

        # ##### Data Type to Use
        data_type = st.sidebar.selectbox(label="Data to use",
                                         options=['Original Data', 'PCA Data'])

        if data_type == "Original Data":
            final_class_features = classification_features_raw.copy()
            final_class_features.extend(['Result', 'Team', 'Opponent', 'Season', 'Season Stage', 'Venue', 'Match Day'])
            df_classification = \
                df_classification_type.loc[(df_classification_type[filter_var] == filter_code),
                                           final_class_features].reset_index(drop=True)
            classification_features = classification_features_raw.copy()
        else:
            df_classification, classification_features = \
                supervised_pca(data=df_classification_type,
                               variables=classification_features_raw,
                               var_filter=filter_var,
                               code_filter=filter_code,
                               dep_var='Result')

        # ##### Classification Page
        classification_application(data=df_classification,
                                   data_map=filter_map,
                                   type_data=data_type,
                                   game_prediction=prediction_type,
                                   sample_filter=filter_stat,
                                   dep_var=prediction_stat,
                                   indep_var=classification_features)

    # ##### Regression Analysis
    elif supervised_algo == "Regression Analysis":

        # ##### Final Regression Data
        df_raw = df_supervised.copy()
        regression_features_raw = all_stats.copy()

        # ##### Analysis Type
        prediction_type = st.sidebar.selectbox(label="Prediction by",
                                               options=['Game', 'Home Team', 'Away Team'])

        if prediction_type == "Home Team":
            df_regression_type = home_away_data(data=df_raw,
                                                features=regression_features_raw,
                                                type_game=prediction_type)
        elif prediction_type == "Away Team":
            df_regression_type = home_away_data(data=df_raw,
                                                features=regression_features_raw,
                                                type_game=prediction_type)
        else:
            df_regression_type = df_raw.copy()

        # ##### Sample Group
        main_stats = list(filter_map['Statistics'].unique())

        if prediction_type == "All Games":
            main_stats.remove("Result")
            main_stats.remove("Team")
        else:
            main_stats.remove("Venue")
            main_stats.remove("Result")
            main_stats.remove("Team")

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
        df_size = df_regression_type[(df_regression_type[filter_var] == filter_code)].shape[0]
        st.sidebar.markdown(f'<b>No of Games</b>: <b><font color=#c3110f>{df_size}</font></b>', unsafe_allow_html=True)

        # ##### Prediction Stat
        prediction_stat = st.sidebar.selectbox(label="Prediction Stat",
                                               options=regression_features_raw)
        regression_features_type = [stat for stat in regression_features_raw if stat != prediction_stat]

        # ##### Data Type to Use
        data_type = st.sidebar.selectbox(label="Data to use",
                                         options=['Original Data', 'PCA Data'])

        if data_type == "Original Data":
            final_reg_features = regression_features_type.copy()
            final_reg_features.extend([prediction_stat,
                                       'Team', 'Opponent', 'Result', 'Season', 'Season Stage', 'Venue', 'Match Day'])
            df_regression = \
                df_regression_type.loc[(df_regression_type[filter_var] == filter_code),
                                       final_reg_features].reset_index(drop=True)
            regression_features = regression_features_type.copy()
        else:
            df_regression, regression_features = \
                supervised_pca(data=df_regression_type,
                               variables=regression_features_type,
                               var_filter=filter_var,
                               code_filter=filter_code,
                               dep_var=prediction_stat)

        if prediction_type == "All Games":
            st.markdown(f"<b><font color=#c3110f>Regression</font></b> Analysis: predict the <b>"
                        f"<font color=#c3110f>{prediction_stat}</font></b> based on a teams Game Stats.",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<b><font color=#c3110f>Regression</font></b> Analysis: predict the <b><font color=#c3110f>"
                        f"{prediction_stat}</font></b> based on the Game Stats difference between the "
                        "<b><font color=#c3110f>Home</font></b> and <b><font color=#c3110f>Away</font></b> team.",
                        unsafe_allow_html=True)

        # ##### Regression Page
        regression_application(data=df_regression,
                               data_map=filter_map,
                               type_data=data_type,
                               game_prediction=prediction_type,
                               sample_filter=filter_stat,
                               dep_var=prediction_stat,
                               indep_var=regression_features)
