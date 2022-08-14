import streamlit as st
import numpy as np
from python_scripts.algo_page.algo_scripts.supervised_algo.utilities_supervised import class_algo_options, \
    class_algo_name, plot_downloader, data_download, hyperparameters_linear, hyperparameters_nonlinear, svg_write, \
    download_button_tree, display_tree, display_rf_tree, display_tree_xgb
from python_scripts.algo_page.algo_scripts.supervised_algo.classification_algo import classification_all_models, \
    linear_class_application, svm_class_application, knn_class_application, naive_class_application, \
    dt_class_application, rf_class_application, xgb_class_application


def classification_application(data, data_map, type_data, game_prediction, sample_filter, dep_var, indep_var):
    config = {'displayModeBar': False}
    # ##### Algorithm Selection
    st.sidebar.subheader("Algorithm")
    classification_algo = st.sidebar.selectbox(label="Classification Algorithm",
                                               options=class_algo_options)

    st.sidebar.markdown("")

    # ##### Algo Description
    if classification_algo == "":
        pass
    elif classification_algo == "All":
        st.subheader("Classification Algorithms")
        st.markdown(f"<b><font color=#c3110f>Classification Algorithms</font></b> "
                    f"{class_algo_name[class_algo_options.index(classification_algo)]} <b><font color=#c3110f>"
                    f"{dep_var}</font></b>.", unsafe_allow_html=True)
    else:
        st.markdown(f"<h4>{classification_algo} Analysis</h5>", unsafe_allow_html=True)
        st.markdown(f"<b><font color=#c3110f>{classification_algo}</font></b> "
                    f"{class_algo_name[class_algo_options.index(classification_algo)]}", unsafe_allow_html=True)

    if classification_algo != "":
        # ##### Features
        feature_col, result_col = st.columns([3, 9])
        with feature_col:
            st.markdown("<b>Features</b>", unsafe_allow_html=True)
            analysis_stats = [col for col in indep_var if st.checkbox(col, True)]
    else:
        st.info(f"Please select one of the Classification Algorithms from the available options.")
        analysis_stats = [""]

    if len(analysis_stats) > 0:

        # ##### ''' All Classification Models '''
        if classification_algo == "All":
            with result_col:
                with st.spinner("Running Classification Algorithms ....."):
                    progress_bar = st.progress(0)
                    fig_class_plot, class_scores_df, top_class_algo = \
                        classification_all_models(data=data,
                                                  data_type=type_data,
                                                  features=analysis_stats,
                                                  predictor="Result",
                                                  progress=progress_bar,
                                                  all_algo=class_algo_options,
                                                  plot_name=sample_filter,
                                                  prediction_type=game_prediction)

                    progress_bar.empty()
                    st.plotly_chart(fig_class_plot,
                                    config=config,
                                    use_container_width=True)
                    st.markdown(f"The best performing Classification Algorithms are <b><font color=#c3110f>"
                                f"{top_class_algo[0]}</font></b>, <b><font color=#c3110f>{top_class_algo[1]}"
                                f"</font></b> and <b><font color=#c3110f>{top_class_algo[2]}</font></b>.",
                                unsafe_allow_html=True)

                with feature_col:
                    download_plot_class = plot_downloader(fig_class_plot)
                    st.download_button(
                        label='📥 Download Plot Classification',
                        data=download_plot_class,
                        file_name=f"{sample_filter.replace('_', '').replace(': ', '_')}_Plot Classification.html",
                        mime='text/html')

                    df_scores_all = data_download(class_scores_df, sample_filter.replace(': ', '_'))
                    st.download_button(label='📥 Download Data Classification',
                                       data=df_scores_all,
                                       file_name=f"{sample_filter.replace(': ', '_')}_Data Results.xlsx")

        # ##### ''' Logistic Regression '''
        elif classification_algo == "Logistic Regression":
            # ##### Hyperparameters
            with st.sidebar.expander(f"Hyperparameter Tuning"):
                train_size, std_data = hyperparameters_linear(model_type=type_data)
                solver_param = st.selectbox(label="Solver",
                                            options=["lbfgs", "newton-cg", "liblinear", "sag", "saga"])
                if solver_param != 'liblinear':
                    multi_class_param = st.selectbox(label="Loss",
                                                     options=["ovr", "multinomial"])
                else:
                    multi_class_param = "ovr"
                if solver_param == 'liblinear':
                    penalty_param = st.selectbox(label="Penalty",
                                                 options=["l1", "l2"])
                elif solver_param == "saga":
                    penalty_param = st.selectbox(label="Penalty",
                                                 options=["elasticnet", "l1", "l2", "none"])
                else:
                    penalty_param = st.selectbox(label="Penalty",
                                                 options=["l2", "none"])
                if penalty_param != "none":
                    c_param = st.select_slider(label="Regularization Strength",
                                               options=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
                                               value=1)
                else:
                    c_param = 1
                if penalty_param == "elasticnet":
                    l1_param = st.slider(label="L1 Ration",
                                         min_value=0.0,
                                         max_value=1.0,
                                         step=0.1)
                else:
                    l1_param = None
                final_params = [solver_param, multi_class_param, penalty_param, c_param, l1_param]

            # ##### Classification Regression Model
            st.sidebar.subheader("Prediction Options")
            if game_prediction != "Game Result":
                game_prediction += " Result"
            linear_plot, linear_metrics, linear_matrix, linear_pred_plot, linear_teams = \
                linear_class_application(data=data,
                                         data_type=type_data,
                                         team_map=data_map,
                                         hyperparams=final_params,
                                         features=analysis_stats,
                                         predictor="Result",
                                         predictor_map=data_map,
                                         train_sample=train_size,
                                         standardize_data=std_data,
                                         plot_name=sample_filter,
                                         prediction_type=game_prediction)

            with result_col:
                st.plotly_chart(linear_plot,
                                config=config,
                                use_container_width=True)
            with feature_col:
                download_plot_linear = plot_downloader(linear_plot)
                st.download_button(
                    label='📥 Download LR Plot',
                    data=download_plot_linear,
                    file_name=f"{sample_filter.replace('_', '').replace(': ', '_')}_Plot LR.html",
                    mime='text/html')

            # ##### Classification Results
            st.subheader("Regression Prediction Results")
            metrics_col, pred_col = st.columns([4.5, 5.5])
            with metrics_col:
                st.markdown(f"<b><font color=#c3110f>{classification_algo}</font></b> Metrics for Predicting "
                            f"<b><font color=#c3110f>{game_prediction}</font></b>",
                            unsafe_allow_html=True)
                st.table(linear_metrics.style.format(formatter="{:.2%}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#c3110f'), ('color', '#ffffff')]}]))

                st.markdown(f"<b><font color=#c3110f>{linear_teams}</font></b> <b>Observed</b> vs "
                            f"<b>Predicted</b> {sample_filter} <b><font color=#c3110f>{game_prediction}</font></b>",
                            unsafe_allow_html=True)
                st.table(linear_matrix.style.format(subset=["Defeat %", "Draw %", "Win %"], formatter="{:.2%}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#c3110f'), ('color', '#ffffff')]}]))

            with pred_col:
                st.plotly_chart(linear_pred_plot,
                                config=config,
                                use_container_width=True)
            with metrics_col:
                download_plot_prediction = plot_downloader(linear_pred_plot)
                st.download_button(
                    label='📥 Download Prediction Plot',
                    data=download_plot_prediction,
                    file_name=f"{sample_filter.replace('_', '').replace(': ', '_')}_Prediction Plot.html",
                    mime='text/html')

        # ##### ''' Support Vector Machine '''
        elif classification_algo == "Support Vector Machine":
            # ##### Hyperparameters
            with st.sidebar.expander(f"Hyperparameter Tuning"):
                train_size, std_data = hyperparameters_linear(model_type=type_data)
                svm_kernel = st.selectbox(label="Kernel",
                                          options=['rbf', "poly", 'linear', 'sigmoid'])
                if svm_kernel == 'poly':
                    svm_c = st.select_slider(label='Regularization',
                                             options=[0.1, 1, 5, 10, 50, 100],
                                             value=1)
                    svm_degree = st.select_slider(label='Poly Degree',
                                                  options=[1, 2, 3, 4, 5],
                                                  value=1)
                    svm_gamma = 'scale'
                elif svm_kernel == 'rbf':
                    svm_c = st.select_slider(label='Regularization',
                                             options=[0.1, 1, 5, 10, 50, 100],
                                             value=1)
                    svm_degree = 1
                    svm_gamma = st.select_slider(label='Kernel coefficient',
                                                 options=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10])
                elif svm_kernel == 'sigmoid':
                    svm_c = st.select_slider(label='Regularization',
                                             options=[0.1, 1, 5, 10, 50, 100],
                                             value=1)
                    svm_degree = 1
                    svm_gamma = st.select_slider(label='Kernel coefficient',
                                                 options=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10])
                else:
                    svm_c = None
                    svm_degree = None
                    svm_gamma = None
                final_params = [svm_kernel, svm_gamma, svm_degree, svm_c]

            # ##### Classification SVM Model
            st.sidebar.subheader("Prediction Options")
            if game_prediction != "Game Result":
                game_prediction += " Result"

            with result_col:
                with st.spinner("Running Model..."):
                    svm_plot, svm_metrics, svm_matrix, svm_pred_plot, svm_teams = \
                        svm_class_application(data=data,
                                              data_type=type_data,
                                              team_map=data_map,
                                              hyperparams=final_params,
                                              features=analysis_stats,
                                              predictor="Result",
                                              predictor_map=data_map,
                                              train_sample=train_size,
                                              standardize_data=std_data,
                                              plot_name=sample_filter,
                                              prediction_type=game_prediction)

            if svm_plot is not None:
                with result_col:
                    st.plotly_chart(svm_plot,
                                    config=config,
                                    use_container_width=True)
                with feature_col:
                    download_plot_linear = plot_downloader(svm_plot)
                    st.download_button(
                        label='📥 Download SVM Plot',
                        data=download_plot_linear,
                        file_name=f"{sample_filter.replace('_', '').replace(': ', '_')}_Plot SVM.html",
                        mime='text/html')
            else:
                with result_col:
                    st.info("Feature Coefficients are only available when the Kernel is Linear.")

            # ##### Classification Results
            st.subheader("SVM Prediction Results")
            metrics_col, pred_col = st.columns([4.5, 5.5])
            with metrics_col:
                st.markdown(f"<b><font color=#c3110f>{classification_algo}</font></b> Metrics for Predicting "
                            f"<b><font color=#c3110f>{game_prediction}</font></b>",
                            unsafe_allow_html=True)
                st.table(svm_metrics.style.format(formatter="{:.2%}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#c3110f'), ('color', '#ffffff')]}]))

            if svm_plot is not None:
                with metrics_col:
                    st.markdown(f"<b><font color=#c3110f>{svm_teams}</font></b> <b>Observed</b> vs "
                                f"<b>Predicted</b> {sample_filter} <b><font color=#c3110f>{game_prediction}</font></b>",
                                unsafe_allow_html=True)
                    st.table(svm_matrix.style.format(subset=["Defeat %", "Draw %", "Win %"], formatter="{:.2%}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#c3110f'), ('color', '#ffffff')]}]))
            else:
                with pred_col:
                    st.markdown(f"<b><font color=#c3110f>{svm_teams}</font></b> <b>Observed</b> vs "
                                f"<b>Predicted</b> {sample_filter} <b><font color=#c3110f>{game_prediction}</font></b>",
                                unsafe_allow_html=True)
                    st.table(svm_matrix.style.format(subset=["Defeat %", "Draw %", "Win %"], formatter="{:.2%}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#c3110f'), ('color', '#ffffff')]}]))
            if svm_plot is not None:
                with pred_col:
                    st.plotly_chart(svm_pred_plot,
                                    config=config,
                                    use_container_width=True)
            else:
                with result_col:
                    st.plotly_chart(svm_pred_plot,
                                    config=config,
                                    use_container_width=True)
            with metrics_col:
                download_plot_prediction = plot_downloader(svm_pred_plot)
                st.download_button(
                    label='📥 Download Prediction Plot',
                    data=download_plot_prediction,
                    file_name=f"{sample_filter.replace('_', '').replace(': ', '_')}_Prediction Plot.html",
                    mime='text/html')

        # ##### ''' Naive Bayes '''
        elif classification_algo == "Naive Bayes":
            # ##### Hyperparameters
            with st.sidebar.expander(f"Hyperparameter Tuning"):
                train_size = hyperparameters_nonlinear()
                nb_smoothing = st.select_slider(label='Smoothing',
                                                options=np.logspace(0, -9, num=10),
                                                value=1.e-09)
            final_params = [nb_smoothing]

            # ##### Classification Naive Bayes Model
            st.sidebar.subheader("Prediction Options")
            if game_prediction != "Game Result":
                game_prediction += " Result"

            with result_col:
                with st.spinner("Running Model..."):
                    nb_metrics, nb_matrix, nb_pred_plot, nb_teams = \
                        naive_class_application(data=data,
                                                team_map=data_map,
                                                hyperparams=final_params,
                                                features=analysis_stats,
                                                predictor="Result",
                                                predictor_map=data_map,
                                                train_sample=train_size,
                                                plot_name=sample_filter,
                                                prediction_type=game_prediction)

            # ##### Classification Results
            st.subheader("Naive Bayes Prediction Results")
            metrics_col, pred_col = st.columns([4.5, 5.5])
            with metrics_col:
                st.markdown(f"<b><font color=#c3110f>{classification_algo}</font></b> Metrics for Predicting "
                            f"<b><font color=#c3110f>{game_prediction}</font></b>",
                            unsafe_allow_html=True)
                st.table(nb_metrics.style.format(formatter="{:.2%}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#c3110f'), ('color', '#ffffff')]}]))

            with pred_col:
                st.markdown(f"<b><font color=#c3110f>{nb_teams}</font></b> <b>Observed</b> vs "
                            f"<b>Predicted</b> {sample_filter} <b><font color=#c3110f>{game_prediction}</font></b>",
                            unsafe_allow_html=True)
                st.table(
                    nb_matrix.style.format(subset=["Defeat %", "Draw %", "Win %"], formatter="{:.2%}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#c3110f'), ('color', '#ffffff')]}]))

            with result_col:
                st.info(f"{classification_algo} Classifier does not have Feature Coefficients.")
                st.plotly_chart(nb_pred_plot,
                                config=config,
                                use_container_width=True)
            with metrics_col:
                download_plot_prediction = plot_downloader(nb_pred_plot)
                st.download_button(
                    label='📥 Download Prediction Plot',
                    data=download_plot_prediction,
                    file_name=f"{sample_filter.replace('_', '').replace(': ', '_')}_Prediction Plot.html",
                    mime='text/html')

        # ##### ''' K-Nearest Neighbors '''
        elif classification_algo == "K-Nearest Neighbors":

            # ##### Hyperparameters
            with st.sidebar.expander(f"Hyperparameter Tuning"):
                train_size, std_data = hyperparameters_linear(model_type=type_data)
                knn_neighbors = st.slider("Neighbors", min_value=1, max_value=50, value=5)
                knn_weights = st.selectbox(label="Weight", options=["uniform", "distance"])
                knn_algorithm = st.selectbox(label="Algorithm",
                                             options=["auto", "ball_tree", "kd_tree", "brute"])
                knn_metric = st.selectbox("Distance Metric",
                                          ["minkowski", "euclidean", "manhattan"])
                final_params = [knn_neighbors, knn_weights, knn_algorithm, knn_metric]

                # ##### Classification KNN Model
                st.sidebar.subheader("Prediction Options")
                if game_prediction != "Game Result":
                    game_prediction += " Result"

                with result_col:
                    with st.spinner("Running Model..."):
                        knn_metrics, knn_matrix, knn_pred_plot, knn_teams = \
                            knn_class_application(data=data,
                                                  data_type=type_data,
                                                  team_map=data_map,
                                                  hyperparams=final_params,
                                                  features=analysis_stats,
                                                  predictor="Result",
                                                  predictor_map=data_map,
                                                  train_sample=train_size,
                                                  standardize_data=std_data,
                                                  plot_name=sample_filter,
                                                  prediction_type=game_prediction)

            # ##### Classification Results
            st.subheader("KNN Prediction Results")
            metrics_col, pred_col = st.columns([4.5, 5.5])
            with metrics_col:
                st.markdown(f"<b><font color=#c3110f>{classification_algo}</font></b> Metrics for Predicting "
                            f"<b><font color=#c3110f>{game_prediction}</font></b>",
                            unsafe_allow_html=True)
                st.table(knn_metrics.style.format(formatter="{:.2%}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#c3110f'), ('color', '#ffffff')]}]))

            with pred_col:
                st.markdown(f"<b><font color=#c3110f>{knn_teams}</font></b> <b>Observed</b> vs "
                            f"<b>Predicted</b> {sample_filter} <b><font color=#c3110f>{game_prediction}</font></b>",
                            unsafe_allow_html=True)
                st.table(
                    knn_matrix.style.format(subset=["Defeat %", "Draw %", "Win %"], formatter="{:.2%}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#c3110f'), ('color', '#ffffff')]}]))

            with result_col:
                st.info(f"{classification_algo} Classifier does not have Feature Coefficients.")
                st.plotly_chart(knn_pred_plot,
                                config=config,
                                use_container_width=True)
            with metrics_col:
                download_plot_prediction = plot_downloader(knn_pred_plot)
                st.download_button(
                    label='📥 Download Prediction Plot',
                    data=download_plot_prediction,
                    file_name=f"{sample_filter.replace('_', '').replace(': ', '_')}_Prediction Plot.html",
                    mime='text/html')

        # ##### ''' Decision Tree '''
        elif classification_algo == "Decision Tree":
            # ##### Hyperparameters
            with st.sidebar.expander(f"Hyperparameter Tuning"):
                train_size = hyperparameters_nonlinear()
                df_criterion = st.selectbox(label="Criterion",
                                            options=["gini", "entropy", "log_loss"])
                dt_max_depth = int(st.select_slider(label="Max Depth",
                                                    options=np.linspace(2, 10, 9),
                                                    value=5))
                dt_min_sample_split = int(st.select_slider(label="Min Sample Split",
                                                           options=np.linspace(2, 40, 20),
                                                           value=2))
                dt_min_samples_leaf = int(st.select_slider(label="Min Sample Leaf",
                                                           options=np.linspace(1, 30, 30),
                                                           value=1))
                dt_max_leaf = st.select_slider(label="Max Leaf Nodes",
                                               options=[None, 5, 10, 15, 20])
                dt_max_feature = st.selectbox(label="Max Features",
                                              options=[None, "log2", "sqrt"])
                final_params = [df_criterion, dt_max_depth, dt_min_sample_split,
                                dt_min_samples_leaf, dt_max_leaf, dt_max_feature]
                st.sidebar.subheader("Prediction Options")

                # ##### Classification Decision Tree Model
                if game_prediction != "Game Result":
                    game_prediction += " Result"
                tree_plot, tree_metrics, tree_matrix, tree_pred_plot, tree_teams, tree_params = \
                    dt_class_application(data=data,
                                         data_type=type_data,
                                         team_map=data_map,
                                         hyperparams=final_params,
                                         features=analysis_stats,
                                         predictor="Result",
                                         predictor_map=data_map,
                                         train_sample=train_size,
                                         plot_name=sample_filter,
                                         prediction_type=game_prediction)

                with result_col:
                    st.plotly_chart(tree_plot,
                                    config=config,
                                    use_container_width=True)
                with feature_col:
                    download_plot_tree = plot_downloader(tree_plot)
                    st.download_button(
                        label='📥 Download DT Plot',
                        data=download_plot_tree,
                        file_name=f"{sample_filter.replace('_', '').replace(': ', '_')}_Plot DT.html",
                        mime='text/html')

            # ##### Classification Results
            st.subheader("Decision Tree Prediction Results")
            metrics_col, pred_col = st.columns([4.5, 5.5])
            with metrics_col:
                st.markdown(f"<b><font color=#c3110f>{classification_algo}</font></b> Metrics for Predicting "
                            f"<b><font color=#c3110f>{game_prediction}</font></b>",
                            unsafe_allow_html=True)
                st.table(tree_metrics.style.format(formatter="{:.2%}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#c3110f'), ('color', '#ffffff')]}]))

                st.markdown(f"<b><font color=#c3110f>{tree_teams}</font></b> <b>Observed</b> vs "
                            f"<b>Predicted</b> {sample_filter} <b><font color=#c3110f>{game_prediction}</font></b>",
                            unsafe_allow_html=True)
                st.table(
                    tree_matrix.style.format(subset=["Defeat %", "Draw %", "Win %"], formatter="{:.2%}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#c3110f'), ('color', '#ffffff')]}]))

            with pred_col:
                st.plotly_chart(tree_pred_plot,
                                config=config,
                                use_container_width=True)
            with metrics_col:
                download_plot_prediction = plot_downloader(tree_pred_plot)
                st.download_button(
                    label='📥 Download Prediction Plot',
                    data=download_plot_prediction,
                    file_name=f"{sample_filter.replace('_', '').replace(': ', '_')}_Prediction Plot.html",
                    mime='text/html')

            # ##### Displaying the Tree
            st.subheader("Display Decision Tree")
            button_col, description_col = st.columns([1, 10])
            with button_col:
                show_tree = st.checkbox(label="Display Tree",
                                        value=False)
            with description_col:
                st.markdown("<b>Selecting</b> the <b>Show</b> Tree Option will display the Final <b>"
                            "<font color=#c3110f>Decision Tree </font></b> based on the Model the user created.",
                            unsafe_allow_html=True)
            if show_tree:
                final_tree = display_tree(final_model=tree_params[0],
                                          x_train=tree_params[1],
                                          y_train=tree_params[2],
                                          target=tree_params[3],
                                          class_labels=tree_params[4],
                                          features=tree_params[5],
                                          plot_label=tree_params[6],
                                          tree_depth=tree_params[7])

                tree_svg_plot = final_tree.svg()
                tree_show_plot = svg_write(tree_svg_plot)
                st.write(tree_show_plot, unsafe_allow_html=True)

                download_tree = download_button_tree(
                    object_to_download=tree_svg_plot,
                    download_filename=f"Decision Tree - {sample_filter}.svg",
                    button_text="📥 Download Decision Tree")

                st.markdown(download_tree, unsafe_allow_html=True)

        # ##### ''' Random Forest '''
        elif classification_algo == "Random Forest":

            # ##### Hyperparameters
            with st.sidebar.expander(f"Hyperparameter Tuning"):
                train_size = hyperparameters_nonlinear()
                rf_n_estimators = int(st.select_slider(label="No of Trees",
                                                       options=[10, 50, 100, 250, 500, 1000],
                                                       value=100))
                rf_criterion = st.selectbox(label="Criterion",
                                            options=["gini", "entropy", "log_loss"])
                rf_max_depth = int(st.select_slider(label="Max Depth",
                                                    options=np.linspace(2, 10, 9),
                                                    value=5))
                rf_min_sample_split = int(st.select_slider(label="Min Sample Split",
                                                           options=np.linspace(2, 20, 10),
                                                           value=2))
                rf_min_samples_leaf = int(st.select_slider(label="Min Sample Leaf",
                                                           options=np.linspace(1, 10, 10),
                                                           value=1))
                rf_max_leaf = st.select_slider(label="Max Leaf Nodes",
                                               options=[None, 5, 10, 15, 20, 25])
                rf_max_feature = st.selectbox(label="Max Features",
                                              options=[None, "log2", "sqrt"])
                final_params = [rf_n_estimators, rf_criterion, rf_max_depth, rf_min_sample_split,
                                rf_min_samples_leaf, rf_max_leaf, rf_max_feature]
                st.sidebar.subheader("Prediction Options")

                # ##### Classification Random Forest Model
                if game_prediction != "Game Result":
                    game_prediction += " Result"
                rf_plot, rf_metrics, rf_matrix, rf_pred_plot, rf_teams, rf_params = \
                    rf_class_application(data=data,
                                         data_type=type_data,
                                         team_map=data_map,
                                         hyperparams=final_params,
                                         features=analysis_stats,
                                         predictor="Result",
                                         predictor_map=data_map,
                                         train_sample=train_size,
                                         plot_name=sample_filter,
                                         prediction_type=game_prediction)

                with result_col:
                    st.plotly_chart(rf_plot,
                                    config=config,
                                    use_container_width=True)
                with feature_col:
                    download_plot_tree = plot_downloader(rf_plot)
                    st.download_button(
                        label='📥 Download RF Plot',
                        data=download_plot_tree,
                        file_name=f"{sample_filter.replace('_', '').replace(': ', '_')}_Plot RF.html",
                        mime='text/html')

            # ##### Classification Results
            st.subheader("Random Forest Prediction Results")
            metrics_col, pred_col = st.columns([4.5, 5.5])
            with metrics_col:
                st.markdown(f"<b><font color=#c3110f>{classification_algo}</font></b> Metrics for Predicting "
                            f"<b><font color=#c3110f>{game_prediction}</font></b>",
                            unsafe_allow_html=True)
                st.table(rf_metrics.style.format(formatter="{:.2%}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#c3110f'), ('color', '#ffffff')]}]))

                st.markdown(f"<b><font color=#c3110f>{rf_teams}</font></b> <b>Observed</b> vs "
                            f"<b>Predicted</b> {sample_filter} <b><font color=#c3110f>{game_prediction}</font></b>",
                            unsafe_allow_html=True)
                st.table(
                    rf_matrix.style.format(subset=["Defeat %", "Draw %", "Win %"], formatter="{:.2%}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#c3110f'), ('color', '#ffffff')]}]))

            with pred_col:
                st.plotly_chart(rf_pred_plot,
                                config=config,
                                use_container_width=True)
            with metrics_col:
                download_plot_prediction = plot_downloader(rf_pred_plot)
                st.download_button(
                    label='📥 Download Prediction Plot',
                    data=download_plot_prediction,
                    file_name=f"{sample_filter.replace('_', '').replace(': ', '_')}_Prediction Plot.html",
                    mime='text/html')

            # ##### Displaying the Tree
            st.subheader("Display Random Forest Tree")
            button_col, description_col, tree_no_col = st.columns([1, 8, 2])
            with button_col:
                show_tree = st.checkbox(label="Display Tree",
                                        value=False)
            with description_col:
                st.markdown(f"<b>Selecting</b> the <b>Show</b> Tree Option will display the Final <b>"
                            f"<font color=#c3110f>Random Forest Tree </font></b> based on the Model the user created "
                            f"and the <font color=#c3110f>Tree No</font></b> that was selected.",
                            unsafe_allow_html=True)
            if show_tree:
                with tree_no_col:
                    tree_no = st.selectbox("Tree No", options=[i+1 for i in range(rf_n_estimators)])
                final_rf_tree = display_rf_tree(final_model=rf_params[0],
                                                x_train=rf_params[1],
                                                y_train=rf_params[2],
                                                target=rf_params[3],
                                                class_labels=rf_params[4],
                                                features=rf_params[5],
                                                plot_label=f"{rf_params[6]}: Tree No {tree_no}",
                                                tree_depth=rf_params[7],
                                                tree_no=tree_no)

                rf_tree_svg_plot = final_rf_tree.svg()
                rf_tree_show_plot = svg_write(rf_tree_svg_plot)
                st.write(rf_tree_show_plot, unsafe_allow_html=True)

                download_tree = download_button_tree(
                    object_to_download=rf_tree_show_plot,
                    download_filename=f"Random Forest - {sample_filter}.svg",
                    button_text="📥 Download Random Forest Tree")

                st.markdown(download_tree, unsafe_allow_html=True)

        # ##### ''' XgBoost '''
        elif classification_algo == "XgBoost":
            # ##### Hyperparameters
            with st.sidebar.expander(f"Hyperparameter Tuning"):
                train_size = hyperparameters_nonlinear()
                xgb_n_estimators = int(st.select_slider(label="No of Trees",
                                                        options=[10, 50, 100, 250, 500, 1000],
                                                        value=100))
                xgb_booster = st.selectbox(label="Booster",
                                           options=["gbtree", "gblinear"])
                xgb_lr = st.select_slider(label="Learning Rate",
                                          options=[0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
                                          value=0.3)
                xgb_max_depth = int(st.select_slider(label="Max Depth",
                                                     options=np.linspace(2, 10, 9),
                                                     value=6))
                xgb_colsample = st.select_slider(label="Max Features",
                                                 options=[0.5, 0.6, 0.7, 0.8, 0.9, 1],
                                                 value=1)
                xgb_loss = st.selectbox(label="Loss Function",
                                        options=["multi:softprob",
                                                 "multi:softmax"])
                final_params = [xgb_n_estimators, xgb_booster, xgb_lr, xgb_max_depth, xgb_colsample, xgb_loss]
                st.sidebar.subheader("Prediction Options")

                # ##### Classification XgBoost Model
                if game_prediction != "Game Result":
                    game_prediction += " Result"
                xgb_plot, xgb_metrics, xgb_matrix, xgb_pred_plot, xgb_teams, xgb_params = \
                    xgb_class_application(data=data,
                                          data_type=type_data,
                                          team_map=data_map,
                                          hyperparams=final_params,
                                          features=analysis_stats,
                                          predictor="Result",
                                          predictor_map=data_map,
                                          train_sample=train_size,
                                          plot_name=sample_filter,
                                          prediction_type=game_prediction)

                with result_col:
                    st.plotly_chart(xgb_plot,
                                    config=config,
                                    use_container_width=True)
                with feature_col:
                    download_plot_tree = plot_downloader(xgb_plot)
                    st.download_button(
                        label='📥 Download XgB Plot',
                        data=download_plot_tree,
                        file_name=f"{sample_filter.replace('_', '').replace(': ', '_')}_Plot XgB.html",
                        mime='text/html')

            # ##### Classification Results
            st.subheader("XgBoost Prediction Results")
            metrics_col, pred_col = st.columns([4.5, 5.5])
            with metrics_col:
                st.markdown(f"<b><font color=#c3110f>{classification_algo}</font></b> Metrics for Predicting "
                            f"<b><font color=#c3110f>{game_prediction}</font></b>",
                            unsafe_allow_html=True)
                st.table(xgb_metrics.style.format(formatter="{:.2%}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#c3110f'), ('color', '#ffffff')]}]))

                st.markdown(f"<b><font color=#c3110f>{xgb_teams}</font></b> <b>Observed</b> vs "
                            f"<b>Predicted</b> {sample_filter} <b><font color=#c3110f>{game_prediction}</font></b>",
                            unsafe_allow_html=True)
                st.table(
                    xgb_matrix.style.format(subset=["Defeat %", "Draw %", "Win %"], formatter="{:.2%}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #c3110f'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#c3110f'), ('color', '#ffffff')]}]))

            with pred_col:
                st.plotly_chart(xgb_pred_plot,
                                config=config,
                                use_container_width=True)
            with metrics_col:
                download_plot_prediction = plot_downloader(xgb_pred_plot)
                st.download_button(
                    label='📥 Download Prediction Plot',
                    data=download_plot_prediction,
                    file_name=f"{sample_filter.replace('_', '').replace(': ', '_')}_Prediction Plot.html",
                    mime='text/html')

            # ##### Displaying the Tree
            if xgb_booster == "gbtree":
                st.subheader("Display XgBoost Tree")
                button_col, description_col, tree_no_col = st.columns([2, 8, 2])
                with button_col:
                    show_tree = st.checkbox(label="Display Tree",
                                            value=False)
                with description_col:
                    st.markdown(f"<b>Selecting</b> the <b>Show</b> Tree Option will display the Final <b>"
                                f"<font color=#c3110f>XgBoost Tree </font></b> based on the Model the user created "
                                f"and the <font color=#c3110f>Tree No</font></b> that was selected.",
                                unsafe_allow_html=True)
                if show_tree:
                    with tree_no_col:
                        tree_no = st.selectbox("Tree No", options=[i + 1 for i in range(xgb_n_estimators)])
                    final_xgb_tree = display_tree_xgb(final_model=xgb_params[0],
                                                      num_tree=tree_no,
                                                      x_train=xgb_params[1],
                                                      y_train=xgb_params[2],
                                                      target=xgb_params[3],
                                                      class_labels=xgb_params[4],
                                                      features=xgb_params[5],
                                                      plot_label=f"{xgb_params[6]}: Tree No {tree_no}",
                                                      tree_depth=xgb_params[7])

                    xgb_tree_svg_plot = final_xgb_tree.svg()
                    xgb_tree_show_plot = svg_write(xgb_tree_svg_plot)
                    st.write(xgb_tree_show_plot, unsafe_allow_html=True)

                    download_tree = download_button_tree(
                        object_to_download=xgb_tree_show_plot,
                        download_filename=f"XgBoost - {sample_filter}.svg",
                        button_text="📥 Download XgBoost Tree")

                    st.markdown(download_tree, unsafe_allow_html=True)

        st.sidebar.markdown("")
    else:
        with result_col:
            st.info("You need at least 1 feature to run the algorithm!")
