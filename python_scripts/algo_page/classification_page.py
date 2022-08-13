import streamlit as st
import numpy as np
from python_scripts.algo_page.algo_scripts.supervised_algo.utilities_supervised import class_algo_options, \
    class_algo_name, plot_downloader, data_download, hyperparameters_linear, hyperparameters_nonlinear
from python_scripts.algo_page.algo_scripts.supervised_algo.classification_algo import classification_all_models, \
    linear_class_application, svm_class_application


def classification_application(data, data_map, type_data, game_prediction, sample_filter, dep_var, indep_var):
    config = {'displayModeBar': False}
    # ##### Algorithm Selection
    st.sidebar.subheader("Algorithm")
    classification_algo = st.sidebar.selectbox(label="Classification Algorithm",
                                               options=class_algo_options)

    st.sidebar.markdown("")

    # ##### Algo Description
    if classification_algo == "All":
        st.subheader("Classification Algorithms")
        st.markdown(f"<b><font color=#6600cc>Classification Algorithms</font></b> "
                    f"{class_algo_name[class_algo_options.index(classification_algo)]} <b><font color=#6600cc>"
                    f"{dep_var}</font></b>.", unsafe_allow_html=True)
    else:
        st.markdown(f"<h4>{classification_algo} Analysis</h5>", unsafe_allow_html=True)
        st.markdown(f"<b><font color=#6600cc>{classification_algo}</font></b> "
                    f"{class_algo_name[class_algo_options.index(classification_algo)]}", unsafe_allow_html=True)

    # ##### Features
    feature_col, result_col = st.columns([3, 9])
    with feature_col:
        st.markdown("<b>Features</b>", unsafe_allow_html=True)
        analysis_stats = [col for col in indep_var if st.checkbox(col, True)]

    if len(analysis_stats) > 0:
        # ##### ''' All Classification Models '''
        if classification_algo == "All":
            pass
            # with result_col:
            #     with st.spinner("Running Classification Algorithms ....."):
            #         progress_bar = st.progress(0)
            #         fig_class_plot, class_scores_df, top_class_algo = \
            #             classification_all_models(data=data,
            #                                       data_type=type_data,
            #                                       features=analysis_stats,
            #                                       predictor="Result",
            #                                       progress=progress_bar,
            #                                       all_algo=class_algo_options,
            #                                       plot_name=sample_filter,
            #                                       prediction_type=game_prediction)
            #
            #         progress_bar.empty()
            #         st.plotly_chart(fig_class_plot,
            #                         config=config,
            #                         use_container_width=True)
            #         st.markdown(f"The best performing Classification Algorithms are <b><font color=#6600cc>{top_class_algo[0]}"
            #                     f"</font></b>, <b><font color=#6600cc>{top_class_algo[1]}</font></b> and "
            #                     f"<b><font color=#6600cc>{top_class_algo[2]}</font></b>.", unsafe_allow_html=True)
            #
            #     with feature_col:
            #         download_plot_class = plot_downloader(fig_class_plot)
            #         st.download_button(
            #             label='📥 Download Plot Classification',
            #             data=download_plot_class,
            #             file_name=f"{sample_filter.replace('_', '').replace(': ', '_')}_Plot Classification.html",
            #             mime='text/html')
            #
            #         df_scores_all = data_download(class_scores_df, sample_filter.replace(': ', '_'))
            #         st.download_button(label='📥 Download Data Classification',
            #                            data=df_scores_all,
            #                            file_name=f"{sample_filter.replace(': ', '_')}_Data Results.xlsx")

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
                st.markdown(f"<b><font color=#6600cc>{classification_algo}</font></b> Metrics for Predicting "
                            f"<b><font color=#6600cc>{game_prediction}</font></b>",
                            unsafe_allow_html=True)
                st.table(linear_metrics.style.format(formatter="{:.2%}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #6600cc'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#aeaec5'), ('color', '#ffffff')]}]))

                st.markdown(f"<b><font color=#6600cc>{linear_teams}</font></b> <b>Observed</b> vs "
                            f"<b>Predicted</b> {sample_filter} <b><font color=#6600cc>{game_prediction}</font></b>",
                            unsafe_allow_html=True)
                st.table(linear_matrix.style.format(subset=["Defeat %", "Draw %", "Win %"], formatter="{:.2%}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #6600cc'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#aeaec5'), ('color', '#ffffff')]}]))

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
                st.markdown(f"<b><font color=#6600cc>{classification_algo}</font></b> Metrics for Predicting "
                            f"<b><font color=#6600cc>{game_prediction}</font></b>",
                            unsafe_allow_html=True)
                st.table(svm_metrics.style.format(formatter="{:.2%}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #6600cc'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#aeaec5'), ('color', '#ffffff')]}]))

            if svm_plot is not None:
                with metrics_col:
                    st.markdown(f"<b><font color=#6600cc>{svm_teams}</font></b> <b>Observed</b> vs "
                                f"<b>Predicted</b> {sample_filter} <b><font color=#6600cc>{game_prediction}</font></b>",
                                unsafe_allow_html=True)
                    st.table(svm_matrix.style.format(subset=["Defeat %", "Draw %", "Win %"], formatter="{:.2%}").apply(
                        lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                                   for i in range(len(x))], axis=0).apply(
                        lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #6600cc'
                                   for i in range(len(x))], axis=0).set_table_styles(
                        [{'selector': 'th',
                          'props': [('background-color', '#aeaec5'), ('color', '#ffffff')]}]))
            else:
                with pred_col:
                    st.markdown(f"<b><font color=#6600cc>{svm_teams}</font></b> <b>Observed</b> vs "
                                f"<b>Predicted</b> {sample_filter} <b><font color=#6600cc>{game_prediction}</font></b>",
                                unsafe_allow_html=True)
                    st.table(svm_matrix.style.format(subset=["Defeat %", "Draw %", "Win %"], formatter="{:.2%}").apply(
                        lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                                   for i in range(len(x))], axis=0).apply(
                        lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #6600cc'
                                   for i in range(len(x))], axis=0).set_table_styles(
                        [{'selector': 'th',
                          'props': [('background-color', '#aeaec5'), ('color', '#ffffff')]}]))
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
                train_size = hyperparameters_linear(model_type=type_data)
                nb_smoothing = st.select_slider(label='Smoothing',
                                                options=np.logspace(0, -9, num=10),
                                                value=1.e-09)
            final_params = [nb_smoothing]

        # ##### ''' K-Nearest Neighbors '''
        elif classification_algo == "K-Nearest Neighbors":
            # ##### Hyperparameters
            with st.sidebar.expander(f"Hyperparameter Tuning"):
                train_size, std_data = hyperparameters_linear(model_type=type_data)
                knn_neighbors = st.slider("Neighbors", min_value=1, max_value=30, value=5)
                knn_weights = st.selectbox(label="Weight", options=["uniform", "distance"])
                knn_algorithm = st.selectbox(label="Algorithm",
                                             options=["auto", "ball_tree", "kd_tree", "brute"])
                knn_metric = st.selectbox("Distance Metric",
                                          ["minkowski", "euclidean", "manhattan"])
                final_params = [knn_neighbors, knn_algorithm, knn_metric]

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
                                                           options=np.linspace(2, 20, 10),
                                                           value=2))
                dt_min_samples_leaf = int(st.select_slider(label="Min Sample Leaf",
                                                           options=np.linspace(1, 10, 10),
                                                           value=1))
                dt_max_leaf = st.select_slider(label="Max Leaf Nodes",
                                               options=[None, 5, 10, 15, 20, 25])
                dt_max_feature = st.selectbox(label="Max Features",
                                              options=[None, "log2", "sqrt"])
                final_params = [df_criterion, dt_max_depth, dt_min_sample_split,
                                dt_min_samples_leaf, dt_max_leaf, dt_max_feature]

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

        st.sidebar.markdown("")
    else:
        with result_col:
            st.info("You need at least 1 feature to run the algorithm!")
