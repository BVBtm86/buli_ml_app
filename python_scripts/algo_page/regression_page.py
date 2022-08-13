import numpy as np
import streamlit as st
from python_scripts.algo_page.algo_scripts.supervised_algo.utilities_supervised import reg_algo_options, \
    reg_algo_name, plot_downloader, data_download, hyperparameters_linear, hyperparameters_nonlinear
from python_scripts.algo_page.algo_scripts.supervised_algo.regression_algo import regression_all_models, \
    linear_reg_application, svm_reg_application, knn_reg_application, tree_reg_application


def regression_application(data, data_map, type_data, game_prediction, sample_filter, dep_var, indep_var):
    config = {'displayModeBar': False}
    # ##### Algorithm Selection
    st.sidebar.subheader("Algorithm")
    regression_algo = st.sidebar.selectbox(label="Regression Algorithm",
                                           options=reg_algo_options)
    st.sidebar.markdown("")

    # ##### Algo Description
    if regression_algo == "":
        pass
    elif regression_algo == "All":
        st.subheader("Regression Algorithms")
        st.markdown(f"<b><font color=#6600cc>Regression Algorithms</font></b> "
                    f"{reg_algo_name[reg_algo_options.index(regression_algo)]} <b><font color=#6600cc>"
                    f"{dep_var}</font></b>.", unsafe_allow_html=True)
    else:
        st.markdown(f"<h4>{regression_algo} Analysis</h5>", unsafe_allow_html=True)
        st.markdown(f"<b><font color=#6600cc>{regression_algo}</font></b> "
                    f"{reg_algo_name[reg_algo_options.index(regression_algo)]}", unsafe_allow_html=True)

    # ##### Features
    if regression_algo != "":
        feature_col, result_col = st.columns([3, 9])
        with feature_col:
            st.markdown("<b>Features</b>", unsafe_allow_html=True)
            analysis_stats = [col for col in indep_var if st.checkbox(col, True)]
    else:
        st.info(f"Please select one of the Regression Algorithms from the available options.")
        analysis_stats = [""]

    if len(analysis_stats) > 0:
        # ##### ''' All Regression Models '''
        if regression_algo == "All":
            with result_col:
                with st.spinner("Running Regression Algorithms ....."):
                    progress_bar = st.progress(0)
                    fig_reg_plot, reg_scores_df, top_reg_algo = \
                        regression_all_models(data=data,
                                              data_type=type_data,
                                              features=analysis_stats,
                                              predictor=dep_var,
                                              progress=progress_bar,
                                              all_algo=reg_algo_options,
                                              plot_name=sample_filter,
                                              prediction_type=game_prediction)

                    progress_bar.empty()
                    st.plotly_chart(fig_reg_plot,
                                    config=config,
                                    use_container_width=True)
                    st.markdown(f"The best performing Regression Algorithms are <b><font color=#6600cc>{top_reg_algo[0]}"
                                f"</font></b>, <b><font color=#6600cc>{top_reg_algo[1]}</font></b> and "
                                f"<b><font color=#6600cc>{top_reg_algo[2]}</font></b>.", unsafe_allow_html=True)

                with feature_col:
                    download_plot_reg = plot_downloader(fig_reg_plot)
                    st.download_button(
                        label='📥 Download Plot Regression',
                        data=download_plot_reg,
                        file_name=f"{sample_filter.replace('_', '').replace(': ', '_')}_Plot Regression.html",
                        mime='text/html')

                    df_scores_all = data_download(reg_scores_df, sample_filter.replace(': ', '_'))
                    st.download_button(label='📥 Download Data Regression',
                                       data=df_scores_all,
                                       file_name=f"{sample_filter.replace(': ', '_')}_Data Results.xlsx")

        # ##### ''' Linear Regression '''
        elif regression_algo == "Linear Regression":
            # ##### Hyperparameters
            with st.sidebar.expander(f"Hyperparameter Tuning"):
                train_size, std_data = hyperparameters_linear(model_type=type_data)
                model_type = st.selectbox(label="Linear Model",
                                          options=["Linear", "Lasso", "Ridge"])

                if model_type == "Ridge":
                    solver_param = st.selectbox(label="Solver",
                                                options=["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])
                else:
                    solver_param = None

                if model_type != "Linear":
                    alpha_param = st.select_slider(label="Regularization Strength",
                                                   options=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
                                                   value=1)
                else:
                    alpha_param = None

                final_params = [model_type, solver_param, alpha_param]
                st.sidebar.subheader("Prediction Options")

                # ##### Regression Linear Model
                linear_plot, linear_metrics, linear_pred_plot = \
                    linear_reg_application(data=data,
                                           data_type=type_data,
                                           team_map=data_map,
                                           hyperparams=final_params,
                                           features=analysis_stats,
                                           predictor=dep_var,
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

            # ##### Regression Results
            st.subheader("Regression Prediction Results")
            metrics_col, pred_col = st.columns([4, 6])
            with metrics_col:
                st.markdown(f"<b><font color=#6600cc>{regression_algo}</font></b> Prediction Metrics by <b>"
                            f"<font color=#6600cc>{game_prediction}</font></b>",
                            unsafe_allow_html=True)
                st.table(linear_metrics.style.format(subset=["R2 Score"], formatter="{:.2%}").format(
                    subset=["MAE", "RMSE"], formatter="{:.3f}").apply(
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
        elif regression_algo == "Support Vector Machine":

            # ##### Hyperparameters
            with st.sidebar.expander(f"Hyperparameter Tuning"):
                train_size, std_data = hyperparameters_linear(model_type=type_data)
                svm_kernel = st.selectbox(label="Kernel",
                                          options=["rbf", "poly", 'linear', 'sigmoid'])
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
                st.sidebar.subheader("Prediction Options")

                # ##### Regression SVM Model
                with result_col:
                    with st.spinner("Running Model..."):
                        svm_plot, svm_metrics, svm_pred_plot = \
                            svm_reg_application(data=data,
                                                data_type=type_data,
                                                team_map=data_map,
                                                hyperparams=final_params,
                                                features=analysis_stats,
                                                predictor=dep_var,
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
                            file_name=f"{sample_filter.replace('_', '').replace(': ', '_')}_Plot LR.html",
                            mime='text/html')
                else:
                    with result_col:
                        st.info("Feature Coefficients are only available when the Kernel is Linear.")

            # ##### SVM Results
            st.subheader("SVM Prediction Results")
            metrics_col, pred_col = st.columns([4, 6])
            with metrics_col:
                st.markdown(f"<b><font color=#6600cc>{regression_algo}</font></b> Prediction Metrics by <b>"
                            f"<font color=#6600cc>{game_prediction}</font></b>",
                            unsafe_allow_html=True)
                st.table(svm_metrics.style.format(subset=["R2 Score"], formatter="{:.2%}").format(
                    subset=["MAE", "RMSE"], formatter="{:.3f}").apply(
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

        # ##### ''' K-Nearest Neighbors '''
        elif regression_algo == "K-Nearest Neighbors":

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
                st.sidebar.subheader("Prediction Options")

                # ##### Regression KNN Model
                with result_col:
                    knn_metrics, knn_pred_plot = \
                        knn_reg_application(data=data,
                                            data_type=type_data,
                                            team_map=data_map,
                                            hyperparams=final_params,
                                            features=analysis_stats,
                                            predictor=dep_var,
                                            train_sample=train_size,
                                            standardize_data=std_data,
                                            plot_name=sample_filter,
                                            prediction_type=game_prediction)

            # ##### KNN Results
            st.subheader("KNN Prediction Results")
            metrics_col, pred_col = st.columns([4, 6])
            with metrics_col:
                st.markdown(f"<b><font color=#6600cc>{regression_algo}</font></b> Prediction Metrics by <b>"
                            f"<font color=#6600cc>{game_prediction}</font></b>",
                            unsafe_allow_html=True)
                st.table(knn_metrics.style.format(subset=["R2 Score"], formatter="{:.2%}").format(
                    subset=["MAE", "RMSE"], formatter="{:.3f}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #6600cc'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#aeaec5'), ('color', '#ffffff')]}]))

            with result_col:
                st.info(f"{regression_algo} Regression does not have Feature Coefficients.")
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
        elif regression_algo == "Decision Tree":
            # ##### Hyperparameters
            with st.sidebar.expander(f"Hyperparameter Tuning"):
                train_size = hyperparameters_nonlinear()
                df_criterion = st.selectbox(label="Criterion",
                                            options=["squared_error", "friedman_mse", "absolute_error", "poisson"])
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

                # ##### Regression Decision Tree Model
                tree_plot, tree_metrics, tree_pred_plot = \
                    tree_reg_application(data=data,
                                         data_type=type_data,
                                         team_map=data_map,
                                         hyperparams=final_params,
                                         features=analysis_stats,
                                         predictor=dep_var,
                                         train_sample=train_size,
                                         plot_name=sample_filter,
                                         prediction_type=game_prediction)

                with result_col:
                    st.plotly_chart(tree_plot,
                                    config=config,
                                    use_container_width=True)
                with feature_col:
                    download_plot_linear = plot_downloader(tree_plot)
                    st.download_button(
                        label='📥 Download DT Plot',
                        data=download_plot_linear,
                        file_name=f"{sample_filter.replace('_', '').replace(': ', '_')}_Plot LR.html",
                        mime='text/html')

            # ##### Decision Tree Results
            st.subheader("Decision Tree Prediction Results")
            metrics_col, pred_col = st.columns([4, 6])
            with metrics_col:
                st.markdown(f"<b><font color=#6600cc>{regression_algo}</font></b> Prediction Metrics by <b>"
                            f"<font color=#6600cc>{game_prediction}</font></b>",
                            unsafe_allow_html=True)
                st.table(tree_metrics.style.format(subset=["R2 Score"], formatter="{:.2%}").format(
                    subset=["MAE", "RMSE"], formatter="{:.3f}").apply(
                    lambda x: ['background: #ffffff' if i % 2 == 0 else 'background: #e7e7e7'
                               for i in range(len(x))], axis=0).apply(
                    lambda x: ['color: #1e1e1e' if i % 2 == 0 else 'color: #6600cc'
                               for i in range(len(x))], axis=0).set_table_styles(
                    [{'selector': 'th',
                      'props': [('background-color', '#aeaec5'), ('color', '#ffffff')]}]))
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

        # ##### ''' Random Forest '''
        elif regression_algo == "Random Forest":
            # ##### Hyperparameters
            with st.sidebar.expander(f"Hyperparameter Tuning"):
                train_size = hyperparameters_nonlinear()
                rf_n_estimators = int(st.select_slider(label="No of Trees",
                                                       options=[10, 50, 100, 250, 500, 1000],
                                                       value=100))
                rf_criterion = st.selectbox(label="Criterion",
                                            options=["squared_error", "absolute_error", "poisson"])
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

        # ##### ''' XgBoost '''
        elif regression_algo == "XgBoost":
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
                                        options=["reg:squarederror",
                                                 "reg:squaredlogerror"])
                final_params = [xgb_n_estimators, xgb_booster, xgb_lr, xgb_max_depth, xgb_colsample, xgb_loss]
                st.sidebar.subheader("Prediction Options")

        st.sidebar.markdown("")
    else:
        with result_col:
            st.info("You need at least 1 feature to run the algorithm!")
