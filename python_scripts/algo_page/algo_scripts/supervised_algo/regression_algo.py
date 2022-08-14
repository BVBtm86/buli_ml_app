from python_scripts.algo_page.algo_scripts.supervised_algo.utilities_supervised import colors_plot, \
    filter_model_team_reg, regression_metrics, plot_y_reg
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.simplefilter("ignore", FutureWarning)


def regression_all_models(data, data_type, features, predictor, progress, all_algo, plot_name, prediction_type):
    # ##### Create Model Data
    x = data[features].values
    y = data[predictor].values

    reg_models = [("LR", LinearRegression()), ("SVM", SVR()), ("KNN", KNeighborsRegressor()),
                  ("DT", DecisionTreeRegressor(max_depth=5, random_state=1909)),
                  ("RF", RandomForestRegressor(max_depth=5, random_state=1909)),
                  ("XgB", XGBRegressor(silent=True, verbosity=0, random_state=1909))]

    no_cv = 10
    # ##### Run Cross Val Score
    final_reg_scores = []
    final_reg_plot = pd.DataFrame()
    final_reg_names = []
    no_of_runs = len(reg_models)
    current_run = 0
    for name_model, reg_model in reg_models:
        if data_type == "Original Data":
            transform_sc = StandardScaler()
            if name_model == "LR" or name_model == "SVM" or name_model == "KNN":
                transform_sc.fit(x)
                x_train = transform_sc.transform(x)
            else:
                x_train = x.copy()
        else:
            x_train = x.copy()
        reg_scores = cross_val_score(estimator=reg_model,
                                     X=x_train,
                                     y=y,
                                     cv=no_cv,
                                     scoring="neg_mean_absolute_error")
        final_reg_scores.append(reg_scores)
        final_reg_names.append(name_model)
        # ##### Data for Plot
        reg_df = pd.DataFrame(reg_scores, columns=["Score"])
        reg_df['Algorithm'] = name_model
        final_reg_plot = pd.concat([final_reg_plot, reg_df], axis=0)
        current_run += 1
        progress.progress((current_run / no_of_runs))

    # ##### Final Data
    final_reg_df = pd.DataFrame(final_reg_scores, index=final_reg_names)
    final_reg_df['Average'] = np.mean(final_reg_df, axis=1)

    fig_reg_all = px.box(final_reg_plot,
                         x="Algorithm",
                         y="Score",
                         color="Algorithm",
                         color_discrete_map=dict(zip(final_reg_names, colors_plot[:len(reg_models)])),
                         title=f"<b>{plot_name}</b> Games - {no_cv} Cross Validation Test Scores for "
                               f"<b>Regression</b> Algorithms by <b>{prediction_type}</b>")
    fig_reg_all.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                              height=600)
    fig_reg_all.update_yaxes(title_text="Test Score MAE",
                             tickformat='.2f')

    final_cols = [f'Test Score {i}' for i in range(1, 11)]
    final_cols.append('Average')
    final_reg_df.columns = final_cols

    # ##### Top 3 Algorithms
    best_algo = pd.Series(final_reg_df.nlargest(3, columns=['Average']).index)
    final_best_algo = best_algo.map(dict(zip(final_reg_names, all_algo[1:]))).values

    return fig_reg_all, final_reg_df, final_best_algo


def linear_reg_application(data, data_type, team_map, hyperparams, features, predictor, train_sample, standardize_data,
                           plot_name, prediction_type):
    # ##### Create X, y Feature
    x = data[features].values
    y = data[predictor].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_sample, random_state=1909)

    if (data_type == "Original Data") and (standardize_data == "Yes"):
        sc = StandardScaler()
        sc.fit(x_train)
        x_train = sc.transform(x_train)
        x_test = sc.transform(x_test)
        x = sc.transform(x)

    # ##### Data Model
    if hyperparams[0] == "Linear":
        model = LinearRegression()
    elif hyperparams[0] == "Lasso":
        model = Lasso(alpha=hyperparams[2])
    else:
        model = Ridge(solver=hyperparams[1], alpha=hyperparams[2])

    model.fit(x_train, y_train)

    # ##### Create Final Data
    final_coef_df = pd.DataFrame(model.coef_, index=features)
    final_coef_df.reset_index(inplace=True)
    final_coef_df.columns = ["Features", "Coefficient"]

    # ##### Plot Coefficients
    linear_reg_plot = px.bar(final_coef_df,
                             x="Coefficient",
                             y="Features",
                             orientation='h')

    if data_type == "Original Data":
        plot_height = 750
    else:
        plot_height = 500
    linear_reg_plot.update_layout(
        title=f"<b>{plot_name}</b> Games - Linear Regression Coefficients for <b>{predictor}</b> by "
              f"<b>{prediction_type}</b>",
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(tickformat='.2f',
                   hoverformat=".3f"),
        height=plot_height)
    linear_reg_plot.update_traces(marker_color="#6612cc")

    # ##### Prediction Team Filter
    team_filter, team_names = filter_model_team_reg(data=data,
                                                    data_filter=team_map)

    # ##### Prediction Metrics
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # ##### Regression Metrics
    final_reg_metrics = regression_metrics(y_train=y_train,
                                           y_train_pred=y_train_pred,
                                           y_test=y_test,
                                           y_test_pred=y_test_pred)

    # ##### Plot Observed vs Predicted
    y_pred = model.predict(x)

    plot_prediction = plot_y_reg(data=data,
                                 y=y,
                                 y_pred=y_pred,
                                 plot_title=plot_name,
                                 pred_label=predictor,
                                 filter_team=team_filter,
                                 filter_name=team_names,
                                 prediction_type=prediction_type)

    return linear_reg_plot, final_reg_metrics, plot_prediction


def svm_reg_application(data, data_type, team_map, hyperparams, features, predictor, train_sample, standardize_data,
                        plot_name, prediction_type):
    # ##### Create X, y Feature
    x = data[features].values
    y = data[predictor].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_sample, random_state=1909)

    if (data_type == "Original Data") and (standardize_data == "Yes"):
        sc = StandardScaler()
        sc.fit(x_train)
        x_train = sc.transform(x_train)
        x_test = sc.transform(x_test)
        x = sc.transform(x)

    # ##### Data Model
    if hyperparams[0] == "linear":
        model = SVR(kernel='linear')
    else:
        model = SVR(kernel=hyperparams[0],
                    gamma=hyperparams[1],
                    degree=hyperparams[2],
                    C=hyperparams[3])

    model.fit(x_train, y_train)

    if hyperparams[0] == 'linear':
        model_coef = pd.DataFrame(model.coef_, columns=features).T
        model_coef.columns = ["Coefficient"]
        model_coef = model_coef.reset_index()
        model_coef.columns = ["Features", 'Coefficient']

        # ##### Plot Results
        svm_reg_plot = px.bar(model_coef,
                              x="Coefficient",
                              y='Features')

        if data_type == "Original Data":
            plot_height = 750
        else:
            plot_height = 500
        svm_reg_plot.update_layout(
            title=f"<b>{plot_name}</b> Games - SVM Regression Coefficients for <b>{predictor}</b> by "
                  f"<b>{prediction_type}</b>",
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(tickformat='.2f',
                       hoverformat=".3f"),
            height=plot_height)
        svm_reg_plot.update_traces(marker_color="#6612cc")
    else:
        svm_reg_plot = None

    # ##### Prediction Team Filter
    team_filter, team_names = filter_model_team_reg(data=data,
                                                    data_filter=team_map)

    # ##### Prediction Metrics
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # ##### Regression Metrics
    final_reg_metrics = regression_metrics(y_train=y_train,
                                           y_train_pred=y_train_pred,
                                           y_test=y_test,
                                           y_test_pred=y_test_pred)

    # ##### Plot Observed vs Predicted
    y_pred = model.predict(x)

    plot_prediction = plot_y_reg(data=data,
                                 y=y,
                                 y_pred=y_pred,
                                 plot_title=plot_name,
                                 pred_label=predictor,
                                 filter_team=team_filter,
                                 filter_name=team_names,
                                 prediction_type=prediction_type,
                                 plot_features=None)

    return svm_reg_plot, final_reg_metrics, plot_prediction


def knn_reg_application(data, data_type, team_map, hyperparams, features, predictor, train_sample, standardize_data,
                        plot_name, prediction_type):
    # ##### Create X, y Feature
    x = data[features].values
    y = data[predictor].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_sample, random_state=1909)

    if (data_type == "Original Data") and (standardize_data == "Yes"):
        sc = StandardScaler()
        sc.fit(x_train)
        x_train = sc.transform(x_train)
        x_test = sc.transform(x_test)
        x = sc.transform(x)

    # ##### Data Model
    model = KNeighborsRegressor(n_neighbors=hyperparams[0], weights=hyperparams[1],
                                algorithm=hyperparams[2], metric=hyperparams[3])
    model.fit(x_train, y_train)

    # ##### Prediction Team Filter
    team_filter, team_names = filter_model_team_reg(data=data,
                                                    data_filter=team_map)

    # ##### Prediction Metrics
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # ##### Regression Metrics
    final_reg_metrics = regression_metrics(y_train=y_train,
                                           y_train_pred=y_train_pred,
                                           y_test=y_test,
                                           y_test_pred=y_test_pred)

    # ##### Plot Observed vs Predicted
    y_pred = model.predict(x)

    plot_prediction = plot_y_reg(data=data,
                                 y=y,
                                 y_pred=y_pred,
                                 plot_title=plot_name,
                                 pred_label=predictor,
                                 filter_team=team_filter,
                                 filter_name=team_names,
                                 prediction_type=prediction_type,
                                 plot_features=None)

    return final_reg_metrics, plot_prediction


def tree_reg_application(data, data_type, team_map, hyperparams, features, predictor, train_sample, plot_name,
                         prediction_type):
    # ##### Create X, y Feature
    x = data[features].values
    y = data[predictor].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_sample, random_state=1909)

    # ##### Data Model
    model = DecisionTreeRegressor(criterion=hyperparams[0],
                                  max_depth=hyperparams[1],
                                  min_samples_split=hyperparams[2],
                                  min_samples_leaf=hyperparams[3],
                                  max_leaf_nodes=hyperparams[4],
                                  max_features=hyperparams[5],
                                  random_state=1909)

    model.fit(x_train, y_train)

    # ##### Create Final Data
    final_coef_df = pd.DataFrame(model.feature_importances_, index=features)
    final_coef_df.reset_index(inplace=True)
    final_coef_df.columns = ['Features', 'Importance']
    final_coef_df = final_coef_df.sort_values(by="Importance")

    # ##### Plot Coefficients
    tree_reg_plot = px.bar(final_coef_df,
                           x="Importance",
                           y="Features",
                           orientation='h')

    if data_type == "Original Data":
        plot_height = 750
    else:
        plot_height = 500
    tree_reg_plot.update_layout(
        title=f"<b>{plot_name}</b> Games - Decision Tree Feature Importance by <b>{predictor}</b> by "
              f"<b>{prediction_type}</b>",
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(tickformat='.2f',
                   hoverformat=".3f"),
        height=plot_height)
    tree_reg_plot.update_traces(marker_color="#6612cc")

    # ##### Prediction Team Filter
    team_filter, team_names = filter_model_team_reg(data=data,
                                                    data_filter=team_map)

    # ##### Prediction Metrics
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # ##### Regression Metrics
    final_reg_metrics = regression_metrics(y_train=y_train,
                                           y_train_pred=y_train_pred,
                                           y_test=y_test,
                                           y_test_pred=y_test_pred)

    # ##### Plot Observed vs Predicted
    y_pred = model.predict(x)

    plot_prediction = plot_y_reg(data=data,
                                 y=y,
                                 y_pred=y_pred,
                                 plot_title=plot_name,
                                 pred_label=predictor,
                                 filter_team=team_filter,
                                 filter_name=team_names,
                                 prediction_type=prediction_type)

    # ##### Tree Parameters for Plot
    tree_params = [model, x_train, y_train, predictor, None,
                   features, f"Decision Tree - {plot_name} Games", hyperparams[1]]

    return tree_reg_plot, final_reg_metrics, plot_prediction, tree_params


def rf_reg_application(data, data_type, team_map, hyperparams, features, predictor, train_sample, plot_name,
                       prediction_type):
    # ##### Create X, y Feature
    x = data[features].values
    y = data[predictor].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_sample, random_state=1909)

    # ##### Data Model
    model = RandomForestRegressor(n_estimators=hyperparams[0],
                                  criterion=hyperparams[1],
                                  max_depth=hyperparams[2],
                                  min_samples_split=hyperparams[3],
                                  min_samples_leaf=hyperparams[4],
                                  max_leaf_nodes=hyperparams[5],
                                  max_features=hyperparams[6],
                                  random_state=1909)
    model.fit(x_train, y_train)

    # ##### Create Final Data
    final_coef_df = pd.DataFrame(model.feature_importances_, index=features)
    final_coef_df.reset_index(inplace=True)
    final_coef_df.columns = ['Features', 'Importance']
    final_coef_df = final_coef_df.sort_values(by="Importance")

    # ##### Plot Coefficients
    tree_reg_plot = px.bar(final_coef_df,
                           x="Importance",
                           y="Features",
                           orientation='h')

    if data_type == "Original Data":
        plot_height = 750
    else:
        plot_height = 500
    tree_reg_plot.update_layout(
        title=f"<b>{plot_name}</b> Games - Decision Tree Feature Importance by <b>{predictor}</b> by "
              f"<b>{prediction_type}</b>",
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(tickformat='.2f',
                   hoverformat=".3f"),
        height=plot_height)
    tree_reg_plot.update_traces(marker_color="#6612cc")

    # ##### Prediction Team Filter
    team_filter, team_names = filter_model_team_reg(data=data,
                                                    data_filter=team_map)

    # ##### Prediction Metrics
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # ##### Regression Metrics
    final_reg_metrics = regression_metrics(y_train=y_train,
                                           y_train_pred=y_train_pred,
                                           y_test=y_test,
                                           y_test_pred=y_test_pred)

    # ##### Plot Observed vs Predicted
    y_pred = model.predict(x)

    plot_prediction = plot_y_reg(data=data,
                                 y=y,
                                 y_pred=y_pred,
                                 plot_title=plot_name,
                                 pred_label=predictor,
                                 filter_team=team_filter,
                                 filter_name=team_names,
                                 prediction_type=prediction_type)

    return tree_reg_plot, final_reg_metrics, plot_prediction


def xgb_reg_application(data, data_type, team_map, hyperparams, features, predictor, train_sample, plot_name,
                        prediction_type):
    # ##### Create X, y Feature
    x = data[features].values
    y = data[predictor].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_sample, random_state=1909)

    # ##### Data Model
    model = XGBRegressor(n_estimators=hyperparams[0],
                         booster=hyperparams[1],
                         learning_rate=hyperparams[2],
                         max_depth=hyperparams[3],
                         colsample_bytree=hyperparams[4],
                         objective=hyperparams[5],
                         random_state=1909)
    model.fit(x_train, y_train)

    if hyperparams[1] == "gblinear":
        feature_importance = np.abs(model.feature_importances_) / np.sum(np.abs(model.feature_importances_))
    else:
        feature_importance = model.feature_importances_

    # ##### Create Final Data
    final_coef_df = pd.DataFrame(feature_importance, index=features)
    final_coef_df.reset_index(inplace=True)
    final_coef_df.columns = ['Features', 'Importance']
    final_coef_df = final_coef_df.sort_values(by="Importance")

    # ##### Plot Coefficients
    xgb_reg_plot = px.bar(final_coef_df,
                          x="Importance",
                          y="Features",
                          orientation='h')

    if data_type == "Original Data":
        plot_height = 750
    else:
        plot_height = 500
    xgb_reg_plot.update_layout(
        title=f"<b>{plot_name}</b> Games - XgBoosting Feature Importance by <b>{predictor}</b> by "
              f"<b>{prediction_type}</b>",
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(tickformat='.2f',
                   hoverformat=".3f"),
        height=plot_height)
    xgb_reg_plot.update_traces(marker_color="#6612cc")

    # ##### Prediction Team Filter
    team_filter, team_names = filter_model_team_reg(data=data,
                                                    data_filter=team_map)

    # ##### Prediction Metrics
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # ##### Regression Metrics
    final_reg_metrics = regression_metrics(y_train=y_train,
                                           y_train_pred=y_train_pred,
                                           y_test=y_test,
                                           y_test_pred=y_test_pred)

    # ##### Plot Observed vs Predicted
    y_pred = model.predict(x)

    plot_prediction = plot_y_reg(data=data,
                                 y=y,
                                 y_pred=y_pred,
                                 plot_title=plot_name,
                                 pred_label=predictor,
                                 filter_team=team_filter,
                                 filter_name=team_names,
                                 prediction_type=prediction_type)

    return xgb_reg_plot, final_reg_metrics, plot_prediction
