from python_scripts.algo_page.algo_scripts.supervised_algo.utilities_supervised import colors_plot, target_color, \
    filter_model_team_class, classification_metrics, conf_matrix, plot_y_class
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.simplefilter("ignore", FutureWarning)


def classification_all_models(data, data_type, features, predictor, progress, all_algo, plot_name, prediction_type):
    # ##### Create Model Data
    x = data[features].values
    y = data[predictor].values

    class_models = [("LR", LogisticRegression()), ("SVM", SVC(random_state=1909)), ("NB", GaussianNB()),
                    ("KNN", KNeighborsClassifier()), ("DT", DecisionTreeClassifier(max_depth=5, random_state=1909)),
                    ("RF", RandomForestClassifier(max_depth=5, random_state=1909)),
                    ("XgB", XGBClassifier(silent=True, verbosity=0, random_state=1909))]

    no_cv = 10
    # ##### Run Cross Val Score
    final_class_scores = []
    final_class_plot = pd.DataFrame()
    final_class_names = []
    no_of_runs = len(class_models)
    current_run = 0
    for name_model, class_model in class_models:
        if data_type == "Original Data":
            transform_sc = StandardScaler()
            if name_model == "LR" or name_model == "SVM" or name_model == "KNN":
                transform_sc.fit(x)
                x_train = transform_sc.transform(x)
            else:
                x_train = x.copy()
        else:
            x_train = x.copy()
        class_scores = cross_val_score(estimator=class_model,
                                       X=x_train,
                                       y=y,
                                       cv=StratifiedKFold(n_splits=no_cv),
                                       scoring="accuracy")
        final_class_scores.append(class_scores)
        final_class_names.append(name_model)
        # ##### Data for Plot
        class_df = pd.DataFrame(class_scores, columns=["Score"])
        class_df['Algorithm'] = name_model
        final_class_plot = pd.concat([final_class_plot, class_df], axis=0)
        current_run += 1
        progress.progress((current_run / no_of_runs))

    # ##### Final Data
    final_class_df = pd.DataFrame(final_class_scores, index=final_class_names)
    final_class_df['Average'] = np.mean(final_class_df, axis=1)

    if prediction_type != "Game Result":
        prediction_type += " Result"

    fig_class_all = px.box(final_class_plot,
                           x="Algorithm",
                           y="Score",
                           color="Algorithm",
                           color_discrete_map=dict(zip(final_class_names, colors_plot[:len(class_models)])),
                           title=f"<b>{plot_name}</b> Games - {no_cv} Cross Validation Test Scores for "
                                 f"<b>Classification</b> Algorithms on Predicting <b>{prediction_type}</b>")
    fig_class_all.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                                height=600)
    fig_class_all.update_yaxes(title_text="Test Score Accuracy",
                               tickformat='.0%',
                               hoverformat='.2%')

    final_cols = [f'Test Score {i}' for i in range(1, no_cv + 1)]
    final_cols.append('Average')
    final_class_df.columns = final_cols

    # ##### Top 3 Algorithms
    best_algo = pd.Series(final_class_df.nlargest(3, columns=['Average']).index)
    final_best_algo = best_algo.map(dict(zip(final_class_names, all_algo[2:]))).values

    return fig_class_all, final_class_df, final_best_algo


def linear_class_application(data, data_type, team_map, hyperparams, features, predictor, predictor_map,
                             train_sample, standardize_data, plot_name, prediction_type):

    # ##### Create X, y Feature
    x = data[features]
    y = data[predictor]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_sample, random_state=1909, stratify=y)

    if (data_type == "Original Data") and (standardize_data == "Yes"):
        sc_transform = StandardScaler()
        sc_transform.fit(x_train)
        x_train = sc_transform.transform(x_train)
        x_test = sc_transform.transform(x_test)
        x = sc_transform.transform(x)

    # ##### Data Model
    model = LogisticRegression(solver=hyperparams[0],
                               multi_class=hyperparams[1],
                               penalty=hyperparams[2],
                               C=hyperparams[3],
                               l1_ratio=hyperparams[4])
    model.fit(x_train, y_train)

    # ##### Create Final Data
    class_labels = predictor_map[predictor_map['Statistics'] == predictor]['Label'].values
    final_coef_df = pd.DataFrame(model.coef_, columns=features, index=class_labels)
    final_coef_df = final_coef_df.T

    # ##### Plot Coefficients
    coef_plot = final_coef_df.reset_index().melt(id_vars=['index'], value_vars=final_coef_df.columns)
    coef_plot.columns = ["Features", "Game Result", "Coefficient"]

    linear_class_plot = px.bar(coef_plot,
                               x="Coefficient",
                               y="Features",
                               color="Game Result",
                               barmode='group',
                               orientation='h',
                               color_discrete_map=dict(zip(class_labels, target_color)))

    if data_type == "Original Data":
        plot_height = 750
    else:
        plot_height = 500

    linear_class_plot.update_layout(
        title=f"<b>{plot_name}</b> Games - Logistic Regression Coefficients by <b>{prediction_type}</b>",
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(tickformat='.2f',
                   hoverformat=".3f"),
        height=plot_height)
    if len(features) > 40:
        linear_class_plot.update_layout(yaxis=dict(tickfont=dict(size=8)))

    # ##### Prediction Team Filter
    team_filter, team_names, feature_x_var, feature_y_var = filter_model_team_class(data=data,
                                                                                    data_filter=team_map,
                                                                                    stats=features)

    # ##### Prediction Metrics
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # ##### Classification Metrics
    final_class_metrics = classification_metrics(y_train=y_train,
                                                 y_train_pred=y_train_pred,
                                                 y_test=y_test,
                                                 y_test_pred=y_test_pred)

    # ##### Confusion Matrix
    y_pred = model.predict(x)
    final_class_matrix = conf_matrix(data=data,
                                     y=y,
                                     y_pred=y_pred,
                                     pred_labels=class_labels,
                                     filter_team=team_filter,
                                     filter_name=team_names)

    # ##### Plot prediction
    predict_class_plot = plot_y_class(data=data,
                                      data_filter_map=team_map,
                                      feature_x=feature_x_var,
                                      feature_y=feature_y_var,
                                      pred_var=y_pred,
                                      plot_title=plot_name,
                                      pred_label=class_labels,
                                      filter_team=team_filter,
                                      filter_name=team_names,
                                      prediction_type=prediction_type)

    # ##### Most important Coefficient
    coef_df_imact = pd.DataFrame(np.mean(np.abs(final_coef_df), axis=1), columns=['Coef'])
    coef_impact = coef_df_imact.nlargest(1, 'Coef').index.values[0]

    return linear_class_plot, final_class_metrics, final_class_matrix, predict_class_plot, team_filter, coef_impact


def svm_class_application(data, data_type, team_map, hyperparams, features, predictor, predictor_map,
                          train_sample, standardize_data, plot_name, prediction_type):
    # ##### Create X, y Feature
    x = data[features]
    y = data[predictor]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_sample, random_state=1909, stratify=y)

    if (data_type == "Original Data") and (standardize_data == "Yes"):
        sc_transform = StandardScaler()
        sc_transform.fit(x_train)
        x_train = sc_transform.transform(x_train)
        x_test = sc_transform.transform(x_test)
        x = sc_transform.transform(x)

    # ##### Data Model
    if hyperparams[0] == "linear":
        model = SVC(kernel='linear', random_state=1909)
    else:
        model = SVC(kernel=hyperparams[0],
                    gamma=hyperparams[1],
                    degree=hyperparams[2],
                    C=hyperparams[3],
                    random_state=1909)

    model.fit(x_train, y_train)

    # ##### Create Final Data
    if hyperparams[0] == 'linear':
        class_labels = predictor_map[predictor_map['Statistics'] == predictor]['Label'].values
        final_coef_df = pd.DataFrame(model.coef_, columns=features, index=class_labels)
        final_coef_df = final_coef_df.T

        # ##### Plot Coefficients
        coef_plot = final_coef_df.reset_index().melt(id_vars=['index'], value_vars=final_coef_df.columns)
        coef_plot.columns = ["Features", "Game Result", "Coefficient"]

        svm_class_plot = px.bar(coef_plot,
                                x="Coefficient",
                                y="Features",
                                color="Game Result",
                                barmode='group',
                                orientation='h',
                                color_discrete_map=dict(zip(class_labels, target_color)))

        if data_type == "Original Data":
            plot_height = 750
        else:
            plot_height = 500

        svm_class_plot.update_layout(
            title=f"<b>{plot_name}</b> Games - Support Vector Machine Coefficients by <b>{prediction_type}</b>",
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(tickformat='.2f',
                       hoverformat=".3f"),
            height=plot_height)
        if len(features) > 40:
            svm_class_plot.update_layout(yaxis=dict(tickfont=dict(size=8)))
    else:
        svm_class_plot = None
        final_coef_df = None

    # ##### Prediction Team Filter
    class_labels = predictor_map[predictor_map['Statistics'] == predictor]['Label'].values
    team_filter, team_names, feature_x_var, feature_y_var = filter_model_team_class(data=data,
                                                                                    data_filter=team_map,
                                                                                    stats=features)

    # ##### Prediction Metrics
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # ##### Classification Metrics
    final_class_metrics = classification_metrics(y_train=y_train,
                                                 y_train_pred=y_train_pred,
                                                 y_test=y_test,
                                                 y_test_pred=y_test_pred)

    # ##### Confusion Matrix
    y_pred = model.predict(x)
    final_class_matrix = conf_matrix(data=data,
                                     y=y,
                                     y_pred=y_pred,
                                     pred_labels=class_labels,
                                     filter_team=team_filter,
                                     filter_name=team_names)

    # ##### Plot prediction
    predict_class_plot = plot_y_class(data=data,
                                      data_filter_map=team_map,
                                      feature_x=feature_x_var,
                                      feature_y=feature_y_var,
                                      pred_var=y_pred,
                                      plot_title=plot_name,
                                      pred_label=class_labels,
                                      filter_team=team_filter,
                                      filter_name=team_names,
                                      prediction_type=prediction_type,
                                      plot_features=None)

    if hyperparams[0] == 'linear':
        # ##### Most important Coefficient
        coef_df_imact = pd.DataFrame(np.mean(np.abs(final_coef_df), axis=1), columns=['Coef'])
        coef_impact = coef_df_imact.nlargest(1, 'Coef').index.values[0]
    else:
        coef_impact = None

    return svm_class_plot, final_class_metrics, final_class_matrix, predict_class_plot, team_filter, coef_impact


def naive_class_application(data, team_map, hyperparams, features, predictor, predictor_map,
                            train_sample, plot_name, prediction_type):
    # ##### Create X, y Feature
    x = data[features]
    y = data[predictor]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_sample, random_state=1909, stratify=y)

    # ##### Data Model
    model = GaussianNB(var_smoothing=hyperparams[0])
    model.fit(x_train, y_train)

    # ##### Prediction Team Filter
    class_labels = predictor_map[predictor_map['Statistics'] == predictor]['Label'].values
    team_filter, team_names, feature_x_var, feature_y_var = filter_model_team_class(data=data,
                                                                                    data_filter=team_map,
                                                                                    stats=features)

    # ##### Prediction Metrics
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # ##### Classification Metrics
    final_class_metrics = classification_metrics(y_train=y_train,
                                                 y_train_pred=y_train_pred,
                                                 y_test=y_test,
                                                 y_test_pred=y_test_pred)

    # ##### Confusion Matrix
    y_pred = model.predict(x)
    final_class_matrix = conf_matrix(data=data,
                                     y=y,
                                     y_pred=y_pred,
                                     pred_labels=class_labels,
                                     filter_team=team_filter,
                                     filter_name=team_names)

    # ##### Plot prediction
    predict_class_plot = plot_y_class(data=data,
                                      data_filter_map=team_map,
                                      feature_x=feature_x_var,
                                      feature_y=feature_y_var,
                                      pred_var=y_pred,
                                      plot_title=plot_name,
                                      pred_label=class_labels,
                                      filter_team=team_filter,
                                      filter_name=team_names,
                                      prediction_type=prediction_type,
                                      plot_features=None)

    return final_class_metrics, final_class_matrix, predict_class_plot, team_filter


def knn_class_application(data, data_type, team_map, hyperparams, features, predictor, predictor_map,
                          train_sample, standardize_data, plot_name, prediction_type):
    # ##### Create X, y Feature
    x = data[features]
    y = data[predictor]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_sample, random_state=1909, stratify=y)

    if (data_type == "Original Data") and (standardize_data == "Yes"):
        sc_transform = StandardScaler()
        sc_transform.fit(x_train)
        x_train = sc_transform.transform(x_train)
        x_test = sc_transform.transform(x_test)
        x = sc_transform.transform(x)

    # ##### Data Model
    model = KNeighborsClassifier(n_neighbors=hyperparams[0], weights=hyperparams[1],
                                 algorithm=hyperparams[2], metric=hyperparams[3])
    model.fit(x_train, y_train)

    # ##### Prediction Team Filter
    class_labels = predictor_map[predictor_map['Statistics'] == predictor]['Label'].values
    team_filter, team_names, feature_x_var, feature_y_var = filter_model_team_class(data=data,
                                                                                    data_filter=team_map,
                                                                                    stats=features)

    # ##### Prediction Metrics
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # ##### Classification Metrics
    final_class_metrics = classification_metrics(y_train=y_train,
                                                 y_train_pred=y_train_pred,
                                                 y_test=y_test,
                                                 y_test_pred=y_test_pred)

    # ##### Confusion Matrix
    y_pred = model.predict(x)
    final_class_matrix = conf_matrix(data=data,
                                     y=y,
                                     y_pred=y_pred,
                                     pred_labels=class_labels,
                                     filter_team=team_filter,
                                     filter_name=team_names)

    # ##### Plot prediction
    predict_class_plot = plot_y_class(data=data,
                                      data_filter_map=team_map,
                                      feature_x=feature_x_var,
                                      feature_y=feature_y_var,
                                      pred_var=y_pred,
                                      plot_title=plot_name,
                                      pred_label=class_labels,
                                      filter_team=team_filter,
                                      filter_name=team_names,
                                      prediction_type=prediction_type,
                                      plot_features=None)

    return final_class_metrics, final_class_matrix, predict_class_plot, team_filter


def dt_class_application(data, data_type, team_map, hyperparams, features, predictor, predictor_map,
                         train_sample, plot_name, prediction_type):
    # ##### Create X, y Feature
    x = data[features]
    y = data[predictor]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_sample, random_state=1909, stratify=y)

    # ##### Data Model
    model = DecisionTreeClassifier(criterion=hyperparams[0],
                                   max_depth=hyperparams[1],
                                   min_samples_split=hyperparams[2],
                                   min_samples_leaf=hyperparams[3],
                                   max_leaf_nodes=hyperparams[4],
                                   max_features=hyperparams[5],
                                   random_state=1909)
    model.fit(x_train, y_train)

    # ##### Create Final Data
    class_labels = predictor_map[predictor_map['Statistics'] == predictor]['Label'].values
    final_coef_df = pd.DataFrame(model.feature_importances_, index=features)
    final_coef_df.reset_index(inplace=True)
    final_coef_df.columns = ['Features', 'Importance']
    final_coef_df = final_coef_df.sort_values(by="Importance")

    # ##### Plot Coefficients
    dt_class_plot = px.bar(final_coef_df,
                           x="Importance",
                           y="Features",
                           orientation='h')

    if data_type == "Original Data":
        plot_height = 750
    else:
        plot_height = 500

    dt_class_plot.update_layout(
        title=f"<b>{plot_name}</b> Games - Decision Tree Feature Importance by <b>{prediction_type}</b>",
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickformat='.0%',
                   hoverformat=".2%"),
        height=plot_height)
    dt_class_plot.update_traces(marker_color="#c3110f")
    if len(features) > 40:
        dt_class_plot.update_layout(yaxis=dict(tickfont=dict(size=8)))

    # ##### Prediction Team Filter
    team_filter, team_names, feature_x_var, feature_y_var = filter_model_team_class(data=data,
                                                                                    data_filter=team_map,
                                                                                    stats=features)

    # ##### Prediction Metrics
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # ##### Classification Metrics
    final_class_metrics = classification_metrics(y_train=y_train,
                                                 y_train_pred=y_train_pred,
                                                 y_test=y_test,
                                                 y_test_pred=y_test_pred)

    # ##### Confusion Matrix
    y_pred = model.predict(x)
    final_class_matrix = conf_matrix(data=data,
                                     y=y,
                                     y_pred=y_pred,
                                     pred_labels=class_labels,
                                     filter_team=team_filter,
                                     filter_name=team_names)

    # ##### Plot prediction
    predict_class_plot = plot_y_class(data=data,
                                      data_filter_map=team_map,
                                      feature_x=feature_x_var,
                                      feature_y=feature_y_var,
                                      pred_var=y_pred,
                                      plot_title=plot_name,
                                      pred_label=class_labels,
                                      filter_team=team_filter,
                                      filter_name=team_names,
                                      prediction_type=prediction_type)

    # ##### Tree Parameters for Plot
    tree_params = [model, x_train, y_train, predictor, list(class_labels),
                   features, f"Decision Tree - {plot_name} Games", hyperparams[1]]

    # ##### Most important Feature
    coef_impact = final_coef_df.set_index('Features').nlargest(1, 'Importance').index.values[0]

    return dt_class_plot, final_class_metrics, final_class_matrix, \
        predict_class_plot, team_filter, tree_params, coef_impact


def rf_class_application(data, data_type, team_map, hyperparams, features, predictor, predictor_map,
                         train_sample, plot_name, prediction_type):
    # ##### Create X, y Feature
    x = data[features]
    y = data[predictor]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_sample, random_state=1909, stratify=y)

    # ##### Data Model
    model = RandomForestClassifier(n_estimators=hyperparams[0],
                                   criterion=hyperparams[1],
                                   max_depth=hyperparams[2],
                                   min_samples_split=hyperparams[3],
                                   min_samples_leaf=hyperparams[4],
                                   max_leaf_nodes=hyperparams[5],
                                   max_features=hyperparams[6],
                                   random_state=1909)
    model.fit(x_train, y_train)

    # ##### Create Final Data
    class_labels = predictor_map[predictor_map['Statistics'] == predictor]['Label'].values
    final_coef_df = pd.DataFrame(model.feature_importances_, index=features)
    final_coef_df.reset_index(inplace=True)
    final_coef_df.columns = ['Features', 'Importance']
    final_coef_df = final_coef_df.sort_values(by="Importance")

    # ##### Plot Coefficients
    rf_class_plot = px.bar(final_coef_df,
                           x="Importance",
                           y="Features",
                           orientation='h')

    if data_type == "Original Data":
        plot_height = 750
    else:
        plot_height = 500

    rf_class_plot.update_layout(
        title=f"<b>{plot_name}</b> Games - Random Forest Feature Importance by <b>{prediction_type}</b>",
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickformat='.0%',
                   hoverformat=".2%"),
        height=plot_height)
    rf_class_plot.update_traces(marker_color="#c3110f")
    if len(features) > 40:
        rf_class_plot.update_layout(yaxis=dict(tickfont=dict(size=8)))

    # ##### Prediction Team Filter
    team_filter, team_names, feature_x_var, feature_y_var = filter_model_team_class(data=data,
                                                                                    data_filter=team_map,
                                                                                    stats=features)

    # ##### Prediction Metrics
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # ##### Classification Metrics
    final_class_metrics = classification_metrics(y_train=y_train,
                                                 y_train_pred=y_train_pred,
                                                 y_test=y_test,
                                                 y_test_pred=y_test_pred)

    # ##### Confusion Matrix
    y_pred = model.predict(x)
    final_class_matrix = conf_matrix(data=data,
                                     y=y,
                                     y_pred=y_pred,
                                     pred_labels=class_labels,
                                     filter_team=team_filter,
                                     filter_name=team_names)

    # ##### Plot prediction
    predict_class_plot = plot_y_class(data=data,
                                      data_filter_map=team_map,
                                      feature_x=feature_x_var,
                                      feature_y=feature_y_var,
                                      pred_var=y_pred,
                                      plot_title=plot_name,
                                      pred_label=class_labels,
                                      filter_team=team_filter,
                                      filter_name=team_names,
                                      prediction_type=prediction_type)

    # ##### Tree Parameters for Plot
    tree_params = [model, x_train, y_train, predictor, list(class_labels),
                   features, f"Random Forest - {plot_name} Games", hyperparams[2]]

    # ##### Most important Feature
    coef_impact = final_coef_df.set_index('Features').nlargest(1, 'Importance').index.values[0]

    return rf_class_plot, final_class_metrics, final_class_matrix, predict_class_plot, \
        team_filter, tree_params, coef_impact


def xgb_class_application(data, data_type, team_map, hyperparams, features, predictor, predictor_map,
                          train_sample, plot_name, prediction_type):
    # ##### Create X, y Feature
    x = data[features]
    y = data[predictor]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_sample, random_state=1909, stratify=y)

    # ##### Data Model
    model = XGBClassifier(n_estimators=hyperparams[0],
                          booster=hyperparams[1],
                          learning_rate=hyperparams[2],
                          max_depth=hyperparams[3],
                          colsample_bytree=hyperparams[4],
                          objective=hyperparams[5],
                          random_state=1909)
    model.fit(x_train, y_train)

    if hyperparams[1] == "gblinear":
        feature_importance = \
            np.sum(np.abs(model.feature_importances_), axis=1) / np.sum(np.abs(model.feature_importances_))
    else:
        feature_importance = model.feature_importances_

    # ##### Create Final Data
    class_labels = predictor_map[predictor_map['Statistics'] == predictor]['Label'].values
    final_coef_df = pd.DataFrame(feature_importance, index=features)
    final_coef_df.reset_index(inplace=True)
    final_coef_df.columns = ['Features', 'Importance']
    final_coef_df = final_coef_df.sort_values(by="Importance")

    # ##### Plot Coefficients
    xgb_class_plot = px.bar(final_coef_df,
                            x="Importance",
                            y="Features",
                            orientation='h')

    if data_type == "Original Data":
        plot_height = 750
    else:
        plot_height = 500

    xgb_class_plot.update_layout(
        title=f"<b>{plot_name}</b> Games - XgBoosting Feature Importance by <b>{prediction_type}</b>",
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickformat='.0%',
                   hoverformat=".2%"),
        height=plot_height)
    xgb_class_plot.update_traces(marker_color="#c3110f")
    if len(features) > 40:
        xgb_class_plot.update_layout(yaxis=dict(tickfont=dict(size=8)))

    # ##### Prediction Team Filter
    team_filter, team_names, feature_x_var, feature_y_var = filter_model_team_class(data=data,
                                                                                    data_filter=team_map,
                                                                                    stats=features)

    # ##### Prediction Metrics
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # ##### Classification Metrics
    final_class_metrics = classification_metrics(y_train=y_train,
                                                 y_train_pred=y_train_pred,
                                                 y_test=y_test,
                                                 y_test_pred=y_test_pred)

    # ##### Confusion Matrix
    y_pred = model.predict(x)
    final_class_matrix = conf_matrix(data=data,
                                     y=y,
                                     y_pred=y_pred,
                                     pred_labels=class_labels,
                                     filter_team=team_filter,
                                     filter_name=team_names)

    # ##### Plot prediction
    predict_class_plot = plot_y_class(data=data,
                                      data_filter_map=team_map,
                                      feature_x=feature_x_var,
                                      feature_y=feature_y_var,
                                      pred_var=y_pred,
                                      plot_title=plot_name,
                                      pred_label=class_labels,
                                      filter_team=team_filter,
                                      filter_name=team_names,
                                      prediction_type=prediction_type)

    # ##### Tree Parameters for Plot
    tree_params = [model, x_train, y_train, predictor, list(class_labels),
                   features, f"XgBoosting - {plot_name} Games", hyperparams[3]]

    # ##### Most important Feature
    coef_impact = final_coef_df.set_index('Features').nlargest(1, 'Importance').index.values[0]

    return xgb_class_plot, final_class_metrics, final_class_matrix, \
        predict_class_plot, team_filter, tree_params, coef_impact
