import pandas as pd
import numpy as np
import math
import streamlit as st
from io import StringIO, BytesIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, \
    r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import base64
from dtreeviz.trees import dtreeviz
import uuid
import re

# ##### Info
colors_plot = ["#1a202a", "#2596be", '#c3110f', "#6612cc", '#72bc75', "#f7f705", "#FFBF00"]

target_color = ["#1a202a", "#2596be", '#c3110f']

class_algo_options = ["", "All", "Logistic Regression", "Support Vector Machine", "Naive Bayes",
                      "K-Nearest Neighbors", "Decision Tree", "Random Forest", "XgBoost"]

reg_algo_options = ["", "All", "Linear Regression", "Support Vector Machine",
                    "K-Nearest Neighbors", "Decision Tree", "Random Forest", "XgBoost"]

class_algo_name = [
    "",
    "a collection of algorithms to identify the best performing base model for predicting",
    "is a statistical analysis method to predict a binary outcome, such as yes or no, based on "
    "prior observations of a data set.",
    "has the objective of finding a hyperplane in an N-dimensional space(N — the number of "
    "features) that distinctly classifies the data points.",
    "classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with "
    "strong (naive) independence assumptions between the features ",
    "also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses "
    "proximity to make classifications about the grouping of an individual data point.",
    "is a tree-like model that acts as a decision support tool, visually displaying decisions and their"
    " potential outcomes, consequences and costs.",
    "consists of a large number of individual decision trees that operate as an ensemble. Each "
    "individual tree in the random forest spits out a class prediction and the class with the most votes becomes our "
    "model’s prediction.",
    "which stands for Extreme Gradient Boosting, is a scalable, distributed gradient-boosted decision tree "
    "(GBDT) machine learning library. It provides parallel tree boosting."]

reg_algo_name = [
    "",
    "a collection of algorithms to identify the best performing base model on predicting",
    "is used to predict the value of a variable based on the value of another variable or variables.",
    "has the objective of finding a hyperplane in an N-dimensional space(N — the number of "
    "features) that distinctly classifies the data points.",
    "also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses "
    "proximity to make classifications about the grouping of an individual data point.",
    "is a tree-like model that acts as a decision support tool, visually displaying decisions and their"
    " potential outcomes, consequences and costs.",
    "consists of a large number of individual decision trees that operate as an ensemble. Each "
    "individual tree in the random forest spits out a class prediction and the class with the most votes becomes our "
    "model’s prediction.",
    "which stands for Extreme Gradient Boosting, is a scalable, distributed gradient-boosted decision tree "
    "(GBDT) machine learning library. It provides parallel tree boosting."]


# ##### Plot and Data Download
def plot_downloader(fig):
    buffer = StringIO()
    fig.write_html(buffer, include_plotlyjs='cdn')
    html_bytes = buffer.getvalue().encode()
    return html_bytes


def data_download(df, sheet_name):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.reset_index(inplace=True)
    df.to_excel(writer, index=False, sheet_name=sheet_name)
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def plot_all_downloader(fig, filter_text, button_label, plot_type):
    # ##### Create Plot Object
    my_plot = StringIO()
    fig.write_html(my_plot, include_plotlyjs='cdn')
    my_plot = BytesIO(my_plot.getvalue().encode())
    b64 = base64.b64encode(my_plot.read()).decode()
    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
            <style>
                #{button_id} {{
                    background-color: #ffffff;
                    color: #1e1e1e;
                    padding: 0.25em 0.38em;
                    position: relative;
                    text-decoration: none;
                    border-radius: 4px;
                    border-width: 1px;
                    border-style: solid;
                    border-color: #e5e5e6;
                    border-image: initial;
                }} 
                #{button_id}:hover {{
                    background-color: #ffffff;
                    border-color: #c3110f;
                    color: #c3110f;
                }}
            </style> """

    button_text = button_label

    download_filename = f"Supervised {plot_type} - {filter_text}.html"
    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" ' \
                           f'href="data:text/html;charset=utf-8;base64,{b64}">{button_text}</a><br></br>'

    return dl_link


def data_all_downloader(data, filter_text, button_label, plot_type):
    # ##### Create Data Object
    csv_file = data.to_csv(index=False)
    b64 = base64.b64encode(csv_file.encode()).decode()
    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
            <style>
                #{button_id} {{
                    background-color: #ffffff;
                    color: #1e1e1e;
                    padding: 0.25em 0.38em;
                    position: relative;
                    text-decoration: none;
                    border-radius: 4px;
                    border-width: 1px;
                    border-style: solid;
                    border-color: #e5e5e6;
                    border-image: initial;
                }} 
                #{button_id}:hover {{
                    background-color: #ffffff;
                    border-color: #c3110f;
                    color: #c3110f;
                }}
            </style> """

    button_text = button_label
    download_filename = f"Supervised {plot_type} - {filter_text}.csv"
    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/csv;base64,{b64}">' \
                           f'{button_text}</a><br></br>'

    return dl_link


# ##### Data Processing
def home_away_data(data, features, type_game):
    # ##### Filter Home vs Away Team
    home_team_df = data[data['Venue'] == 1].reset_index(drop=True)
    away_team_df = data[data['Venue'] == 2].reset_index(drop=True)

    # ##### Rename Columns
    home_team_df.columns = [f"Home {col}" for col in home_team_df.columns]
    away_team_df.columns = [f"Away {col}" for col in away_team_df.columns]
    final_df = pd.merge(left=home_team_df,
                        right=away_team_df,
                        left_index=True,
                        right_index=True)

    if type_game == "Home Team":
        final_df.drop(columns=['Away Match Day', 'Away Total', 'Away Season', 'Away Season Stage', 'Away Venue',
                               'Away Result', 'Away Opponent', 'Home Opponent'], inplace=True)
        final_df.rename(columns={'Home Match Day': 'Match Day', 'Home Total': 'Total', 'Home Season': 'Season',
                                 'Home Season Stage': 'Season Stage', 'Home Result': 'Result', 'Home Team': 'Team',
                                 'Home Venue': 'Venue', 'Away Team': 'Opponent'}, inplace=True)
        # ##### Create Features
        features_remove = []
        for col in features:
            final_df[col] = final_df[f"Home {col}"] - final_df[f"Away {col}"]
            features_remove.append(f"Home {col}")
            features_remove.append(f"Away {col}")

    elif type_game == "Away Team":
        final_df.drop(columns=['Home Match Day', 'Home Total', 'Home Season', 'Home Season Stage', 'Home Venue',
                               'Home Result', 'Home Opponent', 'Away Opponent'],
                      inplace=True)
        final_df.rename(columns={'Away Match Day': 'Match Day', 'Away Total': 'Total', 'Away Season': 'Season',
                                 'Away Season Stage': 'Season Stage', 'Away Result': 'Result', 'Away Team': 'Team',
                                 'Away Venue': 'Venue', 'Home Team': 'Opponent'}, inplace=True)

        # ##### Create Features
        features_remove = []
        for col in features:
            final_df[col] = final_df[f"Away {col}"] - final_df[f"Home {col}"]
            features_remove.append(f"Home {col}")
            features_remove.append(f"Away {col}")
    else:
        features_remove = []

    final_df.drop(columns=features_remove, inplace=True)

    return final_df


def supervised_pca(data, variables, var_filter, code_filter, dep_var):
    # ##### Standardize Data
    data_raw = data[variables]
    transform_supervised = StandardScaler()
    transformed_data = pd.DataFrame(transform_supervised.fit_transform(data_raw), columns=data_raw.columns)
    df_stats_filter = data.drop(columns=variables)
    final_df = pd.merge(left=transformed_data, right=df_stats_filter, left_index=True, right_index=True)

    # ##### Filter Data
    filter_df = final_df.loc[(final_df[var_filter] == code_filter), variables].reset_index(drop=True)

    # ##### Final Principal Component Analysis
    pca_final = PCA(n_components=len(variables))
    pca_final.fit(filter_df)
    pca_scores = pd.DataFrame(pca_final.transform(filter_df),
                              columns=[f"PCA {i}" for i in range(1, len(variables) + 1)])
    final_pca_no = np.sum(np.array(pca_final.explained_variance_ >= 1))

    # ##### Create Final PCA Data
    final_pca_df = pca_scores.iloc[:, :final_pca_no]
    final_pca_features = final_pca_df.columns.to_list()

    if dep_var == 'Result':
        pred_df = data.loc[(data[var_filter] == code_filter),
                           [dep_var, 'Team', 'Opponent', 'Season',
                            'Season Stage', 'Venue', 'Match Day']].reset_index(drop=True)
    else:
        pred_df = data.loc[(data[var_filter] == code_filter),
                           [dep_var, 'Team', 'Opponent', 'Result', 'Season',
                            'Season Stage', 'Venue', 'Match Day']].reset_index(drop=True)

    final_pca_df = pd.merge(left=final_pca_df, right=pred_df, left_index=True, right_index=True)

    return final_pca_df, final_pca_features


# ##### Hyperparameters
def hyperparameters_linear(model_type, sample_size):
    # ##### Min and Max Train Values
    if sample_size <= 300:
        min_games = math.ceil(30 / sample_size * 100)
        max_games = math.floor((sample_size - 3) / sample_size * 100)
    else:
        min_games = 1
        max_games = 99

    # ##### Hyperparameter Tuning
    train_sample = st.slider("% of Sample to use for Training",
                             min_value=min_games,
                             max_value=max_games,
                             value=80,
                             step=1,
                             format="%d%%")
    train_sample = np.round(train_sample / 100, 2)
    if model_type == "Original Data":
        standardize_data = st.selectbox("Standardize Data", ["No", "Yes"])
    else:
        standardize_data = "No"

    return train_sample, standardize_data


def hyperparameters_nonlinear(sample_size):
    # ##### Min and Max Train Values
    if sample_size <= 300:
        min_games = math.ceil(30 / sample_size * 100)
        max_games = math.floor((sample_size - 3) / sample_size * 100)
    else:
        min_games = 1
        max_games = 99

    # ##### Hyperparameter Tuning
    train_sample = st.slider("% of Sample to use for Training",
                             min_value=min_games,
                             max_value=max_games,
                             value=80,
                             step=1,
                             format="%d%%")
    train_sample = np.round(train_sample / 100, 2)
    return train_sample


# ##### Team Filter
def filter_model_team_class(data, data_filter, stats):
    # ##### Team Filter
    data_copy = data.copy()
    x_features = stats.copy()
    plot_x_var = st.sidebar.selectbox(label="Prediction X Feature",
                                      options=x_features)
    y_features = [col for col in stats if col != plot_x_var]
    plot_y_var = st.sidebar.selectbox(label="Prediction Y Feature",
                                      options=y_features)

    team_names = dict(zip(data_filter[data_filter["Statistics"] == "Team"]['Code'].values,
                          data_filter[data_filter["Statistics"] == "Team"]['Label'].values))
    available_teams = ["All Teams"]
    available_teams.extend(list(data_copy['Team'].map(team_names).unique()))

    team_app = st.sidebar.selectbox(label="Team Filter",
                                    options=available_teams)

    return team_app, team_names, plot_x_var, plot_y_var


def filter_model_team_reg(data, data_filter):
    # ##### Team Filter
    data_copy = data.copy()

    team_names = dict(zip(data_filter[data_filter["Statistics"] == "Team"]['Code'].values,
                          data_filter[data_filter["Statistics"] == "Team"]['Label'].values))
    available_teams = ["All Teams"]
    available_teams.extend(list(data_copy['Team'].map(team_names).unique()))

    team_app = st.sidebar.selectbox(label="Team Filter",
                                    options=available_teams)

    return team_app, team_names


# ##### Model Metrics
def classification_metrics(y_train, y_train_pred, y_test, y_test_pred):
    metric_average = "weighted"

    # ##### Train Metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average=metric_average)
    train_recall = recall_score(y_train, y_train_pred, average=metric_average)
    train_f1 = f1_score(y_train, y_train_pred, average=metric_average)

    # ##### Test Metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average=metric_average)
    test_recall = recall_score(y_test, y_test_pred, average=metric_average)
    test_f1 = f1_score(y_test, y_test_pred, average=metric_average)

    # ##### Total Metrics
    total_class_metrics = [[train_accuracy, train_precision, train_recall, train_f1],
                           [test_accuracy, test_precision, test_recall, test_f1]]

    final_metrics = pd.DataFrame(total_class_metrics, columns=['Accuracy', "Precision", "Recall", "F1 Score"],
                                 index=["Train Games", "Test Games"])

    return final_metrics


def conf_matrix(data, y, y_pred, pred_labels, filter_team, filter_name):
    # ##### Data
    data_metric = data.copy()
    data_metric['Observed'] = y
    data_metric['Predicted'] = y_pred

    # ##### Filter if needed
    if filter_team != "All Teams":
        data_metric['Team'] = data_metric['Team'].map(filter_name)
        data_metric = data_metric[data_metric['Team'] == filter_team].reset_index(drop=True)

    # ##### Count Matrix
    final_count_df = pd.DataFrame(confusion_matrix(y_true=data_metric['Observed'].values,
                                                   y_pred=data_metric['Predicted'].values,
                                                   labels=[0, 1, 2]))
    final_count_df.columns = final_count_df.columns.map(dict(zip([0, 1, 2], pred_labels)))
    final_count_df.index = final_count_df.index.map(dict(zip([0, 1, 2], pred_labels)))

    # ##### Percentage Matrix
    final_perc_df = pd.DataFrame(confusion_matrix(y_true=data_metric['Observed'].values,
                                                  y_pred=data_metric['Predicted'].values,
                                                  normalize='true',
                                                  labels=[0, 1, 2]))
    final_perc_df.columns = final_perc_df.columns.map(dict(zip([0, 1, 2], pred_labels)))
    final_perc_df.index = final_perc_df.index.map(dict(zip([0, 1, 2], pred_labels)))

    # ##### Final Matrix
    final_matrix_df = pd.merge(left=final_count_df,
                               right=final_perc_df,
                               left_index=True,
                               right_index=True)
    final_matrix_df.columns = ["Defeat #", "Draw #", "Win #", "Defeat %", "Draw %", "Win %"]
    final_matrix_df = final_matrix_df[["Defeat #", "Defeat %", "Draw #", "Draw %", "Win #", "Win %"]]

    return final_matrix_df


def regression_metrics(data, y_train, y_train_pred, y_test, y_test_pred, team_metric, pred_label):
    # #### Train Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    if y_test is not None:
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
    else:
        test_r2 = None
        test_rmse = None
        test_mae = None

    # ##### Total Metrics
    total_reg_metrics = [[train_r2, train_rmse, train_mae], [test_r2, test_rmse, test_mae]]

    final_metrics = pd.DataFrame(total_reg_metrics, columns=['R2 Score', "MAE", "RMSE"],
                                 index=["Train Games", "Test Games"])

    # ##### Team Metric
    if team_metric == "All Teams":
        team_df = data.copy()
    else:
        team_df = data[data['Team'] == team_metric]
    team_y = team_df[pred_label].values
    team_y_pred = team_df['y_pred'].values
    team_r2 = r2_score(team_y, team_y_pred)
    team_rmse = mean_squared_error(team_y, team_y_pred)
    team_mae = mean_absolute_error(team_y, team_y_pred)
    team_reg_metrics = [[team_r2, team_rmse, team_mae]]
    team_metrics = pd.DataFrame(team_reg_metrics, columns=['R2 Score', "MAE", "RMSE"],
                                index=[f"{team_metric} Results"])

    return final_metrics, team_metrics


# ##### Plotting Predictions
def plot_y_reg(data, data_filter_map, y, y_pred, plot_title, pred_label, filter_team, filter_name, prediction_type,
               plot_features=False):
    # ##### Add Prediction to Data
    data_plot = data.copy()
    data_plot[f"Observed {pred_label}"] = y
    data_plot[f"Predicted {pred_label}"] = y_pred

    # ##### Filter if needed
    if filter_team != "All Teams":
        data_plot['Team'] = data_plot['Team'].map(filter_name)
        data_plot = data_plot[data_plot['Team'] == filter_team].reset_index(drop=True)

        # ##### Map Data with labels
        team_map = dict(zip(data_filter_map[data_filter_map['Statistics'] == 'Team']['Code'].values,
                            data_filter_map[data_filter_map['Statistics'] == 'Team']['Label'].values))
        season_map = dict(zip(data_filter_map[data_filter_map['Statistics'] == 'Season']['Code'].values,
                              data_filter_map[data_filter_map['Statistics'] == 'Season']['Label'].values))
        stage_map = dict(zip(data_filter_map[data_filter_map['Statistics'] == 'Season Stage']['Code'].values,
                             data_filter_map[data_filter_map['Statistics'] == 'Season Stage']['Label'].values))
        venue_map = dict(zip(data_filter_map[data_filter_map['Statistics'] == 'Venue']['Code'].values,
                             data_filter_map[data_filter_map['Statistics'] == 'Venue']['Label'].values))
        data_plot['Opponent'] = data_plot['Opponent'].map(team_map)
        data_plot['Season'] = data_plot['Season'].map(season_map)
        data_plot['Season Stage'] = data_plot['Season Stage'].map(stage_map)
        data_plot['Venue'] = data_plot['Venue'].map(venue_map)

    result_map = dict(zip(data_filter_map[data_filter_map['Statistics'] == 'Result']['Code'].values,
                          data_filter_map[data_filter_map['Statistics'] == 'Result']['Label'].values))
    data_plot['Result'] = data_plot['Result'].map(result_map)

    # #### Plot Predictions
    if plot_features:
        plot_height = 500
    else:
        plot_height = 550
    if filter_team != "All Teams":
        plot_y = px.scatter(data_plot,
                            x=f"Observed {pred_label}",
                            y=f"Predicted {pred_label}",
                            color='Result',
                            color_discrete_map=dict(zip(['Defeat', 'Draw', 'Win'], target_color)),
                            title=f"{plot_title} Games <b>Observed {pred_label}</b> vs <b>Predicted {pred_label}</b> "
                                  f"for <b>{filter_team}</b> by <b>{prediction_type}</b>",
                            trendline="ols",
                            hover_data=['Team', 'Opponent', 'Season', 'Season Stage', 'Venue',
                                        'Match Day'])
    else:
        plot_y = px.scatter(data_plot,
                            x=f"Observed {pred_label}",
                            y=f"Predicted {pred_label}",
                            color='Result',
                            color_discrete_map=dict(zip(['Defeat', 'Draw', 'Win'], target_color)),
                            title=f"{plot_title} Games <b>Observed {pred_label}</b> vs <b>Predicted {pred_label}</b> "
                                  f"for <b>{filter_team}</b> by <b>{prediction_type}</b>",
                            trendline="ols")

    plot_y.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                         xaxis_title=f"Observed {pred_label}",
                         yaxis_title=f"Predicted {pred_label}",
                         height=plot_height)
    plot_y.update_xaxes(showgrid=False)
    plot_y.update_yaxes(showgrid=False)

    return plot_y


def plot_y_class(data, data_filter_map, feature_x, feature_y, pred_var, plot_title, pred_label, filter_team,
                 filter_name, prediction_type, plot_features=False):

    # ##### Add Prediction to Data
    data_plot = data.copy()
    data_plot['Prediction Result'] = pred_var
    data_plot['Prediction Result'] = data_plot['Prediction Result'].map({0: "Defeat", 1: "Draw", 2: "Win"})
    data_plot['Result'] = data_plot['Result'].map({0: "Defeat", 1: "Draw", 2: "Win"})
    data_plot.rename(columns={'Result': "Actual Result"}, inplace=True)

    # ##### Filter if needed
    if filter_team != "All Teams":
        data_plot['Team'] = data_plot['Team'].map(filter_name)
        data_plot = data_plot[data_plot['Team'] == filter_team].reset_index(drop=True)

        # ##### Map Data with labels
        team_map = dict(zip(data_filter_map[data_filter_map['Statistics'] == 'Team']['Code'].values,
                            data_filter_map[data_filter_map['Statistics'] == 'Team']['Label'].values))
        season_map = dict(zip(data_filter_map[data_filter_map['Statistics'] == 'Season']['Code'].values,
                              data_filter_map[data_filter_map['Statistics'] == 'Season']['Label'].values))
        stage_map = dict(zip(data_filter_map[data_filter_map['Statistics'] == 'Season Stage']['Code'].values,
                             data_filter_map[data_filter_map['Statistics'] == 'Season Stage']['Label'].values))
        venue_map = dict(zip(data_filter_map[data_filter_map['Statistics'] == 'Venue']['Code'].values,
                             data_filter_map[data_filter_map['Statistics'] == 'Venue']['Label'].values))
        data_plot['Opponent'] = data_plot['Opponent'].map(team_map)
        data_plot['Season'] = data_plot['Season'].map(season_map)
        data_plot['Season Stage'] = data_plot['Season Stage'].map(stage_map)
        data_plot['Venue'] = data_plot['Venue'].map(venue_map)

    # #### Plot Predictions
    if plot_features:
        plot_height = 500
    else:
        plot_height = 550

    if filter_team != "All Teams":
        plot_y = px.scatter(data_plot,
                            x=feature_x,
                            y=feature_y,
                            color="Prediction Result",
                            color_discrete_map=dict(zip(pred_label, target_color)),
                            title=f"{plot_title} - Predicted <b>{prediction_type}</b> {feature_x} <b>vs</b> {feature_y}"
                                  f" for <br><b>{filter_team}</b>",
                            hover_data=["Actual Result", 'Team', 'Opponent', 'Season', 'Season Stage', 'Venue',
                                        'Match Day'])
    else:
        plot_y = px.scatter(data_plot,
                            x=feature_x,
                            y=feature_y,
                            color="Prediction Result",
                            color_discrete_map=dict(zip(pred_label, target_color)),
                            title=f"{plot_title} - Predicted <b>{prediction_type}</b> {feature_x} <b>vs</b> {feature_y}"
                                  f" for <br><b>{filter_team}</b>",
                            hover_data=["Actual Result"])
    plot_y.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                         xaxis_title=f"{feature_x}",
                         yaxis_title=f"{feature_y}",
                         height=plot_height)
    plot_y.update_xaxes(showgrid=False)
    plot_y.update_yaxes(showgrid=False)

    return plot_y


# ##### Tree Plotting
def display_tree(final_model, x_train, y_train, target, class_labels, features, plot_label, tree_depth):
    # ##### Display Parameters
    if tree_depth == 2:
        scale_fig = 1.5
    elif tree_depth == 3:
        scale_fig = 1.4
    elif tree_depth == 4:
        scale_fig = 1.1
    else:
        scale_fig = 1

    position_plot = "LR"

    tree_graph = dtreeviz(final_model,
                          x_data=x_train,
                          y_data=y_train,
                          target_name=target,
                          feature_names=features,
                          class_names=class_labels,
                          orientation=position_plot,
                          title=plot_label,
                          show_node_labels=True,
                          fontname="Arial",
                          title_fontsize=16,
                          label_fontsize=10,
                          ticks_fontsize=8,
                          scale=scale_fig,
                          colors={'scatter_marker': '#c3110f'})
    return tree_graph


def display_rf_tree(final_model, x_train, y_train, target, class_labels, features, plot_label, tree_depth, tree_no):
    # ##### Display Parameters
    if tree_depth == 2:
        scale_fig = 1.5
    elif tree_depth == 3:
        scale_fig = 1.4
    elif tree_depth == 4:
        scale_fig = 1.1
    else:
        scale_fig = 1

    position_plot = "LR"

    tree_graph = dtreeviz(final_model.estimators_[tree_no - 1],
                          x_data=x_train,
                          y_data=y_train,
                          target_name=target,
                          feature_names=features,
                          class_names=class_labels,
                          orientation=position_plot,
                          title=plot_label,
                          show_node_labels=True,
                          fontname="Arial",
                          title_fontsize=16,
                          label_fontsize=10,
                          ticks_fontsize=8,
                          scale=scale_fig,
                          colors={'scatter_marker': '#c3110f'})
    return tree_graph


def display_tree_xgb(final_model, num_tree, x_train, y_train, target, class_labels, features, plot_label, tree_depth):

    # ##### Display Parameters
    if tree_depth == 2:
        scale_fig = 1.5
    else:
        scale_fig = 0.5

    position_plot = "LR"

    tree_graph = dtreeviz(final_model,
                          tree_index=num_tree,
                          x_data=x_train,
                          y_data=y_train,
                          target_name=target,
                          feature_names=features,
                          class_names=class_labels,
                          orientation=position_plot,
                          title=plot_label,
                          show_node_labels=True,
                          fontname="Arial",
                          title_fontsize=16,
                          label_fontsize=10,
                          ticks_fontsize=8,
                          scale=scale_fig,
                          colors={'scatter_marker': '#c3110f'})

    return tree_graph


def svg_write(svg, center=True):
    """
    Disable center to left-margin align like other objects.
    """
    # Encode as base 64
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")

    # Add some CSS on top
    css_justify = "center" if center else "left"
    css = f'<p style="text-align:center; display: flex; justify-content: {css_justify};">'
    html = f'{css}<img src="data:image/svg+xml;base64,{b64}"/>'

    # Write the HTML
    return html


def download_button_tree(object_to_download, download_filename, button_text):
    b64 = base64.b64encode(object_to_download.encode()).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
            <style>
                #{button_id} {{
                    background-color: #ffffff;
                    color: #1e1e1e;
                    padding: 0.25em 0.38em;
                    position: relative;
                    text-decoration: none;
                    border-radius: 4px;
                    border-width: 1px;
                    border-style: solid;
                    border-color: #e5e5e6;
                    border-image: initial;
                }} 
                #{button_id}:hover {{
                    background-color: #ffffff;
                    border-color: #c3110f;
                    color: #c3110f;
                }}
            </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/svg;base64,{b64}">' \
                           f'{button_text}</a><br></br>'

    return dl_link
