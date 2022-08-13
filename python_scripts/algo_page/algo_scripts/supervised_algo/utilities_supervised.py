import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO, BytesIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, \
    r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px

# ##### Info
colors_plot = ["#6612cc", '#72bc75', '#c3110f', "#2596be", "#1a202a",
               "#d2a2f4", "#6e9df0"]

target_color = ["#c3110f", "#2596be", "#6612cc"]

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


# ##### Load Data
@st.cache
def load_data_supervised():
    df_stats = pd.read_excel("./data/Supervised Bundesliga Team Statistics.xlsx", sheet_name=0)
    df_filter = pd.read_excel("./data/Bundesliga Statistics Filter.xlsx", sheet_name=0)
    df_map = pd.read_excel("./data/Bundesliga Statistics Filter.xlsx", sheet_name=1)

    # ##### Merge Data and File stats
    main_stats = df_stats.columns.to_list()
    final_df = pd.merge(df_stats, df_filter, left_index=True, right_index=True)
    df_map['Option'] = df_map['Statistics'] + ": " + df_map['Label']
    df_map.loc[0, 'Option'] = "Total"

    return final_df, df_map, main_stats


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
        final_df.drop(columns=['Home Venue', 'Away Total', 'Away Season', 'Away Season Stage', 'Away Venue'],
                      inplace=True)
        final_df.rename(columns={'Home Total': 'Total', 'Home Season': 'Season', 'Home Season Stage': 'Season Stage',
                                 'Home Result': 'Result', 'Home Team': 'Team'},
                        inplace=True)

        # ##### Create Features
        features_remove = []
        for col in features:
            final_df[col] = final_df[f"Home {col}"] - final_df[f"Away {col}"]
            features_remove.append(f"Home {col}")
            features_remove.append(f"Away {col}")

    elif type_game == "Away Team":
        final_df.drop(columns=['Away Venue', 'Home Total', 'Home Season', 'Home Season Stage', 'Home Venue'],
                      inplace=True)
        final_df.rename(columns={'Away Total': 'Total', 'Away Season': 'Season', 'Away Season Stage': 'Season Stage',
                                 'Away Result': 'Result', 'Away Team': 'Team'},
                        inplace=True)

        # ##### Create Features
        features_remove = []
        for col in features:
            final_df[col] = final_df[f"Away {col}"] - final_df[f"Home {col}"]
            features_remove.append(f"Home {col}")
            features_remove.append(f"Away {col}")

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

    pred_df = data.loc[(data[var_filter] == code_filter), [dep_var, 'Team']].reset_index(drop=True)
    final_pca_df = pd.merge(left=final_pca_df, right=pred_df, left_index=True, right_index=True)

    return final_pca_df, final_pca_features


# ##### Hyperparameters
def hyperparameters_linear(model_type):
    # ##### Hyperparameter Tuning
    train_sample = st.slider("% of Sample to use for Training",
                             min_value=1,
                             max_value=99,
                             value=80,
                             step=1,
                             format="%d%%")
    train_sample = np.round(train_sample / 100, 2)
    if model_type == "Original Data":
        standardize_data = st.selectbox("Standardize Data", ["No", "Yes"])
    else:
        standardize_data = "No"

    return train_sample, standardize_data


def hyperparameters_nonlinear():
    # ##### Hyperparameter Tuning
    train_sample = st.slider("% of Sample to use for Training",
                             min_value=1,
                             max_value=99,
                             value=80,
                             step=1,
                             format="%d%%")

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
                                                   y_pred=data_metric['Predicted'].values),
                                  columns=pred_labels,
                                  index=pred_labels)

    # ##### Percentage Matrix
    final_perc_df = pd.DataFrame(confusion_matrix(y_true=data_metric['Observed'].values,
                                                  y_pred=data_metric['Predicted'].values,
                                                  normalize='true'),
                                 columns=pred_labels,
                                 index=pred_labels)

    # ##### Final Matrix
    final_matrix_df = pd.merge(left=final_count_df,
                               right=final_perc_df,
                               left_index=True,
                               right_index=True)
    final_matrix_df.columns = ["Defeat #", "Draw #", "Win #", "Defeat %", "Draw %", "Win %"]
    final_matrix_df = final_matrix_df[["Defeat #", "Defeat %", "Draw #", "Draw %", "Win #", "Win %"]]

    return final_matrix_df


def regression_metrics(y_train, y_train_pred, y_test, y_test_pred):
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

    return final_metrics


def plot_y_reg(data, y, y_pred, plot_title, pred_label, filter_team, filter_name, prediction_type, plot_features=False):

    # ##### Add Prediction to Data
    data_plot = data.copy()
    data_plot[f"Observed {pred_label}"] = y
    data_plot[f"Predicted {pred_label}"] = y_pred

    # ##### Filter if needed
    if filter_team != "All Teams":
        data_plot['Team'] = data_plot['Team'].map(filter_name)
        data_plot = data_plot[data_plot['Team'] == filter_team].reset_index(drop=True)

    # #### Plot Predictions
    if plot_features:
        plot_height = 500
    else:
        plot_height = 550
    plot_y = px.scatter(data_plot,
                        x=f"Observed {pred_label}",
                        y=f"Predicted {pred_label}",
                        title=f"{plot_title} Games <b>Observed {pred_label}</b> vs <b>Predicted {pred_label}</b> "
                              f"for <b>{filter_team}</b> by <b>{prediction_type}</b>",
                        trendline="ols",
                        trendline_color_override="#c3110f")
    plot_y.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                         xaxis_title=f"Observed {pred_label}",
                         yaxis_title=f"Predicted {pred_label}",
                         height=plot_height)
    plot_y.update_traces(marker_color='#6612cc')
    plot_y.update_xaxes(showgrid=False)
    plot_y.update_yaxes(showgrid=False)

    return plot_y


def plot_y_class(data, feature_x, feature_y, pred_var, plot_title, pred_label, filter_team, filter_name,
                 prediction_type, plot_features=False):

    # ##### Add Prediction to Data
    data_plot = data.copy()
    data_plot['Prediction Result'] = pred_var
    data_plot['Prediction Result'] = data_plot['Prediction Result'].map({0: "Defeat", 1: "Draw", 2: "Win"})
    data_plot['Result'] = data_plot['Result'].map({0: "Defeat", 1: "Draw", 2: "Win"})
    data_plot.rename(columns={'Result':"Actual Result"}, inplace=True)

    # ##### Filter if needed
    if filter_team != "All Teams":
        data_plot['Team'] = data_plot['Team'].map(filter_name)
        data_plot = data_plot[data_plot['Team'] == filter_team].reset_index(drop=True)

    # #### Plot Predictions
    if plot_features:
        plot_height = 500
    else:
        plot_height = 550
    plot_y = px.scatter(data_plot,
                        x=feature_x,
                        y=feature_y,
                        color="Prediction Result",
                        color_discrete_map=dict(zip(pred_label, target_color)),
                        title=f"{plot_title} - Predicted <b>{prediction_type}</b> {feature_x} <b>vs</b> {feature_y} for"
                              f" <br><b>{filter_team}</b>",
                        hover_data=["Actual Result"])
    plot_y.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                         xaxis_title=f"{feature_x}",
                         yaxis_title=f"{feature_y}",
                         height=plot_height)
    plot_y.update_xaxes(showgrid=False)
    plot_y.update_yaxes(showgrid=False)

    return plot_y