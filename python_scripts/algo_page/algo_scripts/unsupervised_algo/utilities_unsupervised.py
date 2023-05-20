import pandas as pd
import numpy as np
from io import StringIO, BytesIO
from sklearn.decomposition import PCA

# ##### Info
colors_plot = ["#1a202a", "#2596be", '#c3110f', "#6612cc", '#72bc75',
               "#f7f705", "#FFBF00", "#d2a2f4", "#6e9df0", "#dbf700"]


linkage_info = ["the distance between two clusters is computed as the increase in the 'error sum of squares' "
                "after fusing two clusters into a single cluster",
                "the shortest distance between a pair of observations in two clusters",
                "the distance is measured between the farthest pair of observations in two clusters",
                "the distance between each pair of observations in each cluster are added up and divided by the number "
                "of pairs to get an average inter-cluster distance"]

metric_info = ["the distance between between two points in Euclidean space is the length of a line segment between "
               "the two points",
               "the sum of absolute difference between the measures in all dimensions of two points",
               "a metric in a normed vector space which can be considered as a generalization of both the Euclidean "
               "and Manhattan distance"]

sig_markers = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]


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
    writer.close()
    processed_data = output.getvalue()
    return processed_data


def cluster_pca(data, variables, var_filter, code_filter):
    # ##### Filter Data
    filter_df = data.loc[(data[var_filter] == code_filter), variables].reset_index(drop=True)

    # ##### Final Principal Component Analysis
    pca_final = PCA(n_components=len(variables))
    pca_final.fit(filter_df)
    pca_scores = pd.DataFrame(pca_final.transform(filter_df),
                              columns=[f"PCA {i}" for i in range(1, len(variables) + 1)])
    final_pca_no = np.sum(np.array(pca_final.explained_variance_ >= 1))

    # ##### Create Final PCA Data
    stats_filter_df = data.loc[(data[var_filter] == code_filter)].reset_index(drop=True)
    stats_filter_df.drop(columns=variables, inplace=True)
    final_pca_df = pca_scores.iloc[:, :final_pca_no]
    pca_features = final_pca_df.columns.to_list()
    final_pca_df = pd.merge(left=final_pca_df,
                            right=stats_filter_df,
                            left_index=True,
                            right_index=True)
    return final_pca_df, pca_features
