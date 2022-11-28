import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from supabase import create_client


# ##### Supabase Connection
@st.experimental_singleton(show_spinner=False)
def init_connection():
    url = st.secrets["supabase_url"]
    key = st.secrets["supabase_key"]
    return create_client(url, key)


supabase = init_connection()


# ##### Supabase Load Filter Data
@st.experimental_memo(ttl=600, show_spinner=False)
def filter_stats_query():
    """ Return Filter Statistics """
    filter_ml_query = supabase.table('buli_ml_filter_stats').select('*').execute().data
    filter_map_query = supabase.table('buli_ml_map').select('*').execute().data
    data_filter = pd.DataFrame(filter_ml_query)
    data_map = pd.DataFrame(filter_map_query)
    data_map['Option'] = data_map['Statistics'] + ": " + data_map['Label']
    data_map.loc[0, 'Option'] = "Total"

    return data_filter, data_map


@st.experimental_memo(ttl=600, show_spinner=False)
def load_data_unsupervised(data_file, data_filter):
    """ Return Top and All Statistics """
    if data_file == "Top Statistics":
        stats_ml_query = supabase.table('buli_ml_top_stats').select('*').execute().data
    else:
        stats_ml_query = supabase.table('buli_ml_all_stats').select('*').execute().data

    df_stats = pd.DataFrame(stats_ml_query)
    df_stats.drop(columns=["Id"], inplace=True)
    df_filter = data_filter.copy()
    df_filter.drop(columns=["Id"], inplace=True)

    # ##### Transform Data
    sc = StandardScaler()
    df_transformed = sc.fit_transform(df_stats)
    df_transformed = pd.DataFrame(df_transformed, columns=df_stats.columns)

    # ##### Merge Data and File stats
    main_stats = df_stats.columns.to_list()
    final_df = pd.merge(df_transformed, df_filter, left_index=True, right_index=True)
    orig_df = pd.merge(df_stats, df_filter, left_index=True, right_index=True)

    return final_df, orig_df, main_stats


@st.experimental_memo(ttl=600, show_spinner=False)
def load_data_supervised(data_file, data_filter):
    """ Return Top and All Statistics """
    if data_file == "Top Statistics":
        stats_ml_query = supabase.table('buli_ml_top_stats').select('*').execute().data
    else:
        stats_ml_query = supabase.table('buli_ml_all_stats').select('*').execute().data

    df_stats = pd.DataFrame(stats_ml_query)
    df_stats.drop(columns=["Id"], inplace=True)
    df_filter = data_filter.copy()
    df_filter.drop(columns=["Id"], inplace=True)

    # ##### Merge Data and File stats
    main_stats = df_stats.columns.to_list()
    final_df = pd.merge(df_stats, df_filter, left_index=True, right_index=True)

    return final_df, main_stats
