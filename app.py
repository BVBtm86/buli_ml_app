import streamlit as st
from streamlit_option_menu import option_menu
from python_scripts.unsupervised_app import unsupervised_application
from python_scripts.supervised_app import supervised_application
from PIL import Image

logo = Image.open('images/Bundesliga.png')

# # ##### Layout App
st.set_page_config(layout="wide",
                   page_title="Buli ML App",
                   page_icon=logo,
                   initial_sidebar_state="expanded")


# ##### Hide Streamlit info
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ##### Progress Bar Color
st.markdown(
    """
    <style>
        .stProgress > div > div > div > div {
            background-image: linear-gradient(to right, #e5e5e6 , #c3110f);
        }
    </style>""",
    unsafe_allow_html=True,
)

# ##### Button Color
button_color = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #ffffff;
    color:#7575a3;
    width: 100%;
    border-color: #ffffff;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #0db518;
    color:#ffffff;
    border-color: #ffffff;
    font-weight: bold;
    width: 100%;
    }
</style>""", unsafe_allow_html=True)


# ##### Main Application
def main():
    # ##### Login to App
    image_col, title_col, logout_col = st.columns([1, 11, 1])
    with image_col:
        st.markdown("")
        st.image(logo, use_column_width=True)

    with title_col:
        st.markdown("")
        st.markdown("<h1><font color=#c3110f>Bundesliga</font> <font color=#1e1e1e>Machine Learning</font> "
                    "Application</h1>", unsafe_allow_html=True)

    # ##### Option Menu Bar
    menu_options = ['Home', 'Unsupervised Learning', 'Supervised Learning']
    with st.sidebar:
        st.subheader("App Menu")
        analysis_menu = option_menu(menu_title=None,
                                    options=menu_options,
                                    icons=["house-fill", "upc-scan", "ui-checks-grid", "graph-up"],
                                    styles={"container": {"background-color": "#ffffff"},
                                            "nav-link": {"--hover-color": "#e5e5e6"}})
        # ##### Data File to Use
        statistics_used = st.selectbox("Game Stats Data",
                                       options=['Top Statistics', "All Statistics"])

    if analysis_menu == "Home":
        st.markdown(
            "<b><font color=#c3110f>Machine Learning Application</font></b> that allows the user to perform both "
            "<b><font color=#c3110f>Classification</font></b> and <b><font color=#c3110f>Regression</font></b> "
            "analysis thru different <b><font color=#c3110f>Supervised</font></b> algorithms and to learn hidden "
            "patterns from data thru different <b><font color=#c3110f>Unsupervised</font></b> algorithms based on "
            "the <b><font color=#c3110f>Bundesliga</font></b> data. "
            "<br> <br> <font color=#d20614><b>App Features</b></font>",
            unsafe_allow_html=True)
        """
        * Able to run different types of algorithms
          * Unsupervised Learning
          * Supervised Learning Classification
          * Supervised Learning Regression
        * Include / Exclude Features
        * Standardize / Un-standardize data    
        * Filter by different sample groups
        * Tune different Hyperparameters depending on algorithm
        * Charts and Tables based on the analysis available for download
        """
        if statistics_used == "Top Statistics":
            st.markdown(
                "<b><font color=#c3110f>Data</font></b>: is based on the <b><font color=#c3110f>Bundesliga</font></b> "
                "<b><font color=#1e1e1e>2017-2023</font></b> Seasons </b> <b><font color=#c3110f>15</font></b> "
                "Most Important Team Statistics.", unsafe_allow_html=True)
        else:
            st.markdown(
                "<b><font color=#c3110f>Data</font></b>: is based on the <b><font color=#c3110f>Bundesliga</font></b> "
                "<b><font color=#1e1e1e>2017-2023</font></b> Seasons <b><font color=#c3110f>60</font></b> Team "
                "Statistics.", unsafe_allow_html=True)

        # ##### App Description
        st.markdown(
            f"<b><font color=#c3110f>Data Reference</font></b><ul><li><a href='https://fbref.com' "
            "style='text-decoration: none; '>Team Stats</a></li><li><a href='https://www.bundesliga.com' "
            "style='text-decoration: none; '>Tracking Stats</a></li>", unsafe_allow_html=True)

        st.markdown(
                f"<b><font color=#d20614>App Development</font></b><ul><li><a href='https://supabase.com' "
                "style='text-decoration: none; '>Database Storage</a></li><li><a href='https://streamlit.io' "
                "style='text-decoration: none; '>UI Framework</a></li><li>"
                "<a href='https://github.com/BVBtm86/buli_ml_app' style='text-decoration: none; '>"
                "Code Repo</a></li>", unsafe_allow_html=True)

        # ##### Footer Page
        _, fan_club_name, fan_club_logo = st.columns([10, 1, 1])
        with fan_club_name:
            st.markdown(f"<p style='text-align: left;'p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: right;'p>Created By: ", unsafe_allow_html=True)
        with fan_club_logo:
            bvb_ro_logo = Image.open('images/BVB_Romania.png')
            st.image(bvb_ro_logo, width=50)
            st.markdown("@ <b><font color = #d20614 style='text-align: center;'>"
                        "<a href='mailto:omescu.mario.lucian@gmail.com' style='text-decoration: none; '>"
                        "Mario Omescu</a></font></b>", unsafe_allow_html=True)

    elif analysis_menu == "Unsupervised Learning":
        unsupervised_application(data_file=statistics_used)
    elif analysis_menu == "Supervised Learning":
        supervised_application(data_file=statistics_used)


if __name__ == '__main__':
    main()
