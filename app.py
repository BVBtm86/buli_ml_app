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
            background-image: linear-gradient(to right, #d5e3d6 , #0db518);
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
        st.markdown("<h1><font color=#c3110f>Bundesliga</font> <font color=#6600cc>Machine Learning</font> "
                    "Application</h1>", unsafe_allow_html=True)

    # ##### Option Menu Bar
    menu_options = ['Home', 'Unsupervised Learning', 'Supervised Learning']
    with st.sidebar:
        st.subheader("App Menu")
        analysis_menu = option_menu(menu_title=None,
                                    options=menu_options,
                                    icons=["house-fill", "upc-scan", "ui-checks-grid", "graph-up"],
                                    styles={"container": {"background-color": "#e7e7e7"},
                                            "nav-link": {"--hover-color": "#ffffff"}})

    if analysis_menu == "Home":
        st.markdown(
            "<b><font color=#6600cc>Machine Learning Application</font></b> that allows the user to perform both "
            "<b><font color=#6600cc>Classification</font></b> and <b><font color=#6600cc>Regression</font></b> "
            "analysis thru different <b><font color=#6600cc>Supervised</font></b> algorithms and to learn hidden "
            "patterns from data thru different <b><font color=#6600cc>Unsupervised</font></b> algorithms based on "
            "the <b><font color=#c3110f>Bundesliga</font></b> data. "
            "<br> <br> <b>App Features</b>",
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
        st.markdown(
            "<b><font color=#6600cc>Data</font></b>: is based on the <b><font color=#6600cc>Bundesliga</font></b> "
            "<b><font color=#1e1e1e>2017-2022</font></b> Seasons Team Statistics.",
            unsafe_allow_html=True)
        # ##### Footer Page
        st.markdown(
            f"<b><font color=#6600cc>Data Reference:</font></b><ul><li><a href='https://fbref.com' "
            "style='text-decoration: none; '>Team Stats</a></li><li><a href='https://www.bundesliga.com' "
            "style='text-decoration: none; '>Tracking Stats</a></li>", unsafe_allow_html=True)

    elif analysis_menu == "Unsupervised Learning":
        unsupervised_application()
    elif analysis_menu == "Supervised Learning":
        supervised_application()


if __name__ == '__main__':
    main()
