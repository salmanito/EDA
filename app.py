import streamlit as st
from data_preprocessing import data_preprocessing_page
from visualization import visualization_page
from dashboard import dashboard_page

# Streamlit UI setup (this must be the first Streamlit command)
st.set_page_config(layout="wide")

def main():
    st.title("Exploratory Data Analysis Application")

    # Page selection
    page = st.sidebar.selectbox("Choose a page:", ["Data Preprocessing", "Visualization", "Dashboard"])

    if page == "Data Preprocessing":
        data_preprocessing_page()
    elif page == "Visualization":
        visualization_page()
    elif page == "Dashboard":
        dashboard_page()

if __name__ == "__main__":
    main()
