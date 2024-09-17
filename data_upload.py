import streamlit as st
import pandas as pd
import json


def update_data_summary():
    """Update the data summary in the session state."""
    if 'df' in st.session_state and st.session_state.df is not None:
        st.session_state.data_summary = json.dumps({
            "numeric": st.session_state.df.describe(include=['number']).to_string(),
            "categorical": st.session_state.df.describe(include=['object', 'category']).to_string()
        })
    else:
        st.warning("No dataset found in session state.")


def upload_data():
    """Handle data file upload and store in session state."""
    uploaded_file = st.sidebar.file_uploader("Please upload your dataset to start the process (CSV or XLSX)", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        # Initialize session state for 'df' if not already present
        if 'df' not in st.session_state:
            st.session_state.df = None

        # Read the dataset and store it in session state
        try:
            if uploaded_file.type == "text/csv":
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)

            update_data_summary()  # Update the data summary after loading the dataset
            st.sidebar.success("Dataset loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading dataset: {e}")
