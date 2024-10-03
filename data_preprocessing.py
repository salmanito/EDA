import streamlit as st
import pandas as pd
from rename_column import rename_column
from drop_column import drop_column
from remove_duplicates import remove_duplicates
from split_data_ranges import split_data_ranges
from clean_text import clean_text
from set_data_type import set_data_type
from correct_spelling import correct_spelling
from encode_categorical import encode_categorical
from handle_missing_values import fill_with_mean, fill_with_median, fill_with_mode, fill_with_custom_value, fill_with_decision_tree, drop_missing_values
from discretize_data import discretize_column
from chatbot import get_chatbot_response
from data_upload import upload_data, update_data_summary
from langchain.memory import ChatMessageHistory


def generate_statistical_summary(df):
    desc = df.describe(include='all').transpose()
    desc['range'] = desc['max'] - desc['min']
    desc['iqr'] = desc['75%'] - desc['25%']

    # Calculate skewness and kurtosis only for numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    desc['skewness'] = df[numeric_columns].skew()
    desc['kurtosis'] = df[numeric_columns].kurt()

    return desc

def generate_detailed_summary(df):
    detailed_summary = {}
    detailed_summary['basic_stats'] = df.describe(include='all').transpose().to_dict()
    detailed_summary['dtypes'] = df.dtypes.apply(lambda x: x.name).to_dict()

    return detailed_summary

def data_preprocessing_page():
    # Initialize session state to keep the dataset and conversation history
    if 'df' not in st.session_state:
        st.session_state.df = None
        st.session_state.detailed_summary = None

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = ChatMessageHistory()

    # Upload data
    upload_data()

    if st.session_state.df is not None:
        # Create a detailed summary of the dataset
        st.session_state.detailed_summary = generate_detailed_summary(st.session_state.df)

        # Create two columns
        col1, col2 = st.columns([8, 4])

        with col1:
            # Sidebar for column selection
            st.sidebar.title("Column Selection")
            columns = st.session_state.df.columns.tolist()
            selected_column = st.sidebar.radio("Select column", columns)

            # Display the selected column for debugging
            st.sidebar.write(f"Selected column: {selected_column}")

            # Data cleaning methods
            st.subheader("Data Preprocessing:")
            cleaning_methods = [
                "Dropping unnecessary columns", "Renaming columns", "Removing duplicates",
                "Splitting data ranges", "Cleaning text", "Setting data types", "Treating outliers",
                "Correcting spelling errors", "Normalizing/standardizing data", "Encoding categorical values",
                "Discretizing data", "Missing value", "Advanced Statistical Summary"
            ]
            selected_method = st.selectbox("Choose a preprocessing method:", cleaning_methods)

            if selected_method == "Advanced Statistical Summary":
                st.subheader("Advanced Statistical Summary")
                try:
                    summary = generate_statistical_summary(st.session_state.df)
                    st.write(summary)
                except Exception as e:
                    st.error(f"Error generating statistical summary: {e}")

            elif selected_method == "Renaming columns":
                new_name = st.text_input("Enter new name for the column:", key='new_column_name')
                if st.button("Apply", key='rename'):
                    if selected_column:
                        try:
                            st.session_state.df = rename_column(st.session_state.df, selected_column, new_name)
                            update_data_summary()
                            st.success(f"Column '{selected_column}' renamed to '{new_name}'.")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error renaming column: {e}")
                    else:
                        st.error("No column selected.")

            elif selected_method == "Dropping unnecessary columns":
                if st.button("Apply", key='drop'):
                    try:
                        st.session_state.df = drop_column(st.session_state.df, selected_column)
                        update_data_summary()
                        st.success(f"Column '{selected_column}' dropped.")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error dropping column: {e}")

            elif selected_method == "Removing duplicates":
                if st.button("Apply", key='remove_duplicates'):
                    try:
                        st.session_state.df, duplicates_removed = remove_duplicates(st.session_state.df)
                        update_data_summary()
                        if duplicates_removed:
                            st.success("Duplicates removed.")
                        else:
                            st.info("No duplicates found.")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error removing duplicates: {e}")

            elif selected_method == "Splitting data ranges":
                split_options = st.multiselect("Select split options:", ["Upper", "Average", "Lower"])
                if st.button("Apply", key='split'):
                    if selected_column and split_options:
                        try:
                            st.session_state.df = split_data_ranges(st.session_state.df, selected_column, split_options)
                            update_data_summary()
                            st.success(
                                f"Data ranges in column '{selected_column}' split into {', '.join(split_options)}.")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error splitting data ranges: {e}")
                    else:
                        st.error("No column selected or no split options selected.")

            elif selected_method == "Cleaning text":
                cleaning_options = [
                    "Removing Punctuation", "Lowercasing", "Removing Numbers", "Removing Characters"
                ]
                selected_cleaning_option = st.selectbox("Choose a cleaning option:", cleaning_options)
                if st.button("Apply", key='clean_text'):
                    if selected_column and selected_cleaning_option:
                        try:
                            st.session_state.df = clean_text(st.session_state.df, selected_column,
                                                             selected_cleaning_option)
                            update_data_summary()
                            st.success(f"Text in column '{selected_column}' cleaned with '{selected_cleaning_option}'.")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error cleaning text: {e}")
                    else:
                        st.error("No column selected or no cleaning option selected.")

            elif selected_method == "Setting data types":
                if st.button("Show Data Summary"):
                    st.write(st.session_state.df.dtypes)

                data_types = ["int", "float", "str", "category", "datetime64[ns]"]
                selected_dtype = st.selectbox("Select new data type:", data_types)
                if st.button("Apply", key='set_dtype'):
                    if selected_column and selected_dtype:
                        try:
                            st.session_state.df = set_data_type(st.session_state.df, selected_column, selected_dtype)
                            update_data_summary()
                            st.success(f"Column '{selected_column}' set to data type '{selected_dtype}'.")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error setting data type: {e}")
                    else:
                        st.error("No column selected or no data type selected.")

            elif selected_method == "Correcting spelling errors":
                misspelled = st.text_input("Enter misspelled value:", key='misspelled_value')
                corrected = st.text_input("Enter corrected value:", key='corrected_value')
                if st.button("Apply", key='correct_spelling'):
                    if selected_column and misspelled and corrected:
                        try:
                            st.session_state.df = correct_spelling(st.session_state.df, selected_column, misspelled,
                                                                   corrected)
                            update_data_summary()
                            st.success(
                                f"All instances of '{misspelled}' corrected to '{corrected}' in column '{selected_column}'.")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error correcting spelling: {e}")
                    else:
                        st.error("Please provide the misspelled and corrected values.")

            elif selected_method == "Encoding categorical values":
                unique_count = st.session_state.df[selected_column].nunique()
                if unique_count > 10:
                    st.warning("Not eligible for encoding categorical values. More than 10 unique values.")
                else:
                    if st.button("Encode"):
                        try:
                            st.session_state.df, unique_mapping = encode_categorical(st.session_state.df,
                                                                                     selected_column)
                            update_data_summary()
                            st.success(f"Column '{selected_column}' encoded successfully.")
                            st.write(f"Encoding Mapping: {unique_mapping}")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error encoding categorical values: {e}")

            elif selected_method == "Missing value":
                missing_percentage = st.session_state.df[selected_column].isnull().mean() * 100
                st.write(f"Percentage of missing values in column '{selected_column}': {missing_percentage:.2f}%")

                missing_value_options = [
                    "Fill with mean value", "Fill with median value", "Fill with mode value",
                    "Fill with decision tree method", "Fill with custom value", "Drop rows with missing values"
                ]
                selected_missing_value_option = st.selectbox("Choose an option to handle missing values:",
                                                             missing_value_options)

                if selected_missing_value_option == "Fill with custom value":
                    custom_value = st.text_input("Enter the custom value to fill missing values:", key='custom_value')

                if st.button("Apply", key='handle_missing'):
                    if selected_column and selected_missing_value_option:
                        try:
                            if selected_missing_value_option == "Fill with mean value":
                                st.session_state.df = fill_with_mean(st.session_state.df, selected_column)
                            elif selected_missing_value_option == "Fill with median value":
                                st.session_state.df = fill_with_median(st.session_state.df, selected_column)
                            elif selected_missing_value_option == "Fill with mode value":
                                st.session_state.df = fill_with_mode(st.session_state.df, selected_column)
                            elif selected_missing_value_option == "Fill with decision tree method":
                                st.session_state.df = fill_with_decision_tree(st.session_state.df, selected_column)
                            elif selected_missing_value_option == "Fill with custom value":
                                if custom_value:
                                    st.session_state.df = fill_with_custom_value(st.session_state.df, selected_column,
                                                                                 custom_value)
                                else:
                                    st.error("Please enter a custom value to fill missing values.")
                                    return
                            elif selected_missing_value_option == "Drop rows with missing values":
                                st.session_state.df = drop_missing_values(st.session_state.df, selected_column)
                            update_data_summary()
                            st.success(f"Missing values in column '{selected_column}' handled successfully.")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error handling missing values: {e}")
                    else:
                        st.error("Please provide the necessary options.")

            elif selected_method == "Discretizing data":
                if pd.api.types.is_numeric_dtype(st.session_state.df[selected_column]):
                    num_categories = st.number_input("Enter number of categories:", min_value=2, step=1)
                    if num_categories > 1:
                        categories = []
                        ranges = []
                        for i in range(num_categories):
                            category_name = st.text_input(f"Enter name for category {i + 1}:", key=f'category_name_{i}')
                            categories.append(category_name)
                        for i in range(num_categories):
                            if i == 0:
                                range_start = st.number_input(f"Enter start range for category {i + 1}:",
                                                              value=float(st.session_state.df[selected_column].min()),
                                                              key=f'range_start_{i}')
                            else:
                                range_start = st.number_input(f"Enter start range for category {i + 1}:",
                                                              key=f'range_start_{i}')
                            ranges.append(range_start)
                        ranges.append(st.number_input(f"Enter end range for category {num_categories}:",
                                                      value=float(st.session_state.df[selected_column].max()),
                                                      key=f'range_end_{num_categories}'))
                        if st.button("Apply", key='discretize'):
                            try:
                                st.session_state.df = discretize_column(st.session_state.df, selected_column,
                                                                        categories, ranges)
                                update_data_summary()
                                st.success(f"Column '{selected_column}' discretized successfully.")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error discretizing data: {e}")
                else:
                    st.error("Selected column is not numeric!")

            # Display the dataset
            st.subheader("Dataset Dashboard")
            st.write(st.session_state.df)

        with col2:
            st.subheader("AI Chatbot")
            user_input = st.text_input("Ask your data question here:", key="chat_input")
            if user_input and st.session_state.df is not None:
                st.write("Calling chatbot...")
                response = get_chatbot_response(user_input, st.session_state.df, st.session_state.conversation_history)
                st.write(response)
