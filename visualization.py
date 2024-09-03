import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def visualization_page():
    if 'df' not in st.session_state:
        st.session_state.df = None
        st.session_state.data_summary = None

    if st.session_state.df is not None:
        st.subheader("Data Visualization")
        st.sidebar.title("Visualization Settings")

        viz_types = ["Univariate", "Bivariate", "Multivariate"]
        selected_viz_type = st.sidebar.selectbox("Select visualization type", viz_types)

        columns = st.session_state.df.columns.tolist()

        if selected_viz_type == "Univariate":
            viz_types_uni = ["Histogram", "Box Plot", "Violin Plot", "Bar Plot", "Pie Chart"]
            selected_viz = st.sidebar.selectbox("Select univariate visualization type", viz_types_uni)

            x_column = st.sidebar.selectbox("Select column", columns, key="x_col_uni")

            scale_factor = st.sidebar.slider("Scale Factor", min_value=1, max_value=50, value=10)

            title = st.sidebar.text_input("Enter chart title", key="title_uni")

            if selected_viz == "Histogram":
                if pd.api.types.is_numeric_dtype(st.session_state.df[x_column]):
                    fig = px.histogram(st.session_state.df, x=x_column, nbins=scale_factor, title=title)
                    fig.update_traces(marker_line_width=1, marker_line_color="black")
                else:
                    st.error("Histogram requires a numeric column.")
            elif selected_viz == "Box Plot":
                if pd.api.types.is_numeric_dtype(st.session_state.df[x_column]):
                    fig = px.box(st.session_state.df, y=x_column, title=title)
                else:
                    st.error("Box Plot requires a numeric column.")
            elif selected_viz == "Violin Plot":
                if pd.api.types.is_numeric_dtype(st.session_state.df[x_column]):
                    fig = px.violin(st.session_state.df, y=x_column, box=True, title=title)
                else:
                    st.error("Violin Plot requires a numeric column.")
            elif selected_viz == "Bar Plot":
                if pd.api.types.is_categorical_dtype(st.session_state.df[x_column]) or st.session_state.df[x_column].dtype == object:
                    value_counts = st.session_state.df[x_column].value_counts()[:scale_factor]
                    fig = px.bar(x=value_counts.index, y=value_counts.values, labels={'x': x_column, 'y': 'count'}, title=title)
                else:
                    st.error("Bar Plot requires a categorical column.")
            elif selected_viz == "Pie Chart":
                if pd.api.types.is_categorical_dtype(st.session_state.df[x_column]) or st.session_state.df[x_column].dtype == object:
                    data = st.session_state.df[x_column].value_counts()[:scale_factor]
                    fig = px.pie(values=data.values, names=data.index, title=title)
                else:
                    st.error("Pie Chart requires a categorical column.")

            st.plotly_chart(fig)

            # Save the plot configuration to the session state
            if st.button("Save Chart"):
                if 'saved_charts' not in st.session_state:
                    st.session_state.saved_charts = []
                st.session_state.saved_charts.append({
                    "title": title,
                    "type": selected_viz,
                    "x_column": x_column,
                    "y_column": None,  # No y_column for univariate plots
                    "scale_factor": scale_factor
                })

        elif selected_viz_type == "Bivariate":
            viz_types_bi = ["Side by Side Box Plot", "Overlapping Histogram", "Scatter Plot", "Side by Side Bar Plot", "Stacked Bar Plot"]
            selected_viz = st.sidebar.selectbox("Select bivariate visualization type", viz_types_bi)

            x_column = st.sidebar.selectbox("Select X-axis column", columns, key="x_col_bi")
            y_column = st.sidebar.selectbox("Select Y-axis column", columns, key="y_col_bi")

            scale_factor = st.sidebar.slider("Scale Factor", min_value=1, max_value=50, value=10)

            title = st.sidebar.text_input("Enter chart title", key="title_bi")

            if selected_viz == "Side by Side Box Plot":
                if pd.api.types.is_numeric_dtype(st.session_state.df[x_column]) and (pd.api.types.is_categorical_dtype(st.session_state.df[y_column]) or st.session_state.df[y_column].dtype == object):
                    fig = px.box(st.session_state.df, x=y_column, y=x_column, title=title)
                else:
                    st.error("Side by Side Box Plot requires a numeric column and a categorical column.")
            elif selected_viz == "Overlapping Histogram":
                if pd.api.types.is_numeric_dtype(st.session_state.df[x_column]) and (pd.api.types.is_categorical_dtype(st.session_state.df[y_column]) or st.session_state.df[y_column].dtype == object):
                    fig = go.Figure()
                    for category in st.session_state.df[y_column].unique():
                        subset = st.session_state.df[st.session_state.df[y_column] == category]
                        fig.add_trace(go.Histogram(x=subset[x_column], nbinsx=scale_factor, name=str(category), opacity=0.75, marker_line_width=1, marker_line_color="black"))
                    fig.update_layout(barmode='overlay', title=title)
                else:
                    st.error("Overlapping Histogram requires a numeric column and a categorical column.")
            elif selected_viz == "Scatter Plot":
                if pd.api.types.is_numeric_dtype(st.session_state.df[x_column]) and pd.api.types.is_numeric_dtype(st.session_state.df[y_column]):
                    fig = px.scatter(st.session_state.df, x=x_column, y=y_column, title=title)
                else:
                    st.error("Scatter Plot requires two numeric columns.")
            elif selected_viz == "Side by Side Bar Plot":
                if (pd.api.types.is_categorical_dtype(st.session_state.df[x_column]) or st.session_state.df[x_column].dtype == object) and (pd.api.types.is_categorical_dtype(st.session_state.df[y_column]) or st.session_state.df[y_column].dtype == object):
                    crosstab = pd.crosstab(st.session_state.df[x_column], st.session_state.df[y_column]).iloc[:, :scale_factor]
                    fig = go.Figure(data=[
                        go.Bar(name=str(col), x=crosstab.index, y=crosstab[col]) for col in crosstab.columns
                    ])
                    fig.update_layout(barmode='group', title=title)
                else:
                    st.error("Side by Side Bar Plot requires two categorical columns.")
            elif selected_viz == "Stacked Bar Plot":
                if (pd.api.types.is_categorical_dtype(st.session_state.df[x_column]) or st.session_state.df[x_column].dtype == object) and (pd.api.types.is_categorical_dtype(st.session_state.df[y_column]) or st.session_state.df[y_column].dtype == object):
                    crosstab = pd.crosstab(st.session_state.df[x_column], st.session_state.df[y_column]).iloc[:, :scale_factor]
                    fig = go.Figure(data=[
                        go.Bar(name=str(col), x=crosstab.index, y=crosstab[col]) for col in crosstab.columns
                    ])
                    fig.update_layout(barmode='stack', title=title)
                else:
                    st.error("Stacked Bar Plot requires two categorical columns.")

            st.plotly_chart(fig)

            # Save the plot configuration to the session state
            if st.button("Save Chart"):
                if 'saved_charts' not in st.session_state:
                    st.session_state.saved_charts = []
                st.session_state.saved_charts.append({
                    "title": title,
                    "type": selected_viz,
                    "x_column": x_column,
                    "y_column": y_column,
                    "scale_factor": scale_factor
                })

        elif selected_viz_type == "Multivariate":
            viz_types_multi = ["Heatmap"]
            selected_viz = st.sidebar.selectbox("Select multivariate visualization type", viz_types_multi)

            title = st.sidebar.text_input("Enter chart title", key="title_multi")

            if selected_viz == "Heatmap":
                scale_factor = st.sidebar.slider("Scale Factor", min_value=1, max_value=50, value=10)

                if st.button("Generate Multivariate Plot"):
                    numeric_df = st.session_state.df.select_dtypes(include=['float64', 'int64'])
                    if not numeric_df.empty:
                        corr = numeric_df.corr()
                        fig = px.imshow(corr, color_continuous_scale='RdBu_r', aspect='auto', title=title)
                        st.plotly_chart(fig)

                        # Save the plot configuration to the session state
                        if 'saved_charts' not in st.session_state:
                            st.session_state.saved_charts = []
                        st.session_state.saved_charts.append({
                            "title": title,
                            "type": selected_viz,
                            "x_column": None,  # No x_column for multivariate plots
                            "y_column": None,  # No y_column for multivariate plots
                            "scale_factor": scale_factor
                        })
                    else:
                        st.warning("No numeric columns available for correlation heatmap.")
                # Move the Save Chart button outside the conditional block to ensure it always appears
                if st.button("Save Chart"):
                    if 'saved_charts' not in st.session_state:
                        st.session_state.saved_charts = []
                    st.session_state.saved_charts.append({
                        "title": title,
                        "type": selected_viz,
                        "x_column": None,
                        "y_column": None,
                        "scale_factor": scale_factor
                    })
