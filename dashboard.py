import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from chatbot import get_chatbot_response
from langchain.memory import ChatMessageHistory

def is_numeric_column(column):
    return pd.api.types.is_numeric_dtype(column)

def dashboard_page():
    if 'df' not in st.session_state:
        st.session_state.df = None
        st.session_state.data_summary = None

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = ChatMessageHistory()

    if st.session_state.df is not None:
        st.subheader("Dashboard")

        st.sidebar.title("Dashboard Settings")
        columns = st.session_state.df.columns.tolist()

        # Chatbot visibility toggle
        show_chatbot = st.sidebar.checkbox("Show Chatbot", value=True)

        # Metrics Selection
        metric1 = st.sidebar.selectbox("Select metric 1", ["Min", "Max", "Range", "Count", "Avg", "Mode", "Median"])
        column1 = st.sidebar.selectbox("Select column for metric 1", columns)

        metric2 = st.sidebar.selectbox("Select metric 2", ["Min", "Max", "Range", "Count", "Avg", "Mode", "Median"])
        column2 = st.sidebar.selectbox("Select column for metric 2", columns)

        manual_metric_text = st.sidebar.text_input("Enter manual metric text (max 25 characters)", max_chars=25)

        # Saved Charts Selection
        st.sidebar.subheader("Saved Charts")
        selected_chart_titles = st.sidebar.multiselect("Select saved charts to display", [chart['title'] for chart in st.session_state.get('saved_charts', [])])

        # Extract number from the manual metric text
        import re
        manual_metric_number = re.search(r'\d+[%]?', manual_metric_text)
        manual_metric_value = manual_metric_number.group() if manual_metric_number else ""

        # Metrics Calculation
        metrics = {
            "Min": lambda x: np.min(x),
            "Max": lambda x: np.max(x),
            "Range": lambda x: np.ptp(x),
            "Count": lambda x: len(x),
            "Avg": lambda x: np.mean(x),
            "Mode": lambda x: pd.Series.mode(x)[0] if len(pd.Series.mode(x)) > 0 else 'N/A',
            "Median": lambda x: np.median(x)
        }

        def calculate_metric(metric, column):
            if is_numeric_column(st.session_state.df[column]):
                return metrics[metric](st.session_state.df[column])
            else:
                return "N/A"

        metric1_value = calculate_metric(metric1, column1)
        metric2_value = calculate_metric(metric2, column2)

        # Format metric values with commas and no decimal points
        metric1_display = f"{metric1_value:,.0f}" if metric1_value != "N/A" else "N/A"
        metric2_display = f"{metric2_value:,.0f}" if metric2_value != "N/A" else "N/A"
        manual_metric_display = manual_metric_value

        # Layout for dashboard and chatbot
        if show_chatbot:
            col1, col2 = st.columns([3, 1])
        else:
            col1 = st.container()

        with col1:
            # Display Metrics
            col1_1, col1_2, col1_3 = st.columns(3)

            col1_1.markdown(f"""
                <div style="text-align:center">
                    <span style="font-size: 50px; font-weight: bold;">{metric1_display}</span><br>
                    <span style="font-size: 20px;">{metric1} of {column1}</span>
                </div>
                """, unsafe_allow_html=True)

            col1_2.markdown(f"""
                <div style="text-align:center">
                    <span style="font-size: 50px; font-weight: bold;">{metric2_display}</span><br>
                    <span style="font-size: 20px;">{metric2} of {column2}</span>
                </div>
                """, unsafe_allow_html=True)

            col1_3.markdown(f"""
                <div style="text-align:center">
                    <span style="font-size: 50px; font-weight: bold;">{manual_metric_display}</span><br>
                    <span style="font-size: 20px;">{manual_metric_text.replace(manual_metric_display, '')}</span>
                </div>
                """, unsafe_allow_html=True)

            # Display Saved Charts
            if selected_chart_titles:
                st.subheader("Saved Charts")
                num_plots = min(len(selected_chart_titles), 4)  # Limit to 4 charts
                rows = (num_plots + 1) // 2
                cols = 2 if num_plots > 1 else 1

                specs = [[{'type': 'xy'} if not selected_chart_titles[i * 2 + j].endswith("Pie Chart") else {'type': 'domain'}
                          for j in range(cols) if i * 2 + j < num_plots]
                         for i in range(rows)]

                fig = make_subplots(rows=rows, cols=cols, subplot_titles=selected_chart_titles[:4], specs=specs)

                for i, title in enumerate(selected_chart_titles[:4]):
                    chart_config = next(chart for chart in st.session_state.saved_charts if chart["title"] == title)
                    viz_type = chart_config["type"]
                    x_col = chart_config["x_column"]
                    y_col = chart_config["y_column"]
                    scale_factor = chart_config["scale_factor"]

                    row = i // 2 + 1
                    col = i % 2 + 1

                    if viz_type == "Histogram":
                        fig.add_trace(go.Histogram(x=st.session_state.df[x_col], nbinsx=scale_factor, marker_line_width=1, marker_line_color="black"), row=row, col=col)
                        fig.update_xaxes(title_text=x_col, row=row, col=col)
                    elif viz_type == "Box Plot":
                        fig.add_trace(go.Box(y=st.session_state.df[x_col]), row=row, col=col)
                        fig.update_yaxes(title_text=x_col, row=row, col=col)
                    elif viz_type == "Violin Plot":
                        fig.add_trace(go.Violin(y=st.session_state.df[x_col], box_visible=True), row=row, col=col)
                        fig.update_yaxes(title_text=x_col, row=row, col=col)
                    elif viz_type == "Bar Plot":
                        value_counts = st.session_state.df[x_col].value_counts()[:scale_factor]
                        fig.add_trace(go.Bar(x=value_counts.index, y=value_counts.values), row=row, col=col)
                        fig.update_xaxes(title_text=x_col, row=row, col=col)
                        fig.update_yaxes(title_text='Count', row=row, col=col)
                    elif viz_type == "Pie Chart":
                        data = st.session_state.df[x_col].value_counts()[:scale_factor]
                        fig.add_trace(go.Pie(labels=data.index, values=data.values), row=row, col=col)
                    elif viz_type == "Side by Side Box Plot":
                        fig.add_trace(go.Box(y=st.session_state.df[x_col], x=st.session_state.df[y_col]), row=row, col=col)
                        fig.update_xaxes(title_text=y_col, row=row, col=col)
                        fig.update_yaxes(title_text=x_col, row=row, col=col)
                    elif viz_type == "Overlapping Histogram":
                        for category in st.session_state.df[y_col].unique():
                            subset = st.session_state.df[st.session_state.df[y_col] == category]
                            fig.add_trace(go.Histogram(x=subset[x_col], nbinsx=scale_factor, name=str(category), opacity=0.75, marker_line_width=1, marker_line_color="black"), row=row, col=col)
                        fig.update_layout(barmode='overlay')
                        fig.update_xaxes(title_text=x_col, row=row, col=col)
                    elif viz_type == "Scatter Plot":
                        fig.add_trace(go.Scatter(x=st.session_state.df[x_col], y=st.session_state.df[y_col], mode='markers'), row=row, col=col)
                        fig.update_xaxes(title_text=x_col, row=row, col=col)
                        fig.update_yaxes(title_text=y_col, row=row, col=col)
                    elif viz_type == "Side by Side Bar Plot":
                        crosstab = pd.crosstab(st.session_state.df[x_col], st.session_state.df[y_col]).iloc[:, :scale_factor]
                        for col_name in crosstab.columns:
                            fig.add_trace(go.Bar(x=crosstab.index, y=crosstab[col_name], name=str(col_name)), row=row, col=col)
                        fig.update_layout(barmode='group')
                        fig.update_xaxes(title_text=x_col, row=row, col=col)
                        fig.update_yaxes(title_text=y_col, row=row, col=col)
                    elif viz_type == "Stacked Bar Plot":
                        crosstab = pd.crosstab(st.session_state.df[x_col], st.session_state.df[y_col]).iloc[:, :scale_factor]
                        for col_name in crosstab.columns:
                            fig.add_trace(go.Bar(x=crosstab.index, y=crosstab[col_name], name=str(col_name)), row=row, col=col)
                        fig.update_layout(barmode='stack')
                        fig.update_xaxes(title_text=x_col, row=row, col=col)
                        fig.update_yaxes(title_text=y_col, row=row, col=col)
                    elif viz_type == "Heatmap":
                        numeric_df = st.session_state.df.select_dtypes(include=['float64', 'int64'])
                        if not numeric_df.empty:
                            corr = numeric_df.corr()
                            fig.add_trace(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorbar=dict(len=0.75, x=0.95, y=0.5, thickness=15)), row=row, col=col)
                            fig.update_xaxes(title_text='Correlation', row=row, col=col)

                fig.update_layout(height=600 * rows, showlegend=False)
                st.plotly_chart(fig)

        if show_chatbot:
            with col2:
                st.subheader("Chatbot")
                user_input = st.text_input("Ask your data question here:", key="chat_input")
                if user_input and st.session_state.data_summary:
                    st.write("Calling chatbot...")
                    response = get_chatbot_response(user_input, st.session_state.data_summary, st.session_state.conversation_history)
                    st.write(response)
                else:
                    st.write("Please upload data to enable data-driven responses.")


if __name__ == "__main__":
    dashboard_page()
