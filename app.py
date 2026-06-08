import os
import google.generativeai as genai
import streamlit as st
import pandas as pd
import plotly.express as px

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)
model = genai.GenerativeModel("gemini-2.5-flash")


# Page title
st.set_page_config(page_title="AI Data Analyst Assistant")

st.title("🤖 AI Data Analyst Assistant")

# Upload file
st.sidebar.title("📂 Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)
st.sidebar.markdown("---")

st.sidebar.info(
    "Upload a CSV file and explore your data with AI."
)
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df)

    # Dashboard Metrics
    st.subheader("Dashboard")

    col1, col2, col3 , col4 = st.columns(4)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
        
    with col4:
        st.metric(
        "Duplicates",
         df.duplicated().sum()
        )    
    st.subheader("📋 Dataset Health Report")

    col1, col2, col3 = st.columns(3)

    with col1:
     st.metric(
        "Missing Values",
        df.isnull().sum().sum()
     )

    with col2:
     st.metric(
        "Duplicate Rows",
        df.duplicated().sum()
    )

    with col3:
     st.metric(
        "Features",
        len(df.columns)
    )
    # Columns Available
    st.subheader("Columns Available")

    selected_column = st.selectbox(
    "Select a Column",
    df.columns
    )
st.subheader("Column Distribution")

try:
    chart_data = (
        df[selected_column]
        .value_counts()
        .head(10)
        .reset_index()
    )

    chart_data.columns = [
        selected_column,
        "Count"
    ]

    fig = px.bar(
        chart_data,
        x=selected_column,
        y="Count",
        title=f"Top 10 {selected_column}"
    )

    st.plotly_chart(fig)

except:
    st.warning("Cannot create chart for this column")
    
st.subheader("Ask Questions")

question = st.text_input(
    "Ask about your dataset"
)
if question:

    prompt = f"""
    Dataset Information:
    Rows: {df.shape[0]}
    Columns: {df.shape[1]}

    Column Names:
    {list(df.columns)}

    Sample Data:
    {df.head(3).to_string()}

    Question:
    {question}

    Answer the question based on the dataset.
    """

    try:
     response = model.generate_content(prompt)
     st.write(response.text)

    except Exception:
      st.warning(
        "⏳ Gemini API limit reached. Please wait 1 minute and ask your question again."
       )
    
    if st.button("Generate Insights"):

        insights_prompt = f"""
        Analyze this dataset.

        Columns:
        {list(df.columns)}

        Sample Data:
        {df.head(3).to_string()}

        Give:
        1. Key observations
        2. Trends
        3. Business insights
        4. Recommendations
        """

        try:
            insights = model.generate_content(insights_prompt)
            st.write(insights.text)

        except Exception:
           st.warning(
           "⏳ Gemini API limit reached. Please wait 1 minute and ask your question again."
        )
    # Dataset information
    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    # Column names
    st.subheader("Columns")
    st.write(list(df.columns))

    # Missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Statistics
    st.subheader("Statistics")
    st.write(df.describe())
    st.subheader("📥 Download Dataset Summary")

    summary = f"""
    Dataset Summary

    Rows: {df.shape[0]}
    Columns: {df.shape[1]}
    Missing Values: {df.isnull().sum().sum()}
    Duplicate Rows: {df.duplicated().sum()}

    Column Names:
    {list(df.columns)}
    """

    st.download_button(
    label="Download Summary Report",
    data=summary,
    file_name="dataset_summary.txt",
    mime="text/plain"
)
