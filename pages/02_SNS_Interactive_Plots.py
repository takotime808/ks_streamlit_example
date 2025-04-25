import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Interactive Pairwise Plot", layout="wide")

# Page title
st.title("🔀 Interactive Pairwise Distribution Plot")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load DataFrame
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Data Preview")
    st.dataframe(df.head())

    # Select numeric columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Please upload a CSV with at least two numeric columns.")
    else:
        # Let user select which columns to include
        selected_cols = st.multiselect("Select numeric columns to include in pairplot", numeric_cols, default=numeric_cols)

        if len(selected_cols) >= 2:
            st.subheader("📈 Pairwise Distribution Plot with KDE")
            plot_data = df[selected_cols]

            # Create pairplot with KDE on lower triangle
            pairplot = sns.pairplot(plot_data, kind='scatter', corner=True, diag_kind='kde', plot_kws={'alpha': 0.6})
            
            # Render in Streamlit
            st.pyplot(pairplot.figure)
        else:
            st.info("Please select at least two columns to generate the pairplot.")
else:
    st.info("👈 Upload a CSV file with numeric columns to get started.")
