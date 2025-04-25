import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.stats import zscore, ks_2samp, anderson_ksamp

# === File Upload ===
st.sidebar.header("Step 1: Upload Files")
uploaded_file1 = st.sidebar.file_uploader("Upload CSV for Sample 1", type=["csv"])
uploaded_file2 = st.sidebar.file_uploader("Upload CSV for Sample 2", type=["csv"])

if uploaded_file1 and uploaded_file2:
    df1 = pd.read_csv(uploaded_file1)
    df2 = pd.read_csv(uploaded_file2)
    
    # === Select Columns ===
    st.sidebar.header("Step 2: Select Columns")
    col1 = st.sidebar.selectbox("Column from Sample 1", df1.select_dtypes(include=np.number).columns)
    col2 = st.sidebar.selectbox("Column from Sample 2", df2.select_dtypes(include=np.number).columns)

    data1 = df1[col1].dropna()
    data2 = df2[col2].dropna()

    # === Plot Options ===
    st.sidebar.header("Step 3: Select Figures")
    plot_type = st.sidebar.selectbox(
        "Select Plot Type", 
        ["None", "ECDF", "Histogram", "Cross-Correlation", "RMSE Plot"]
    )
    
    fig = None  # Initialize the fig variable to avoid errors if no plot is selected.

    if plot_type == "ECDF":
        st.subheader("📈 ECDF")
        st.markdown("""
            The **Empirical Cumulative Distribution Function (ECDF)** shows the cumulative probability of a dataset 
            by plotting the fraction of data points that are less than or equal to each value. 
            It helps to visualize the distribution and compare the shapes of two samples.
        """)
        def ecdf(data):
            x = np.sort(data)
            y = np.arange(1, len(data) + 1) / len(data)
            return x, y

        x1, y1 = ecdf(data1)
        x2, y2 = ecdf(data2)
        
        fig, ax = plt.subplots()
        ax.step(x1, y1, where='post', label="Sample 1", color="blue")
        ax.step(x2, y2, where='post', label="Sample 2", color="red")
        ax.set_title("Empirical CDFs")
        ax.legend()
        st.pyplot(fig)

    elif plot_type == "Histogram":
        st.subheader("📊 Histogram")
        st.markdown("""
            The **Histogram** shows the distribution of data in bins and is used to understand the frequency 
            of data values within a range. Comparing histograms for two samples helps in determining their 
            similarities or differences in distribution.
        """)
        fig, ax = plt.subplots()
        ax.hist(data1, bins=30, alpha=0.6, label="Sample 1", color="skyblue")
        ax.hist(data2, bins=30, alpha=0.6, label="Sample 2", color="salmon")
        ax.set_title("Histogram of Samples")
        ax.legend()
        st.pyplot(fig)

    elif plot_type == "Cross-Correlation":
        st.subheader("🔄 Cross-Correlation")
        st.markdown("""
            **Cross-correlation** is used to measure the similarity between two signals as a function of the time-lag 
            applied to one of them. It can identify how aligned or phase-shifted the datasets are.
        """)
        corr = np.correlate(data1 - np.mean(data1), data2 - np.mean(data2), mode='full')
        lag = np.arange(-len(data1)+1, len(data1))
        
        fig, ax = plt.subplots()
        ax.plot(lag, corr, label="Cross-Correlation", color="purple")
        ax.set_title("Cross-Correlation between Samples")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Correlation")
        ax.legend()
        st.pyplot(fig)

    elif plot_type == "RMSE Plot":
        st.subheader("📉 RMSE Plot")
        st.markdown("""
            **Root Mean Square Error (RMSE)** quantifies the difference between two datasets by measuring the average 
            squared differences between the predicted and actual values. A lower RMSE indicates better similarity between the samples.
        """)
        rmse = np.sqrt(np.mean((data1 - data2)**2))
        
        fig, ax = plt.subplots()
        ax.plot([0, len(data1)], [rmse, rmse], label="RMSE", color="green", linestyle="--")
        ax.set_title(f"Root Mean Square Error: {rmse:.4f}")
        ax.set_xlabel("Index")
        ax.set_ylabel("Error")
        ax.legend()
        st.pyplot(fig)

    # === Download Options ===
    def save_plot(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return buf.getvalue()

    if fig is not None:  # Ensure the plot is created before attempting to download
        st.download_button(
            "🖼️ Download Plot", 
            save_plot(fig), 
            file_name="plot.png", 
            mime="image/png"
        )

else:
    st.info("""
    📂 Please upload both CSV files to begin.

    If you don't have the CSV files yet, you can generate them using the **Generate Distribution** page on the left.
    
    To do so, navigate to the **Generate Distribution** page, select the desired distribution (Normal, Uniform, Exponential, or Binomial), 
    set the parameters, and then download the generated data as a CSV file. 
    Once you have your files, you can upload them here to compare distributions.
""")
