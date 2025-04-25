import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pandas.plotting import lag_plot

# Set page title
st.set_page_config(page_title="Generate Plots for Two Samples", layout="wide")

# Title of the app
st.title("📊 Generate Plots and Visualize Two Samples")

# === Sidebar for File Uploads ===
st.sidebar.header("Upload CSV Files")
uploaded_file1 = st.sidebar.file_uploader("Upload CSV for Sample 1", type=["csv"])
uploaded_file2 = st.sidebar.file_uploader("Upload CSV for Sample 2", type=["csv"])

# === Read Files and Handle Data ===
if uploaded_file1 and uploaded_file2:
    df1 = pd.read_csv(uploaded_file1)
    df2 = pd.read_csv(uploaded_file2)
    
    # === Select Columns for Comparison ===
    st.sidebar.header("Select Columns for Comparison")
    col1 = st.sidebar.selectbox("Column from Sample 1", df1.select_dtypes(include=np.number).columns)
    col2 = st.sidebar.selectbox("Column from Sample 2", df2.select_dtypes(include=np.number).columns)

    data1 = df1[col1].dropna()
    data2 = df2[col2].dropna()

    # === Plot 1: Histogram ===
    st.subheader("📊 Histogram")
    fig1, ax1 = plt.subplots()
    ax1.hist(data1, bins=30, alpha=0.6, label="Sample 1", color="skyblue")
    ax1.hist(data2, bins=30, alpha=0.6, label="Sample 2", color="salmon")
    ax1.set_title("Histogram of Samples")
    ax1.legend()
    st.pyplot(fig1)
    st.write("A histogram shows the distribution of each sample, allowing you to compare their shapes and frequency distribution.")

    # === Plot 2: ECDF ===
    st.subheader("📈 ECDF")
    x1, y1 = np.sort(data1), np.arange(1, len(data1) + 1) / len(data1)
    x2, y2 = np.sort(data2), np.arange(1, len(data2) + 1) / len(data2)
    fig2, ax2 = plt.subplots()
    ax2.step(x1, y1, where='post', label="Sample 1", color="blue")
    ax2.step(x2, y2, where='post', label="Sample 2", color="red")
    ax2.set_title("Empirical CDFs")
    ax2.legend()
    st.pyplot(fig2)
    st.write("The ECDF (Empirical Cumulative Distribution Function) shows the cumulative distribution of each sample.")

    # === Plot 3: Box Plot ===
    st.subheader("📦 Box Plot")
    fig3, ax3 = plt.subplots()
    ax3.boxplot([data1, data2], labels=["Sample 1", "Sample 2"])
    ax3.set_title("Box Plot of Samples")
    st.pyplot(fig3)
    st.write("A box plot summarizes the distribution by showing the median, interquartile range (IQR), and potential outliers.")

    # === Plot 4: Violin Plot ===
    st.subheader("🎻 Violin Plot")
    fig4, ax4 = plt.subplots()
    sns.violinplot(data=[data1, data2], ax=ax4, inner="quart", palette="muted")
    ax4.set_title("Violin Plot of Samples")
    ax4.set_xticklabels(["Sample 1", "Sample 2"])
    st.pyplot(fig4)
    st.write("A violin plot combines a box plot with a kernel density estimate, providing insight into the distribution of data.")

    # === Plot 5: Q-Q Plot ===
    st.subheader("📊 Q-Q Plot")
    fig5, ax5 = plt.subplots()
    stats.probplot(data1, dist="norm", plot=ax5)
    stats.probplot(data2, dist="norm", plot=ax5)
    ax5.set_title("Q-Q Plot of Sample 1 and Sample 2")
    st.pyplot(fig5)
    st.write("A Q-Q plot compares the quantiles of the two samples against a normal distribution, helping to check for normality.")

    # === Plot 6: KDE Plot ===
    st.subheader("🌫️ KDE Plot")
    fig6, ax6 = plt.subplots()
    sns.kdeplot(data1, ax=ax6, shade=True, label="Sample 1", color="blue")
    sns.kdeplot(data2, ax=ax6, shade=True, label="Sample 2", color="red")
    ax6.set_title("Kernel Density Estimation Plot")
    ax6.legend()
    st.pyplot(fig6)
    st.write("A KDE plot provides a smoothed version of the histogram and shows the probability density function of the samples.")

    # === Plot 7: Cumulative Distribution Plot ===
    st.subheader("📈 Cumulative Distribution Plot")
    fig7, ax7 = plt.subplots()
    ax7.plot(np.sort(data1), np.linspace(0, 1, len(data1), endpoint=False), label="Sample 1", color="blue")
    ax7.plot(np.sort(data2), np.linspace(0, 1, len(data2), endpoint=False), label="Sample 2", color="red")
    ax7.set_title("Cumulative Distribution Function (CDF) Comparison")
    ax7.legend()
    st.pyplot(fig7)
    st.write("The CDF plot shows the cumulative probability of each sample and allows comparison of their distributions.")

    # # === Plot 8: Pairwise Distribution Plot ===
    # st.subheader("🔀 Pairwise Distribution Plot")
    # pairplot_data = pd.concat([df1, df2], ignore_index=True)
    # sns.pairplot(pairplot_data)
    # st.pyplot()
    # st.write("The pairwise distribution plot shows the relationships between different features, helpful for multivariate comparisons.")

    # === Plot 8: Pairwise Distribution Plot ===
    st.subheader("🔀 Pairwise Distribution Plot")
    pairplot_data = pd.concat([df1, df2], ignore_index=True)
    # Generate the pairplot
    pairplot = sns.pairplot(pairplot_data)
    # Display the figure explicitly
    st.pyplot(pairplot.fig)
    st.write("The pairwise distribution plot shows the relationships between different features, helpful for multivariate comparisons.")


    # === Plot 9: Lag Plot (for Time Series Data) ===
    st.subheader("⏳ Lag Plot")
    fig9, ax9 = plt.subplots()
    lag_plot(data1, ax=ax9)
    ax9.set_title("Lag Plot for Sample 1")
    st.pyplot(fig9)
    st.write("A lag plot is used to examine time series data for autocorrelation. It helps check the relationship between data points at different time lags.")

else:
    st.info("""
    📂 Please upload both CSV files to begin.

    If you don't have the CSV files yet, you can generate them using the **Generate Distribution** page on the left.
    
    To do so, navigate to the **Generate Distribution** page, select the desired distribution (Normal, Uniform, Exponential, or Binomial), 
    set the parameters, and then download the generated data as a CSV file. 
    Once you have your files, you can upload them here to compare distributions.
""")

