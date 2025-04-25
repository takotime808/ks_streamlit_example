import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title="KS Test App", layout="centered")
st.title("📊 Kolmogorov-Smirnov Test: Compare Two Distributions")

st.sidebar.header("Step 1: Data Input")
input_method = st.sidebar.radio("Choose data input method", ("Generate Data", "Upload CSV"))

def generate_data(dist_type, size, params):
    if dist_type == "Normal":
        return np.random.normal(loc=params['mean'], scale=params['std'], size=size)
    elif dist_type == "Uniform":
        return np.random.uniform(low=params['min'], high=params['max'], size=size)
    elif dist_type == "Exponential":
        return np.random.exponential(scale=1/params['lambda'], size=size)
    return np.random.normal(size=size)

def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(data)+1) / len(data)
    return x, y

if input_method == "Generate Data":
    st.sidebar.subheader("Sample 1 Settings")
    dist1 = st.sidebar.selectbox("Distribution 1", ["Normal", "Uniform", "Exponential"])
    size1 = st.sidebar.slider("Sample size 1", 10, 5000, 500)

    if dist1 == "Normal":
        mean1 = st.sidebar.number_input("Mean (μ)", value=0.0, key="mean1")
        std1 = st.sidebar.number_input("Std Dev (σ)", value=1.0, min_value=0.01, key="std1")
        params1 = {"mean": mean1, "std": std1}
    elif dist1 == "Uniform":
        min1 = st.sidebar.number_input("Min", value=0.0, key="min1")
        max1 = st.sidebar.number_input("Max", value=1.0, key="max1")
        params1 = {"min": min1, "max": max1}
    else:
        lam1 = st.sidebar.number_input("Lambda (rate)", value=1.0, min_value=0.01, key="lambda1")
        params1 = {"lambda": lam1}

    st.sidebar.subheader("Sample 2 Settings")
    dist2 = st.sidebar.selectbox("Distribution 2", ["Normal", "Uniform", "Exponential"])
    size2 = st.sidebar.slider("Sample size 2", 10, 5000, 500)

    if dist2 == "Normal":
        mean2 = st.sidebar.number_input("Mean (μ)", value=0.0, key="mean2")
        std2 = st.sidebar.number_input("Std Dev (σ)", value=1.0, min_value=0.01, key="std2")
        params2 = {"mean": mean2, "std": std2}
    elif dist2 == "Uniform":
        min2 = st.sidebar.number_input("Min", value=0.0, key="min2")
        max2 = st.sidebar.number_input("Max", value=1.0, key="max2")
        params2 = {"min": min2, "max": max2}
    else:
        lam2 = st.sidebar.number_input("Lambda (rate)", value=1.0, min_value=0.01, key="lambda2")
        params2 = {"lambda": lam2}

    data1 = generate_data(dist1, size1, params1)
    data2 = generate_data(dist2, size2, params2)

else:
    uploaded_file1 = st.sidebar.file_uploader("Upload CSV for Sample 1", type=["csv"])
    uploaded_file2 = st.sidebar.file_uploader("Upload CSV for Sample 2", type=["csv"])

    if uploaded_file1 is not None and uploaded_file2 is not None:
        df1 = pd.read_csv(uploaded_file1)
        df2 = pd.read_csv(uploaded_file2)
        column1 = st.sidebar.selectbox("Select column for Sample 1", df1.columns)
        column2 = st.sidebar.selectbox("Select column for Sample 2", df2.columns)
        data1 = df1[column1].dropna().values
        data2 = df2[column2].dropna().values
    else:
        st.warning("Please upload both CSV files to proceed.")
        st.stop()

# Perform KS Test
st.header("📈 KS Test Result")
stat, p_value = ks_2samp(data1, data2)
st.success(f"**KS Statistic:** {stat:.4f}")
st.success(f"**P-value:** {p_value:.4f}")

# Option to download results
results_df = pd.DataFrame({
    "KS Statistic": [stat],
    "P-value": [p_value]
})
csv_buffer = StringIO()
results_df.to_csv(csv_buffer, index=False)
st.download_button("📥 Download Results as CSV", csv_buffer.getvalue(), file_name="ks_test_results.csv", mime="text/csv")

# Plotting
st.header("📊 Distribution Comparison")

# Histogram
st.subheader("Histogram")
fig1, ax1 = plt.subplots()
ax1.hist(data1, bins=30, alpha=0.6, label="Sample 1", color='skyblue')
ax1.hist(data2, bins=30, alpha=0.6, label="Sample 2", color='salmon')
ax1.legend()
st.pyplot(fig1)

# ECDF
st.subheader("Empirical CDFs")
x1, y1 = ecdf(data1)
x2, y2 = ecdf(data2)
fig2, ax2 = plt.subplots()
ax2.step(x1, y1, where='post', label='Sample 1', color='blue')
ax2.step(x2, y2, where='post', label='Sample 2', color='red')
ax2.legend()
st.pyplot(fig2)
