import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

st.title("Kolmogorov-Smirnov Test for Two Distributions")

st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose data input method", ("Generate Data", "Upload CSV"))

def generate_data(dist_type, size):
    if dist_type == "Normal":
        return np.random.normal(loc=0, scale=1, size=size)
    elif dist_type == "Uniform":
        return np.random.uniform(low=0, high=1, size=size)
    elif dist_type == "Exponential":
        return np.random.exponential(scale=1.0, size=size)
    else:
        return np.random.normal(size=size)

if input_method == "Generate Data":
    st.sidebar.subheader("Sample 1")
    dist1 = st.sidebar.selectbox("Distribution 1", ["Normal", "Uniform", "Exponential"])
    size1 = st.sidebar.slider("Sample size 1", 10, 1000, 200)

    st.sidebar.subheader("Sample 2")
    dist2 = st.sidebar.selectbox("Distribution 2", ["Normal", "Uniform", "Exponential"])
    size2 = st.sidebar.slider("Sample size 2", 10, 1000, 200)

    data1 = generate_data(dist1, size1)
    data2 = generate_data(dist2, size2)

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
        st.warning("Please upload both CSV files.")
        st.stop()

# Run KS test
st.subheader("Kolmogorov-Smirnov Test Results")
stat, p_value = ks_2samp(data1, data2)
st.write(f"KS Statistic: **{stat:.4f}**")
st.write(f"P-value: **{p_value:.4f}**")

# Plot histogram
st.subheader("Histogram of Both Samples")
fig, ax = plt.subplots()
ax.hist(data1, bins=30, alpha=0.5, label="Sample 1")
ax.hist(data2, bins=30, alpha=0.5, label="Sample 2")
ax.legend()
st.pyplot(fig)

# Plot ECDF
def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(data)+1) / len(data)
    return x, y

st.subheader("Empirical Cumulative Distribution Functions (ECDFs)")
x1, y1 = ecdf(data1)
x2, y2 = ecdf(data2)

fig2, ax2 = plt.subplots()
ax2.step(x1, y1, label="Sample 1", where='post')
ax2.step(x2, y2, label="Sample 2", where='post')
ax2.legend()
st.pyplot(fig2)
