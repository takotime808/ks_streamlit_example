import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kstest, anderson
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set page title
st.set_page_config(page_title="Generate and Save Distribution", layout="wide")

# Title of the app
st.title("🎲 Generate a Random Distribution and Save as CSV")

# Sidebar options to choose the type of distribution
st.sidebar.header("Choose Distribution Parameters")

# Distribution type
dist_type = st.sidebar.selectbox(
    "Select Distribution Type", ["Normal", "Uniform", "Exponential", "Binomial"]
)

# Parameters for the selected distribution
if dist_type == "Normal":
    mean = st.sidebar.slider("Mean", -10.0, 10.0, 0.0)
    std_dev = st.sidebar.slider("Standard Deviation", 0.1, 10.0, 1.0)
    size = st.sidebar.slider("Sample Size", 100, 10000, 1000)
elif dist_type == "Uniform":
    low = st.sidebar.slider("Low", -10.0, 10.0, -5.0)
    high = st.sidebar.slider("High", -10.0, 10.0, 5.0)
    size = st.sidebar.slider("Sample Size", 100, 10000, 1000)
elif dist_type == "Exponential":
    scale = st.sidebar.slider("Scale", 0.1, 10.0, 1.0)
    size = st.sidebar.slider("Sample Size", 100, 10000, 1000)
elif dist_type == "Binomial":
    n = st.sidebar.slider("Number of Trials", 1, 100, 10)
    p = st.sidebar.slider("Probability of Success", 0.0, 1.0, 0.5)
    size = st.sidebar.slider("Sample Size", 100, 10000, 1000)

# Generate data based on selected distribution
if dist_type == "Normal":
    data = np.random.normal(loc=mean, scale=std_dev, size=size)
elif dist_type == "Uniform":
    data = np.random.uniform(low=low, high=high, size=size)
elif dist_type == "Exponential":
    data = np.random.exponential(scale=scale, size=size)
elif dist_type == "Binomial":
    data = np.random.binomial(n=n, p=p, size=size)

# Display histogram
st.subheader(f"Histogram of {dist_type} Distribution")
fig, ax = plt.subplots()
ax.hist(data, bins=30, alpha=0.7, color='skyblue')
ax.set_title(f"{dist_type} Distribution (n={size})")
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Convert data to DataFrame
df = pd.DataFrame(data, columns=["Value"])

# Button to download the data as CSV
st.subheader("Download the Data as CSV")
csv = df.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name=f"{dist_type}_distribution.csv",
    mime="text/csv",
)

# Show the metrics table
st.subheader("Validation Metrics from Sandia 2016-1421 Paper")

# Define metrics with placeholders (these can be calculated based on data comparison)
metrics = {
    "Mean Bias (MB)": "Mean of the differences between observed and predicted data.",
    "Root Mean Squared Error (RMSE)": "Square root of the mean of squared errors.",
    "Mean Absolute Error (MAE)": "Mean of the absolute errors.",
    "Percent Bias (PBIAS)": "Percentage bias between observed and predicted values.",
    "R-squared (R²)": "Measure of the proportion of variance explained by the model.",
    "Kullback-Leibler Divergence (KLD)": "Divergence between two distributions.",
    "Kolmogorov-Smirnov Statistic (KS)": "Statistical test to compare distributions.",
    "Anderson-Darling Test Statistic": "Test for the goodness of fit."
}

# Create a table with the metrics descriptions
metric_data = pd.DataFrame(metrics.items(), columns=["Metric", "Description"])

st.table(metric_data)

# Placeholder to compute and show sample metrics based on generated data
# For demonstration, we will compare against a standard normal distribution for simplicity.

# Compare with standard normal distribution for some metrics
obs = np.random.normal(loc=0, scale=1, size=size)  # "Observed" values for comparison

# Mean Bias
mean_bias = np.mean(data - obs)

# RMSE
rmse = np.sqrt(mean_squared_error(obs, data))

# MAE
mae = mean_absolute_error(obs, data)

# Percent Bias (PBIAS)
p_bias = 100 * np.sum(data - obs) / np.sum(obs)

# R-squared
r2 = r2_score(obs, data)

# Kolmogorov-Smirnov Test
ks_stat, ks_p_value = kstest(data, 'norm')

# Anderson-Darling Test
ad_result = anderson(data, dist='norm')

# Show the calculated metrics
st.subheader("Calculated Metrics (compared with standard normal distribution)")

metrics_values = {
    "Mean Bias (MB)": mean_bias,
    "Root Mean Squared Error (RMSE)": rmse,
    "Mean Absolute Error (MAE)": mae,
    "Percent Bias (PBIAS)": p_bias,
    "R-squared (R²)": r2,
    "Kolmogorov-Smirnov Statistic (KS)": ks_stat,
    "KS Test p-value": ks_p_value,
    "Anderson-Darling Test Statistic": ad_result.statistic
}

# Show the metrics values in a table
metrics_values_df = pd.DataFrame(list(metrics_values.items()), columns=["Metric", "Value"])
st.table(metrics_values_df)
