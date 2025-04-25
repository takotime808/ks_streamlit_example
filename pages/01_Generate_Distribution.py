import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

