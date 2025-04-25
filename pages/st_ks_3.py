import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, anderson_ksamp
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title="Two-Sample Statistical Test App", layout="centered")
st.title("📊 Compare Two Samples: KS & Anderson-Darling Test")

st.sidebar.header("Step 1: Upload Data")
uploaded_file1 = st.sidebar.file_uploader("Upload CSV for Sample 1", type=["csv"])
uploaded_file2 = st.sidebar.file_uploader("Upload CSV for Sample 2", type=["csv"])

if uploaded_file1 is not None and uploaded_file2 is not None:
    df1 = pd.read_csv(uploaded_file1)
    df2 = pd.read_csv(uploaded_file2)

    st.sidebar.header("Step 2: Select Columns")
    column1 = st.sidebar.selectbox("Column from Sample 1", df1.select_dtypes(include=np.number).columns)
    column2 = st.sidebar.selectbox("Column from Sample 2", df2.select_dtypes(include=np.number).columns)

    data1 = df1[column1].dropna().values
    data2 = df2[column2].dropna().values

    st.sidebar.header("Step 3: Choose Test")
    test_type = st.sidebar.selectbox("Statistical Test", ["Kolmogorov-Smirnov", "Anderson-Darling"])
    alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.10, 0.05, step=0.01)

    st.header("📈 Test Results")
    if test_type == "Kolmogorov-Smirnov":
        stat, p_value = ks_2samp(data1, data2)
        st.write(f"**Test:** Kolmogorov-Smirnov")
        st.write(f"**Statistic:** {stat:.4f}")
        st.write(f"**P-value:** {p_value:.4f}")
        result = "❌ Reject H₀: Distributions are different" if p_value < alpha else "✅ Fail to reject H₀: Distributions are similar"

    else:
        result = ""
        try:
            anderson_result = anderson_ksamp([data1, data2])
            stat = anderson_result.statistic
            p_value = anderson_result.significance_level / 100  # convert to 0-1
            st.write(f"**Test:** Anderson-Darling k-sample")
            st.write(f"**Statistic:** {stat:.4f}")
            st.write(f"**Approx. P-value:** {p_value:.4f}")
            result = "❌ Reject H₀: Distributions are different" if p_value < alpha else "✅ Fail to reject H₀: Distributions are similar"
        except Exception as e:
            st.error(f"Error running Anderson-Darling test: {e}")
            st.stop()

    st.subheader("🔍 Interpretation")
    st.info(f"**Alpha:** {alpha:.2f}\n\n**Conclusion:** {result}")

    # Downloadable result
    result_df = pd.DataFrame({
        "Test": [test_type],
        "Statistic": [stat],
        "P-value": [p_value],
        "Alpha": [alpha],
        "Result": [result]
    })
    buffer = StringIO()
    result_df.to_csv(buffer, index=False)
    st.download_button("📥 Download Result as CSV", buffer.getvalue(), "test_results.csv", mime="text/csv")

    # Plot histograms
    st.subheader("📊 Histogram Comparison")
    fig1, ax1 = plt.subplots()
    ax1.hist(data1, bins=30, alpha=0.6, label="Sample 1", color='skyblue')
    ax1.hist(data2, bins=30, alpha=0.6, label="Sample 2", color='salmon')
    ax1.set_title("Histogram of Samples")
    ax1.legend()
    st.pyplot(fig1)

    # ECDF function
    def ecdf(data):
        x = np.sort(data)
        y = np.arange(1, len(data)+1) / len(data)
        return x, y

    # Plot ECDFs
    st.subheader("📈 Empirical CDF Comparison")
    x1, y1 = ecdf(data1)
    x2, y2 = ecdf(data2)
    fig2, ax2 = plt.subplots()
    ax2.step(x1, y1, where='post', label='Sample 1', color='blue')
    ax2.step(x2, y2, where='post', label='Sample 2', color='red')
    ax2.set_title("ECDFs of the Samples")
    ax2.legend()
    st.pyplot(fig2)

else:
    st.info("Please upload two CSV files to begin.")

