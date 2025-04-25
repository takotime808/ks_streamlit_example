import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, anderson_ksamp, zscore
import matplotlib.pyplot as plt
from io import StringIO, BytesIO

st.set_page_config(page_title="Two Sample Comparison App", layout="wide")
st.title("📊 Compare Two Uploaded Samples")

# === File Uploads ===
st.sidebar.header("Step 1: Upload Files")
uploaded_file1 = st.sidebar.file_uploader("Upload CSV for Sample 1", type=["csv"])
uploaded_file2 = st.sidebar.file_uploader("Upload CSV for Sample 2", type=["csv"])

def get_filterable_columns(df):
    return df.select_dtypes(include=[np.number, "category", "object"]).columns.tolist()

def filter_dataframe(df, label):
    st.subheader(f"🔍 Filter: {label}")
    filtered_df = df.copy()
    with st.expander(f"Filter options for {label}", expanded=False):
        for col in get_filterable_columns(df):
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val, max_val = float(df[col].min()), float(df[col].max())
                selected_range = st.slider(f"{label}: {col}", min_val, max_val, (min_val, max_val))
                filtered_df = filtered_df[(df[col] >= selected_range[0]) & (df[col] <= selected_range[1])]
            else:
                options = df[col].dropna().unique().tolist()
                selected = st.multiselect(f"{label}: {col}", options, default=options)
                filtered_df = filtered_df[df[col].isin(selected)]
    return filtered_df

if uploaded_file1 and uploaded_file2:
    df1 = pd.read_csv(uploaded_file1)
    df2 = pd.read_csv(uploaded_file2)

    # === Optional Filters ===
    df1 = filter_dataframe(df1, "Sample 1")
    df2 = filter_dataframe(df2, "Sample 2")

    st.sidebar.header("Step 2: Select Columns")
    col1 = st.sidebar.selectbox("Column from Sample 1", df1.select_dtypes(include=np.number).columns)
    col2 = st.sidebar.selectbox("Column from Sample 2", df2.select_dtypes(include=np.number).columns)

    data1 = df1[col1].dropna()
    data2 = df2[col2].dropna()

    # === Transform Options ===
    st.sidebar.header("Step 3: Preprocessing")
    log_transform = st.sidebar.checkbox("Log transform")
    log_base = st.sidebar.selectbox("Log base", ["e", "10"], disabled=not log_transform)
    standardize = st.sidebar.checkbox("Standardize (z-score)")

    def preprocess(data, label):
        transformed = data.copy()
        if log_transform:
            if (transformed <= 0).any():
                st.warning(f"{label}: Log transform skipped due to non-positive values.")
            else:
                transformed = np.log(transformed) if log_base == "e" else np.log10(transformed)
        if standardize:
            transformed = zscore(transformed)
        return transformed

    data1 = preprocess(data1, "Sample 1")
    data2 = preprocess(data2, "Sample 2")

    # === Statistical Test ===
    st.sidebar.header("Step 4: Testing")
    test_type = st.sidebar.selectbox("Test Type", ["Kolmogorov-Smirnov", "Anderson-Darling"])
    alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.10, 0.05, step=0.01)

    st.header("📈 Test Results")
    if test_type == "Kolmogorov-Smirnov":
        stat, p_value = ks_2samp(data1, data2)
        test_name = "Kolmogorov-Smirnov"
    else:
        result = anderson_ksamp([data1, data2])
        stat = result.statistic
        p_value = result.significance_level / 100  # percent to decimal
        test_name = "Anderson-Darling"

    conclusion = (
        "❌ Reject H₀: Distributions are different"
        if p_value < alpha
        else "✅ Fail to reject H₀: Distributions are similar"
    )

    st.markdown(f"**Test:** {test_name}")
    st.markdown(f"**Statistic:** `{stat:.4f}`")
    st.markdown(f"**P-value:** `{p_value:.4f}`")
    st.markdown(f"**Conclusion (α = {alpha:.2f}):** {conclusion}")

    # === Download Results ===
    result_df = pd.DataFrame({
        "Test": [test_name],
        "Statistic": [stat],
        "P-value": [p_value],
        "Alpha": [alpha],
        "Conclusion": [conclusion]
    })
    csv_buf = StringIO()
    result_df.to_csv(csv_buf, index=False)
    st.download_button("📥 Download Results CSV", csv_buf.getvalue(), file_name="test_results.csv", mime="text/csv")

    # === Plotting Functions ===
    def ecdf(data):
        x = np.sort(data)
        y = np.arange(1, len(data) + 1) / len(data)
        return x, y

    def save_plot(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png")
        return buf.getvalue()

    # === Histogram ===
    st.subheader("📊 Histogram")
    fig1, ax1 = plt.subplots()
    ax1.hist(data1, bins=30, alpha=0.6, label="Sample 1", color="skyblue")
    ax1.hist(data2, bins=30, alpha=0.6, label="Sample 2", color="salmon")
    ax1.set_title("Histogram of Samples")
    ax1.legend()
    st.pyplot(fig1)
    st.download_button("🖼️ Download Histogram", save_plot(fig1), file_name="histogram.png", mime="image/png")

    # === ECDF ===
    st.subheader("📈 ECDF")
    x1, y1 = ecdf(data1)
    x2, y2 = ecdf(data2)
    fig2, ax2 = plt.subplots()
    ax2.step(x1, y1, where='post', label="Sample 1", color="blue")
    ax2.step(x2, y2, where='post', label="Sample 2", color="red")
    ax2.set_title("Empirical CDFs")
    ax2.legend()
    st.pyplot(fig2)
    st.download_button("🖼️ Download ECDF", save_plot(fig2), file_name="ecdf.png", mime="image/png")

else:
    st.info("📂 Please upload both CSV files to begin.")
