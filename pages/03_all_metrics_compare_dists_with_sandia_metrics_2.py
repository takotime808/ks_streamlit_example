import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, anderson_ksamp, zscore, entropy
from scipy.spatial.distance import minkowski
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

    data1 = df1[col1].dropna().values
    data2 = df2[col2].dropna().values

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

    st.sidebar.header("Step 4: Choose Validation Metrics")

    with st.sidebar.expander("📀 Deterministic Validation Metrics", expanded=False):
        use_rmse = st.checkbox("Root Mean Square Error")
        use_minkowski = st.checkbox("Minkowski Distance (Lp)")

    with st.sidebar.expander("📊 Probability-Based Validation Metrics", expanded=False):
        use_neuclidean = st.checkbox("Normalized Euclidean Metric")
        use_mahalanobis = st.checkbox("Mahalanobis Distance")
        use_kl = st.checkbox("Kullback-Leibler Divergence")
        use_sym_div = st.checkbox("Symmetrized Divergence")
        use_jsd = st.checkbox("Jensen-Shannon Divergence")
        use_hellinger = st.checkbox("Hellinger Metric")
        use_ks = st.checkbox("Kolmogorov-Smirnov Test", value=True)
        use_tv = st.checkbox("Total Variation Distance")

    with st.sidebar.expander("🎚️ Signal Processing Validation Metrics", expanded=False):
        use_xcorr = st.checkbox("Simple Cross Correlation")
        use_nxcorr = st.checkbox("Normalized Cross Correlation")
        use_nzssd = st.checkbox("Normalized Zero-Mean Sum of Squared Distances")
        use_moravec = st.checkbox("Moravec Correlation")
        use_ioa = st.checkbox("Index of Agreement")
        use_sg = st.checkbox("Sprague-Geers Metric")

    submitted = st.sidebar.button("✅ Submit and Run Metrics")

    if submitted:
        st.header("📊 Selected Validation Metrics")
        metrics_results = {}

        if use_rmse:
            metrics_results["Root Mean Square Error"] = np.sqrt(np.mean((data1 - data2) ** 2))

        if use_minkowski:
            metrics_results["Minkowski Distance (Lp=2)"] = minkowski(data1, data2, p=2)

        if use_neuclidean:
            metrics_results["Normalized Euclidean Metric"] = np.linalg.norm(data1 - data2) / len(data1)

        if use_mahalanobis:
            diff = np.mean(data1) - np.mean(data2)
            cov = np.cov(np.vstack([data1, data2]), rowvar=False)
            try:
                metrics_results["Mahalanobis Distance (Approx.)"] = np.sqrt(diff**2 / cov.mean())
            except:
                metrics_results["Mahalanobis Distance (Approx.)"] = "Error: singular matrix"

        if use_kl or use_sym_div or use_jsd or use_hellinger or use_tv:
            p = data1 / np.sum(data1)
            q = data2 / np.sum(data2)

        if use_kl:
            metrics_results["Kullback-Leibler Divergence"] = entropy(p, q)

        if use_sym_div:
            metrics_results["Symmetrized Divergence"] = entropy(p, q) + entropy(q, p)

        if use_jsd:
            m = 0.5 * (p + q)
            metrics_results["Jensen-Shannon Divergence"] = 0.5 * (entropy(p, m) + entropy(q, m))

        if use_hellinger:
            metrics_results["Hellinger Metric"] = np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))

        if use_ks:
            stat, _ = ks_2samp(data1, data2)
            metrics_results["Kolmogorov-Smirnov Test (D)"] = stat

        if use_tv:
            metrics_results["Total Variation Distance"] = 0.5 * np.sum(np.abs(p - q))

        if use_xcorr:
            metrics_results["Simple Cross Correlation"] = np.correlate(data1, data2, mode="valid")[0]

        if use_nxcorr:
            metrics_results["Normalized Cross Correlation"] = np.corrcoef(data1, data2)[0, 1]

        if use_nzssd:
            d1_z = data1 - np.mean(data1)
            d2_z = data2 - np.mean(data2)
            metrics_results["NZSSD"] = np.sum((d1_z - d2_z)**2)

        if use_moravec:
            metrics_results["Moravec Correlation"] = np.sum(np.abs(np.diff(data1) - np.diff(data2)))

        if use_ioa:
            denom = np.sum((np.abs(data2 - np.mean(data1)) + np.abs(data1 - np.mean(data1)))**2)
            metrics_results["Index of Agreement"] = 1 - np.sum((data1 - data2)**2) / denom if denom != 0 else np.nan

        if use_sg:
            A = 100 * (data1 - data2) / (data2 + 1e-9)
            B = 100 * (data2 - data1) / (data1 + 1e-9)
            metrics_results["Sprague-Geers Metric"] = np.sqrt(np.mean(A**2)) + np.sqrt(np.mean(B**2))

        metrics_df = pd.DataFrame(metrics_results.items(), columns=["Metric", "Value"])
        st.dataframe(metrics_df)

        csv_buf = StringIO()
        metrics_df.to_csv(csv_buf, index=False)
        st.download_button("\ud83d\udcc5 Download Validation Metrics", csv_buf.getvalue(), file_name="validation_metrics.csv", mime="text/csv")

    # === Plotting ===
    def ecdf(data):
        x = np.sort(data)
        y = np.arange(1, len(data) + 1) / len(data)
        return x, y

    def save_plot(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png")
        return buf.getvalue()

    st.subheader("📊 Histogram")
    fig1, ax1 = plt.subplots()
    ax1.hist(data1, bins=30, alpha=0.6, label="Sample 1", color="skyblue")
    ax1.hist(data2, bins=30, alpha=0.6, label="Sample 2", color="salmon")
    ax1.set_title("Histogram of Samples")
    ax1.legend()
    st.pyplot(fig1)
    st.download_button("🖼️ Download Histogram", save_plot(fig1), file_name="histogram.png", mime="image/png")

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
