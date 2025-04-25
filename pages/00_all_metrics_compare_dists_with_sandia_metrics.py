# [Start of Code]

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, anderson_ksamp, zscore, entropy
from scipy.spatial.distance import minkowski, mahalanobis
from scipy.signal import correlate
from sklearn.metrics import mean_squared_error
from io import StringIO, BytesIO
import matplotlib.pyplot as plt

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

# === If both files uploaded ===
if uploaded_file1 and uploaded_file2:
    df1 = pd.read_csv(uploaded_file1)
    df2 = pd.read_csv(uploaded_file2)

    df1 = filter_dataframe(df1, "Sample 1")
    df2 = filter_dataframe(df2, "Sample 2")

    st.sidebar.header("Step 2: Select Columns")
    col1 = st.sidebar.selectbox("Column from Sample 1", df1.select_dtypes(include=np.number).columns)
    col2 = st.sidebar.selectbox("Column from Sample 2", df2.select_dtypes(include=np.number).columns)

    data1 = df1[col1].dropna().to_numpy()
    data2 = df2[col2].dropna().to_numpy()

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

    # === Step 4a: Choose which Sandia metrics you want reported back ===
    st.sidebar.header("Step 4a: Choose Validation Metrics")

    with st.sidebar.expander("📐 Deterministic Validation Metrics", expanded=False):
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

        from scipy.spatial.distance import minkowski, euclidean, hellinger
        from scipy.stats import entropy
        from numpy.linalg import inv
        import math

        metrics_results = {}

        if use_rmse:
            rmse = np.sqrt(np.mean((data1 - data2) ** 2))
            metrics_results["Root Mean Square Error"] = rmse

        if use_minkowski:
            mink = minkowski(data1, data2, p=2)
            metrics_results["Minkowski Distance (Lp=2)"] = mink

        if use_neuclidean:
            neuclid = np.linalg.norm(data1 - data2) / len(data1)
            metrics_results["Normalized Euclidean Metric"] = neuclid

        if use_mahalanobis:
            data = np.vstack([data1, data2])
            cov = np.cov(data, rowvar=False)
            diff = np.mean(data1) - np.mean(data2)
            try:
                m_dist = np.sqrt(diff**2 / cov.mean())
                metrics_results["Mahalanobis Distance (Approx.)"] = m_dist
            except:
                metrics_results["Mahalanobis Distance (Approx.)"] = "Error: singular matrix"

        if use_kl:
            p = data1 / np.sum(data1)
            q = data2 / np.sum(data2)
            kl_div = entropy(p, q)
            metrics_results["Kullback-Leibler Divergence"] = kl_div

        if use_sym_div:
            p = data1 / np.sum(data1)
            q = data2 / np.sum(data2)
            sym_div = entropy(p, q) + entropy(q, p)
            metrics_results["Symmetrized Divergence"] = sym_div

        if use_jsd:
            m = 0.5 * (p + q)
            jsd = 0.5 * (entropy(p, m) + entropy(q, m))
            metrics_results["Jensen-Shannon Divergence"] = jsd

        if use_hellinger:
            h = np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))
            metrics_results["Hellinger Metric"] = h

        if use_ks:
            ks_stat, _ = ks_2samp(data1, data2)
            metrics_results["Kolmogorov-Smirnov Test (D)"] = ks_stat

        if use_tv:
            tv = 0.5 * np.sum(np.abs(p - q))
            metrics_results["Total Variation Distance"] = tv

        if use_xcorr:
            xc = np.correlate(data1, data2, mode="valid")[0]
            metrics_results["Simple Cross Correlation"] = xc

        if use_nxcorr:
            nxc = np.corrcoef(data1, data2)[0, 1]
            metrics_results["Normalized Cross Correlation"] = nxc

        if use_nzssd:
            d1_z = data1 - np.mean(data1)
            d2_z = data2 - np.mean(data2)
            nzssd = np.sum((d1_z - d2_z)**2)
            metrics_results["NZSSD"] = nzssd

        if use_moravec:
            moravec = np.sum(np.abs(np.diff(data1) - np.diff(data2)))
            metrics_results["Moravec Correlation"] = moravec

        if use_ioa:
            denom = np.sum((np.abs(data2 - np.mean(data1)) + np.abs(data1 - np.mean(data1)))**2)
            ioa = 1 - np.sum((data1 - data2)**2) / denom if denom != 0 else np.nan
            metrics_results["Index of Agreement"] = ioa

        if use_sg:
            A = 100 * (data1 - data2) / (data2 + 1e-9)
            B = 100 * (data2 - data1) / (data1 + 1e-9)
            SG = np.sqrt(np.mean(A**2)) + np.sqrt(np.mean(B**2))
            metrics_results["Sprague-Geers Metric"] = SG

        metrics_df = pd.DataFrame(metrics_results.items(), columns=["Metric", "Value"])
        st.dataframe(metrics_df)

        csv_buf = StringIO()
        metrics_df.to_csv(csv_buf, index=False)
        st.download_button("📥 Download Validation Metrics", csv_buf.getvalue(), file_name="validation_metrics.csv", mime="text/csv")


    # === Step 4b: Choose Statistical Test ===
    st.sidebar.header("Step 4b: Statistical Test")
    test_type = st.sidebar.selectbox("Test Type", ["Kolmogorov-Smirnov", "Anderson-Darling"])
    alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.10, 0.05, step=0.01)

    st.header("📈 Test Results")
    if test_type == "Kolmogorov-Smirnov":
        stat, p_value = ks_2samp(data1, data2)
        test_name = "Kolmogorov-Smirnov"
    else:
        result = anderson_ksamp([data1, data2])
        stat = result.statistic
        p_value = result.significance_level / 100
        test_name = "Anderson-Darling"

    conclusion = (
        "❌ Reject H₀: Distributions are different"
        if p_value < alpha else "✅ Fail to reject H₀: Distributions are similar"
    )

    st.markdown(f"**Test:** {test_name}")
    st.markdown(f"**Statistic:** `{stat:.4f}`")
    st.markdown(f"**P-value:** `{p_value:.4f}`")
    st.markdown(f"**Conclusion (α = {alpha:.2f}):** {conclusion}")

    # === Step 5: Select Validation Metrics ===
    st.sidebar.header("Step 5: Select Validation Metrics")

    with st.sidebar.expander("Deterministic Metrics", expanded=True):
        rmse = st.checkbox("Root Mean Square Error")
        minkowski_dist = st.checkbox("Minkowski Distance")

    with st.sidebar.expander("Probability-Based Metrics", expanded=True):
        kl_div = st.checkbox("Kullback-Leibler Divergence")
        js_div = st.checkbox("Jensen-Shannon Divergence")
        hellinger = st.checkbox("Hellinger Metric")
        total_variation = st.checkbox("Total Variation Distance")
        mahal_dist = st.checkbox("Mahalanobis Distance")
        norm_euclidean = st.checkbox("Normalized Euclidean Metric")
        sym_div = st.checkbox("Symmetrized Divergence")

    with st.sidebar.expander("Signal Processing Metrics", expanded=True):
        cross_corr = st.checkbox("Simple Cross Correlation")
        norm_cross_corr = st.checkbox("Normalized Cross Correlation")
        nzmssd = st.checkbox("Normalized Zero-Mean SSD")
        moravec = st.checkbox("Moravec Correlation")
        index_agreement = st.checkbox("Index of Agreement")
        sprague_geers = st.checkbox("Sprague-Geers Metric")

    # === Metric Calculation Functions ===
    def compute_metrics(data1, data2):
        results = {}
        if rmse:
            results["RMSE"] = np.sqrt(mean_squared_error(data1, data2))
        if minkowski_dist:
            results["Minkowski (p=2)"] = minkowski(data1, data2, p=2)
        if kl_div:
            hist1, _ = np.histogram(data1, bins=50, density=True)
            hist2, _ = np.histogram(data2, bins=50, density=True)
            hist1 += 1e-10
            hist2 += 1e-10
            results["KL Divergence"] = entropy(hist1, hist2)
        if js_div:
            M = 0.5 * (hist1 + hist2)
            results["JS Divergence"] = 0.5 * entropy(hist1, M) + 0.5 * entropy(hist2, M)
        if hellinger:
            results["Hellinger"] = np.sqrt(0.5 * np.sum((np.sqrt(hist1) - np.sqrt(hist2))**2))
        if total_variation:
            results["Total Variation"] = 0.5 * np.sum(np.abs(hist1 - hist2))
        if mahal_dist:
            try:
                cov = np.cov(np.stack([data1, data2]), rowvar=False)
                cov_inv = np.linalg.inv(cov)
                diff = np.mean(data1) - np.mean(data2)
                results["Mahalanobis"] = np.sqrt(diff.T @ cov_inv @ diff)
            except np.linalg.LinAlgError:
                results["Mahalanobis"] = np.nan
        if norm_euclidean:
            norm1 = np.linalg.norm(data1)
            norm2 = np.linalg.norm(data2)
            results["Normalized Euclidean"] = np.linalg.norm(data1 - data2) / (norm1 + norm2)
        if sym_div:
            results["Symmetrized Divergence"] = entropy(hist1, hist2) + entropy(hist2, hist1)
        if cross_corr:
            results["Cross Corr"] = np.correlate(data1, data2, mode="valid")[0]
        if norm_cross_corr:
            results["Norm Cross Corr"] = np.correlate(zscore(data1), zscore(data2), mode="valid")[0]
        if nzmssd:
            results["NZMSSD"] = np.sum((zscore(data1) - zscore(data2))**2)
        if moravec:
            results["Moravec"] = np.mean(np.abs(np.gradient(data1) - np.gradient(data2)))
        if index_agreement:
            mean_obs = np.mean(data1)
            denom = np.sum((np.abs(data2 - mean_obs) + np.abs(data1 - mean_obs))**2)
            results["Index of Agreement"] = 1 - np.sum((data1 - data2)**2) / denom if denom != 0 else np.nan
        if sprague_geers:
            diff = data1 - data2
            sum_sq = np.sum((diff / (data1 + data2 + 1e-10))**2)
            results["Sprague-Geers"] = np.sqrt(sum_sq / len(data1))
        return results

    metrics_results = compute_metrics(data1, data2)
    if metrics_results:
        st.subheader("📐 Validation Metrics")
        metrics_df = pd.DataFrame.from_dict(metrics_results, orient="index", columns=["Value"])
        st.dataframe(metrics_df)

        csv_buf2 = StringIO()
        metrics_df.to_csv(csv_buf2)
        st.download_button("📥 Download Metrics CSV", csv_buf2.getvalue(), file_name="metrics_results.csv", mime="text/csv")

    # === Plots ===
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

# [End of Code]
