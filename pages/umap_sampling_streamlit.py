# Copyright (c) 2025 takotime808
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import KDTree
from scipy.stats import entropy
import umap
import plotly.express as px

st.set_page_config(page_title='UMAP Sampling Inference', layout='wide')
st.title('🔍 UMAP Sampling Method Inference')
st.markdown("""This app generates synthetic data from common sampling strategies (Grid, Random, LHS, Sobol),  
performs UMAP + clustering, and infers which sampling method was used based on structural patterns.  
You can also upload your own CSV (2 numeric columns) to test the inference!
""")

# --- Sidebar options ---
n_samples = st.sidebar.slider('Samples (for non-grid methods)', 50, 500, 100, step=10)
st.sidebar.markdown('---')
uploaded_file = st.sidebar.file_uploader('Upload CSV (2 columns)', type='csv')
if not uploaded_file:
    method_options = ["Grid", "Random", "LHS", "Sobol"]
    synth_method = st.sidebar.selectbox("Choose synthetic sampling method", method_options)
else:
    synth_method = None

run_button = st.sidebar.button('Run Inference')

if uploaded_file:
    st.info('Custom CSV uploaded!')

# --- Generate sampling data ---
def generate_synthetic(method, n=100):
    np.random.seed(42)
    if method == "Grid":
        X = np.array(np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))).T.reshape(-1, 2)
        label = ["Grid"] * len(X)
    elif method == "Random":
        X = np.random.rand(n, 2)
        label = ["Random"] * n
    elif method == "LHS":
        X = qmc.LatinHypercube(d=2).random(n)
        label = ["LHS"] * n
    elif method == "Sobol":
        X = qmc.Sobol(d=2).random(n)
        label = ["Sobol"] * n
    else:
        raise ValueError(f"Unknown method: {method}")
    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['TrueSampling'] = label
    return df

# --- Stricter Grid detection function ---
def is_grid_sample(X, tol=1e-3, min_grid=8):
    """
    Detects if data X is approximately on a grid.
    Now stricter: checks for near-equal spacing and enough unique grid lines.
    """
    x_unique = np.unique(np.round(X[:, 0], 4))
    y_unique = np.unique(np.round(X[:, 1], 4))
    def is_regular_spacing(arr):
        if len(arr) < 2:
            return False
        diffs = np.diff(np.sort(arr))
        return np.std(diffs) < tol
    return (len(x_unique) >= min_grid and len(y_unique) >= min_grid and
            is_regular_spacing(x_unique) and is_regular_spacing(y_unique))

# --- Inference function ---
def classify_sampling(embedding, clusters):
    dists = KDTree(embedding).query(embedding, k=2)[0][:, 1]
    std = np.std(dists)
    # Only compute silhouette if there are at least 2 clusters present
    n_labels = len(np.unique(clusters))
    if n_labels < 2:
        sil = np.nan
    else:
        sil = silhouette_score(embedding, clusters)
    ent = entropy(np.histogram(dists, bins=30, density=True)[0])
    if std < 0.05 and sil > 0.6:
        return 'Grid', 'Low std and high silhouette -> Grid'
    if std > 0.2 and sil < 0.3:
        return 'Random', 'High std and low silhouette -> Random'
    if 0.05 <= std <= 0.15:
        return ('Sobol', 'Moderate spread, low entropy -> Sobol') if ent < 2.0 else ('LHS', 'Moderate spread, higher entropy -> LHS')
    return 'Uncertain', 'Pattern unclear'


# --- Main run ---
if run_button:
    # Data: user-uploaded or synthetic
    if uploaded_file:
        try:
            df_user = pd.read_csv(uploaded_file)
            # Expect at least two columns; take first two numeric ones
            num_cols = df_user.select_dtypes(include=np.number).columns
            if len(num_cols) < 2:
                st.error("Uploaded CSV must have at least 2 numeric columns.")
                df = None
            else:
                df = df_user.loc[:, num_cols[:2]].copy()
                df.columns = ['x1', 'x2']
                df['TrueSampling'] = "UserData"
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None
    else:
        df = generate_synthetic(synth_method, n_samples)

    # Proceed only if df loaded
    if df is not None:
        X = df[['x1', 'x2']].values

        # --- Debug info: grid detection thresholds ---
        x_unique = np.unique(np.round(X[:, 0], 4))
        y_unique = np.unique(np.round(X[:, 1], 4))
        x_diff_std = np.std(np.diff(np.sort(x_unique))) if len(x_unique) > 1 else np.nan
        y_diff_std = np.std(np.diff(np.sort(y_unique))) if len(y_unique) > 1 else np.nan
        st.caption(
            f"Unique x: {len(x_unique)}, std of x diffs: {x_diff_std:.5f} | "
            f"Unique y: {len(y_unique)}, std of y diffs: {y_diff_std:.5f}"
        )

        # --- Grid detection (before UMAP/clustering) ---
        if is_grid_sample(X):
            st.success("**Detected Sampling Method: Grid**")
            st.caption("*Justification: Points are arranged in a regular grid.*")
            fig = px.scatter(df, x='x1', y='x2', color='TrueSampling' if 'TrueSampling' in df.columns else None,
                             title="Detected Grid Sampling")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Continue with UMAP + clustering as before
            X_scaled = StandardScaler().fit_transform(X)
            X_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit_transform(X_scaled)
            # n_clusters = 2 if uploaded_file else 1 if synth_method == "Grid" else 1 if len(df['TrueSampling'].unique()) == 1 else 4
            n_clusters = 2 if len(df) > 2 else 1  # Always use at least 2 if possible
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_umap)
            df_umap = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])
            df_umap['Cluster'] = clusters
            df_umap['TrueSampling'] = df['TrueSampling']
            method, explanation = classify_sampling(X_umap, clusters)
            st.success(f'**Inferred Sampling Method:** {method}')
            st.caption(f'*Justification:* {explanation}')
            fig = px.scatter(
                df_umap, x='UMAP1', y='UMAP2', color='Cluster', symbol='TrueSampling',
                title=f'UMAP 2D Projection - Inferred: {method}', hover_data=['TrueSampling']
            )
            st.plotly_chart(fig, use_container_width=True)
