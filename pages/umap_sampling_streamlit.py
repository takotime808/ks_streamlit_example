# Copyright (c) 2025 takotime808

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import qmc, entropy
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import KDTree
import umap
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title='UMAP Sampling Inference', layout='wide')
st.title('🔍 UMAP Sampling Method Inference')

st.markdown("""This app generates synthetic data from common sampling strategies (Grid, Random, LHS, Sobol),  
performs UMAP + clustering, and infers which sampling method was used based on structural patterns.  
It also shows a decision boundary plot based on nearest-neighbor distance and entropy.
""")

# --- Sidebar options ---
n_samples = st.sidebar.slider('Samples (for non-grid methods)', 50, 500, 100, step=10)
uploaded_file = st.sidebar.file_uploader('Upload CSV (2 columns)', type='csv')
if not uploaded_file:
    method_options = ["Grid", "Random", "LHS", "Sobol"]
    synth_method = st.sidebar.selectbox("Choose synthetic sampling method", method_options)
else:
    synth_method = None

run_button = st.sidebar.button('Run Inference')


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


def is_grid_sample(X, tol=1e-3, min_grid=8):
    x_unique = np.unique(np.round(X[:, 0], 4))
    y_unique = np.unique(np.round(X[:, 1], 4))

    def is_regular_spacing(arr):
        if len(arr) < 2:
            return False
        diffs = np.diff(np.sort(arr))
        return np.std(diffs) < tol

    return (len(x_unique) >= min_grid and len(y_unique) >= min_grid and
            is_regular_spacing(x_unique) and is_regular_spacing(y_unique))


def classify_sampling(embedding, clusters):
    dists = KDTree(embedding).query(embedding, k=2)[0][:, 1]
    std = np.std(dists)
    ent = entropy(np.histogram(dists, bins=30, density=True)[0] + 1e-9)
    sil = silhouette_score(embedding, clusters) if len(np.unique(clusters)) >= 2 else 0

    st.caption(f"STD of NN distances: {std:.4f} | Silhouette: {sil:.4f} | Entropy: {ent:.4f}")

    if std < 0.05 and sil > 0.5 and ent < 1.5:
        return 'Grid', 'Low std, low entropy, and structured clusters → Grid'
    elif std > 0.25 and sil < 0.25 and ent > 2.5:
        return 'Random', 'High std, high entropy, low silhouette → Random'
    elif 0.10 <= std <= 0.25:
        if ent < 2.0:
            return 'Sobol', 'Moderate std and entropy → Sobol'
        else:
            return 'LHS', 'Moderate std and higher entropy → LHS'
    return 'Uncertain', 'Pattern did not match any known structure'


def plot_decision_boundary():
    std_vals = np.linspace(0.01, 0.35, 200)
    ent_vals = np.linspace(0.5, 3.5, 200)
    STD, ENT = np.meshgrid(std_vals, ent_vals)
    sil = 0.6 * np.ones_like(STD)

    def classify(std, sil, ent):
        if std < 0.05 and sil > 0.5 and ent < 1.5:
            return 'Grid'
        elif std > 0.25 and sil < 0.25 and ent > 2.5:
            return 'Random'
        elif 0.10 <= std <= 0.25:
            if ent < 2.0:
                return 'Sobol'
            else:
                return 'LHS'
        return 'Uncertain'

    label_map = {'Grid': 0, 'Sobol': 1, 'LHS': 2, 'Random': 3, 'Uncertain': 4}
    labels = np.empty_like(STD, dtype=object)
    for i in range(STD.shape[0]):
        for j in range(STD.shape[1]):
            labels[i, j] = classify(STD[i, j], sil[i, j], ENT[i, j])
    int_labels = np.vectorize(label_map.get)(labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    c = ax.contourf(STD, ENT, int_labels, levels=np.arange(len(label_map)+1)-0.5, cmap='tab10', alpha=0.8)
    cbar = fig.colorbar(c, ticks=list(label_map.values()))
    cbar.ax.set_yticklabels(list(label_map.keys()))
    ax.set_xlabel('Standard Deviation of NN Distances')
    ax.set_ylabel('Entropy of Distances Histogram')
    ax.set_title('Sampling Method Decision Boundaries')
    st.pyplot(fig)


if run_button:
    if uploaded_file:
        try:
            df_user = pd.read_csv(uploaded_file)
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

    if df is not None:
        X = df[['x1', 'x2']].values

        if is_grid_sample(X):
            st.success("**Detected Sampling Method: Grid**")
            st.caption("*Justification: Points are arranged in a regular grid.*")
            fig = px.scatter(df, x='x1', y='x2', color='TrueSampling', title="Detected Grid Sampling")
            st.plotly_chart(fig, use_container_width=True)
        else:
            X_scaled = StandardScaler().fit_transform(X)
            X_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit_transform(X_scaled)
            kmeans = KMeans(n_clusters=2, random_state=42)
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

        st.subheader("📊 Decision Boundary Plot")
        plot_decision_boundary()
