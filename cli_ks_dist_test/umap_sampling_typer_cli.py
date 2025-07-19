#!/usr/bin/env python3
# Typer CLI: UMAP Sampling Method Inference
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
import plotly.io as pio
import typer

app = typer.Typer()

def generate_data(n: int = 100):
    np.random.seed(42)
    grid = np.array(np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))).T.reshape(-1, 2)
    lhs = qmc.LatinHypercube(d=2).random(n)
    sobol = qmc.Sobol(d=2).random(n)
    rand = np.random.rand(n, 2)
    X = np.vstack([grid, rand, lhs, sobol])
    labels = ['Grid'] * len(grid) + ['Random'] * n + ['LHS'] * n + ['Sobol'] * n
    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['TrueSampling'] = labels
    return df

def classify_sampling(embedding, clusters):
    dists = KDTree(embedding).query(embedding, k=2)[0][:, 1]
    std = np.std(dists)
    sil = silhouette_score(embedding, clusters)
    ent = entropy(np.histogram(dists, bins=30, density=True)[0])
    if std < 0.05 and sil > 0.6:
        return 'Grid', 'Low std and high silhouette -> Grid'
    if std > 0.2 and sil < 0.3:
        return 'Random', 'High std and low silhouette -> Random'
    if 0.05 <= std <= 0.15:
        return ('Sobol', 'Moderate spread, low entropy -> Sobol') if ent < 2.0 else ('LHS', 'Moderate spread, higher entropy -> LHS')
    return 'Uncertain', 'Pattern unclear'

@app.command()
def main(
    html: str = typer.Option("umap_output.html", help="Output HTML file path"),
    png: str = typer.Option("umap_output.png", help="Output PNG file path"),
    n: int = typer.Option(100, help="Number of samples for each method (except grid)")
):
    df = generate_data(n)
    X_scaled = StandardScaler().fit_transform(df[['x1', 'x2']])
    X_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_umap)
    df_umap = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])
    df_umap['Cluster'] = clusters
    df_umap['TrueSampling'] = df['TrueSampling']
    method, explanation = classify_sampling(X_umap, clusters)
    print(f'Inferred Sampling Method: {method}')
    print(f'Justification: {explanation}')
    fig = px.scatter(df_umap, x='UMAP1', y='UMAP2', color='Cluster', symbol='TrueSampling',
                     title=f'UMAP 2D Projection - Inferred: {method}', hover_data=['TrueSampling'])
    fig.write_html(html)
    pio.write_image(fig, png, width=1000, height=800)
    fig.show()

if __name__ == "__main__":
    app()