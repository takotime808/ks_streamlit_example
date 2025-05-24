# distribution_classifier.py
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

def classify_distribution(data, dist_names=None, top_n=3, verbose=False):
    data = np.asarray(data)
    data = data[np.isfinite(data)]

    if data.size == 0:
        raise ValueError("Input data must contain at least one finite number.")

    if dist_names is None:
        dist_names = [
            'norm', 'expon', 'uniform', 'lognorm', 'gamma', 'beta',
            'weibull_min', 'weibull_max', 'pareto', 'triang'
        ]

    results = []

    for dist_name in dist_names:
        dist = getattr(stats, dist_name)
        try:
            params = dist.fit(data)
            D, p = stats.kstest(data, dist_name, args=params)
            results.append({
                'Distribution': dist_name,
                'KS_Statistic': D,
                'p_value': p,
                'Parameters': params
            })
            if verbose:
                print(f"{dist_name}: KS={D:.4f}, p={p:.4f}, params={params}")
        except Exception as e:
            if verbose:
                print(f"{dist_name} fitting failed: {e}")

    if not results:
        raise RuntimeError("No distributions could be fitted to the input data.")

    results_df = pd.DataFrame(results).sort_values(by='KS_Statistic').head(top_n)
    return results_df

def plot_distributions(data, top_results):
    x = np.linspace(np.min(data), np.max(data), 1000)
    fig, ax = plt.subplots()
    ax.hist(data, bins=30, density=True, alpha=0.5, label='Data')

    for _, row in top_results.iterrows():
        dist = getattr(stats, row['Distribution'])
        y = dist.pdf(x, *row['Parameters'])
        ax.plot(x, y, label=row['Distribution'])

    ax.set_title("Top Fitted Distributions")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)
    return fig
