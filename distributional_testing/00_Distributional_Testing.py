import numpy as np
import scipy.stats as stats
import pandas as pd

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

    results_df = pd.DataFrame(results).sort_values(by='KS_Statistic')
    return results_df.head(top_n)


import pytest
import numpy as np
import pandas as pd
from scipy import stats
# from your_module import classify_distribution  # Replace 'your_module' with the actual module name

@pytest.mark.parametrize("dist_name, generator, params", [
    ("norm", np.random.normal, {"loc": 0, "scale": 1}),
    ("expon", np.random.exponential, {"scale": 1}),
    ("uniform", np.random.uniform, {"low": 0, "high": 1}),
    ("lognorm", lambda size: stats.lognorm.rvs(s=0.954, scale=np.exp(0), size=size), {}),
    ("gamma", lambda size: stats.gamma.rvs(a=2, scale=1, size=size), {}),
    ("beta", lambda size: stats.beta.rvs(a=2, b=5, size=size), {}),
    ("weibull_min", lambda size: stats.weibull_min.rvs(1.5, size=size), {}),
    ("weibull_max", lambda size: stats.weibull_max.rvs(1.5, size=size), {}),
    ("pareto", lambda size: stats.pareto.rvs(2.62, size=size), {}),
    ("triang", lambda size: stats.triang.rvs(c=0.5, loc=0, scale=1, size=size), {}),
])
def test_classify_known_distribution(dist_name, generator, params):
    size = 1000
    if isinstance(generator, np.random.Generator) or callable(generator) and params:
        data = generator(size=size, **params)
    else:
        data = generator(size)

    result = classify_distribution(data, dist_names=[dist_name])
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.iloc[0]['Distribution'] == dist_name
    assert 'KS_Statistic' in result.columns
    assert 'p_value' in result.columns
    assert isinstance(result.iloc[0]['Parameters'], tuple)


def test_classify_distribution_default_list():
    data = np.random.normal(0, 1, 1000)
    result = classify_distribution(data, top_n=5)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5
    assert set(['Distribution', 'KS_Statistic', 'p_value', 'Parameters']).issubset(result.columns)


def test_classify_invalid_data():
    data = [1, 2, 3, np.nan, np.inf, -np.inf]
    result = classify_distribution(data)
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_classify_empty_input():
    with pytest.raises(ValueError):
        classify_distribution([])


def test_classify_verbose_output(capsys):
    data = np.random.normal(0, 1, 100)
    classify_distribution(data, dist_names=['norm', 'expon'], verbose=True)
    captured = capsys.readouterr()
    assert "KS=" in captured.out or "fitting failed" in captured.out


import pytest
import numpy as np
import pandas as pd
# from your_module import classify_distribution  # Replace with actual import path

@pytest.fixture
def normal_data():
    return np.random.normal(loc=0, scale=1, size=2000)

@pytest.fixture
def uniform_data():
    return np.random.uniform(0, 1, size=1000)

def test_returns_dataframe(normal_data):
    result = classify_distribution(normal_data)
    assert isinstance(result, pd.DataFrame), "Output should be a pandas DataFrame"

def test_top_n_results(normal_data):
    result = classify_distribution(normal_data, top_n=5)
    assert len(result) == 5, "Should return top_n rows"

def test_distribution_columns(normal_data):
    result = classify_distribution(normal_data)
    expected_columns = {'Distribution', 'KS_Statistic', 'p_value', 'Parameters'}
    assert expected_columns.issubset(result.columns), "Missing expected result columns"

def test_known_best_fit(normal_data):
    result = classify_distribution(normal_data)
    top_dist = result.iloc[0]['Distribution']
    assert top_dist in ['norm', 'lognorm'], f"Expected 'norm' or similar, got {top_dist}"

def test_handles_nan_inf():
    data = np.array([1.0, 2.0, np.nan, 3.0, np.inf, 4.0])
    result = classify_distribution(data, top_n=2)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty, "Should handle NaN and inf and still return results"

def test_custom_distribution_list(uniform_data):
    dists = ['uniform', 'expon']
    result = classify_distribution(uniform_data, dist_names=dists)
    assert all(dist in dists for dist in result['Distribution']), "Only specified dists should be tested"

def test_verbose_does_not_crash(normal_data, capsys):
    classify_distribution(normal_data, verbose=True)
    captured = capsys.readouterr()
    assert "KS=" in captured.out or "fitting failed" in captured.out

import pytest
import numpy as np
import pandas as pd
from scipy import stats
# from your_module import classify_distribution  # Replace with actual module path

# List of all distributions to test
all_distributions = [
    'norm', 'expon', 'uniform', 'lognorm', 'gamma', 'beta',
    'weibull_min', 'weibull_max', 'pareto', 'triang'
]

@pytest.mark.parametrize("dist_name", all_distributions)
def test_each_distribution_runs(dist_name):
    """
    For each supported distribution, generate sample data and check that
    the function classifies it without error.
    """
    rng = np.random.default_rng(seed=42)
    
    # Generate synthetic data for each distribution
    if dist_name == 'norm':
        data = rng.normal(loc=0, scale=1, size=1000)
    elif dist_name == 'expon':
        data = rng.exponential(scale=1.0, size=1000)
    elif dist_name == 'uniform':
        data = rng.uniform(0, 1, size=1000)
    elif dist_name == 'lognorm':
        data = rng.lognormal(mean=0.0, sigma=1.0, size=1000)
    elif dist_name == 'gamma':
        data = rng.gamma(shape=2.0, scale=2.0, size=1000)
    elif dist_name == 'beta':
        data = rng.beta(a=2.0, b=5.0, size=1000)
    elif dist_name == 'weibull_min':
        data = stats.weibull_min.rvs(1.5, size=1000, random_state=rng)
    elif dist_name == 'weibull_max':
        data = stats.weibull_max.rvs(1.5, size=1000, random_state=rng)
    elif dist_name == 'pareto':
        data = stats.pareto.rvs(b=2.5, size=1000, random_state=rng)
    elif dist_name == 'triang':
        data = stats.triang.rvs(c=0.5, loc=0, scale=1, size=1000, random_state=rng)
    else:
        pytest.fail(f"Unhandled distribution: {dist_name}")

    # Run classification
    result = classify_distribution(data, dist_names=all_distributions, top_n=3)

    # Ensure result is a DataFrame
    assert isinstance(result, pd.DataFrame), "Expected result to be a DataFrame"
    assert not result.empty, "Result DataFrame should not be empty"

    # Check that tested dist appears in results (not necessarily at the top)
    assert dist_name in result['Distribution'].values, \
        f"Expected '{dist_name}' to be in top 3 fits for its own sample"
