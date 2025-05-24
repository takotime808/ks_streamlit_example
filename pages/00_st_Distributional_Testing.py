import pandas as pd
import streamlit as st
from scipy import stats
from distribution_classifier import classify_distribution, plot_distributions

st.title("Distribution Classifier & Visualizer")

st.sidebar.header("Upload Data or Generate Random")

upload = st.sidebar.file_uploader("Upload CSV with one numeric column", type=["csv"])
col = st.sidebar.text_input("Column name to use (if uploaded)", "")

# -- Distribution options
available_dists = [
    'norm', 'expon', 'uniform', 'lognorm', 'gamma', 'beta',
    'weibull_min', 'weibull_max', 'pareto', 'triang'
]


or_generate = st.sidebar.checkbox("Generate Synthetic Data")
gen_dist = st.sidebar.selectbox("Select Distribution", available_dists)
n_points = st.sidebar.slider("Number of data points", 100, 5000, 1000)

# -- Parameter inputs per distribution
dist_params = {}
st.sidebar.markdown("### Distribution Parameters")

if upload:
    df = pd.read_csv(upload)
    if col and col in df.columns:
        data = df[col].dropna().values
    else:
        st.warning("Please enter a valid column name.")
        st.stop()
elif or_generate:
    if gen_dist == 'norm':
        loc = st.sidebar.number_input("Mean (loc)", value=0.0)
        scale = st.sidebar.number_input("Std dev (scale)", value=1.0)
        dist_params = dict(loc=loc, scale=scale)

    elif gen_dist == 'expon':
        scale = st.sidebar.number_input("Scale", value=1.0)
        dist_params = dict(scale=scale)

    elif gen_dist == 'uniform':
        loc = st.sidebar.number_input("Start (loc)", value=0.0)
        scale = st.sidebar.number_input("Width (scale)", value=1.0)
        dist_params = dict(loc=loc, scale=scale)

    elif gen_dist == 'lognorm':
        s = st.sidebar.number_input("Shape (s)", value=0.954)
        scale = st.sidebar.number_input("Scale", value=1.0)
        dist_params = dict(s=s, scale=scale)

    elif gen_dist == 'gamma':
        a = st.sidebar.number_input("Shape (a)", value=2.0)
        scale = st.sidebar.number_input("Scale", value=1.0)
        dist_params = dict(a=a, scale=scale)

    elif gen_dist == 'beta':
        a = st.sidebar.number_input("Alpha (a)", value=2.0)
        b = st.sidebar.number_input("Beta (b)", value=5.0)
        dist_params = dict(a=a, b=b)

    elif gen_dist == 'weibull_min':
        c = st.sidebar.number_input("Shape (c)", value=1.5)
        dist_params = dict(c=c)

    elif gen_dist == 'weibull_max':
        c = st.sidebar.number_input("Shape (c)", value=1.5)
        dist_params = dict(c=c)

    elif gen_dist == 'pareto':
        b = st.sidebar.number_input("Shape (b)", value=2.62)
        dist_params = dict(b=b)

    elif gen_dist == 'triang':
        c = st.sidebar.number_input("Shape (c)", value=0.5, min_value=0.0, max_value=1.0)
        dist_params = dict(c=c, loc=0, scale=1)

    try:
        dist = getattr(stats, gen_dist)
        data = dist.rvs(size=n_points, **dist_params)
    except Exception as e:
        st.error(f"Error generating data: {e}")
        st.stop()
else:
    st.info("Upload a CSV or generate synthetic data.")
    st.stop()

st.success("Data loaded successfully!")
st.write("Preview of input data:")
st.write(data[:10])

top_n = st.slider("Number of top distributions to show", 1, 10, 3)

try:
    results_df = classify_distribution(data, top_n=top_n)
    st.subheader("Top Fitting Distributions")
    st.dataframe(results_df)
    
    st.subheader("Distribution Fit Plot")
    fig = plot_distributions(data, results_df)
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error: {e}")
