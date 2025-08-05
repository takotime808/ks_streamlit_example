import pandas as pd
import numpy as np
import streamlit as st
import yaml
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import energy_distance
import matplotlib.pyplot as plt

from audit_sampling import (
    build_bounds_dists,
    create_lhs_samples,
    create_grid_samples,
    create_random_samples,
    ks_multivariate_test,
    ad_multivariate_test,
    chi2_multivariate_test,
    entropy_score,
)

st.title("Sampling Audit")

st.write(
    "Compare a user dataset against synthetic samples generated via Latin Hypercube,"
    " random, and grid sampling strategies. Upload your own CSV/YAML pair or run the"
    " built-in example."
)

uploaded_csv = st.file_uploader("CSV file", type=["csv"])
uploaded_yaml = st.file_uploader("YAML config", type=["yml", "yaml"])
n_samples = st.number_input(
    "Samples to generate (defaults to size of CSV)", min_value=1, value=100
)


def run_audit(df: pd.DataFrame, yaml_cfg: dict, n_samples: int):
    dep_vars = yaml_cfg["dependent_variables"]
    var_descs = build_bounds_dists(yaml_cfg)
    user_data = df[dep_vars].values
    n_samples = n_samples or len(user_data)
    scaler = MinMaxScaler()
    user_data_scaled = scaler.fit_transform(user_data.astype(float))

    methods, scaled = {}, {}
    for name, func in [
        ("LHS", create_lhs_samples),
        ("Grid", create_grid_samples),
        ("Random", create_random_samples),
    ]:
        try:
            smpl = func(var_descs, n_samples)
            methods[name] = smpl
            scaled[name] = scaler.transform(smpl.astype(float))
        except Exception as e:
            st.warning(f"Could not create {name} samples: {e}")

    rows = []
    best_method, min_total = None, float("inf")

    for method, synth in methods.items():
        synth_scaled = scaled[method]
        ks = ks_multivariate_test(user_data_scaled, synth_scaled)
        ad = ad_multivariate_test(user_data_scaled, synth_scaled)
        chi2 = chi2_multivariate_test(user_data_scaled, synth_scaled)
        energy = energy_distance(user_data_scaled, synth_scaled)
        entropy = entropy_score(synth_scaled)
        total = ks + ad + energy
        if total < min_total:
            min_total = total
            best_method = method
        rows.append(
            dict(
                method=method,
                ks_stat=ks,
                ad_stat=ad,
                chi2_stat=chi2,
                energy_stat=energy,
                coverage_score=entropy,
            )
        )

        with st.expander(f"{method} diagnostics"):
            for i, var in enumerate(dep_vars):
                st.markdown(f"**{var}**")
                col1, col2, col3 = st.columns(3)

                with col1:
                    fig, ax = plt.subplots()
                    ax.hist(user_data_scaled[:, i], bins=20, alpha=0.5, label="User")
                    ax.hist(synth_scaled[:, i], bins=20, alpha=0.5, label=method)
                    ax.set_title("Histogram")
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)

                with col2:
                    fig, ax = plt.subplots()
                    ecdf_user = np.sort(user_data_scaled[:, i])
                    ecdf_synth = np.sort(synth_scaled[:, i])
                    ax.plot(ecdf_user, np.linspace(0, 1, len(ecdf_user)), label="User")
                    ax.plot(ecdf_synth, np.linspace(0, 1, len(ecdf_synth)), label=method)
                    ax.set_title("ECDF")
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)

                with col3:
                    fig, ax = plt.subplots()
                    qq_user = np.sort(user_data_scaled[:, i])
                    qq_synth = np.sort(synth_scaled[:, i])
                    n = min(len(qq_user), len(qq_synth))
                    ax.plot(qq_user[:n], qq_synth[:n], "o")
                    ax.plot(
                        [qq_user[0], qq_user[-1]],
                        [qq_user[0], qq_user[-1]],
                        "k--",
                    )
                    ax.set_title("QQ")
                    st.pyplot(fig)
                    plt.close(fig)

    st.subheader("Summary")
    st.dataframe(pd.DataFrame(rows))
    st.success(f"Best method: {best_method}")


if st.button("Run Audit"):
    if uploaded_csv and uploaded_yaml:
        df = pd.read_csv(uploaded_csv)
        yaml_cfg = yaml.safe_load(uploaded_yaml)
        run_audit(df, yaml_cfg, n_samples)
    else:
        st.error("Please upload both CSV and YAML files.")

if st.button("Run Example"):
    df = pd.read_csv("examples/sample_data.csv")
    with open("examples/config.yaml") as f:
        yaml_cfg = yaml.safe_load(f)
    run_audit(df, yaml_cfg, n_samples)
