import argparse
import pandas as pd
import numpy as np
import yaml
import sys
from scipy.stats import (
    kstest, anderson_ksamp, chisquare, energy_distance
)
from sklearn.preprocessing import MinMaxScaler
from itertools import product
from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
import os


def load_csv(csv_path):
    return pd.read_csv(csv_path)


def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_bounds_dists(yaml_cfg):
    dep_vars = yaml_cfg["dependent_variables"]
    dists = yaml_cfg["distributions"]
    stat1 = yaml_cfg["stat1"]
    stat2 = yaml_cfg["stat2"]
    stat3 = yaml_cfg["stat3"]
    results = []
    for i, var in enumerate(dep_vars):
        distrib = dists[i]
        s1, s2, s3 = stat1[i], stat2[i], stat3[i]
        results.append(dict(name=var, dist=distrib, stat1=s1, stat2=s2, stat3=s3))
    return results


def sample_from_dist(desc, n):
    from scipy.stats import norm, gamma, beta
    if isinstance(desc["dist"], list):
        values = np.array(desc["dist"])
        idxs = np.random.choice(len(values), size=n, replace=True)
        return values[idxs]
    elif desc["dist"] == "uniform":
        return np.random.uniform(desc["stat1"], desc["stat2"], size=n)
    elif desc["dist"] == "normal":
        return np.random.normal(loc=desc["stat1"], scale=np.sqrt(desc["stat2"]), size=n)
    elif desc["dist"] == "gamma":
        return gamma.rvs(a=desc["stat3"], loc=desc["stat1"], scale=desc["stat2"], size=n)
    elif desc["dist"] == "beta":
        a, b = desc["stat1"], desc["stat2"]
        loc = desc["stat3"] if desc["stat3"] is not None else 0
        return beta.rvs(a, b, loc=loc, size=n)
    else:
        raise ValueError(f"Unsupported distribution: {desc['dist']}")


def create_lhs_samples(var_descs, n_samples):
    from scipy.stats import qmc, norm, gamma, beta
    num_vars = len(var_descs)
    sampler = qmc.LatinHypercube(d=num_vars)
    lhs_points = sampler.random(n=n_samples)
    result = []
    for i, desc in enumerate(var_descs):
        if isinstance(desc["dist"], list):
            vals = np.array(desc["dist"])
            idxs = (lhs_points[:, i] * len(vals)).astype(int).clip(0, len(vals)-1)
            result.append(vals[idxs])
        elif desc["dist"] == "uniform":
            low, high = desc["stat1"], desc["stat2"]
            vals = lhs_points[:, i] * (high - low) + low
            result.append(vals)
        elif desc["dist"] == "normal":
            vals = norm.ppf(lhs_points[:, i], loc=desc["stat1"], scale=np.sqrt(desc["stat2"]))
            result.append(vals)
        elif desc["dist"] == "gamma":
            vals = gamma.ppf(lhs_points[:, i], a=desc["stat3"], loc=desc["stat1"], scale=desc["stat2"])
            result.append(vals)
        elif desc["dist"] == "beta":
            a, b = desc["stat1"], desc["stat2"]
            loc = desc["stat3"] if desc["stat3"] is not None else 0
            vals = beta.ppf(lhs_points[:, i], a, b, loc=loc)
            result.append(vals)
        else:
            raise ValueError(f"LHS not supported for {desc['dist']}")
    return np.column_stack(result)


def create_grid_samples(var_descs, n_samples):
    grids = []
    for desc in var_descs:
        if isinstance(desc["dist"], list):
            vals = np.array(desc["dist"])
        elif desc["dist"] == "uniform":
            vals = np.linspace(desc["stat1"], desc["stat2"], int(np.cbrt(n_samples)))
        elif desc["dist"] == "normal":
            mean, std = desc["stat1"], np.sqrt(desc["stat2"])
            vals = np.linspace(mean - 2*std, mean + 2*std, int(np.cbrt(n_samples)))
        elif desc["dist"] == "gamma":
            a = desc["stat3"]
            mean = desc["stat1"] + desc["stat2"]*a
            std = np.sqrt(a)*desc["stat2"]
            vals = np.linspace(mean - 2*std, mean + 2*std, int(np.cbrt(n_samples)))
        elif desc["dist"] == "beta":
            a, b = desc["stat1"], desc["stat2"]
            loc = desc["stat3"] if desc["stat3"] is not None else 0
            mean = a/(a+b)
            std = np.sqrt(a*b/((a+b)**2*(a+b+1)))
            vals = np.linspace(mean - 2*std, mean + 2*std, int(np.cbrt(n_samples)))
        else:
            raise ValueError(f"Grid not supported for {desc['dist']}")
        grids.append(vals)
    grid_points = np.array(list(product(*grids)))
    if len(grid_points) > n_samples:
        idx = np.random.choice(len(grid_points), n_samples, replace=False)
        grid_points = grid_points[idx]
    return grid_points


def create_random_samples(var_descs, n_samples):
    return np.column_stack([sample_from_dist(desc, n_samples) for desc in var_descs])


def ks_multivariate_test(data, sampled):
    D_sum = 0
    for i in range(data.shape[1]):
        try:
            D, _ = kstest(data[:, i], sampled[:, i])
        except Exception:
            vals = np.unique(np.concatenate([data[:, i], sampled[:, i]]))
            freq_data = np.array([np.sum(data[:, i] == x)/len(data) for x in vals])
            freq_sampled = np.array([np.sum(sampled[:, i] == x)/len(sampled) for x in vals])
            D = np.max(np.abs(np.cumsum(freq_data) - np.cumsum(freq_sampled)))
        D_sum += D
    return D_sum


def ad_multivariate_test(data, sampled):
    D_sum = 0
    for i in range(data.shape[1]):
        try:
            ad_result = anderson_ksamp([data[:, i], sampled[:, i]])
            D_sum += ad_result.statistic
        except Exception:
            D_sum += 0
    return D_sum


def chi2_multivariate_test(data, sampled, bins=10):
    D_sum = 0
    for i in range(data.shape[1]):
        try:
            _data = data[:, i]
            _sampled = sampled[:, i]
            all_x = np.concatenate([_data, _sampled])
            if np.issubdtype(_data.dtype, np.number):
                bins_ = np.histogram_bin_edges(all_x, bins=bins)
                obs, _ = np.histogram(_data, bins=bins_)
                exp, _ = np.histogram(_sampled, bins=bins_)
            else:
                cats = list(set(_data) | set(_sampled))
                obs = [np.sum(_data == c) for c in cats]
                exp = [np.sum(_sampled == c) for c in cats]
            stat, _ = chisquare(f_obs=obs, f_exp=exp+np.finfo(float).eps)
            D_sum += stat
        except Exception:
            D_sum += 0
    return D_sum


def entropy_score(data, bins=10):
    entropies = []
    for i in range(data.shape[1]):
        x = data[:, i]
        if np.issubdtype(x.dtype, np.number):
            p = np.histogram(x, bins=bins, density=True)[0]
        else:
            vals, counts = np.unique(x, return_counts=True)
            p = counts / counts.sum()
        p = p[p > 0]
        entropies.append(-np.sum(p * np.log(p)))
    return np.mean(entropies)


def plot_diagnostics(user_data, synthetic, dep_vars, out_dir, method):
    os.makedirs(out_dir, exist_ok=True)
    filenames = {"histograms":[], "ecdf":[], "qq":[]}
    for i, var in enumerate(dep_vars):
        plt.figure()
        plt.hist(user_data[:, i], bins=20, alpha=0.5, label="UserData")
        plt.hist(synthetic[:, i], bins=20, alpha=0.5, label=method)
        plt.legend()
        fn = f"{out_dir}/hist_{method}_{var}.png"
        plt.title(f"Histogram {var} ({method})")
        plt.savefig(fn); filenames["histograms"].append(fn); plt.close()
        plt.figure()
        ecdf_user = np.sort(user_data[:, i])
        ecdf_synth = np.sort(synthetic[:, i])
        plt.plot(ecdf_user, np.linspace(0, 1, len(ecdf_user)), label="UserData")
        plt.plot(ecdf_synth, np.linspace(0, 1, len(ecdf_synth)), label=method)
        plt.title(f"ECDF {var} ({method})")
        plt.legend()
        fn = f"{out_dir}/ecdf_{method}_{var}.png"
        plt.savefig(fn); filenames["ecdf"].append(fn); plt.close()
        plt.figure()
        qq_user = np.sort(user_data[:, i])
        qq_synth = np.sort(synthetic[:, i])
        n = min(len(qq_user), len(qq_synth))
        plt.plot(qq_user[:n], qq_synth[:n], 'o')
        plt.plot([qq_user[0], qq_user[-1]], [qq_user[0], qq_user[-1]], 'k--')
        plt.title(f"QQ {var} ({method})")
        fn = f"{out_dir}/qq_{method}_{var}.png"
        plt.savefig(fn); filenames["qq"].append(fn); plt.close()
    return filenames


def main():
    parser = argparse.ArgumentParser(description='Audit sampling method of data against LHS/Random/Grid, with report.')
    parser.add_argument('--csv', required=True, help='Path to CSV file')
    parser.add_argument('--yaml', required=True, help='Path to YAML config')
    parser.add_argument('--samples', type=int, default=None, help='Samples to generate (default: size of CSV)')
    parser.add_argument('--report', default='sampling_report.txt', help='Report output path')
    parser.add_argument('--template', default='report_template.j2', help='Jinja2 report template')
    parser.add_argument('--plots', default='plots', help='Directory for diagnostic plots')
    args = parser.parse_args()

    df = load_csv(args.csv)
    yaml_cfg = load_yaml(args.yaml)
    dep_vars = yaml_cfg["dependent_variables"]
    var_descs = build_bounds_dists(yaml_cfg)
    user_data = df[dep_vars].values
    n_samples = args.samples if args.samples else len(user_data)
    scaler = MinMaxScaler()
    user_data_scaled = scaler.fit_transform(user_data.astype(float))

    methods, diagnostics, scaled = {}, {}, {}
    for name, func in [
        ("LHS", create_lhs_samples),
        ("Grid", create_grid_samples),
        ("Random", create_random_samples)
    ]:
        try:
            smpl = func(var_descs, n_samples)
            methods[name] = smpl
            scaled[name] = scaler.transform(smpl.astype(float))
        except Exception as e:
            print(f"Could not create {name} samples: {e}", file=sys.stderr)

    table_rows = []
    best_method, min_total = None, np.inf
    summary_comment = ""
    plot_filenames = {"histograms": [], "ecdf": [], "qq": []}
    for method, synth in methods.items():
        synth_scaled = scaled[method]
        ks = ks_multivariate_test(user_data_scaled, synth_scaled)
        ad = ad_multivariate_test(user_data_scaled, synth_scaled)
        chi2 = chi2_multivariate_test(user_data_scaled, synth_scaled)
        energy = energy_distance(user_data_scaled, synth_scaled)
        entropy = entropy_score(synth_scaled)
        comment = ""
        total = ks + energy + ad
        if total < min_total:
            min_total = total
            best_method = method
        files = plot_diagnostics(user_data_scaled, synth_scaled, dep_vars, args.plots, method)
        for k in plot_filenames:
            plot_filenames[k].extend(files[k])
        if ks < 0.1 and energy < 0.1:
            comment = "Very strong distributional similarity."
        elif ad < 0.25:
            comment = "Tails are similar; further checks may be needed."
        else:
            comment = "Distributional differences detected; probable mismatch."
        table_rows.append({
            'method': method,
            'ks_stat': f"{ks:.3f}",
            'ad_stat': f"{ad:.3f}",
            'chi2_stat': f"{chi2:.3f}",
            'energy_stat': f"{energy:.3f}",
            'coverage_score': f"{entropy:.3f}",
            'comment': comment
        })

    justification = {
        "ks": "KS scores measure the basic fit of marginal distributions.",
        "chi2": "Chi-squared adds sensitivity on discrete/categorical bins.",
        "ad": "Anderson-Darling is sensitive to distribution tails.",
        "energy": "Energy distance captures global similarity in multiple dimensions.",
        "coverage": "Entropy close to 0 implies structured (grid), near 1 means highly random/uniform."
    }
    summary_comment = (
        "Overall, the method with the lowest composite divergence (KS + Anderson-Darling + Energy) "
        f"was '{best_method}'. See diagnostic plots for additional context."
    )

    env = Environment(loader=FileSystemLoader(os.path.dirname(args.template) or "."))
    tmplt = env.get_template(os.path.basename(args.template))
    with open(args.report, 'w') as f:
        f.write(tmplt.render(
            csv_path=args.csv,
            yaml_path=args.yaml,
            variables=dep_vars,
            extra_methods=['chi-squared', 'Anderson-Darling', 'energy distance', 'entropy'],
            table_rows=table_rows,
            best_method=best_method,
            justification=justification,
            plot_filenames=plot_filenames,
            summary_comment=summary_comment
        ))
    print(f"\nReport written to {args.report}\nBest guess: {best_method}")


if __name__ == '__main__':
    main()
