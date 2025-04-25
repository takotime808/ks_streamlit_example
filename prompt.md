
The following code needs to be updated.

```python
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, anderson_ksamp, zscore
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

    data1 = df1[col1].dropna()
    data2 = df2[col2].dropna()

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

    # === Statistical Test ===
    st.sidebar.header("Step 4: Testing")
    test_type = st.sidebar.selectbox("Test Type", ["Kolmogorov-Smirnov", "Anderson-Darling"])
    alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.10, 0.05, step=0.01)

    st.header("📈 Test Results")
    if test_type == "Kolmogorov-Smirnov":
        stat, p_value = ks_2samp(data1, data2)
        test_name = "Kolmogorov-Smirnov"
    else:
        result = anderson_ksamp([data1, data2])
        stat = result.statistic
        p_value = result.significance_level / 100  # percent to decimal
        test_name = "Anderson-Darling"

    conclusion = (
        "❌ Reject H₀: Distributions are different"
        if p_value < alpha
        else "✅ Fail to reject H₀: Distributions are similar"
    )

    st.markdown(f"**Test:** {test_name}")
    st.markdown(f"**Statistic:** `{stat:.4f}`")
    st.markdown(f"**P-value:** `{p_value:.4f}`")
    st.markdown(f"**Conclusion (α = {alpha:.2f}):** {conclusion}")

    # === Download Results ===
    result_df = pd.DataFrame({
        "Test": [test_name],
        "Statistic": [stat],
        "P-value": [p_value],
        "Alpha": [alpha],
        "Conclusion": [conclusion]
    })
    csv_buf = StringIO()
    result_df.to_csv(csv_buf, index=False)
    st.download_button("📥 Download Results CSV", csv_buf.getvalue(), file_name="test_results.csv", mime="text/csv")

    # === Plotting Functions ===
    def ecdf(data):
        x = np.sort(data)
        y = np.arange(1, len(data) + 1) / len(data)
        return x, y

    def save_plot(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png")
        return buf.getvalue()

    # === Histogram ===
    st.subheader("📊 Histogram")
    fig1, ax1 = plt.subplots()
    ax1.hist(data1, bins=30, alpha=0.6, label="Sample 1", color="skyblue")
    ax1.hist(data2, bins=30, alpha=0.6, label="Sample 2", color="salmon")
    ax1.set_title("Histogram of Samples")
    ax1.legend()
    st.pyplot(fig1)
    st.download_button("🖼️ Download Histogram", save_plot(fig1), file_name="histogram.png", mime="image/png")

    # === ECDF ===
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
```

Modify the code above to include all of the following metrics for users that are comparing two distributions. The additional metrics *must* be provided to the end user.

METRICS:

- Validation Metrics:
    - Deterministic Validation Metric:
        - Root Mean Square Error
        - Minkowski Distance ($l_{p}$ Distance)
    - Probability-Based Validation Metrics
        - Normalized Euclidean Metrid
        - Mahalanobis Distance
        - Kullback-Leibler Divergence
        - Symmetrized Divergence
        - Jensen-Shannon Divergence
        - Hellinger Metric
        - Kolmogorov-Smirnov Test
        - Total Variation Distance
    - Signal Processing Validation Metrics
        - Simple Cross Correlation
        - Normalized Cross Correlationl
        - Normalized Zero-Mean Sum of Squared Distances
        - Moravec Correlationl
        - Index of Agreement
        - Sprague-Geers Metric

Stop. Take a breathe and review the metrics above. Use the metrics above as a checklist. Allow users to check and uncheck boxes for what metrics they want to be provided.

All of the metrics above come from the Sandia Report titled "Validation Metrics for Deterministic and Probabilistic Data". The full report is:

SANDIA REPORT
SAND2016-1421
Unlimited Release
Printed January 2017
Validation Metrics for Deterministic and Probabilistic Data
Kathryn A. Maupin and Laura P. Swiler


----
----
Are all of these metrics included? Fill in the checkboxes next to the metrics that are reported next to the users. If a metric is not reported to the user, leave the checkbox blank.

CHECKLIST:

- [ ] Validation Metrics:
    - [ ] Deterministic Validation Metric:
        - [ ] Root Mean Square Error
        - [ ] Minkowski Distance ($l_{p}$ Distance)
    - [ ] Probability-Based Validation Metrics
        - [ ] Normalized Euclidean Metrid
        - [ ] Mahalanobis Distance
        - [ ] Kullback-Leibler Divergence
        - [ ] Symmetrized Divergence
        - [ ] Jensen-Shannon Divergence
        - [ ] Hellinger Metric
        - [ ] Kolmogorov-Smirnov Test
        - [ ] Total Variation Distance
    - [ ] Signal Processing Validation Metrics
        - [ ] Simple Cross Correlation
        - [ ] Normalized Cross Correlationl
        - [ ] Normalized Zero-Mean Sum of Squared Distances
        - [ ] Moravec Correlationl
        - [ ] Index of Agreement
        - [ ] Sprague-Geers Metric

<!-- 
All of the metrics above come from the Sandia Report titled "Validation Metrics for Deterministic and Probabilistic Data". Here is the full report:

```txt
SANDIA REPORT
SAND2016-1421
Unlimited Release
Printed January 2017
Validation Metrics for Deterministic and Probabilistic Data
Kathryn A. Maupin and Laura P. Swiler
Prepared by
Sandia National Laboratories
Albuquerque, New Mexico 87185 and Livermore, California 94550
Sandia National Laboratories is a multi-mission laboratory managed and operated by Sandia Corporation,
a wholly owned subsidiary of Lockheed Martin Corporation, for the U.S. Department of Energy's
National Nuclear Security Administration under contract DE-AC04-94AL85000.
Approved for public release; further dissemination unlimited.
Sandia National Laboratories
SAND2017-1421
Issued by Sandia National Laboratories, operated for the United States Department of Energy
by Sandia Corporation.
NOTICE: This report was prepared as an account of work sponsored by an agency of the United
States Government. Neither the United States Government, nor any agency thereof, nor any
of their employees, nor any of their contractors, subcontractors, or their employees, make any
warranty, express or implied, or assume any legal liability or responsibility for the accuracy,
completeness, or usefulness of any information, apparatus, product, or process disclosed, or represent that its use would not infringe privately owned rights. Reference herein to any specific
commercial product, process, or service by trade name, trademark, manufacturer, or otherwise,
does not necessarily constitute or imply its endorsement, recommendation, or favoring by the
United States Government, any agency thereof, or any of their contractors or subcontractors.
The views and opinions expressed herein do not necessarily state or reflect those of the United
States Government, any agency thereof, or any of their contractors.
Printed in the United States of America. This report has been reproduced directly from the best
available copy.
Available to DOE and DOE contractors from
U.S. Department of Energy
Office of Scientific and Technical Information
P.O. Box 62
Oak Ridge, TN 37831
Telephone:
Facsimile:
E-Mail:
Online ordering:
(865) 576-8401
(865) 576-5728
reports@adonis.osti.gov
http://www.osti.gov/bridge
Available to the public from
U.S. Department of Commerce
National Technical Information Service
5285 Port Royal Rd
Springfield, VA 22161
Telephone:
Facsimile:
E-Mail:
Online ordering:
(800) 553-6847
(703) 605-6900
orders@ntis.fedworld.gov
http://www.ntis.gov/help/ordermethods.asp?loc=7-4-0#online
2
SAND2016-1421
Unlimited Release
Printed January 2017
Validation Metrics for Deterministic and Probabilistic
Data
Kathryn A. Maupin
Optimization and Uncertainty Quant.
Sandia National Laboratories
P.O. Box 5800
Albuquerque, NM 87185-1318
kmaupinAsandia.goy
Laura P. Swiler
Optimization and Uncertainty Quant.
Sandia National Laboratories
P.O. Box 5800
Albuquerque, NM 87185-1318
lpswile©sandia.goy
Abstract
3
Acknowledgment
This work was funded by the Nuclear Energy Advanced Modeling and Simulation (NEAMS)
Program under the Advanced Modeling and Simulation Office (AMSO) in the Nuclear Energy Office in the U.S. Department of Energy. The authors specifically acknowledge the
Program Manager of the Integrated Product Line, Dr. Bradley Rearden (ORNL) and Dr.
Christopher Stanek, National Technical Director of NEAMS, for their support of this work.
4
Contents
7
ObservationaLDBtal 7
Deterministic Validation Metric§
Root Mean Square Error
Minkowski Distance (lp Distance)
Probability-Based Validation Metrics
Normalized Euclidean Metrid
MalLalanobilist,andel
Kullback-Leibler Divergence
Symmetrized Divergence
Jensen-Shannon Divergence
Hellinger Metric
Kolmogorov-Smirnov Test
TataLVanahonDistance
Signal Processing Validation Metrics
Simple Cross Correlation
Normalized Cross Correlationl
Normalized Zero-Mean Sum of Squared Distances
Moravec Correlationl
Index of Agreement
5
9
9
9
10
10
11
12
13
14
15
15
17
17
18
18
19
19
20
20
3 Examples
Sprague-Geers Metric
Data set 11
Deterministic Validation Metrics I
Probability-Based Metrics for Type 2 Data
Probability-Based Metrics for Type 3 Data
Data set 2
IleternunisticValidationMetricS
Probability-Based Metrics
4 Concluding Comments
References1
6
21
23
23
23
25
29
34
35
36
45
46
Chapter 1
Introduction
The purpose of this document is to compare and contrast metrics that may be considered
for use in validating computational models. Metrics suitable for use in one application,
scenario, and/or quantity of interest may not be acceptable in another; these notes merely
provide information that may be used as guidance in selecting a validation metric.
Observational Data
The basis of all model validation lies in the comparison of experimental observations
with model output. Validation metrics seek to quantify the difference between physical and
simulated observations. Whether experimental or virtual, however, these observed values
contain uncertainties. Experimental measurement error, if provided, is usually given as a
percentage, such as yobs± 5%. The quantification of modeling uncertainties is widely studied,
and many methods for addressing these uncertainties exist (see, e.g. [2] for a comprehensive
review). The basic principles to which we subscribe are detailed in the remainder of this
section.
We consider the case in which the desired scenario produces data over a designated
period of time. That is, either the experiment or the simulation is run for a period of
time and measurements are taken periodically throughout the process. Each data point
is therefore linked inextricably to a certain point in time with respect to the initialization
of the (physical or virtual) experiment. The fact that these time-points may not exactly
coincide between physical experiments and computational simulations can be addressed via
interpolation. These details are not the subject of this report, and we therefore assume that
both experimental and model observations are given at the same time points (ti) throughout
the experiment.
Depending on cost, we may have at our disposal data from either a single implementation or multiple implementations of an experiment. Likewise, we may have data from either
a single simulation run or, if computational cost permits, multiple simulation runs. These
observationsn may be considered with or without uncertainties. If uncertainty is acknowledged, it can either be given a nominal value, such as ±a%, or it may be calculated via,
an uncertainty analysis or as the standard deviation in the sample of values. As a rule of
7
thumb, if multiple data sets are available, at the very least the standard deviation in the
sample of values should be acknowledged as an approximate measure of the uncertainty.
With this in mind, we can categorize available data into one of three classes:
• Type 1: Experimental and predicted (model) values are treated as point values, i.e.
they are treated without uncertainty.
• Type 2: Experimental values are treated as uncertain. This uncertainty is usually
given as a nominal value, but may also be calculated as a standard deviation if multiple
samples are available. It is denoted by the covariance matrix ED, which is usually a
diagonal matrix, with each entry being the variance 0-2D, of the data at time point ti.
Note that if nominal uncertainties are assigned (e.g. those of the form ±a%), we can
assume a normal distribution, in which case we could define
1 a T_1
D'i 2100
which corresponds to a 95% confidence interval.
• Type 3: Both experimental and predicted values are treated as uncertain. The description of the experimental uncertainty given for Type 2 still applies. Model uncertainty
is also described with a covariance matrix, Ep. It may be diagonal if each time point
ti is regarded as an independent sample, with the variance cr2p, being derived from a
nominal value, as in (1.1), or from the sample of values. Despite the fact that this is
not the case, it may not always be feasible to conduct a full uncertainty analysis to
produce the full Ep, thus sample independence is often assumed.
A summary of the validation metrics discussed and the types of data to which they apply is
given at the end of Section 2 in Table 2.1
It should be noted that the distinction between these three data types is not the only
means by which to select a validation metric. This choice also depends on the quantity of
interest itself. Model or experimental observations which exhibit discontinuities or contain
vast changes in scale, for example, require different consideration than smooth data. Furthermore, the signal processing metrics presented at the end of Section 2 were designed to
treat high-frequency data. This document provides a sample and comparison of validation
metrics that are currently in use for both deterministic and probabilistic data.
8
Chapter 2
Validation Metrics
In this section, we list and discuss possible validation metrics, along with their assumptions, uses, and limitations. In particular, we discuss which types of data, according to the
categories listed above, each metric may be applied to. For consistency in notation and terminology, we refer to specific points in time as either "data points" or "time points," and denote
them by ti. Experimental values taken at these points are denoted Di, while predicted values
(or interpolations thereof) are denoted Pi. These values may be cast into vectors representing the whole set of outputs, such that D {D1, D2, . . . , DN} and P = {P11 P21 • • • PA •
If multiple sets of data exist, an additional subscript of j or k is given. That is, the jth
set of model output and the kth set of experimental data, respectively, are contained in the
vectors Pi and Dk, with and DiA occurring at ti.
Deterministic Validation Metrics
Metrics which treat each data point as a point value (i. e. without uncertainty) are deterministic validation metrics. These metrics are therefore targeted towards Type 1 data.
They may, of course, be applied to data of Types 2 or 3, but they would not make use of
all available information. Metrics detailed in following sections would be better suited, and,
hence, the metrics presented in this section may be less preferable for use with Type 2 or
Type 3 data.
Root Mean Square Error
Measuring the difference between experimental and model outputs by calculating the
Root Mean Square Error is widely accepted in many fields of science. The Root Mean
Square Error between a single set of experimental and model data is given by [E111]
drmse
1
N E(P
i
— Di)2.
9
(2.1)
If multiple sets of experimental and/or model data are available, the Root Mean Square
Error can be computed for every pair of experimental and model outputs by
=
N Di,k)2"
Then the overall Root Mean Square Error would be
drmse
1
MPMD
(2.2)
(2.3)
Note that d calculates an average error per point, per output, making the values this metric
yields rather intuitive when compared to the average value or order of magnitude of the
experimental data.
Minkowski Distance (lp Distance)
The lp distance, also called the Minkowski Distance, measures the distance between two
points in Rd for any finite d. When considering the current application, d is equal to the
number of time points N so that, for a single set of experimental and model outputs [4],
lp =(E1-732
Note that when p = 2, d corresponds to the Euclidean distance. When multiple sets of data
are available, we calculate for each experiment-simulation pair
di,k =
>_21137
Di,kr)
•
(2.4)
(2.5)
Then the average Minkowski distance per model output may be calculated
1
lp = , , EEd,,k. (2.6)
j k
The values ( ), ( ), and (2.6) naturally increase as the number of time points increases.
Therefore, the average value per point, per output may be calculated by dividing by N to
yield a more intuitive measure of how close the model output is to experimental observations.
2.4 2.5
Probability-Based Validation Metrics
In the presence of uncertainties, measured values are no longer single, deterministic measurements; they are a single sample coming from a distribution of possibly measured values,
10
which is generally unknown. This is true for measurements coming from either physical
experiments or virtual simulations. As previously described, the best guess we usually have
for this distribution is to take the mean to be the measured value and the standard deviation
to be a measured quantity, e.g. through an uncertainty analysis, or as a nominal value (see
1.1)).
If the measured values are considered to be independent from one another, we can think
of each measurement as having a distribution independent of all the others. The set of
values is therefore a concatenation of independent Gaussian distributions. In this case,
the joint distribution is described by a vector of mean values in each direction, p,, and a
covariance matrix E, which is a diagonal matrix whose components are the variances cr2 in
each direction. In general, however, these values are not independent. The set of values may
still be assumed to have a joint Gaussian distribution, with the dimension of the distribution
being equal to the number of time points, N. However, in this case, the covariance matrix
is no longer diagonal, as each time point (dimension) depends on the others.
Normalized Euclidean Metric
The Normalized Euclidean Metric measures the Euclidean distance between experimental
and model-produced data sets and normalizes this distance by the standard deviation of the
data. In particular, it is targeted towards Type 2 data. Although it may be applied to
Type 3 data, it does not make use of model uncertainty and is therefore not preferred. It
cannot be applied to Type 1 data. Note that in using this metric, model output values are
treated without uncertainty, and both model and experimental outputs are implied to be
independent with respect to time.
For a single set of model output, we calculate at each time point
di =
IPi- Dt
(7.13,2,
Then the total normalized Euclidean distance is
dne =
(2.7)
(2.8)
and the average distance per point may be calculated. Adaptation to the case where multiple
sets of experimental and/or model data are available is possible. For example, we calculate
at each time point,
d j,k =
(Pi
,
i — Di,k)2
,-2
`7' D,i,k
Then the average normalized Euclidean distance per experiment-simulation pair is given by
dne =
iffP MD 3.
11
(2.9)
(2.10)
and the average distance per point can be calculated, if desired.
Note that this metric is a generalization of the absolute value of the z-score used in
statistics. For each data point, di measures the number of standard deviations the point Pi,
is from the normal distribution centered at Di with standard deviation equal to CJD,i. A value
of dne < 2 is therefore generally considered acceptable. The total distance (2.8) or (2.9) per
experiment-simulation pair is a geometric average of the contributions at each time point.
Mahalanobis Distance
The relative distance of a single point from the mean of a multivariate distribution is
characterized by the Mahalanobis Distance [9] and is given by
dmaha
tt)T E-1 (x (2.11)
This distance is a multivariate generalization of the z-score used in statistics. In the context
of the applications considered here, p, is given by the experimental observations and its
covariance matrix over time is given by E, while x corresponds to model output. This
metric therefore considers uncertainty only in the data, and is best suited for Type 2 data.
As with the Normalized Euclidean Metric, application to Type 3 data is not preferable since
it does not make use of uncertainty in model output, and it cannot be applied to Type 1.
When a single set of experimental data is available, µ= D and E D is the corresponding
covariance matrix. For a single set of model outputs, x = P, making the Mahalanobis
Distance
dmaha (P — D)T E31(P — D). (2.12)
If we have multiple sets of either or both experimental and model output values, we can
calculate the Mahalanobis Distance for each pair,
d = (13j D k)T (P j D k) •
Then the average Mahalanobis Distance per experiment-simulation pair is given by
dmaha
1
MP MD
(2.13)
(2.14)
and the average distance per point can be calculated. Note that if the experimental observations are taken to be independent with respect to time, the covariance matrix is a diagonal
matrix, and the Mahalanobis Distance reduces to the Normalized Euclidean Distance.
12
Kullback-Leibler Divergence
Given two probability distribution functions, f and g, the Kullback-Leibler Divergence
(DKL) between them is defined to be [6,
ÐKL(flIg) = f f(x) f (x) log g(x) dx. (2.15)
If either (or both) f or g is a point value (a delta-function) instead of a continuous distribution, DKL will be infinite. Therefore, this metric is only valid for Type 3 data in which
we can characterize the uncertainty of both experimental and simulation data. One major
advantage DKL has over any previously mentioned metric is that it uses the full covariance
information from both the experiments and model predictions.
For multivariate distributions, (2.15) becomes a multidimensional integral, which can be
difficult to compute. However, if the distributions are multivariate normal distributions (as
is the case we are considering here), such that f f, E f) and g Ar(p,g, Eg), (2.15)
reduces to
1
DKL( flIg) = [tr(E-I-Ef) (Pg ittf)TE;l(pg — PI) — k + ln
det g
)1 ,
2 g
(2.16)
det Ef
where k is the dimension of the distributions. When f = g, DKL = 0, however, this "metric"
is not symmetric: it measures the information lost when g approximates f . Furthermore,
the values produced by this pseudo-metric are unintuitive. Every modeler has their own
definition of what a good or acceptable value of DKL is, which may further depend on the
application. However, it is possible to scale DKL by either the self-entropy of the "truth"
distribution,
DKL DKL(flIg)
= (2.17)
H(f)
or by location and spread measures (such as the mean and standard deviation, respectively)
of the "truth" distribution. These options often yield more intuitive values, but are not supported by rigorous derivation. It is important to note that the self-entropy of a distribution
is allowed to be negative, in which case the scaling presented in (2.17) would not be useful.
When a single set of both experiment and model observations are given, we denote
the multivariate distribution describing the experimental values by ArD such that ArD =
Ar(D,ED) and that describing the model output values by Arp such that Alp = Ar(p, Ep).
Then
DKr, (AID MArP) = 2 [tr(E-lp ED) (P — D)TE jT,1(P — D) — k + ln
det
det ED
Ep
)] , (2.18)
where k is the number of time points. If there are multiple sets of experiment and model
outputs, we can calculate the DKL between the jth set of model observations, denoted by
Alp,i such that Afp,i = N(P3, p,j) , and the kth set of experimental values, denoted by AiD,k
13
such that ArD,k = Ar(Dk, ED,O• The average DKL per experiment-simulation pair would
then be given by
DKL =
1
MPMD
DK LGIVD,k11-1VP,i) • (2.19)
If both the model and experimental output values are treated as independent, their
covariance matrices reduce to diagonal matrices. That is, we are comparing two univariate
normal distributions at each data point rather than two multivariable normal distributions
that span several time steps. In this case, (2.18) reduces to a sum over all data points,
D KLGAIDII-ArP) = DKL(DiMP,), (2.20)
where
2 2
DKL(DiMPi) = 1 —
n
[(aD'i) + (Pi ) z 1 +21n(crP'i)]. (2.21)
2 Up,i P,i
A similar reduction holds for (2.19). The average DKL per time point may also be calculated
by dividing by the number of time points, if desired.
Symmetrized Divergence
As mentioned in the previous section, the Kullback-Leibler Divergence is not a symmetric
measure of distance between two probability distribution functions. One way of amending
this problem while maintaining the benefits DKL has as a probability-based metric is to
symmetrize it [1:
SKL (MP, ND) = DKr, (ArD WP) + DK/(AI/3WD), (2.22)
where notations from the previous section have been adopted here. As with the pure DKL,
this metric must be applied to Type 3 data and it makes use of the full covariance matrices
from both the experimental and simulation data. Furthermore, although this metric is
symmetric, the values produced are still unintuitive and may be unbounded.
In a spirit similar to (2.17), the Symmetrized Divergence may be normalized by the
experimental self-entropy such that
SKL(AIP7AID)
H (Alp) •
Furthermore, the average Symmetrized Divergence per experiment-simulation pair in the
presence of multiple sets of observational data is given by
SKL =
1
jmD_ E E DKLGAiD,kwp,,) + DKLGAip, WD,k)-
(2.23)
(2.24)
The simplification for diagonal matrices (i.e. independent data points), given in (
2.21), also holds here when applicable.
14
2.20) and
Jensen-Shannon Divergence
Another DKL-based metric is the Jensen-Shannon Divergence. As with Kullback-Leibler
and the Symmetrized Divergences, it must be applied to Type 3 data, it makes use of the
full covariance matrices of both the experimental and model outputs, and it is symmetric.
Again adopting notation used in previous sections,
Djs = 2DK-Lora* + 2-13KL(ArP11111), (2.25)
where M = -(ArD + Arp) [8]. Since the combination of two normal distributions is also
normal, and using the simplification for normal distributions in (2.18), the Jensen-Shannon
Divergence between two normal distributions reduces to
1
Djs =
2
[k—kln(4) + (D — P)T(ED + Epri(D — P) (2.26)
+ ln (det (ED — Ep)) - 2ln (det (EDEP))] . (2.27)
There exist simplifications similar to (
covariances are diagonal.
2.21) when both the experimental and prediction
As with the metrics discussed previously, in the presence of multiple experimental and
model output data sets, the Jensen-Shannon Divergence between each pair may be calculated
and averaged,
Djs =
1
MDMP
n Ar \ -/-/n Ar
KL(vP,j1livij,k)• (2.28)
The most important feature of the Jensen-Shannon Divergence is that it is finite when
comparing two non-delta function probability distributions [13]. Specifically,
0 < Djs < ln(2), (2.29)
where zero indicates Arp = ArD. This bound affords modelers an intuition regarding what
constitutes a "good" or "close" value of Djs.
Hellinger Metric
The Hellinger metric is also designed to quantify the similarity between two probability
distributions. If f and g are two probability distributions, the squared Hellinger metric is
given by [5]
H2(f, g) = 2 f — \/7g)2 d37. (2.30)
15
It is a bounded metric such that
0 < H(f , g) < 1, (2.31)
with a value of zero indicating exact agreement between f and g. The benefit of having
a bounded metric is that it gives context and intuition to the values of H that may be
produced. This metric targets Type 3 data.
As described earlier, the distributions representing experimental and simulation data are
multivariate distributions and are assumed to be normal. Furthermore, at each time point ti,
the distribution about Pi and Di are considered to be Gaussian. Using notation introduced
previously, the Hellinger distance between single sets of experimental and model outputs is
given by
H2(ArD , Arp) = 1
(det(EDEP))114
exp
(
--
1
(P — D)TE-1(P — D) , (2.32)
(det())1/2 8
where E = (Ep + ED). If multiple data sets are available, the average distance between
experiment-model data pairs can be calculated,
H=
MD MP
1
H(ArD,k,Afp,j).
(2.33)
Furthermore, if the covariance matrices are diagonal, indicating the experimental and
model observations are independent at each time point, (2.32) may be simplified to
where
H2(ATD,Arp) = 1 - fl H(Di, (2.34)
H (Di, Pi) =
2c r p,i0D,i ( 1 (Pi- Di)2
exp „_2
P
_L ,2
,z 4 o-2Pi + o-2D) i •
(2.35)
This simplification may be extended to (2.33). However, since H(Di, Pi) < 1, the product
over all time points may misleadingly approach zero. In this case, it may be more useful to
use the maximum Hellinger metric calculated per point,
or an approximate average,
for validation purposes.
112(ArD, Arp) = max {1 — H (Di, Pi)} , (2.36)
1
H2(ArD, N-p) = 1 —
N
— H (Di, Pi),
16
(2.37)
Kolmogorov-Smirnov Test
The probability-based metrics thus far have used the probability distribution functions, f
and g, to quantify the similarity between two random variables (the data). The KolmogorovSmirnov Test instead calculates the maximum vertical distance between two cumulative
distribution functions,
dKs= sup1F(x) — G(x)1 , (2.38)
where F and G correspond to f and g, respectively [3]. This metric may only be applied
to Type 2 or Type 3 data in which the observations at each time point are treated as
independent, which is not a trivial assumption. In the case of Type 2 data, the cumulative
distribution function of the model output at each time point is simply a vertical jump between
0 and 1 at Pi. Furthermore, when the simplifying assumption that the experimental and
prediction data are normally distributed about the observed values Di and Pi, respectively,
the cumulative distribution function has a known form and
1
= sup
x 2
x — P)i (x — Di
er f er f
N/C1p,i )
(2.39)
where er f(x) is the error function. Note that the supremum is taken over x for each time
point ti. For a whole set of observed values, dKs may be taken to be the maximum observed
di over all time points,
dKs = max{dj, (2.40)
or the average observed di,
1
dxs = — E di. (2.41)
As before, if multiple sets of experimental and model data are available, we can calculate
the average
dKs = E sup liV13,k -IVP,j11 (2.42)
MDMP k
where, in this case, we make an exception in notation and let AiD,k and J\ipj denote the
cumulative distribution functions of experiment set k and model set j, respectively.
Total Variation Distance
The Total Variation Distance calculates the largest possible difference in probabilities
that two probability distributions can assign to the same event N. That is,
dTv = sup (x) — g(x)1. (2.43)
It is best applied to Type 3 data. If experimental and model outputs are considered to be
multivariate distributions, as previously discussed, calculating the Total Variation Distance
amounts to a multidimensional optimization problem. The simplifying assumption that
17
1
d TV --
observations measured at each time point are independent would significantly reduce the
computational cost of this metric. Then
dTv — E sup IND,ti — (2.44)
where ArDA and .1\ip,t, denote the normal probability distributions of the experimental and
model data, respectively, at time point ti. If we have at hand multiple sets of experimental
and model data, we can calculate the average Total Variation Distance per experimentsimulation pair,
MDMP-d
suplAID,k — Al/7,j1. (2.45)
Signal Processing Validation Metrics
The metrics listed in this section were designed to be applied to problems in signal
processing. They target high-frequency and phase-based data. These metrics tend to be
either affine or magnitude invariant such that results from data in which Pi = Di + a or
Pi = aDi for every time point ti give false "exact agreement" results. Despite this fact, these
metrics can be used to understand trends in the differences between the experimental and
simulation measured values. These properties are noted for each metric. In general, these
metrics seek to identify global time-lag behavior between data sets. Mean values P and D
are those for the duration of the entire experiment rather than those for each data set. In
general, they apply to Type 1 data. Of course, application to data of Type 2 or 3 can be
done, but the uncertainty information would not be used.
Simple Cross Correlation
As a metric, Simple Cross Correlation calculates the correlation between experimental
and predicted sets of outputs. In this metric, both experimental and model data are treated
without uncertainty, and each data point is considered to be independent of the others.
Furthermore, this metric is bounded between -1 and 1. In its original form [ 11 11,
E.(13i — P)(Di — D)
.N/Ei(Pi - P)2Ei(Di - D)2'
(2.46)
assuming single experimental and model data sets. Note that the denominator contains the
sample standard deviations of the experimental and model data sets over the duration of
the experiment. For multiple data sets, the Simple Cross Correlation could be calculated for
every pair of experimental and model data,
E (Pi - P3)(Di Dk) d3 k - z
VEz(Pi - P3)2Et(Di Dkr'
18
(2.47)
and the average of the total over all sets may be calculated.
It should be stressed that this metric is designed to calculate lag without regard to affine
shifts in the data values themselves. That is, if Pi, = Di + a for every time point ti, d would
equal 1. This therefore does not imply perfect agreement between the data, this implies
perfect agreement in the timing of the data.
Normalized Cross Correlation
Normalized Cross Correlation is quite similar to Simple Cross Correlation. It applies to
Type 1 data in which each data point is considered to be independent of the others and it
is bounded between -1 and 1. Although it measures the similarity between two sets of data,
it is magnitude invariant. That is, if Pi = aD, for every time point ti, d will equal 1, where
Ez P,Di
driCC
P22E, -Da
if a single experimental and model data set is available [O. Therefore, a value of d = 1 does
not imply perfect agreement with the data, it implies agreement in the timing of the data.
(2.48)
As with the Simple Cross Correlation, this calculation can be modified for multiple data
sets by computing d for every pair of experimental and model data,
E, Pi,jDi,k d,k = ,
VE, P,2,3 E, 14 ,k 7
and the average of the total over all sets may be calculated.
Normalized Zero-Mean Sum of Squared Distances
(2.49)
Another similar metric is the Normalized Zero-Mean Sum of Squared Distances. It again
applies to Type 1 data and each data point is considered to be independent of the others.
For single experimental and model data sets [1].1],
— P) — (Di — D))2
dnssd (2.50)
P)2 E,(Di D)2
For multiple data sets, d can be calculated for each pair of model and experimental data
sets,
E.((Pi — Pj) — (Di — Dk))2
dj,k =
i(Pi — Pi)2Ei(Di — DO2
summed, and then averaged.
(2.51)
As with Simple Cross Correlation, the denominator contains the sample standard deviations of the experimental and model data sets over the duration of the experiment rather
19
than at a single time point. It is simple to show that this metric is affine invariant: model
observations for which Pi = Di + a yield d = 0, which indicates exact agreement in the
timing of the data.
Moravec Correlation
Closely related to the previously described signal processing validation metrics is the
Moravec Correlation. It again applies to Type 1 data, in which experimental and model data
sets are treated without uncertainty, and the data points are considered to be independent
of one another. The metric is given by [I11 I
E. (Pi — P)(Di — D)
Ej(Pi — P)2 Ej(Di — D)2
(2.52)
assuming there is a single experimental and model data set. The denominator contains the
sum of the sample variances for the model and experimental data over the duration of the
experiment.
Again, for multiple data sets, d can be calculated for each pair of experimental and model
data sets
Ei(Pj Pi)(Di Dk) d3,k = (2.53)
z(Pi Ei(Di D kr
summed, and then averaged. Furthermore, this metric is designed to calculate the lag
between two data sets and is affine invariant. That is, if P = Di + a for every time point ti,
d would appear to reflect exact agreement with a value of 1/2.
Index of Agreement
The Index of Agreement metric was designed to calculate the co-variability between two
sets of data about an approximate "true" mean. It is geared towards Type 1 data and each
point is implicitly treated as independent of the others. For single experimental and model
output sets,
Ej(Pi — Di)2
dia — 1 (2.54)
— — D1)2.
As before, the metric may be adapted to the case where multiple experimental and/or model
data sets are available by calculating, for each pair of sets,
—
d
Di,k)2
— 1 (2.55)
- — D i,k IcIr
The average of the total over all sets may then be calculated. Note that although this metric
is neither affine nor magnitude invariant, unlike the preceding signal processing validation
metrics, if Pi and Di are on opposite sides of the mean Di, false zeros may still occur.
20
Sprague-Geers Metric
The Sprague-Geers metric explicitly incorporates error due to phase shift as well as error
due to magnitude differences [al, 121]. Therefore, its usefulness to applications outside of
signal processing is of further interest. By differentiating between these two types of errors,
the metric is able to point towards sources of error in the model.
For single sets of experimental and model data, the measurements of the magnitude and
phase errors, respectively, are given by
dm =
Then the total error is
p? 1
Ez
1, dp = arccos Ez P1171
Di 71
N/Ei Pi2
dSG = d2m d2p .
(2.56)
(2.57)
When multiple sets of experimental and/or model outputs are available, these equations
become
dM,j,k =
p2
i'3 1,
Ei k
dpa k = arccos , 7r
Ei
E i 2,3 E D2i,k
(2.58)
Metric Type 1 Type 2 Type 3
Root Mean Square ./
Minkowski Distance ./
Normalized Euclidean Metric V
Mahalanobis Distance V
Kullback-Leibler Divergence .7
Symmetrized Divergence ,./
Jensen-Shannon Divergence ../
Hellinger Metric V
Kolmogorov-Smirnov Test V V
Total Variation Distance V
Simple Cross Correlation V
Normalized Cross Correlation si
Normalized Zero-Mean Sum of Squared Distances V
Moravec Correlation V
Index of Agreement V
Sprague-Geers Metric V
Table 2.1. Table of validation metrics and the types of
data to which they are targeted.
21
Then the total error per pair of data sets may be found by summing over all pairs the value
= d2m,j,k d2P,j,k.
and taking the average.
(2.59)
As with other signal processing metrics, this metric is meant to be applied to Type 1
data, in which all data sets are treated without uncertainty, and each data point is taken to
be independent of the others.
22
Chapter 3
Examples
We consider two examples consisting of synthetic experimental and model data. For
each, a selection of the various validation metrics described in Section 2 are computed and
discussed for comparison and also to build intuition for those metrics that rely more heavily
on heuristic interpretation to determine model validity.
Data set 1
Preliminary results were calculated for a single set of synthetic experimental and model
simulation data. The experimental data is characterized by the function
f (x) = 1.1 log(10x) + (3.1)
where e represents measurement error with mean zero and standard deviation equal to 1%
of f . Sample values are drawn at xi = 1, 2, ... , 20. The model is defined by
g(x) = log(10x), (3.2)
with sample values taken at the same xi listed above. While it is clear that the model g is
not perfect, our goal is determine whether it is "close enough" to f to be used in its stead.
Deterministic Validation Metrics
We first discuss metrics for which no uncertainty is incorporated and are therefore considered to be applicable only to Type 1 data. The data produced by f (x) and g(x), defined
in ( ) and (3.2), respectively, are shown in Figure 3.1. The Root Mean Square Error is
calculated via (2.1) to be
3.1
cl„,„ = 4.4706 x 10-1. (3.3)
This value already includes a notion of per point averaging, and considering the average
response value is 4.8554, this error will likely be considered acceptable.
The Minkowski Distance (2.4) can be calculated using several different values of p,
/1 = 8.7212, /2 = 1.9993, /3 = 1.2393.
23
(3.4)
Figure 3.1. Graph of experimental and model measurements to be compared using deterministic validation metrics.
The average relative Minkowski Distances, in which each point error is normalized by the
value of the experimental measurement and the sum of these values is averaged over the
number of points, may be more intuitive,
/1 = 8.9337 x 10-2, /2 = 8.9728 x 10-2, /3 = 9.0108 x 10-2. (3.5)
In particular, /1 is the average relative error between the simulation and experimental values.
It is common to consider models with relative error less than 10% to be "valid," making these
values acceptable.
When we consider the magnitude and phase errors in addition to the full Sprague-Geers
Metric, calculated using (2.56) and (2.57)
dM = —9.0137 x 10-2, dp = 3.0483 x 10-3, dsG = 9.0188 x 10-2, (3.6)
we can see that the majority of the error is due to magnitude discrepancy, and the phase
behavior of the model simulation and experimental output coincide well. Recalling (3.1) and
(3.2), the coefficient of the logarithm (i.e. the magnitude) differs while the argument (which
controls the phase) does not. The Sprague-Geers metric is able to capture this.
The deterministic validation metric calculations, along with the minimum and maximum
possible values of each are given in Table 3.1. Note that all of these metrics can take on
infinitely large values.
24
Metric Value Min Max
Root Mean Square 4.4706 x 10-1 0 oo
Average Relative Minkowski Distance
p = 1 8.9337 x 10-2 0 oo
p = 2 3.0483 x 10-3 0 oo
P = 3 9.0188 x 10-2 0 oo
Sprague-Geers Metric
Magitude —9.0137 x 10-2 0 oo
Phase 3.0483 x 10-3 0 oo
Total 9.0188 x 10-2 0 oo
Table 3.1. Table of values calculated using deterministic
validation metrics.
Probability-Based Metrics for Type 2 Data
We now discuss metrics for the purpose of comparing experimental data that is considered to have uncertainty with deterministic model output data. Metric values produced
when nominal uncertainties are taken to be both 5% and 10%, as shown in Figure
considered for further comparison.
3.2, are
Since we are assuming nominal uncertainty values, the covariance matrix is a diagonal
matrix with entries given by (1.1). As previously noted, this yields identical results for the
Normalized Euclidean distance and the Mahalanobis Distance. Over the whole set of data,
the assumption of 5% and 10% nominal uncertainty, respectively, yield
dmah.(5) = 1.6051 x 101, -maha(1o) d = 8.0255. (3.7)
The per-point average of each,
dmaha(5) = 3.5891, dmaha(10) = 1.7946, (3.8)
can be thought of as corresponding to a measurement of how many standard deviations the
model measurement is from the experimental measurement. The 95% confidence interval
for normal distributions is (A — 2o-, p, + 2u), making the case with 5% nominal uncertainty
invalid and the case with 10% nominal uncertainty valid, as illustrated in Figure 3.3. This
corresponds well to the depiction of the data in Figure 3.2, in which the model measurements
lie outside of the 5% error bars but inside the 10% error bars.
When applied to a single set of data, the Kolmogorov-Smirnov Test has been described as
being given by either ( ) or (2.41). Using the average value is, of course, more forgiving,
but using the maximum value may be necessary for some applications. Assuming 5% nominal
uncertainty yields the maximum and average Kolmogorov-Smirnov distances
dKS(5) = 9.9998 x 10-1, -KS(5) d = 9.9965 x 10-1, (3.9)
2.40
25
6
4
3
2
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
♦ Experiment
----Modd
7
6
5
4
3
2
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
♦ Experiment
----Model
Figure 3.2. Graph of experimental and model measurements to be compared using probabilistic validation metrics.
Error bars are calculated using 5% nominal uncertainty (top)
and 10% nominal uncertainty (bottom).
26
3.5
3
2.5
2
1.5
0.5
3.5
3
2.5
2
1.5
0.5
—Experiment
—wee
4 -3a -2a -a a 2a 3a 6.3
—Experiment
—Mole
4 -2a 2a 6.3
Figure 3.3. Illustration of the physical meaning behind the
Mahalanobis Distance at a single, representative data point.
The probability density functions of the experimental data
with considerations of 5% (top) and 10% (bottom) nominal
uncertainties are shown, as well as the model output, which
is taken to be deterministic. The Mahalanobis Distance measures how many standard deviations (a) the model output is
from the mean of the experimental data. For Gaussian distributions, the interval [a — 2u, ,a+ 2u] contains 95% of the mass
of the distribution, making the model output acceptable in
the case of 10% experimental uncertainty but unacceptable
when 5% uncertainty is assumed.
27
0.9 —
0.8 —
0.7 —
0.6 —
0.5 —
0.4 —
0.3 —
0.2 —
0.1 —
0
43 47
—Model
—Experiment, 5% nominal uncertainty
—Experiment, 10% nominal uncertainty
5.16
Figure 3.4. Graph of the cumulative distribution functions
of the experiment and model at a representative data point.
The Kolmogorov-Smirnov distance measures the maximum
vertical distance between two cumulative distribution functions, which for Type 2 data occurs at the calculated value of
the model. This metric is therefore prone to extreme values
when applied to Type 2 data and is better suited for Type 3
data.
respectively. Assuming 10% nominal uncertainty, the maximum and average KolmogorovSmirnov distances are instead
dics(1o)= 9.8061 x 10-1, dxs(m) = 9.6096 x 10-1. (3.10)
It is not surprising that considering additional uncertainty (i.e. when 10% uncertainty is
considered) yields smaller values. However, noting that the maximum value the KolmogorovSmirnov Test can take on is 1, in either case, the value is quite high and would probably
render the model invalid. It is important to recall that the Kolmogorov-Smirnov distance is
better suited for Type 3 data. This is due to the fact that when considering Type 2 data,
as in the present case, the cumulative distribution function of the model data is essentially
a step function, with the jump occurring at the observed value. We will see that using
a continuous cumulative distribution function, as with Type 3 data, is less prone to such
extreme values. An illustration of this metric with the use of Type 2 data is given in Figure
3 4
28
Metric Value (5%) Value (10%) Min Max
Average Mahalanobis Distance 3.5891 1.7946 0 oo
Kolmogorov-Smirnov Test
Max Value 9.9998 x 10-1 9.8061 x 10-1 0 1
Average Value 9.9965 x 10-1 9.6096 x 10-1 0 1
Table 3.2. Table of values calculated using probabilitybased validation metrics targeted toward Type 2 data.
A summary of these results, along with the minimum and maximum possible values for
each metric applied to Type 2 data is given in Table
Probability-Based Metrics for Type 3 Data
3.2
Here we consider metrics whose purpose is to compare experimental and model simulation
data that both contain uncertainty. Metric values are produced for the case in which 5%
nominal uncertainty is assumed for both sets of data and also for the case in which 10%
nominal uncertainty is assumed for both sets of data. Data for each case are shown in
Figure 3.2. Note that since nominal uncertainty values are assumed, both the experimental
and model covariance matrices are diagonal with entries of the form (1.1).
The Kullback-Leibler Divergence is calculated for the 5% nominal uncertainty case. When
calculated over the whole set of data,
DKL(5) = 1.5608 x 102. (3.11)
As previously mentioned, this metric is extremely difficult to build an intuition for, and
it is strongly modeler and application dependent. Section 2 discusses normalizing the KL
Divergence by the self-entropy of the experimental data, as in equation ( ), or taking
the average over the number of data points may yield values that are easier to interpret.
However, in this case, the self-entropy of f is less than one, producing
jjKL(5) = 4.6546 x 102. (3.12)
If instead we take the per-point average of DKL, we obtain
DKL(5) = 7.8041.
This value is moderate, but does not inspire much confidence i
duce the experimental values. Indeed, if we consider Figure
5% nominal uncertainty overlap only slightly.
When 10% nominal uncertainty is considered, each of these values are (not surprisingly)
lower,
3.2
2.17
(3.13)
n the model's ability to repro-
, the error bars representing
DKL(10) = 3.9162 x 101, nico.o) = 2.8950, ÐKL(1o)= 1.9581. (3.14)
29
These values are much more reasonable, an intuition which corresponds to the high overlap
of the error bars in Figure 3.2. To further illustrate this point, consider Figure 3.5. The
experimental and model distributions for both 5% and 10% nominal uncertainties are shown
for a representative data point. From this it is clear that considering additional uncertainty
yields probability densities that appear much more alike.
Past experience suggests that the values of the total, normalized, and average DKL are
still relatively high when 10% uncertainty is assumed. Someone experienced with DKL as
a validation metric would likely render both models invalid, but they may be acceptable
in some cases. As a further note, as the mean values of the experimental and simulation
measurements remain the same between the two cases (that is, the measurement values
themselves are not changing), we can see how strongly the DKL is dependent on the standard
deviation (i.e. uncertainty) of the considered distributions. The physical manifestation of
this can be seen in Figure 3.5
The arguments above may also be applied to the Symmetrized Kullback-Leibler Divergence. Over a whole set of data,
SKL(5) = 2.8506 x 102, SKL(10) = 7.1532 x 101 (3.15)
for the 5% and 10% nominal uncertainty cases, respectively. Recalling that the data for
which 5% uncertainty is assumed has a small self-entropy,
SKL(5) = 8.5035 x 102, ŠKL(1o) = 5.2879, (3.16)
Again, intuition of "loe and "high" values is difficult to define, and the average per-point
Symmetrized Divergence,
SKL(5) = 1.4253 x 101, -KL(10) S = 3.5766, (3.17)
may instead be used. Note that these values are not exactly double those of the corresponding
traditional Kullback-Leibler Divergence since it is not symmetric. As with the traditional
Kullback-Leibler Divergence, the values for the 10% nominal uncertainty case are much lower
than those for the 5% uncertainty case but are still likely to render both models invalid.
The Hellinger Metric, as previously mentioned, has both lower and upper bounds on the
values it may take on. This provides a better intuition on what constitutes a "good" or
"bad" model fit if these values are very close to either bound; however, values in the middle
of the range may still be difficult to qualify since the metric is nonlinear. For the present
example,
H(5) 1, H(io) = 9.9993 x 10-1, (3.18)
for nominal uncertainty values of 5% and 10%, respectively. These values were calculated
using (2.34), as the covariance matrices are diagonal. Since each individual H(Di, Pi) is less
than one, as the number of time points increases, the product will approach zero, yielding
metric values close to 1, as we can see from above. Therefore, it may be better to consider
the average per-point Hellinger distances,
H(5) = 9.0423 x 10-1, nom = 5.9447 x 10-1. (3.19)
30
3.5
2.5
1.5
0.5
—Model
—Expednlent
4.7 5.16
Figure 3.5. Comparison of experimental and model probability distribution functions when 5% (top) and 10% (bottom) nominal uncertainties are assumed. Both plots are produced at the same data point.
31
6
0.1
0.09
0.08
0.07
-0
0.06
0.05
0.04
0.03
10
Nominal uncertainty (percentage)
Figure 3.6. The Hellinger distance decreases monotonically with respect to the nominal uncertainty a, which is
assumed to be equal for both the experiment and model
in this example. Each line represents one data point for
x =1,2,...,20.
One might be able to say that 5.9447 x 10-1 is a reasonable Hellinger metric value, while
9.0423 x 10-1 is not. These conclusions are consistent with the DKL-based metrics previously
discussed.
In this example, the nominal uncertainty percentage is equal for both the model and
the experimental data. Using (1.1) to rewrite (2.35), we can take the derivative of the
Hellinger distance with respect to a, the percentage of uncertainty. From this we can say
that IH < 0 for a > 0. That is, the Hellinger distance monotonically decreases as the
amount of uncertainty increases. See Figure 3.6 for a graphical representation.
In the previous section, we discussed details of the application of the Kolmogorov-Smirnov
Test, all of which extend to Type 3 data. Assuming 5% nominal uncertainty, the maximum
and average Kolmogorov-Smirnov distances, given by ( 2.40) or (2.41), respectively, are
dics(5) = 9.7072 x 10-1,
When 10% nominal uncertainty is assumed,
dics(1o) = 7.2466 x 10-1, jics(io) = 6.4907 x 10-1. (3.21)
dics(5) = 9.3431 x 10-1. (3.20)
32
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
— Model, 5% nominal uncertainty
—Model, 10% nominal uncertainty
—Experiment, 5% nominal uncertainty
— Experiment, 10% nominal uncertainty
4.7 4.92 5.16
Figure 3.7. The cumulative distribution function for the
model output becomes smooth when uncertainty is taken
into account. The Kolmogorov-Smirov distance then becomes
much smaller. This distance is further reduced when greater
nominal uncertainty is assumed. In the graph above, the
greatest vertical distance between the cumulative distribution functions of the model and experiment when 5% (dashed
black line) and 10% (solid black line) nominal uncertainties
are assumed.
6
Not only are the values much lower for 10% nominal uncertainty, unsurprisingly, but all
four of these values are lower than those returned by Type 2 data. Thus, incorporating the
uncertainty for the model in addition to the uncertainty for the data yields more forgiving
Kolmorogov-Smirnov distances. When model uncertainty is included, the cumulative distribution function is smooth instead of jumping sharply at the output value, as demonstrated
in Figure 3.7. The values produced when 5% nominal uncertainty is considered will probably
render the model invalid, while the values produced when 10% nominal uncertainty is assumed may be considered acceptable. These results, along with those discussed throughout
this section are summarized in Table 3.31.
33
Metric Value (5%) Value (10%) Min Max
Kullback-Leibler Divergence
Total
Normalized
Average per point
1.5608 x 102
4.6546 x 102
7.8041
3.9162 x 101
2.8950
1.9581
0
0
0
oo
oo
oo
Symmetrized DKL
Total 2.8506 x 102 7.1532 x 101 0 oo
Normalized 8.5035 x 102 5.2879 0 oo
Average per point 1.4253 x 101 3.5766 0 oo
Hellinger Metric
Total 1 9.9993 x 10-1 0 1
Average per point 9.0423 x 10-1 5.9447 x 10-1 0 1
Kolmogorov-Smirnov Test
Maximum 9.7072 x 10-1 7.2466 x 10-1 0 1
Average 9.3431 x 10-1 6.4907 x 10-1 0 1
Table 3.3. Table of values calculated using probabilitybased validation metrics targeted toward Type 3 data.
Data set 2
We now investigate a second set of data consisting of a single set of experimental measurements and multiple sets of model measurements. The experimental data is produced
using the function
f (x) = sin2(x) + 5 + E, (3.22)
where, as in the previous example, E is meant to represent measurement error, which in this
case has mean zero and standard deviation equal to 0.5% of f . The model is given by
g(x,O) = sin2(x) + B. (3.23)
The parameter 0 is calibrated through the Bayesian calibration tool in Dakota [1], using data
calculated via (3.22). Figure 3.8 shows the calibration data (black) along with the experimental validation data (red) and the set of model responses (blue). The model responses
were produced by sampling the posterior distribution of the parameter 0 and calculating the
corresponding g(x, O). Ideally, analysts would like the experimental measurements to fall in
the middle of the simulation measurements, and the simulation should capture the qualitative behavior of the experiment. The given data set, therefore, would likely be considered
valid by the "eye test." This should be kept in mind as we build intuition for numerical
validation metrics.
34
6.4
6.2
5.8
5.6
5.4
5.2
4.9
o
Figure 3.8. Graph of experimental (red) and model (blue)
measurements. The points indicated in black were used for
the calibration of the model.
Deterministic Validation Metrics
Although the deterministic validation metrics do not make use of all information at
hand, they often provide perspective on metrics that may be less intuitive. For the purpose
of these metrics, the mean of the model output at each data point will be taken as the set
of deterministic model values. The Root Mean Square Error, calculated using (2.1), is
drmse = 1.0242 x 10-1,
which should be considered acceptable. As before, the Minkowski Distance, (
calculated using several different values of p,
2.4
(3.24)
), may be
11 = 1.2044 x 101, /2 = 1.2248, /3= 6.1820 x 10-1, (3.25)
with the average relative values being
11 = 1.5291 x 10-2, /2 = 1.8371 x 10-2, /3 = 2.0916 x 10-2. (3.26)
Recalling that /1 is the average relative error between the experimental and simulation values,
we can say that the simulation differed from the experiment by less than 2%. It is commonly
assumed that models within 10% of the experimental values are "valid," rendering the model
valid by this metric.
35
Metric Value Min Max
Root Mean Square 1.0242 x 10-1 0 oo
Average Relative Minkowski Distance
p = 1 1.5291 x 10-2 0 oo
p = 2 1.8371 x 10-2 0 oo
P = 3 2.0916 x 10-2 0 oo
Sprague-Geers Metric
Magitude —3.6862 x 10-4 0 oo
Phase 5.9480 x 10-3 0 oo
Total 5.9594 x 10-3 0 oo
Table 3.4. Table of values calculated using deterministic
validation metrics.
Calculating the magnitude and phase errors (
metric (2.57),
2.56), as well as the full Sprague-Geers
dm = —3.6862 x 10-4, dp = 5.9480 x 10-3, dsG = 5.9594 x 10-3, (3.27)
we can deduce that the model is able to capture both the magnitude and the phase quite
well. Using this metric, the model would be considered valid yet again. A summary of the
deterministic validation calculations is given in Table
Probability-Based Metrics
3.4
We shall assume in this example that the experimental data is assigned nominal uncertainties, as before, and that the uncertainties in the simulation data may be calculated using
the sample standard deviations at each time point. The experimental covariance matrix is,
therefore, a diagonal matrix with entries given by (1.1), and the model covariance matrix is
a diagonal matrix whose entries are the variances for each time point.
If the data is considered to be Type 2 data such that only the experimental uncertainty is
accounted for, we can calculate the Mahalanobis Distance. Note that here, the Normalized
Euclidean Distance and the Mahalanobis Distance are equal due to the diagonal experimental covariance matrix. When 5% nominal uncertainty is assumed, the per-point average
Mahalanobis distance is
drnaha(5) = 7.3463 X 10-1.
If, instead, 10% uncertainty is assumed,
dmah.(1o) = 3.6817 x 10-1. (3.29)
Recall that dmaha can be interpreted as the number of standard deviations away the model
measurement is from the experimental measurement, on average and that the 95% confidence
36
(3.28)
interval for Gaussian distributions has a radius of two standard deviations. In both the 5%
and 10% uncertainty cases, the model is within one standard deviation, as shown in Figure
3.9, and are therefore both valid.
In this example, however, the model uncertainty plays an important role. Not only is
all available information used when the data is considered to be of Type 3, the key desired
feature is that the experimental data set falls within the span of the simulation data sets. The
Kolmogorov-Smirnov Test compares the cumulative distribution functions of the two data
sets. In both cases, either the maximal value or the average value over the data points may
be considered. When 5% experimental uncertainty is assumed, the Kolmogorov-Smirnov
Test gives
dKS(5) = 7.2276 x 10-1 d-KS(5) = 2.6065 x 10-1. (3.30)
The average values are often more useful, as previously discussed. Since this metric is
bounded above by one, the average value seems reasonable.
When 10% experimental uncertainty is assumed, the total and average KolmogorovSmirnov Test values are
dics(10) = 5.8143 x 10-1 dKS(10) = 3.0401 x 10-1. (3.31)
Although the average value is still quite small and likely to render to model valid, it is
unexpected that the average value for the Kolmogorov-Smirnov Test is higher for the case
in which 10% experimental uncertainty is assumed. For this particular data set, the mean
model response and the experimental measurements are often quite close. In this case,
probability-based metrics rely more heavily on how close the variance in the model response
and experimental error are to one another. When the uncertainty in the model parameter is
propagated through the model, the resulting variance in the responses is quite small and is
therefore much closer to the nominal variance when less experimental uncertainty is assumed.
This would explain why, on average, the Kolmogorov-Smirnov Test yields smaller values when
5% experimental uncertainty is considered. However, there are a few experimental points
in the experimental data set that are quite far from the mean of the model responses. For
these points, the difference in mean dominates the metric, which explains why the maximum
value of the Kolmogorov-Smirnoff Test is lower for the scenario in which 10% experimental
uncertainty is used. A graphical demonstration is given in Figure 3.10
We now consider metrics based on probability distribution functions. The KullbackLeibler Divergence is calculate in both the 5% and 10% cases to be
D KL(5) = 5.8205 x 101, D K L(10) = 2.4860 x 102. (3.32)
As before, we attempt to mitigate the lack of intuition surrounding this metric. Although
the self-entropy of the experimental data is much lower when less uncertainty is considered,
it is negative when 5% nominal uncertainty is considered and positive when 10% nominal
uncertainty is considered. Normalization via (2.17) would therefore be unhelpful. The perpoint average, however, may still be calculated and is found to be
DKL(5) = 4.0703 x 10-1, DKL(10) = 1.7384. (3.33)
37
2.5
2
1.5
0.5
91.3
3
2.5
2
1.5
0.5
-Experiment
-Model
-3a -2a -a a 2a 3a 6 4
°4 3 -2a 2a
-Experiment
-Model
&4
Figure 3.9. Physical interpretation of the Mahalanobis
Distance at a single, representative data point. The probability density functions of the experimental data with considerations of 5% (top) and 10% (bottom) nominal uncertainty
are shown, as well as the model output, which is taken to
be deterministic. The Mahalanobis Distance measures how
many standard deviations (u) the model output is from the
mean of the experimental data. In both cases shown here,
the model is within one standard deviation of the measured
experiment value. Since the interval [it — 2u, + 2u] contains
95% of the mass of Gaussian distributions, the model would
be acceptable in either case.
38
0.9
0.8 -
0.7 -
0.6 -
0.5 -
0.4 -
0.3 -
0.2 -
0.1 -
04 5
—Model
—Experiment, 5% nominal uncertainty
—Experiment, 10% nominal uncertainty
i
0.9
0.8 -
0.7 -
0.6 -
0.5 -
0.4 -
0.3 -
0.2 -
0.1 -
—Model
—Experiment, 5% nominal uncertainty
— Experiment, 10% nominal uncertainty
°5 5
5.5
Figure 3.10. Illustrations of the cumulative distribution
functions of the model and experimental data at points which
yield Kolmogorov-Smirnov distances close to the average values (top) and maximum values (bottom) given in (3.30) and
(3.31). In both cases, the variance of the model is closer
to the variance derived by calculating 5% nominal uncertainty. When the model mean and experimental observation
are close to one another, which is true on average in this
example, the vertical distance between the model and experimental cumulative distribution functions is smaller, resulting
in dKs(5) < dKs(io)• However, when the means are further
away, such as in the bottom graph, greater experimental uncertainty yields smaller Kolmogorov-Smirnov values.
39
3.5 —
3 —
2.5 —
2 —
1.5 —
0.5 —
0
4 5 5.5
—Model
—Experiment, 5% nominal uncertainty
—Experiment, 10% nominal uncertainty
Figure 3.11. Probability distributions of the model and
experiment, for each of the two prescribed uncertainty assumptions, at a representative data point. The distributions
shown here produce values of the Kullback-Leibler Divergence similar to the average values given in (3.33). From
this illustration, it is clear that the model is more similar
to the experimental data if only 5% nominal uncertainty is
considered.
6.5
Distributions representative of these averages are shown in Figure 3.11. Although the mean
of both experimental distributions is the same, the model distribution appears to be much
more similar to the distribution with less experimental uncertainty. Recalling that the variance of the model responses are the same at every data point, we can analytically determine
that DKL reaches its minimum with respect to the experimental uncertainty when the standard deviations of the model and experimental outputs are equal. See Figure 3.12 for an
illustration. Of course, experimental uncertainty should not actually vary, however, this explains the increase in the Kullback-Leibler Divergence when greater experimental uncertainty
is considered.
Experience has shown that values of DKL less than 0.5 indicate a "good" match of the
model to the experimental data. This standard would render the model valid when 5% experimental uncertainty is assumed, but not if this uncertainty is increased to 10%. In the latter
case, one could argue that the model would be unlikely to capture potential experimental
observations due to its lack of spread. Some analysts, however, may find that the model
40
distribution looks reasonable when greater experimental uncertainty is considered since, e.g.
the model captures enough of the expected values of the experimental measurements. Then
the tolerance on DKL may be raised or another metric may be more appropriate.
Similar behavior is seen for the Symmetric Kullback-Leibler Divergence, which calculates
a total of
SKL(5) = 1.0023 x 102, SKL(1o) = 3.2016 x 102 (3.34)
for the 5% and 10% nominal uncertainty cases, respectively. These have per-point averages
of
SKL(5) = 7.0088 x 10-1, SKL(1o) = 2.2389. (3.35)
Again, these values are not exactly double those found using the traditional Kullback-Leibler
Divergence, but analysis similar to that above holds true here.
The interplay between the importance of the difference in mean and the difference in
variance can again be seen when using the Hellinger Metric, which displays behavior similar to the Kolmogorov-Smirnoff Test. When 5% nominal uncertainty is assumed for the
experimental observations, the maximum and average Hellinger distances are, respectively,
H(5) = 6.6715 x 10-1, H(io) = 5.7666 x 10-1,
while the 10% nominal uncertainty case yields the corresponding values
H(5) = 2.4120 x 10-1, k(io) = 4.1032 x 10-1.
(3.36)
(3.37)
Although all of these values are reasonable and would likely render the model valid in either
situation, the average Hellinger distance is much larger with greater experimental uncertainty. In Section 3, we saw that the Hellinger metric decreased monotonically as a function
of the nominal uncertainty percentage, which was assumed to be the same for both the
experimental and model data. In the present example, the model uncertainty is calculated
according to the set of simulation data, produced by sampling the posterior distribution of
the model parameter 0 and running these samples through the model. The Hellinger metric
therefore depends on the two uncertainty parameters, crD,, and Up" , rather than one, a, at
each time point. Furthermore, the nonlinear relationship between these standard deviations
and the difference in model and experimental means affects the shape of the Hellinger Distance curve, shown in Figure 3.13. From this illustration, it is evident that for data points
with greater discrepancy between model and experimental observations yield curves for which
the Hellinger Distance is lower when greater experimental uncertainty is accepted. For the
majority of points, however, the opposite is true. Furthermore, the spread of Hellinger Distances at 5% nominal uncertainty is much wider than that for 10% nominal uncertainty. This
explains the increase in the average and the decrease in the maximum Hellinger Distances
calculated for this example.
The results for the probability-based validation metrics for this example are summarized
in Table 3,51.
41
(Model - Experimen1)2
0.3
0.25
0.2
0.15
0.05 0.1 0.15 0.2 0.25
Standard Deviation of Experimental Data
0.3 0.35
Figure 3.12. Plot of the graphs of the Kullback-Leibler Divergence at each time point as a function of the standard deviation of the experimental data. The standard deviation of the
model is 1.1795 x 10-1. This supports the analytical determination that the Kullback-Leibler Divergence reaches its minimum when the standard deviation of the experimental data
matches the standard deviation of the model in this example.
Point-wise experimental standard deviations lie in the range
[1.2097 x 10-1, 1.5722 x 10-1] when 5% nominal uncertainty
is assumed, and in the range [2.4194 x 10-1, 3.1444 x 10-1]
when 10% nominal uncertainty is assumed.
42
0.9
0.8
0.7
23 0.6
c
0.5
cn
0.4
0.3
0.2
0.1
0.05 0.1 0.15 0.2 0.25
Standard Deviation of Experimental Data
0.3
(Model - Experiment)2
0.09
0.35
Figure 3.13. Plot of the graphs of the Hellinger Distance at each time point as a function of the standard deviation of the experimental data. Recall that the standard deviation of the model is 1.1795 x 10-1 and that the
point-wise experimental standard deviations lie in the range
[1.2097 x 10-1, 1.5722 x 10-1] when 5% nominal uncertainty
is assumed, and in the range [2.4194 x 10-1, 3.1444 x 10-1]
when 10% nominal uncertainty is assumed. As the difference
between the model and experimental observations decreases,
the influence of the difference between standard deviations
becomes more influential on the behavior of the Hellinger
metric.
43
0 08
0 07
0.06
0.05
0.04
0.03
0.02
0.01
Metric Value (5%) Value (10%) Min Max
Average Mahalanobis Distance 7.3463 x 10-1 3.6817 x 10-1 0 oo
Kolmogorov-Smirnov Test
Maximum 7.2276 x 10-1 5.8143 x 10-1 0 1
Average 2.6065 x 10-1 3.0401 x 10-1 0 1
Kullback-Leibler Divergence
Total 5.8205 x 101 2.4860 x 102 0 oo
Average per point 4.0703 x 10-1 1.7384 0 oo
Symmetrized DKL
Total 1.0023 x 102 3.2016 x 102 0 Do
Average per point 7.0088 x 10-1 2.2389 0 Do
Hellinger Metric
Total 6.6715 x 10-1 5.7666 x 10-1 0 1
Average per point 2.4120 x 10-1 4.1032 x 10-1 0 1
Table 3.5. Table of values calculated using probabilitybased validation metrics targeted toward Type 3 data.
44
Chapter 4
Concluding Comments
It is indisputable that computational models require validation before they can be reliably
used in prediction scenarios. We have described several validation metrics that quantify the
information provided by physical and simulated observations. These metrics were discussed
in the context of three different types of available data: completely deterministic experimental and model data, experimental data treated with uncertainties paired with deterministic
simulation responses, and data that treats both experimental and model measurements with
uncertainties. Each validation metric detailed herein was classified by the type of data to
which it applies, and intuition behind these metrics was described. Two illustrative examples
were provided, which addressed challenges and comparisons among the metrics to further
build insight as to the meaning, usefulness, and application of validation metrics. Although
strongly application, quantity of interest, and modeler dependent, the model validation metrics are a necessary tool in the process of building a predictive model.
45
46
[2] ASME V&V 20-2009. Standard for Verification and Validation in Computational Fluid
Dynamics and Heat Transfer, 2009.
D. A. Darling. The kolmogorov-smirnov, cramer-von mises tests. The Annals of Mathematical Statistics, 28(4):823-838, 1957.
[4] L. Debnath and P. Mikusffiski. Hilbert Spaces with Applications. Elsevier Academic
Press, 2005.
J.K. Ghosh and R.V. Ramamoorthi. Bayesian Nonparametrics. Springer Series in
Statistics. Springer New York, 2006.
[6] S. Kullback. Information Theory and Statistics. Wiley series in probability and mathematical statistics. Probability and mathematical statistics. John Wiley &Sons, 1959.
Solomon Kullback and Richard A Leibler. On information and sufficiency. The annals
of mathematical statistics, 22(1):79-86, 1951.
Jianhua Lin. Divergence measures based on the shannon entropy. IEEE Transactions
on Information theory, 37(1):145-151, 1991.
Prasanta Chandra Mahalanobis. On the generalised distance in statistics. In Proceedings
of the National Institute of Sciences of India, volume 2 of 1, 1936.
[10] Malcolm H Ray, Chuck A Plaxico, and Marco Anghileri. Procedures for Verification
and Validation of Computer Simulations Used for Roadside Safety Applications. National Cooperative Highway Research Program, Transportation Research Board of the
National Academies, 2010.
[11] Nuno Roma, Jose Santos-Victor, and Jose Tome. A comparative analysis of crosscorrelation matching algorithms using a pyramidal resolution approach. 2002.
[12] H Sarin, M Kokkolaras, G Hulbert, P Papalambros, S Barbat, and R-J Yang. A comprehensive metric for comparing time histories in validation of simulation models with
emphasis on vehicle safety applications. In ASME 2008 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference,
pages 1275-1286. American Society of Mechanical Engineers, 2008.
References
[1] Dakota, a multilevel parallel object-oriented framework for design optimization, parameter estimation, uncertainty quantification, and sensitivity analysis: Version 6.4
users manual. Technical Report SAND2014-4633, Sandia National Laboratories, Albuquerque, NM, Updated May 2016. Available online from
O . fi '' $ . ' 0 $ $ fi
http://dakota.sandia.gov/
[3]
[5]
[7]
[8]
[9]
47
[13] Takuya Yamano. A note on bound for Jensen-Shannon Divergence by Jeffreys. In
ECEA-1: lst International Electronic Conference on Entropy and its Applications,
November, pages 3-21, 2014.
48
DISTRIBUTION:
1 Bradley Rearden
Leader, Modeling and Simulation Integration
Oak Ridge National Laboratory
Reactor and Nuclear Systems Division
P. O. Box 2008, Bldg. 5700
Oak Ridge, TN 37831
1 MS 0828 K. Hu, 1544
1 MS 0828 S. Kieweg, 01544
1 MS 0828 J. Mullins, 01544
1 MS 0828 G.E. Orient, 01544
1 MS 0828 V.J. Romero, 01544
1 MS 0828 J. Winokur, 01544
1 MS 0828 W.R. Witkowski, 01544
1 MS 1318 J.R. Stewart, 01441
1 MS 1318 B.M. Adams, 01441
1 MS 1318 M.S. Eldred, 01441
1 MS 1318 J.A. Stephens, 01441
1 MS 1318 V.G. Weirs, 01441
1 MS 1318 V.A. Mousseau, 01444
1 MS 1318 R.M. Summers, 01444
1 MS 0899 Technical Library, 9536 (electronic copy)
50
v1.40
51
Sandia National laboratories
52
``` -->