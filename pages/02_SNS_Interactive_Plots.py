import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")
st.set_page_config(page_title="Seaborn Statistical Plot Gallery", layout="wide")
st.title("📊 Seaborn Statistical Plot Gallery")

# Load datasets
fmri = sns.load_dataset("fmri")
tips = sns.load_dataset("tips")
penguins = sns.load_dataset("penguins")
flights = sns.load_dataset("flights")
diamonds = sns.load_dataset("diamonds")
planets = sns.load_dataset("planets")
titanic = sns.load_dataset("titanic")

def display_plot(title, plot_func):
    st.subheader(title)
    fig = plot_func()
    st.pyplot(fig)

# --- Plots --- #
# 1. Timeseries plot with error bands
display_plot("📈 Timeseries with Error Bands", lambda: sns.lineplot(data=fmri, x="timepoint", y="signal", hue="event").figure)

# 2. Scatterplot with continuous hues and sizes
display_plot("🎯 Scatterplot with Continuous Hues and Sizes", lambda: sns.relplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue="body_mass_g", size="body_mass_g").figure)

# 3. Small multiple time series
display_plot("⏱️ Small Multiple Time Series", lambda: sns.relplot(data=fmri, x="timepoint", y="signal", col="region", hue="event", kind="line").figure)

# 4. Horizontal boxplot with observations
def horizontal_boxplot():
    fig, ax = plt.subplots()
    sns.boxplot(x="total_bill", y="day", data=tips, ax=ax)
    sns.stripplot(x="total_bill", y="day", data=tips, color=".3", size=3, jitter=True, ax=ax)
    sns.despine()
    return fig
display_plot("📦 Horizontal Boxplot with Observations", horizontal_boxplot)

# 5. Linear regression with marginal distributions
display_plot("📐 Linear Regression with Marginals", lambda: sns.jointplot(data=tips, x="total_bill", y="tip", kind="reg").figure)

# 6. Scatterplot with varying point sizes and hues
display_plot("🟡 Scatterplot with Varying Size/Hue", lambda: sns.relplot(data=diamonds.sample(1000), x="carat", y="price", hue="clarity", size="depth").figure)

# 7. Scatterplot with categorical variables
display_plot("📌 Categorical Scatterplot", lambda: sns.swarmplot(x="day", y="total_bill", data=tips).figure)

# 8. Violinplots with observations
def violinplot_with_points():
    fig, ax = plt.subplots()
    sns.violinplot(x="day", y="total_bill", data=tips, ax=ax, inner=None)
    sns.stripplot(x="day", y="total_bill", data=tips, ax=ax, color="white", size=2, jitter=True)
    return fig
display_plot("🎻 Violinplot with Observations", violinplot_with_points)

# 9. Smooth kernel density with marginal histograms
def kde_with_marginals():
    g = sns.JointGrid(data=penguins, x="bill_length_mm", y="bill_depth_mm")
    g.plot(sns.scatterplot, sns.histplot)
    g.plot_marginals(sns.kdeplot, fill=True)
    return g.figure
display_plot("🔍 KDE with Marginal Histograms", kde_with_marginals)

# 10. Annotated heatmap
display_plot("🔥 Annotated Heatmap", lambda: sns.heatmap(flights.pivot("month", "year", "passengers"), annot=True, fmt="d").figure)

# 11. Boxenplot for large distributions
display_plot("📦 Plotting Large Distributions (Boxenplot)", lambda: sns.boxenplot(x="day", y="total_bill", data=tips).figure)

# 12. Stacked histogram on log scale
def stacked_hist_log():
    fig, ax = plt.subplots()
    sns.histplot(data=diamonds, x="price", hue="cut", multiple="stack", log_scale=True, ax=ax)
    sns.despine()
    return fig
display_plot("📊 Stacked Histogram (Log Scale)", stacked_hist_log)

# 13. Paired categorical plots (PairGrid)
def paired_categorical():
    g = sns.PairGrid(titanic, y_vars=["survived"], x_vars=["sex", "class", "embark_t"], height=4, aspect=.7)
    g.map(sns.barplot)
    sns.despine()
    return g.figure
display_plot("🧩 Paired Categorical Plots", paired_categorical)

# 14. Plotting on many facets
display_plot("📁 Plotting Many Facets", lambda: sns.FacetGrid(planets, col="method", col_wrap=4, height=2).map_dataframe(sns.histplot, x="distance").figure)

# 15. Violinplot from wide-form dataset
def wide_violinplot():
    df = sns.load_dataset("iris")
    fig, ax = plt.subplots()
    sns.violinplot(data=df.iloc[:, :-1], ax=ax)
    sns.despine()
    return fig
display_plot("🎻 Violinplot from Wide-form", wide_violinplot)

# 16. Bivariate plot with multiple elements
def bivariate_elements():
    fig, ax = plt.subplots()
    sns.scatterplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", ax=ax)
    sns.kdeplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", ax=ax, levels=5, color="r")
    return fig
display_plot("🧮 Bivariate with Multiple Elements", bivariate_elements)

# 17. Conditional KDE
display_plot("🌊 Conditional KDE", lambda: sns.displot(data=tips, x="total_bill", col="time", kind="kde").figure)

# Footer
st.markdown("---")
st.markdown("Created using 🐍 Seaborn and Streamlit • Inspired by [Sandeep Kumar Patel](https://medium.com/swlh).")

