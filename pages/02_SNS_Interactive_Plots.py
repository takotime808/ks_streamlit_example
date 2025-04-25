import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Set theme and page config
sns.set_theme(style="darkgrid")
st.set_page_config(page_title="Interactive Seaborn Plot Gallery", layout="wide")

st.title("📊 Interactive Seaborn Statistical Plot Gallery")

# File upload
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head())
    
    # Select numeric columns for plotting
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Please upload a CSV with at least two numeric columns.")
    else:
        # Let user select which columns to include
        selected_cols = st.multiselect("Select numeric columns to include in plots", numeric_cols, default=numeric_cols)

        if len(selected_cols) >= 2:
            # Function to display the plot
            def display_plot(title, plot_func):
                st.subheader(title)
                fig = plot_func()
                st.pyplot(fig)

            # --- Plots --- #
            # 1. Timeseries plot with error bands
            display_plot("📈 Timeseries with Error Bands", lambda: sns.lineplot(data=df[selected_cols], x=df[selected_cols].columns[0], y=df[selected_cols].columns[1]).figure)

            # 2. Scatterplot with continuous hues and sizes
            display_plot("🎯 Scatterplot with Continuous Hues and Sizes", lambda: sns.relplot(data=df[selected_cols], x=df[selected_cols].columns[0], y=df[selected_cols].columns[1], hue=df[selected_cols].columns[0], size=df[selected_cols].columns[1]).figure)

            # 3. Small multiple time series
            display_plot("⏱️ Small Multiple Time Series", lambda: sns.relplot(data=df[selected_cols], x=df[selected_cols].columns[0], y=df[selected_cols].columns[1], col="day", hue="timepoint", kind="line").figure)

            # 4. Horizontal boxplot with observations
            def horizontal_boxplot():
                fig, ax = plt.subplots()
                sns.boxplot(x=df[selected_cols].iloc[:, 0], y=df[selected_cols].iloc[:, 1], data=df, ax=ax)
                sns.stripplot(x=df[selected_cols].iloc[:, 0], y=df[selected_cols].iloc[:, 1], data=df, color=".3", size=3, jitter=True, ax=ax)
                sns.despine()
                return fig
            display_plot("📦 Horizontal Boxplot with Observations", horizontal_boxplot)

            # 5. Linear regression with marginal distributions
            display_plot("📐 Linear Regression with Marginals", lambda: sns.jointplot(data=df[selected_cols], x=df[selected_cols].columns[0], y=df[selected_cols].columns[1], kind="reg").figure)

            # 6. Scatterplot with varying point sizes and hues
            display_plot("🟡 Scatterplot with Varying Size/Hue", lambda: sns.relplot(data=df[selected_cols], x=df[selected_cols].columns[0], y=df[selected_cols].columns[1], hue=df[selected_cols].columns[0], size=df[selected_cols].columns[1]).figure)

            # 7. Violinplots with observations
            def violinplot_with_points():
                fig, ax = plt.subplots()
                sns.violinplot(x=df[selected_cols].iloc[:, 0], y=df[selected_cols].iloc[:, 1], data=df, ax=ax, inner=None)
                sns.stripplot(x=df[selected_cols].iloc[:, 0], y=df[selected_cols].iloc[:, 1], data=df, ax=ax, color="white", size=2, jitter=True)
                return fig
            display_plot("🎻 Violinplot with Observations", violinplot_with_points)

            # 8. Smooth kernel density with marginal histograms
            def kde_with_marginals():
                g = sns.JointGrid(data=df, x=df[selected_cols].columns[0], y=df[selected_cols].columns[1])
                g.plot(sns.scatterplot, sns.histplot)
                g.plot_marginals(sns.kdeplot, fill=True)
                return g.figure
            display_plot("🔍 KDE with Marginal Histograms", kde_with_marginals)

            # 9. Annotated heatmap
            display_plot("🔥 Annotated Heatmap", lambda: sns.heatmap(df.corr(), annot=True, fmt="d").figure)

            # 10. Boxenplot for large distributions
            display_plot("📦 Plotting Large Distributions (Boxenplot)", lambda: sns.boxenplot(x=df[selected_cols].iloc[:, 0], y=df[selected_cols].iloc[:, 1], data=df).figure)

            # 11. Stacked histogram on log scale
            def stacked_hist_log():
                fig, ax = plt.subplots()
                sns.histplot(data=df, x=df[selected_cols].iloc[:, 0], hue="cut", multiple="stack", log_scale=True, ax=ax)
                sns.despine()
                return fig
            display_plot("📊 Stacked Histogram (Log Scale)", stacked_hist_log)

            # 12. Paired categorical plots (PairGrid)
            def paired_categorical():
                g = sns.PairGrid(df, y_vars=["survived"], x_vars=["sex", "class", "embark_t"], height=4, aspect=.7)
                g.map(sns.barplot)
                sns.despine()
                return g.figure
            display_plot("🧩 Paired Categorical Plots", paired_categorical)

            # 13. Plotting on many facets
            display_plot("📁 Plotting Many Facets", lambda: sns.FacetGrid(df, col="method", col_wrap=4, height=2).map_dataframe(sns.histplot, x="distance").figure)

            # 14. Violinplot from wide-form dataset
            def wide_violinplot():
                fig, ax = plt.subplots()
                sns.violinplot(data=df.iloc[:, :-1], ax=ax)
                sns.despine()
                return fig
            display_plot("🎻 Violinplot from Wide-form", wide_violinplot)

            # 15. Bivariate plot with multiple elements
            def bivariate_elements():
                fig, ax = plt.subplots()
                sns.scatterplot(data=df, x=df[selected_cols].columns[0], y=df[selected_cols].columns[1], ax=ax)
                sns.kdeplot(data=df, x=df[selected_cols].columns[0], y=df[selected_cols].columns[1], ax=ax, levels=5, color="r")
                return fig
            display_plot("🧮 Bivariate with Multiple Elements", bivariate_elements)

            # 16. Conditional KDE
            display_plot("🌊 Conditional KDE", lambda: sns.displot(data=df, x=df[selected_cols].columns[0], col="time", kind="kde").figure)

else:
    st.info("👈 Upload a CSV file with numeric columns to get started.")

# Footer
st.markdown("---")
st.markdown("Created using 🐍 Seaborn and Streamlit • Inspired by [Sandeep Kumar Patel](https://medium.com/swlh).")
