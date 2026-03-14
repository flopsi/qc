"""Page 2: Global PCA + PERMANOVA for overall QC assessment."""
import streamlit as st
import numpy as np
import polars as pl
from analysis.pca_engine import run_pca
from analysis.permanova import run_permanova, interpret_permanova
from analysis.cluster_metrics import compute_cluster_metrics
from core.transforms import log2_transform, glog_transform, vsn_transform_rpy2
from viz.pca_plots import create_pca_scatter, create_scree_plot
from viz.heatmaps import create_correlation_heatmap
from config import CONDITION_COLORS, PERMANOVA_ALPHA, PERMANOVA_R2_GOOD

st.header("\U0001f50d Global PCA & PERMANOVA")

if "protein_df" not in st.session_state:
    st.warning("\u26a0\ufe0f Please upload data on the Upload page first.")
    st.stop()

df = st.session_state["protein_df"]
intensity_cols = st.session_state["intensity_cols"]
metadata = st.session_state["metadata"]
conditions = st.session_state.get("conditions", ["A", "B"])
n_reps = st.session_state.get("n_reps", len(intensity_cols) // len(conditions))

# --- Help section ---
with st.expander("\u2139\ufe0f How to interpret this page", expanded=False):
    st.markdown("""
### Principal Component Analysis (PCA)

PCA reduces the high-dimensional proteomics data (thousands of protein intensities) into a few
principal components that capture the most variance. Each point on the PCA scatter represents
one sample (replicate).

**What to look for:**
- **Replicates of the same condition should cluster together** \u2014 tight grouping = good technical precision
- **Different conditions should separate** \u2014 the further apart, the stronger the biological effect
- **PC1 variance (%)** \u2014 the higher, the more a single source of variation dominates (ideally condition)
- **95% confidence ellipses** \u2014 visual guide for within-group dispersion

### Why no z-score standardisation by default?

After log2 or glog transformation the data is already variance-stabilised \u2014 that is the whole
purpose of those transforms. Applying z-score scaling on top would:

1. Give every protein **equal weight**, regardless of signal quality
2. **Amplify noise** from low-abundance proteins whose variance is mostly technical
3. Distort the Euclidean distances that PERMANOVA relies on

Z-scoring is appropriate only when features have fundamentally different units (e.g. mixing
mass-spec intensities with retention times). For standard proteomics QC on transformed data,
it is not recommended.

### PERMANOVA (Permutational Multivariate ANOVA)

A non-parametric test that asks: "Are the centroids of the two condition groups significantly different
in multivariate space?"

| Metric | What it measures | How to judge |
|--------|-----------------|--------------|
| **pseudo-F** | Ratio of between-group to within-group variance | Higher = better separation. Think of it as a signal-to-noise ratio. |
| **p-value** | Probability of observing this separation by chance | Lower = more significant. **Important:** With small replicates, the minimum achievable p-value is constrained by limited permutations. |
| **R\u00b2** | Proportion of total variance explained by condition | 0\u20131 scale. >0.5 = condition drives most variance. For a mixed-proteome benchmark (e.g. HYE) where ~50% of proteins are unchanged, global R\u00b2 of 0.3\u20130.5 is expected and normal. |
| **Silhouette** | How well-separated the clusters are in PCA space | \u22120.1 to 1. >0.7 = excellent separation, >0.5 = good, <0.25 = overlapping clusters. |

### Sample Correlation Heatmap

Pearson correlations between all pairs of samples.

- **Replicates of the same condition should have the highest correlations** (close to 1.0)
- **Cross-condition correlations should be slightly lower** if a real biological difference exists
- **A uniformly high correlation matrix (>0.99)** indicates a dominant stable background

### Important: Small-sample considerations

With only 3 replicates per condition (6 samples total), there are only C(6,3) = 20 unique ways to
assign labels. This limits the minimum p-value to 0.10. Always interpret R\u00b2 and silhouette alongside
the p-value. High R\u00b2 + high silhouette + p at minimum = strong evidence of separation, even without
a classical p < 0.05.
    """)

# Sidebar controls
with st.sidebar:
    st.subheader("PCA Settings")
    transform = st.radio("Transformation", ["log2", "glog", "VSN (rpy2 / glog fallback)"],
                         index=0)
    scale_pca = st.checkbox(
        "Z-score standardize",
        value=False,
        help="Not recommended after log/glog transform \u2014 variance is already stabilised. "
             "Enable only for raw data or when mixing different measurement types.",
    )
    n_components = st.slider("Components", 2, 5, 3)
    n_permutations = st.number_input("PERMANOVA permutations",
                                     100, 9999, 999, step=100)

# Apply transformation
if transform == "log2":
    df_t = log2_transform(df, intensity_cols)
elif transform == "glog":
    df_t = glog_transform(df, intensity_cols)
else:
    df_t = vsn_transform_rpy2(df, intensity_cols)

# Filter: drop rows with any NaN in intensity columns
df_t = df_t.drop_nulls(subset=intensity_cols)
for c in intensity_cols:
    df_t = df_t.filter(~pl.col(c).is_nan())

if df_t.shape[0] < 10:
    st.error("Too few proteins remaining after filtering. Check your data.")
    st.stop()

st.caption(f"Analysis based on **{df_t.shape[0]:,}** proteins after removing missing values.")

# Run PCA
pca_result = run_pca(df_t, intensity_cols, n_components=n_components,
                     scale=scale_pca)

# Build condition labels for each sample
cond_labels = [col.split("_")[0] for col in intensity_cols]

# PCA Scatter Plot
scale_label = " (z-scored)" if scale_pca else ""
fig_pca = create_pca_scatter(
    pca_result, cond_labels,
    title=f"Global PCA \u2014 {transform} transformed{scale_label}",
    color_map=CONDITION_COLORS,
)
st.plotly_chart(fig_pca, width='stretch')

# Scree plot
fig_var = create_scree_plot(pca_result)
st.plotly_chart(fig_var, width='stretch')

# PERMANOVA
st.subheader("PERMANOVA Results")
mat = df_t.select(intensity_cols).to_numpy().T  # samples x features
valid_mask = ~np.isnan(mat).any(axis=0)
mat_clean = mat[:, valid_mask]

perm_result = run_permanova(mat_clean, cond_labels,
                            permutations=n_permutations)

col1, col2, col3, col4 = st.columns(4)
col1.metric("pseudo-F", f"{perm_result.test_statistic:.2f}")
col2.metric("p-value", f"{perm_result.p_value:.4f}",
            help=f"Min achievable with this design: {perm_result.min_achievable_p:.3f}")
col3.metric("R\u00b2", f"{perm_result.r_squared:.3f}")

# Cluster metrics
cluster = compute_cluster_metrics(pca_result.scores[:, :2], cond_labels)
col4.metric("Silhouette", f"{cluster.silhouette:.3f}")

# Small-sample notice
if perm_result.is_min_p:
    st.caption(
        f"\u2139\ufe0f **Note:** p-value ({perm_result.p_value:.3f}) is at the theoretical minimum "
        f"({perm_result.min_achievable_p:.3f}) for a "
        f"{n_reps}-vs-{n_reps} design. "
        f"This is the most significant result achievable with this number of replicates. "
        f"Interpret R\u00b2 and silhouette as primary effect-size indicators."
    )

# Interpretation
severity, message = interpret_permanova(perm_result, context="global")
if severity == "success":
    st.success(f"\u2705 {message}")
elif severity == "info":
    st.info(f"\u2139\ufe0f {message}")
elif severity == "warning":
    st.warning(f"\u26a0\ufe0f {message}")
else:
    st.error(f"\u274c {message}")

# Correlation heatmap
st.subheader("Sample Correlation")
fig_corr = create_correlation_heatmap(df_t, intensity_cols)
st.plotly_chart(fig_corr, width='stretch')
