"""Page 7 - ICC & CV Metrics: intraclass correlation and coefficient of variation."""
import streamlit as st
import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from config import CFG

st.header("ICC & CV Metrics")

if "df" not in st.session_state:
    st.warning("Please upload data on the Upload & Preview page first.")
    st.stop()

df = st.session_state["df"]
intensity_cols = st.session_state["intensity_cols"]

mat = df.select(intensity_cols).to_numpy().astype(np.float64)

# --- Coefficient of Variation ---
st.subheader("Coefficient of Variation (CV)")

means = np.nanmean(mat, axis=1)
sds = np.nanstd(mat, axis=1)
cvs = np.where(means > 0, sds / means, np.nan)
cvs_valid = cvs[np.isfinite(cvs)]

col1, col2, col3 = st.columns(3)
col1.metric("Median CV", f"{np.nanmedian(cvs_valid):.3f}")
col2.metric(f"CV < {CFG.cv_good:.0%}", f"{(cvs_valid < CFG.cv_good).sum()} / {len(cvs_valid)}")
col3.metric(f"CV < {CFG.cv_acceptable:.0%}", f"{(cvs_valid < CFG.cv_acceptable).sum()} / {len(cvs_valid)}")

fig_cv = px.histogram(
    x=cvs_valid, nbins=100,
    labels={"x": "CV", "y": "Count"},
    title="CV Distribution (per protein)",
)
fig_cv.add_vline(x=CFG.cv_good, line_dash="dash", line_color="green",
                 annotation_text=f"{CFG.cv_good:.0%}")
fig_cv.add_vline(x=CFG.cv_acceptable, line_dash="dash", line_color="orange",
                 annotation_text=f"{CFG.cv_acceptable:.0%}")
st.plotly_chart(fig_cv, use_container_width=True)

# Per-sample CV
st.subheader("Per-sample CV")
sample_means = np.nanmean(mat, axis=0)
sample_sds = np.nanstd(mat, axis=0)
sample_cvs = np.where(sample_means > 0, sample_sds / sample_means, np.nan)

fig_scv = px.bar(
    x=intensity_cols, y=sample_cvs,
    labels={"x": "Sample", "y": "CV"},
    title="Per-sample CV",
)
fig_scv.add_hline(y=CFG.cv_acceptable, line_dash="dash", line_color="orange")
st.plotly_chart(fig_scv, use_container_width=True)

# --- Sample correlation heatmap ---
st.subheader("Sample correlation matrix")
corr_mat = np.corrcoef(mat.T)
corr_mat = np.where(np.isfinite(corr_mat), corr_mat, 0)

fig_hm = px.imshow(
    corr_mat,
    x=intensity_cols, y=intensity_cols,
    color_continuous_scale="RdBu_r",
    zmin=0.8, zmax=1.0,
    title="Sample-sample Pearson correlation",
)
fig_hm.update_layout(height=600)
st.plotly_chart(fig_hm, use_container_width=True)

upper_tri = corr_mat[np.triu_indices_from(corr_mat, k=1)]
col1, col2 = st.columns(2)
col1.metric("Median pairwise r", f"{np.nanmedian(upper_tri):.4f}")
col2.metric("Min pairwise r", f"{np.nanmin(upper_tri):.4f}")

# --- ICC approximation ---
st.subheader("ICC (one-way random, simplified)")
st.markdown("""
The ICC below uses a one-way random effects model per protein across all samples.
Higher values indicate better reproducibility.
""")

def icc_oneway(values):
    """One-way random ICC for a single protein across samples."""
    vals = values[np.isfinite(values)]
    if len(vals) < 3:
        return np.nan
    k = len(vals)
    grand_mean = np.mean(vals)
    ms_between = np.var(vals, ddof=1)
    if ms_between == 0:
        return 1.0
    # For one-way model with single measurement per rater
    return max(0, 1 - (np.var(vals) / ms_between))

n_calc = min(2000, mat.shape[0])
icc_vals = np.array([icc_oneway(mat[i, :]) for i in range(n_calc)])
icc_valid = icc_vals[np.isfinite(icc_vals)]

if len(icc_valid) > 0:
    col1, col2 = st.columns(2)
    col1.metric("Median ICC", f"{np.nanmedian(icc_valid):.3f}")
    col2.metric("ICC > 0.75 (good)", f"{(icc_valid > 0.75).sum()} / {len(icc_valid)}")

    fig_icc = px.histogram(
        x=icc_valid, nbins=50,
        labels={"x": "ICC", "y": "Count"},
        title=f"ICC Distribution (first {n_calc} proteins)",
    )
    fig_icc.add_vline(x=0.75, line_dash="dash", line_color="green",
                      annotation_text="Good (0.75)")
    st.plotly_chart(fig_icc, use_container_width=True)
else:
    st.info("Could not compute ICC values.")
