"""Page 5 - Transformations: compare log2 vs generalized log (glog)."""
import streamlit as st
import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import CFG

st.header("log2 vs glog Transformations")

if "df" not in st.session_state:
    st.warning("Please upload data on the Upload & Preview page first.")
    st.stop()

df = st.session_state["df"]
intensity_cols = st.session_state["intensity_cols"]

mat = df.select(intensity_cols).to_numpy().astype(np.float64)


def glog(x, lam=1.0):
    """Generalized logarithm transform: asinh(x / lam) * ln(2)."""
    return np.arcsinh(x / lam) / np.log(2)


# Sidebar
with st.sidebar:
    glog_lambda = st.number_input("glog lambda", min_value=0.01, value=1.0, step=0.1)

# Apply transforms
mat_pos = np.where(mat > 0, mat, np.nan)
log2_mat = np.log2(mat_pos)
glog_mat = glog(mat, lam=glog_lambda)

st.subheader("Distribution comparison")

# Flatten for histogram
log2_flat = log2_mat.flatten()
glog_flat = glog_mat.flatten()
log2_flat = log2_flat[np.isfinite(log2_flat)]
glog_flat = glog_flat[np.isfinite(glog_flat)]

fig = make_subplots(rows=1, cols=2, subplot_titles=["log2", "glog"])
fig.add_trace(go.Histogram(x=log2_flat, nbinsx=100, name="log2", marker_color="#636EFA"), row=1, col=1)
fig.add_trace(go.Histogram(x=glog_flat, nbinsx=100, name="glog", marker_color="#EF553B"), row=1, col=2)
fig.update_layout(height=400, showlegend=False, title_text="Intensity distributions after transformation")
st.plotly_chart(fig, use_container_width=True)

# Per-sample boxplots
st.subheader("Per-sample boxplots")
transform_choice = st.radio("Transform", ["log2", "glog"], horizontal=True)

if transform_choice == "log2":
    chosen_mat = log2_mat
else:
    chosen_mat = glog_mat

box_data = []
for i, col_name in enumerate(intensity_cols):
    vals = chosen_mat[:, i]
    vals = vals[np.isfinite(vals)]
    box_data.append(go.Box(y=vals, name=col_name, showlegend=False))

fig_box = go.Figure(data=box_data)
fig_box.update_layout(title=f"Per-sample boxplot ({transform_choice})", height=500)
st.plotly_chart(fig_box, use_container_width=True)

# Mean-SD plot
st.subheader("Mean vs SD (per protein)")
for label, tmat in [("log2", log2_mat), ("glog", glog_mat)]:
    means = np.nanmean(tmat, axis=1)
    sds = np.nanstd(tmat, axis=1)
    valid = np.isfinite(means) & np.isfinite(sds)

    fig_ms = px.scatter(
        x=means[valid], y=sds[valid],
        labels={"x": "Mean", "y": "SD"},
        title=f"Mean-SD plot ({label})",
        opacity=0.3,
    )
    fig_ms.update_layout(height=400)
    st.plotly_chart(fig_ms, use_container_width=True)
