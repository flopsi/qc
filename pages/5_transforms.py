"""Page 5: Transformation comparison — log2 vs glog side-by-side."""
import streamlit as st
import numpy as np
import polars as pl
from core.transforms import log2_transform, glog_transform, vsn_transform_rpy2
from viz.mean_sd_plot import create_mean_sd_plot

st.header("\U0001f504 Transformation Comparison")

if "protein_df" not in st.session_state:
    st.warning("\u26a0\ufe0f Please upload data on the Upload page first.")
    st.stop()

df = st.session_state["protein_df"]
intensity_cols = st.session_state["intensity_cols"]

# --- Help section ---
with st.expander("\u2139\ufe0f How to interpret this page", expanded=False):
    st.markdown("""
### Variance-Stabilizing Transformations

Raw protein intensities span several orders of magnitude, and their variance typically increases 
with mean intensity. This heteroscedasticity violates assumptions of many statistical tests. 
Transformations aim to make variance independent of the mean.

| Transform | Method | Characteristics |
|-----------|--------|----------------|
| **log2** | log\u2082(x + 1) | Simple, widely used. Good at high intensities but variance can still increase at low values. |
| **glog** | log\u2082(x + \u221a(x\u00b2 + \u03bb)) | Generalized log. Approaches log2 at high intensities, compresses variance at low intensities. \u03bb estimated from data. |
| **VSN** | R's vsn::vsn2() via rpy2, or glog fallback | Maximum-likelihood variance stabilization. Gold standard but requires R. Falls back to glog if R is unavailable. |

### Mean-SD Plot

The Mean-SD plot shows the relationship between a protein's mean intensity (x-axis) and its 
standard deviation across samples (y-axis). The red trend line is a running median smoother.

**How to judge:**
- **Flat trend line** = variance is stabilized \u2192 good transformation
- **Rising trend** (SD increases with mean) = variance not fully stabilized \u2192 poor for downstream stats
- **Falling trend at low intensities** = over-compression of low-abundance variance

### Summary statistics

| Metric | What it tells you |
|--------|------------------|
| **Median SD** | Overall noise level after transformation. Lower = less noisy. |
| **SD range** | Spread of noise across proteins. Narrow range = good stabilization. |
| **SD CV** | Coefficient of variation of the SD values themselves. Lower = more uniform variance = better stabilization. |

Choose the transformation with the **flattest trend** and **lowest SD CV**.
    """)

# Apply all transformations
transforms = {
    "log2": log2_transform(df, intensity_cols),
    "glog": glog_transform(df, intensity_cols),
    "VSN (glog fallback)": vsn_transform_rpy2(df, intensity_cols),
}

# Also show raw (no transform)
raw_df = df.clone()

# Create side-by-side plots
st.subheader("Mean-SD Plots")

cols = st.columns(len(transforms) + 1)

# Raw data
with cols[0]:
    st.markdown("**Raw (no transform)**")
    raw_mat = raw_df.select(intensity_cols).to_numpy().astype(np.float64)
    valid = ~np.isnan(raw_mat).any(axis=1)
    raw_valid = raw_mat[valid]
    if len(raw_valid) > 0:
        fig_raw = create_mean_sd_plot(raw_valid, title="Raw")
        fig_raw.update_layout(width=400, height=400)
        st.plotly_chart(fig_raw, use_container_width=True)
        
        row_sds = np.std(raw_valid, axis=1, ddof=1)
        st.write(f"Median SD: **{np.median(row_sds):.1f}**")
        st.write(f"SD CV: **{np.std(row_sds)/np.mean(row_sds)*100:.1f}%**")

# Transformed data
for i, (name, df_t) in enumerate(transforms.items()):
    with cols[i + 1]:
        st.markdown(f"**{name}**")
        df_clean = df_t.drop_nulls(subset=intensity_cols)
        for c in intensity_cols:
            df_clean = df_clean.filter(~pl.col(c).is_nan())
        
        mat = df_clean.select(intensity_cols).to_numpy().astype(np.float64)
        valid = ~np.isnan(mat).any(axis=1)
        mat_valid = mat[valid]
        
        if len(mat_valid) > 0:
            fig = create_mean_sd_plot(mat_valid, title=name)
            fig.update_layout(width=400, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            row_sds = np.std(mat_valid, axis=1, ddof=1)
            st.write(f"Median SD: **{np.median(row_sds):.3f}**")
            st.write(f"SD range: [{np.min(row_sds):.3f}, {np.max(row_sds):.3f}]")
            st.write(f"SD CV: **{np.std(row_sds)/np.mean(row_sds)*100:.1f}%**")

# Recommendation
st.subheader("Recommendation")
st.info(
    "Choose the transformation with the **flattest Mean-SD trend** and **lowest SD CV**. "
    "For most LC-MS proteomics data, **glog** provides good variance stabilization "
    "as a Python-native alternative to R's VSN package. **log2** is simpler but "
    "may not stabilize variance at low intensities."
)
