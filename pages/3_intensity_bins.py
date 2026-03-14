"""Page 3: Intensity-bin PCA — PCA and PERMANOVA per intensity quartile."""
import streamlit as st
import numpy as np
import polars as pl
from core.transforms import log2_transform, glog_transform, compute_intensity_bins
from analysis.pca_engine import run_pca
from analysis.permanova import run_permanova, interpret_permanova
from analysis.cluster_metrics import compute_cluster_metrics
from viz.pca_plots import create_pca_scatter
from config import CONDITION_COLORS, BIN_LABELS

st.header("\U0001f4ca Intensity-Bin PCA")

if "protein_df" not in st.session_state:
    st.warning("\u26a0\ufe0f Please upload data on the Upload page first.")
    st.stop()

df = st.session_state["protein_df"]
intensity_cols = st.session_state["intensity_cols"]
conditions = st.session_state.get("conditions", ["A", "B"])

# --- Help section ---
with st.expander("\u2139\ufe0f How to interpret this page", expanded=False):
    st.markdown("""
### Intensity-Bin Analysis

This page splits proteins into quartiles (Q1\u2013Q4) based on their average intensity and runs 
PCA + PERMANOVA independently within each bin. This reveals whether the measurement system 
resolves conditions equally well across the intensity range, or whether performance degrades 
at low abundance.

| Bin | Meaning | Typical behavior |
|-----|---------|-----------------|
| **Q1** | Lowest-intensity proteins | Often noisiest, may show poor separation. High CVs expected. |
| **Q2** | Low-to-medium intensity | Moderate noise, separation should begin to appear. |
| **Q3** | Medium-to-high intensity | Usually good separation and precision. |
| **Q4** | Highest-intensity proteins | Best precision and strongest separation. |

**What to look for:**
- R\u00b2 should **increase** from Q1 to Q4 — this is normal and expected
- If Q1 shows no separation but Q4 does, the system works but has limited sensitivity at low abundance
- If all bins show separation, the system performs well across the dynamic range
- If Q4 fails to separate, there may be a systematic issue

**Note:** With 3 vs 3 replicates, the minimum p-value is 0.10 in all bins. Focus on R\u00b2 and silhouette.
    """)

# Sidebar
with st.sidebar:
    st.subheader("Settings")
    transform = st.radio("Transformation", ["log2", "glog"], index=0, key="bin_transform")
    n_bins = st.slider("Number of bins", 2, 6, 4, key="n_bins")

# Apply transformation
if transform == "log2":
    df_t = log2_transform(df, intensity_cols)
else:
    df_t = glog_transform(df, intensity_cols)

# Drop NaN
df_t = df_t.drop_nulls(subset=intensity_cols)
for c in intensity_cols:
    df_t = df_t.filter(~pl.col(c).is_nan())

# Compute intensity bins
df_t = compute_intensity_bins(df_t, intensity_cols, n_bins=n_bins)

# Build condition labels
cond_labels = [col.split("_")[0] for col in intensity_cols]

# Create tabs for each bin
bin_names = [f"Q{i+1}" for i in range(n_bins)]
tabs = st.tabs(bin_names)

summary_rows = []

for i, (tab, bin_name) in enumerate(zip(tabs, bin_names)):
    with tab:
        df_bin = df_t.filter(pl.col("intensity_bin") == bin_name)
        n_proteins = df_bin.shape[0]
        st.write(f"**{n_proteins:,} proteins** in {bin_name}")
        
        if n_proteins < 5:
            st.warning(f"Too few proteins in {bin_name} for PCA.")
            summary_rows.append({
                "Bin": bin_name, "N proteins": n_proteins,
                "R\u00b2": "N/A", "Silhouette": "N/A", "p-value": "N/A",
                "Interpretation": "Insufficient data",
            })
            continue
        
        try:
            pca_res = run_pca(df_bin, intensity_cols, n_components=min(3, len(intensity_cols)),
                             scale=False)
            
            fig = create_pca_scatter(
                pca_res, cond_labels,
                title=f"PCA \u2014 {bin_name} ({n_proteins:,} proteins)",
                color_map=CONDITION_COLORS,
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # PERMANOVA
            mat = df_bin.select(intensity_cols).to_numpy().T
            valid = ~np.isnan(mat).any(axis=0)
            mat_clean = mat[:, valid]
            
            if mat_clean.shape[1] >= 3:
                perm_res = run_permanova(mat_clean, cond_labels)
                cluster = compute_cluster_metrics(pca_res.scores[:, :2], cond_labels)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("R\u00b2", f"{perm_res.r_squared:.3f}")
                c2.metric("p-value", f"{perm_res.p_value:.4f}",
                         help=f"Min achievable: {perm_res.min_achievable_p:.2f}")
                c3.metric("Silhouette", f"{cluster.silhouette:.3f}")
                
                # Descriptive interpretation per bin
                severity, message = interpret_permanova(perm_res, context="intensity_bin")
                if severity == "success":
                    st.success(f"\u2705 {message}")
                elif severity == "info":
                    st.info(f"\u2139\ufe0f {message}")
                elif severity == "warning":
                    st.warning(f"\u26a0\ufe0f {message}")
                else:
                    st.error(f"\u274c {message}")
                
                summary_rows.append({
                    "Bin": bin_name, "N proteins": n_proteins,
                    "R\u00b2": f"{perm_res.r_squared:.3f}",
                    "Silhouette": f"{cluster.silhouette:.3f}",
                    "p-value": f"{perm_res.p_value:.4f}",
                    "Interpretation": message[:80] + "..." if len(message) > 80 else message,
                })
            else:
                st.info("Not enough valid features for PERMANOVA.")
                summary_rows.append({
                    "Bin": bin_name, "N proteins": n_proteins,
                    "R\u00b2": "N/A", "Silhouette": "N/A", "p-value": "N/A",
                    "Interpretation": "Insufficient valid features",
                })
        except Exception as e:
            st.error(f"Error in {bin_name}: {e}")
            summary_rows.append({
                "Bin": bin_name, "N proteins": n_proteins,
                "R\u00b2": "Error", "Silhouette": "Error", "p-value": "Error",
                "Interpretation": str(e)[:80],
            })

# Summary table
st.subheader("Summary Across Intensity Bins")
if summary_rows:
    st.dataframe(pl.DataFrame(summary_rows).to_pandas(), use_container_width=True)
    
    # Overall trend assessment
    valid_r2 = [float(r["R\u00b2"]) for r in summary_rows 
                if r["R\u00b2"] not in ("N/A", "Error")]
    if len(valid_r2) >= 2:
        if valid_r2[-1] > valid_r2[0]:
            st.info(
                f"\U0001f4c8 R\u00b2 increases from {valid_r2[0]:.3f} (Q1) to {valid_r2[-1]:.3f} "
                f"(Q{len(valid_r2)}), indicating better condition resolution at higher intensities. "
                f"This is typical — low-abundance proteins are noisier."
            )
        elif all(r > 0.5 for r in valid_r2):
            st.success(
                "\u2705 Strong separation across all intensity bins. The measurement system "
                "resolves conditions well throughout the dynamic range."
            )
