"""Page 6: PVCA — Principal Variance Component Analysis."""
import streamlit as st
import polars as pl
from core.transforms import log2_transform, glog_transform
from analysis.variance_components import run_pvca
from viz.variance_bar import create_pvca_bar

st.header("\U0001f4ca Variance Components (PVCA)")

if "protein_df" not in st.session_state:
    st.warning("\u26a0\ufe0f Please upload data on the Upload page first.")
    st.stop()

df = st.session_state["protein_df"]
intensity_cols = st.session_state["intensity_cols"]
conditions = st.session_state.get("conditions", ["A", "B"])

# --- Help section ---
with st.expander("\u2139\ufe0f How to interpret this page", expanded=False):
    st.markdown("""
### PVCA (Principal Variance Component Analysis)

PVCA decomposes the total variance in the dataset into contributions from known experimental 
factors. It combines PCA (for dimensionality reduction) with variance component estimation 
(like a random-effects ANOVA).

**Algorithm:**
1. Run PCA on all samples
2. Retain PCs until a cumulative variance threshold is met (default 60%)
3. For each retained PC, decompose variance into condition, replicate, and residual components
4. Weight each PC's contributions by its eigenvalue to get overall proportions

### Variance Components

| Component | What it represents | Ideal outcome |
|-----------|-------------------|---------------|
| **Condition** | Biological signal — variance due to the intended experimental manipulation | Should be **dominant** (>50%). This is what you designed the experiment to measure. |
| **Replicate** | Technical variation between replicates within the same condition | Should be **minimal** (<10%). High values indicate poor technical reproducibility. |
| **Residual** | Unexplained variation — measurement noise, protein-level effects, etc. | Moderate levels are normal. Very high residual suggests uncontrolled noise sources. |

### How to judge

| Condition % | Assessment |
|------------|-----------|
| **>70%** | Excellent — the experiment is well-powered and the signal dominates |
| **50\u201370%** | Good — clear biological signal with some noise |
| **20\u201350%** | Moderate — biological signal present but noise is substantial |
| **<20%** | Poor — noise or batch effects overshadow the biological signal |

### Species-specific PVCA

For a HYE benchmark:
- **Human PVCA**: Condition should explain **little** variance (human is the stable background)
- **Yeast/E. coli PVCA**: Condition should explain **most** variance (these species change between conditions)

### Cumulative variance threshold

The slider controls how many PCs are retained. Lower thresholds use fewer PCs (faster, less noise), 
higher thresholds capture more total variance. Default 60% is standard.
    """)

# Sidebar
with st.sidebar:
    st.subheader("PVCA Settings")
    transform = st.radio("Transformation", ["log2", "glog"], index=0, key="pvca_transform")
    threshold = st.slider("Cumulative variance threshold", 0.3, 0.95, 0.6, 0.05,
                          key="pvca_threshold",
                          help="Retain PCs until this proportion of total variance is explained")

# Apply transformation
if transform == "log2":
    df_t = log2_transform(df, intensity_cols)
else:
    df_t = glog_transform(df, intensity_cols)

# Drop NaN
df_t = df_t.drop_nulls(subset=intensity_cols)
for c in intensity_cols:
    df_t = df_t.filter(~pl.col(c).is_nan())

# Run PVCA
try:
    pvca_result = run_pvca(df_t, intensity_cols, conditions, threshold=threshold)
    
    # Bar chart
    fig = create_pvca_bar(pvca_result)
    st.plotly_chart(fig, width='stretch')
    
    # Detailed results
    st.subheader("Detailed Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Condition", f"{pvca_result.components['Condition']*100:.1f}%")
    col2.metric("Replicate", f"{pvca_result.components['Replicate']*100:.1f}%")
    col3.metric("Residual", f"{pvca_result.components['Residual']*100:.1f}%")
    
    st.write(f"**PCs retained:** {pvca_result.n_pcs_used}")
    
    # Interpretation
    cond_pct = pvca_result.components["Condition"] * 100
    rep_pct = pvca_result.components["Replicate"] * 100
    res_pct = pvca_result.components["Residual"] * 100
    
    if cond_pct > 70:
        st.success(
            f"\u2705 Excellent: Condition explains {cond_pct:.1f}% of variance. "
            f"The biological signal strongly dominates over technical noise "
            f"(replicate: {rep_pct:.1f}%, residual: {res_pct:.1f}%)."
        )
    elif cond_pct > 50:
        st.success(
            f"\u2705 Good: Condition explains {cond_pct:.1f}% of variance. "
            f"Clear biological signal with manageable noise."
        )
    elif cond_pct > 20:
        st.info(
            f"\u2139\ufe0f Moderate: Condition explains {cond_pct:.1f}% of variance. "
            f"Biological signal is present but noise is substantial."
        )
    else:
        st.warning(
            f"\u26a0\ufe0f Low: Condition explains only {cond_pct:.1f}% of variance. "
            f"Technical or unexplained variation dominates."
        )
    
    if rep_pct > 20:
        st.warning(
            f"\u26a0\ufe0f Replicate variation is high ({rep_pct:.1f}%). "
            f"This suggests poor technical reproducibility \u2014 investigate sample preparation, "
            f"instrument variability, or batch effects."
        )

    # Species-specific PVCA
    if "species" in df_t.columns:
        st.subheader("Species-Specific PVCA")
        species_list = df_t.select("species").unique().to_series().to_list()
        species_list = [s for s in ["human", "yeast", "ecoli"] if s in species_list]
    
        if species_list:
            sp_cols = st.columns(len(species_list))
            for j, species in enumerate(species_list):
                with sp_cols[j]:
                    st.markdown(f"**{species.capitalize()}**")
                    df_sp = df_t.filter(pl.col("species") == species)
                    if df_sp.shape[0] >= 10:
                        sp_pvca = run_pvca(df_sp, intensity_cols, conditions, threshold=threshold)
                        sp_fig = create_pvca_bar(sp_pvca, title=f"{species.capitalize()} PVCA")
                        sp_fig.update_layout(width=400, height=400)
                        st.plotly_chart(sp_fig, width='stretch')
                        
                        sp_cond = sp_pvca.components["Condition"] * 100
                        if species == "human":
                            if sp_cond < 30:
                                st.success(f"\u2705 Human background is stable (condition: {sp_cond:.1f}%).")
                            else:
                                st.warning(f"\u26a0\ufe0f Human shows unexpected condition effect ({sp_cond:.1f}%).")
                        else:
                            if sp_cond > 50:
                                st.success(f"\u2705 {species.capitalize()} shows strong condition effect ({sp_cond:.1f}%).")
                            else:
                                st.info(f"\u2139\ufe0f {species.capitalize()} condition effect is moderate ({sp_cond:.1f}%).")
                    else:
                        st.write(f"Too few {species} proteins for PVCA.")

except Exception as e:
    st.error(f"PVCA computation failed: {e}")
    import traceback
    st.code(traceback.format_exc())
