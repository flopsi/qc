"""Page 4: Species-specific PCA — separate PCA per species."""
import streamlit as st
import numpy as np
import polars as pl
from core.transforms import log2_transform, glog_transform
from analysis.pca_engine import run_pca
from analysis.permanova import run_permanova, interpret_permanova
from analysis.cluster_metrics import compute_cluster_metrics
from viz.pca_plots import create_pca_scatter
from config import CONDITION_COLORS, SPECIES_COLORS

st.header("\U0001f9ec Species-Specific PCA")

if "protein_df" not in st.session_state:
    st.warning("\u26a0\ufe0f Please upload data on the Upload page first.")
    st.stop()

df = st.session_state["protein_df"]
intensity_cols = st.session_state["intensity_cols"]
conditions = st.session_state.get("conditions", ["A", "B"])
n_reps = st.session_state.get("n_reps", len(intensity_cols) // len(conditions))

# Check if species annotation is available
if "species" not in df.columns:
    st.warning("No species annotations found. This page requires species information in the data.")
    st.stop()

# --- Help section ---
with st.expander("\u2139\ufe0f How to interpret this page", expanded=False):
    st.markdown("""
### Species-Specific PCA

In a three-proteome benchmark (Human/Yeast/E. coli), each species plays a different role:

| Species | Role in HYE benchmark | Expected PCA result |
|---------|----------------------|-------------------|
| **Human** | Stable background (50%, equal in both conditions) | Conditions should **overlap** \u2014 no separation expected. Low R\u00b2. |
| **Yeast** | Variable species (changes between conditions) | Conditions should **separate clearly** along PC1. High R\u00b2. |
| **E. coli** | Variable species (changes between conditions) | Conditions should **separate clearly** along PC1. High R\u00b2. |

### Key metrics per species

| Metric | Interpretation |
|--------|---------------|
| **PC1 variance (%)** | If >90% for yeast/ecoli, nearly all variation is the condition effect \u2014 this is ideal. |
| **R\u00b2** | Fraction of total variance explained by condition. >0.8 = excellent for variable species. |
| **Silhouette** | Cluster separation quality. >0.7 = clear grouping. |
| **p-value** | With small replicates, the minimum achievable p is constrained by permutation count. Always prioritize R\u00b2 and silhouette for effect-size assessment. |

### The small-sample p-value problem

With n=6 (3 per group), PERMANOVA has C(6,3)=20 unique label permutations. The observed assignment
and its mirror (swapping A\u2194B) always produce the same F-statistic, so at least 2/20 permutations
will match, giving a minimum p=0.10. A p-value of 0.10\u20130.11 with R\u00b2>0.9 is extremely strong
evidence of separation \u2014 it means the observed grouping produces the highest possible F-statistic.
    """)

# Sidebar
with st.sidebar:
    st.subheader("Settings")
    transform = st.radio("Transformation", ["log2", "glog"], index=0, key="species_transform")

# Apply transformation
if transform == "log2":
    df_t = log2_transform(df, intensity_cols)
else:
    df_t = glog_transform(df, intensity_cols)

# Drop NaN
df_t = df_t.drop_nulls(subset=intensity_cols)
for c in intensity_cols:
    df_t = df_t.filter(~pl.col(c).is_nan())

# Condition labels
cond_labels = [col.split("_")[0] for col in intensity_cols]

# Get available species (prefer canonical order, then any others)
all_species = df_t.select("species").unique().to_series().to_list()
canonical_order = ["human", "yeast", "ecoli"]
available_species = [s for s in canonical_order if s in all_species]
# Add non-canonical species (including 'unknown' if it's the only one)
other_species = [s for s in all_species if s not in canonical_order and s != "unknown"]
available_species.extend(other_species)
# If all proteins are unknown, include it
if not available_species and "unknown" in all_species:
    available_species = ["unknown"]

if not available_species:
    st.warning("No species annotations found. Check species_annotator configuration.")
    st.stop()

# Define which species are expected to be stable vs variable
# In HYE benchmark: human is stable, yeast and ecoli are variable
stable_species = {"human"}
variable_species = {"yeast", "ecoli"}

# Create columns for each species
n_sp = len(available_species)
cols = st.columns(min(n_sp, 3))

for i, species in enumerate(available_species):
    with cols[i % len(cols)]:
        st.subheader(f"{species.capitalize()}")
        df_sp = df_t.filter(pl.col("species") == species)
        n_proteins = df_sp.shape[0]
        st.write(f"**{n_proteins:,} proteins**")

        if n_proteins < 5:
            st.warning("Too few proteins for PCA.")
            continue

        try:
            pca_res = run_pca(df_sp, intensity_cols,
                              n_components=min(3, len(intensity_cols)),
                              scale=False)

            fig = create_pca_scatter(
                pca_res, cond_labels,
                title=f"{species.capitalize()} PCA",
                color_map=CONDITION_COLORS,
            )
            fig.update_layout(width=400, height=400)
            st.plotly_chart(fig, use_container_width=True)

            # PERMANOVA
            mat = df_sp.select(intensity_cols).to_numpy().T
            valid = ~np.isnan(mat).any(axis=0)
            mat_clean = mat[:, valid]

            if mat_clean.shape[1] >= 3:
                perm_res = run_permanova(mat_clean, cond_labels)
                cluster = compute_cluster_metrics(pca_res.scores[:, :2], cond_labels)

                st.metric("p-value", f"{perm_res.p_value:.4f}",
                         help=f"Min achievable: {perm_res.min_achievable_p:.3f}")
                st.metric("R\u00b2", f"{perm_res.r_squared:.3f}")
                st.metric("Silhouette", f"{cluster.silhouette:.3f}")

                # Variance explained by PC1
                pc1_var = pca_res.explained_variance[0] * 100
                st.caption(f"PC1 captures {pc1_var:.1f}% of variance")

                # Context-aware interpretation
                if species in stable_species:
                    context = "species_stable"
                elif species in variable_species:
                    context = "species_variable"
                else:
                    context = "global"  # unknown species — use generic

                severity, message = interpret_permanova(perm_res, context=context)
                if severity == "success":
                    st.success(f"\u2705 {message}")
                elif severity == "info":
                    st.info(f"\u2139\ufe0f {message}")
                elif severity == "warning":
                    st.warning(f"\u26a0\ufe0f {message}")
                else:
                    st.error(f"\u274c {message}")
        except Exception as e:
            st.error(f"Error: {e}")
