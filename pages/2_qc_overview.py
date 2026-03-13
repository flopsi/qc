"""Page 2 - Global PCA: PCA on all samples, PERMANOVA by Condition."""
import streamlit as st
import polars as pl
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import permanova, DistanceMatrix
from config import CFG

st.header("Global PCA")

if "df" not in st.session_state:
    st.warning("Please upload data on the Upload & Preview page first.")
    st.stop()

df = st.session_state["df"]
intensity_cols = st.session_state["intensity_cols"]

# Build numeric matrix
mat = df.select(intensity_cols).to_numpy().astype(np.float64)
mask = np.isfinite(mat)
mat = np.where(mask, mat, 0)

# Sidebar options
with st.sidebar:
    n_comp = st.slider("PCA components", 2, min(10, mat.shape[1]), CFG.pca_n_components)
    do_scale = st.checkbox("Standardize (z-score)", value=CFG.pca_scale)
    color_col = st.selectbox("Color by", [None] + st.session_state["meta_cols"])

if do_scale:
    scaler = StandardScaler()
    mat_scaled = scaler.fit_transform(mat)
else:
    mat_scaled = mat

pca = PCA(n_components=n_comp)
scores = pca.fit_transform(mat_scaled)

# Scores DataFrame
score_cols = [f"PC{i+1}" for i in range(n_comp)]
scores_df = pl.DataFrame({c: scores[:, i] for i, c in enumerate(score_cols)})

if color_col and color_col in df.columns:
    scores_df = scores_df.with_columns(df[color_col].alias(color_col))

var_explained = pca.explained_variance_ratio_ * 100

# --- Scree plot ---
col1, col2 = st.columns(2)
with col1:
    fig_scree = px.bar(
        x=score_cols, y=var_explained,
        labels={"x": "Component", "y": "% Variance Explained"},
        title="Scree Plot",
    )
    fig_scree.update_layout(showlegend=False)
    st.plotly_chart(fig_scree, use_container_width=True)

# --- 2D Scatter ---
with col2:
    pc_x = st.selectbox("X axis", score_cols, index=0, key="pcx")
    pc_y = st.selectbox("Y axis", score_cols, index=1, key="pcy")

pd_scores = scores_df.to_pandas()
fig_scatter = px.scatter(
    pd_scores, x=pc_x, y=pc_y,
    color=color_col if color_col else None,
    title=f"{pc_x} vs {pc_y}",
    labels={
        pc_x: f"{pc_x} ({var_explained[score_cols.index(pc_x)]:.1f}%)",
        pc_y: f"{pc_y} ({var_explained[score_cols.index(pc_y)]:.1f}%)",
    },
    color_discrete_sequence=list(CFG.color_discrete_sequence),
)
st.plotly_chart(fig_scatter, use_container_width=True)

# --- PERMANOVA ---
if color_col and color_col in df.columns:
    st.subheader("PERMANOVA")
    try:
        grouping = df[color_col].to_list()
        dist_mat = squareform(pdist(mat_scaled, metric="euclidean"))
        dm = DistanceMatrix(dist_mat)
        result = permanova(dm, grouping, permutations=CFG.permanova_permutations)
        st.metric("Test statistic", f"{result['test statistic']:.4f}")
        st.metric("p-value", f"{result['p-value']:.4f}")
        if result["p-value"] < 0.05:
            st.success("Significant difference between groups (p < 0.05).")
        else:
            st.info("No significant difference between groups (p >= 0.05).")
    except Exception as e:
        st.error(f"PERMANOVA failed: {e}")

# --- Loadings ---
with st.expander("PCA Loadings"):
    loadings = pca.components_.T
    load_df = pl.DataFrame(
        {f"PC{i+1}": loadings[:, i] for i in range(n_comp)}
    ).with_columns(pl.Series("Feature", intensity_cols))
    st.dataframe(load_df.to_pandas(), use_container_width=True)
