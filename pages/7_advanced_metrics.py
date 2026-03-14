"""Page 7: ICC, CV distributions, and cluster metrics."""
import streamlit as st
import numpy as np
import polars as pl
from core.transforms import log2_transform, glog_transform, compute_intensity_bins
from analysis.cv_analysis import compute_cvs
from analysis.icc import compute_icc
from analysis.cluster_metrics import compute_cluster_metrics
from analysis.pca_engine import run_pca
from viz.cv_plots import create_cv_violin_plot, create_cv_summary_table
from config import ICC_EXCELLENT, ICC_GOOD, CV_THRESHOLD

st.header("\U0001f4c8 ICC & CV Metrics")

if "protein_df" not in st.session_state:
    st.warning("\u26a0\ufe0f Please upload data on the Upload page first.")
    st.stop()

df = st.session_state["protein_df"]
intensity_cols = st.session_state["intensity_cols"]
conditions = st.session_state.get("conditions", ["A", "B"])

# --- Help section ---
with st.expander("\u2139\ufe0f How to interpret this page", expanded=False):
    st.markdown("""
### Intraclass Correlation Coefficient (ICC)

ICC measures the **consistency** (reproducibility) of replicate measurements. It quantifies how much
of the total variance in protein measurements is due to true biological differences between proteins,
versus technical noise between replicates.

| ICC value | Rating | Meaning |
|-----------|--------|---------|
| **> 0.95** | Excellent | Almost all variance is biological; replicates are nearly identical |
| **0.90 \u2013 0.95** | Very good | Strong reproducibility; minor technical variation |
| **0.75 \u2013 0.90** | Good | Acceptable for most applications |
| **0.50 \u2013 0.75** | Moderate | Noticeable technical variation; may affect quantification |
| **< 0.50** | Poor | Technical noise dominates; consider protocol optimization |

ICC is computed as ICC(3,1) \u2014 two-way mixed model, single measures \u2014 which assesses **consistency**
(not absolute agreement). This is appropriate for replicate LC-MS runs.

### Coefficient of Variation (CV)

CV = (standard deviation / mean) \u00d7 100%, computed per protein across replicates within each condition.
It measures the **precision** of quantification for individual proteins. **CVs are always computed on
raw (untransformed) intensities**, because log or glog transforms distort the SD/mean ratio
non-linearly, making the percentage meaningless.

| Median CV | Assessment |
|-----------|-----------|
| **< 5%** | Excellent precision |
| **5 \u2013 10%** | Good precision |
| **10 \u2013 20%** | Acceptable |
| **20 \u2013 30%** | Marginal \u2014 may limit statistical power |
| **> 30%** | Poor \u2014 quantitative results unreliable |

**CV by intensity bin:** Low-abundance proteins (Q1) typically have higher CVs. If Q4 (highest intensity)
also shows high CVs, there may be a systematic problem.

### Cluster Quality Metrics

| Metric | What it measures | Scale | Good value |
|--------|-----------------|-------|-----------|
| **Silhouette** | How well each sample fits its assigned condition group vs. the other group | \u22121 to 1 | > 0.5 (good), > 0.7 (excellent) |
| **Calinski-Harabasz** | Ratio of between-cluster to within-cluster dispersion | 0 to \u221e | Higher = better. No universal threshold; compare across settings. |

These metrics are computed on PCA scores using different numbers of principal components to assess
robustness. Stable values across PC counts indicate a robust grouping structure.
    """)

# Sidebar
with st.sidebar:
    st.subheader("Settings")
    transform = st.radio("Transformation", ["log2", "glog"], index=0, key="adv_transform")

# Apply transformation (used for ICC and cluster metrics)
if transform == "log2":
    df_t = log2_transform(df, intensity_cols)
else:
    df_t = glog_transform(df, intensity_cols)

# Drop NaN from transformed data
df_t = df_t.drop_nulls(subset=intensity_cols)
for c in intensity_cols:
    df_t = df_t.filter(~pl.col(c).is_nan())

# === ICC Section ===
st.subheader("Intraclass Correlation Coefficients (ICC)")

with st.spinner("Computing ICC..."):
    icc_results = compute_icc(df_t, intensity_cols, conditions)

if icc_results:
    icc_cols = st.columns(len(icc_results))
    for i, (cond, icc_val) in enumerate(icc_results.items()):
        with icc_cols[i]:
            if np.isnan(icc_val):
                st.metric(f"ICC \u2014 Condition {cond}", "N/A")
            else:
                st.metric(f"ICC \u2014 Condition {cond}", f"{icc_val:.4f}")
                if icc_val >= 0.95:
                    st.success("Excellent reproducibility")
                elif icc_val >= ICC_EXCELLENT:
                    st.success("Very good reproducibility")
                elif icc_val >= ICC_GOOD:
                    st.info("Good reproducibility")
                elif icc_val >= 0.5:
                    st.warning("Moderate reproducibility")
                else:
                    st.error("Poor reproducibility")

# === CV Section ===
st.subheader("Coefficient of Variation (CV) Distributions")
st.caption("CVs are computed on **raw (untransformed) intensities** \u2014 log/glog transforms distort the SD/mean ratio.")

# Prepare raw data for CV computation (drop rows with NaN/null in intensity cols)
df_raw = df.drop_nulls(subset=intensity_cols)
for c in intensity_cols:
    df_raw = df_raw.filter(~pl.col(c).is_nan())
    df_raw = df_raw.filter(pl.col(c) > 0)  # CV undefined for zero intensities

# Compute intensity bins on raw data
df_raw_binned = compute_intensity_bins(df_raw, intensity_cols)

# Compute CVs on raw (untransformed) data
cv_df = compute_cvs(df_raw_binned, intensity_cols, conditions)

# CV summary table
st.markdown("**Global CV Summary:**")
cv_summary = create_cv_summary_table(cv_df, conditions)
if cv_summary.shape[0] > 0:
    st.dataframe(cv_summary.to_pandas(), width='stretch')

    # Interpretation
    for row in cv_summary.iter_rows(named=True):
        median_cv = row["Median CV (%)"]
        cond = row["Condition"]
        pct_under_20 = row["CV < 20% (%)"]
        if median_cv < 5:
            st.success(f"\u2705 Condition {cond}: Excellent precision (median CV = {median_cv:.1f}%, "
                      f"{pct_under_20:.0f}% of proteins under 20% CV).")
        elif median_cv < 10:
            st.info(f"\u2139\ufe0f Condition {cond}: Good precision (median CV = {median_cv:.1f}%).")
        elif median_cv < 20:
            st.info(f"\u2139\ufe0f Condition {cond}: Acceptable precision (median CV = {median_cv:.1f}%).")
        else:
            st.warning(f"\u26a0\ufe0f Condition {cond}: Poor precision (median CV = {median_cv:.1f}%). "
                      f"Only {pct_under_20:.0f}% of proteins are under 20% CV.")

# CV violin plot by intensity bin with per-bin interpretation
st.markdown("**CV Distribution by Intensity Bin:**")
fig_cv_bin = create_cv_violin_plot(cv_df, conditions, group_by="intensity_bin",
                                    title="CV (%) by Intensity Bin (raw intensities)")
st.plotly_chart(fig_cv_bin, width='stretch')

# Per-bin interpretation
if "intensity_bin" in cv_df.columns:
    bin_order = ["Q1", "Q2", "Q3", "Q4"]
    bin_labels = {
        "Q1": "lowest-abundance",
        "Q2": "low-to-medium-abundance",
        "Q3": "medium-to-high-abundance",
        "Q4": "highest-abundance",
    }
    for b in bin_order:
        bin_data = cv_df.filter(pl.col("intensity_bin") == b)
        if bin_data.shape[0] == 0:
            continue
        msgs = []
        for cond in conditions:
            cv_col = f"cv_{cond}"
            if cv_col not in bin_data.columns:
                continue
            vals = bin_data[cv_col].drop_nulls().to_numpy()
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            med = float(np.median(vals))
            if med < 5:
                rating = "excellent"
            elif med < 10:
                rating = "good"
            elif med < 20:
                rating = "acceptable"
            elif med < 30:
                rating = "marginal"
            else:
                rating = "poor"
            msgs.append(f"Condition {cond}: median CV = {med:.1f}% ({rating})")
        if msgs:
            desc = bin_labels.get(b, b)
            msg_str = "; ".join(msgs)
            if b == "Q1":
                st.markdown(f"**{b} ({desc}, n={bin_data.shape[0]}):** {msg_str}. "
                           f"Higher CVs in Q1 are expected due to lower signal-to-noise.")
            elif b == "Q4":
                any_high = False
                for c in conditions:
                    cv_c = f"cv_{c}"
                    if cv_c in bin_data.columns and len(bin_data[cv_c].drop_nulls()) > 0:
                        v = bin_data[cv_c].drop_nulls().to_numpy()
                        v = v[~np.isnan(v)]
                        if len(v) > 0 and float(np.median(v)) > 20:
                            any_high = True
                note = " \u26a0\ufe0f High CVs at high intensity suggest a systematic issue." if any_high else ""
                st.markdown(f"**{b} ({desc}, n={bin_data.shape[0]}):** {msg_str}. "
                           f"High-abundance proteins should have the lowest CVs.{note}")
            else:
                st.markdown(f"**{b} ({desc}, n={bin_data.shape[0]}):** {msg_str}.")

# CV violin plot by species
if "species" in cv_df.columns:
    st.markdown("**CV Distribution by Species:**")
    fig_cv_sp = create_cv_violin_plot(cv_df, conditions, group_by="species",
                                      title="CV (%) by Species (raw intensities)")
    st.plotly_chart(fig_cv_sp, width='stretch')

# === Cluster Metrics Section ===
st.subheader("Cluster Quality Metrics")

cond_labels = [col.split("_")[0] for col in intensity_cols]

try:
    pca_res = run_pca(df_t, intensity_cols, n_components=min(5, len(intensity_cols)), scale=False)

    st.markdown("**Metrics across PC dimensions:**")
    metric_rows = []
    for n_pc in range(2, min(6, pca_res.n_components + 1)):
        cluster = compute_cluster_metrics(pca_res.scores[:, :n_pc], cond_labels)
        metric_rows.append({
            "PCs used": n_pc,
            "Silhouette": f"{cluster.silhouette:.3f}",
            "Calinski-Harabasz": f"{cluster.calinski_harabasz:.1f}",
        })

    st.dataframe(pl.DataFrame(metric_rows).to_pandas(), width='stretch')

    # Interpretation
    sil_values = [float(r["Silhouette"]) for r in metric_rows]
    if all(s > 0.7 for s in sil_values):
        st.success("\u2705 Consistently excellent cluster separation across all PC dimensions.")
    elif all(s > 0.5 for s in sil_values):
        st.info("\u2139\ufe0f Good cluster separation across PC dimensions.")
    elif max(sil_values) > 0.5:
        st.info(f"\u2139\ufe0f Best silhouette: {max(sil_values):.3f} (with {metric_rows[sil_values.index(max(sil_values))]['PCs used']} PCs).")
    else:
        st.warning("\u26a0\ufe0f Weak cluster separation. Conditions may not be well-resolved.")

except Exception as e:
    st.error(f"Cluster metrics computation failed: {e}")
