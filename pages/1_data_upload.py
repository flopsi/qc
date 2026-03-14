"""Page 1: Upload protein/peptide matrix, configure metadata, preview data."""
import streamlit as st
import polars as pl
from core.data_loader import (
    parse_uploaded_file, build_metadata, standardize_columns,
    detect_data_level, 
)
from core.species_annotator import annotate_species

st.header("\U0001f4c1 Data Upload & Configuration")

with st.expander("\u2139\ufe0f About this app", expanded=False):
    st.markdown("""
### Proteomics QC Dashboard

This app performs **accuracy and precision qualification** of LC-MS proteomics data,
designed for a three-proteome benchmark (Human/Yeast/E. coli, "HYE") with 2 conditions.

**Supported input:**
- **Protein-level** data (e.g. Spectronaut PG.Quantity, MaxQuant proteinGroups, DIA-NN pg matrix)
- **Peptide-level** data (e.g. Spectronaut EG/FG.Quantity, MaxQuant evidence, DIA-NN report)
  — will be aggregated to protein level before analysis

**Flexible design:** any number of replicates per condition (minimum 3 recommended).
Supports CSV, TSV, and TXT formats with automatic separator detection.

| Page | Purpose |
|------|---------|
| **Upload & Preview** | Load data, map columns, annotate species |
| **Global PCA** | Overall sample clustering and PERMANOVA significance test |
| **Intensity Bins** | PCA per intensity quartile to assess dynamic range performance |
| **Species PCA** | Separate PCA per species to validate expected separation patterns |
| **log2 vs glog** | Compare variance-stabilizing transformations via Mean-SD plots |
| **Variance Components** | PVCA to decompose variance into condition/replicate/residual |
| **ICC & CV Metrics** | Replicate reproducibility (ICC) and per-protein precision (CV) |
    """)

# File uploader
uploaded = st.file_uploader(
    "Upload protein or peptide matrix (TSV, CSV, or TXT)",
    type=["tsv", "csv", "txt"],
    help="Rows = proteins (or peptides); columns include ID column(s) and intensity columns.",
)

if uploaded is not None:
    # --- Parse file ---
    try:
        with st.spinner("Reading file..."):
            df = parse_uploaded_file(uploaded)
    except Exception as e:
        st.error(f"Failed to parse file: {e}")
        st.stop()

    st.success(f"Loaded **{df.shape[0]:,}** rows \u00d7 **{df.shape[1]}** columns")

    # --- Auto-detect data level ---
    data_level = detect_data_level(df)
    if data_level == "peptide":
        st.info(
            "\U0001f50d **Peptide-level data detected.** "
            "Intensities will be aggregated to protein level before analysis."
        )

    # --- Column mapping ---
    with st.expander("\u2699\ufe0f Configure Columns", expanded=True):
        all_cols = df.columns

        # === ID column ===
        # Try to guess protein ID column
        id_guess = 0
        for i, c in enumerate(all_cols):
            cl = c.lower()
            if any(kw in cl for kw in ["protein", "pg.protein", "majority"]):
                id_guess = i
                break

        id_col = st.selectbox("Protein / Protein-group ID column", all_cols, index=id_guess)

        # === Gene name column ===
        gene_guess = 0  # index into ["None"] + all_cols
        for i, c in enumerate(all_cols):
            cl = c.lower()
            if any(kw in cl for kw in ["gene", "proteinname", "pg.proteinnames",
                                         "fasta.header", "protein.name"]):
                gene_guess = i + 1  # +1 because of "None" entry at index 0
                break
        if gene_guess == 0 and len(all_cols) > 1:
            gene_guess = 2 if len(all_cols) > 2 else 1  # reasonable fallback

        gene_col = st.selectbox(
            "Gene name / Protein names column (for species detection) — select 'None' if absent",
            ["None"] + all_cols,
            index=min(gene_guess, len(all_cols)),
        )

        # === Peptide-level extra column (if applicable) ===
        peptide_id_col = None
        if data_level == "peptide":
            pep_candidates = [c for c in all_cols
                              if any(kw in c.lower()
                                     for kw in ["peptide", "sequence", "modified",
                                                 "stripped", "eg.", "fg.", "precursor"])]
            if pep_candidates:
                peptide_id_col = st.selectbox(
                    "Peptide / Precursor ID column (informational)",
                    pep_candidates,
                    index=0,
                )

        # === Intensity columns ===
        exclude = {id_col}
        if gene_col != "None":
            exclude.add(gene_col)
        if peptide_id_col:
            exclude.add(peptide_id_col)

        # Filter to numeric-looking columns only
        candidate_cols = []
        for c in all_cols:
            if c in exclude:
                continue
            dtype = df[c].dtype
            if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt32, pl.UInt64):
                candidate_cols.append(c)

        if not candidate_cols:
            st.error("No numeric intensity columns found. Please check your file format.")
            st.stop()

        st.markdown("**Select intensity columns in order: all replicates for Condition A first, then Condition B.**")
        intensity_cols = st.multiselect(
            "Intensity columns (select all replicates for both conditions, in order)",
            candidate_cols,
            default=candidate_cols[:min(6, len(candidate_cols))],
        )

        # === Condition configuration ===
        st.markdown("---")
        st.markdown("**Condition setup:**")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            cond_a = st.text_input("Condition A label", value="A")
        with col_c2:
            cond_b = st.text_input("Condition B label", value="B")
        with col_c3:
            n_conditions = 2
            if len(intensity_cols) > 0:
                n_reps_total = len(intensity_cols)
                n_reps = n_reps_total // n_conditions
                st.text_input("Replicates per condition", value=str(n_reps), disabled=True)
            else:
                n_reps = 0

        conditions = [cond_a, cond_b]

    # === Validate column selection ===
    if len(intensity_cols) < 4:
        st.warning(
            f"Please select at least 4 intensity columns (minimum 2 replicates \u00d7 2 conditions). "
            f"Currently selected: {len(intensity_cols)}."
        )
        st.stop()

    if len(intensity_cols) % 2 != 0:
        st.warning(
            f"Number of intensity columns ({len(intensity_cols)}) must be even "
            f"(equal replicates per condition)."
        )
        st.stop()

    n_reps = len(intensity_cols) // 2
    st.caption(f"Design: **{n_reps}** replicates \u00d7 **2** conditions = **{len(intensity_cols)}** samples")



    # === Standardize columns ===
    try:
        df_std, std_intensity_cols = standardize_columns(
            df, id_col, gene_col if gene_col != "None" else None,
            intensity_cols, conditions,
        )
    except Exception as e:
        st.error(f"Column standardization failed: {e}")
        st.stop()

    # === Species annotation ===
    try:
        df_std = annotate_species(df_std, id_column="protein_id")
    except Exception as e:
        st.warning(f"Species annotation failed ({e}). All proteins labeled 'unknown'.")
        df_std = df_std.with_columns(pl.lit("unknown").alias("species"))

    # === Species summary ===
    species_counts = df_std.group_by("species").len().sort("len", descending=True)
    st.subheader("Species Summary")
    n_species = min(len(species_counts), 5)
    if n_species > 0:
        cols = st.columns(n_species)
        for i, row in enumerate(species_counts.head(n_species).iter_rows(named=True)):
            with cols[i]:
                pct = row["len"] / df_std.shape[0] * 100
                st.metric(
                    row["species"].capitalize(),
                    f"{row['len']:,}",
                    delta=f"{pct:.1f}%",
                    delta_color="off",
                )

    # === Build metadata ===
    metadata = build_metadata(std_intensity_cols, conditions)

    # === Store in session state ===
    st.session_state["protein_df"] = df_std
    st.session_state["metadata"] = metadata
    st.session_state["intensity_cols"] = std_intensity_cols
    st.session_state["conditions"] = conditions
    st.session_state["data_level"] = data_level
    st.session_state["n_reps"] = n_reps

    # === Preview ===
    st.subheader("Data Preview")
    st.dataframe(df_std.head(50).to_pandas(), width='stretch)

    # === Data quality summary ===
    with st.expander("\U0001f4ca Data Quality Summary"):
        n_total = df_std.shape[0] * len(std_intensity_cols)
        n_missing = sum(
            df_std.select(pl.col(c).is_null().sum()).item()
            for c in std_intensity_cols
        )
        n_nan = 0
        for c in std_intensity_cols:
            try:
                n_nan += df_std.select(pl.col(c).is_nan().sum()).item()
            except Exception:
                pass  # non-float columns won't have is_nan
        total_missing = n_missing + n_nan
        n_zero = sum(
            df_std.select((pl.col(c) == 0).sum()).item()
            for c in std_intensity_cols
            if df_std[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        )

        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        col_q1.metric("Total proteins", f"{df_std.shape[0]:,}")
        col_q2.metric("Completeness", f"{(1 - total_missing / n_total) * 100:.1f}%")
        col_q3.metric("Missing/NaN", f"{total_missing:,}")
        col_q4.metric("Zero values", f"{n_zero:,}")

        if data_level == "peptide":
            st.info(
                f"\U0001f4cb Data was aggregated from peptide level. "
                f"Original file had {uploaded.size / 1e6:.1f} MB."
            )

        st.write("**Sample Metadata:**")
        st.dataframe(metadata.to_pandas(), width="stretch")
else:
    st.info("\U0001f449 Upload a TSV or CSV protein or peptide matrix to get started.")
