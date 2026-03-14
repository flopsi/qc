"""Load and validate protein/peptide matrices using Polars.

Handles CSV, TSV, and TXT formats with automatic separator detection.
Supports large peptide-level files via streaming/lazy evaluation.
"""
import polars as pl
import io


def _detect_separator(content: bytes, n_lines: int = 5) -> str:
    """Auto-detect the column separator from file content."""
    sample = content[:8192].decode("utf-8", errors="replace")
    lines = sample.split("\n")[:n_lines]
    # Count candidate separators in header and first data lines
    tab_count = sum(line.count("\t") for line in lines)
    comma_count = sum(line.count(",") for line in lines)
    semicol_count = sum(line.count(";") for line in lines)
    
    counts = {"\t": tab_count, ",": comma_count, ";": semicol_count}
    best = max(counts, key=lambda x: counts[x])
    # Only return if the best has a reasonable count (at least 1 per line)
    if counts[best] >= len(lines) - 1:
        return best
    return "\t"  # default fallback


def load_protein_matrix(file_path: str | pl.DataFrame) -> pl.DataFrame:
    """Load TSV/CSV protein matrix into Polars DataFrame.
    Accepts file path or already-loaded DataFrame.
    Returns Polars DataFrame with original column names.
    """
    if isinstance(file_path, pl.DataFrame):
        return file_path

    # Read raw bytes for separator detection
    with open(file_path, "rb") as f:
        head = f.read(8192)
    sep = _detect_separator(head)

    return pl.read_csv(
        file_path,
        separator=sep,
        infer_schema_length=10000,
        null_values=["", "NA", "NaN", "null", "NULL", "Filtered", "0", "NAN"],
        try_parse_dates=False,
    )


def parse_uploaded_file(uploaded_file) -> pl.DataFrame:
    """Parse a Streamlit UploadedFile object into Polars DataFrame.
    
    Auto-detects separator (tab, comma, semicolon).
    Handles large files gracefully.
    """
    content = uploaded_file.getvalue()
    sep = _detect_separator(content)

    try:
        df = pl.read_csv(
            io.BytesIO(content),
            separator=sep,
            infer_schema_length=10000,
            null_values=["", "NA", "NaN", "null", "NULL", "Filtered"],
            try_parse_dates=False,
        )
    except Exception:
        # Retry with more lenient settings
        df = pl.read_csv(
            io.BytesIO(content),
            separator=sep,
            infer_schema_length=10000,
            null_values=["", "NA", "NaN", "null", "NULL", "Filtered"],
            try_parse_dates=False,
            ignore_errors=True,
        )

    return df


def detect_data_level(df: pl.DataFrame) -> str:
    """Detect whether the data is protein-level or peptide-level.
    
    Heuristics:
    - Column names containing 'peptide', 'sequence', 'modified', 'precursor' → peptide
    - Duplicate protein IDs → peptide (multiple peptides per protein)
    - Otherwise → protein
    """
    cols_lower = [c.lower() for c in df.columns]

    # Check column names for peptide-level indicators
    peptide_keywords = ["peptide", "sequence", "modified", "precursor",
                        "stripped", "charge", "eg.", "fg."]
    for kw in peptide_keywords:
        if any(kw in c for c in cols_lower):
            return "peptide"

    # Check first column for duplicate IDs (multiple peptides per protein)
    first_col = df.columns[0]
    n_rows = df.shape[0]
    n_unique = df.select(pl.col(first_col).n_unique()).item()
    if n_rows > 100 and n_unique < n_rows * 0.5:
        return "peptide"

    return "protein"




def build_metadata(intensity_cols: list[str], conditions: list[str]) -> pl.DataFrame:
    """Generate metadata DataFrame from the user's column mapping.

    Args:
        intensity_cols: list of column names in order per condition
        conditions: list of condition labels, e.g. ["A", "B"]

    Returns:
        DataFrame with columns: sample, condition, replicate
    """
    rows = []
    n_reps = len(intensity_cols) // len(conditions)
    for i, col in enumerate(intensity_cols):
        cond_idx = i // n_reps
        rep_idx = (i % n_reps) + 1
        rows.append({
            "sample": col,
            "condition": conditions[cond_idx],
            "replicate": rep_idx,
        })
    return pl.DataFrame(rows)


def standardize_columns(
    df: pl.DataFrame,
    id_col: str,
    gene_col: str | None,
    intensity_cols: list[str],
    conditions: list[str],
) -> tuple[pl.DataFrame, list[str]]:
    """Rename user columns to standardized names (A_1, A_2, ... B_3).

    Returns:
        Tuple of (standardized DataFrame, list of new intensity column names)
    """
    n_reps = len(intensity_cols) // len(conditions)
    new_names = []
    rename_map = {id_col: "protein_id"}
    if gene_col and gene_col != "None":
        rename_map[gene_col] = "gene_name"

    for i, col in enumerate(intensity_cols):
        cond_idx = i // n_reps
        rep_idx = (i % n_reps) + 1
        new_name = f"{conditions[cond_idx]}_{rep_idx}"
        rename_map[col] = new_name
        new_names.append(new_name)

    # Select only relevant columns and rename
    cols_to_keep = [id_col]
    if gene_col and gene_col != "None":
        cols_to_keep.append(gene_col)
    cols_to_keep.extend(intensity_cols)

    df = df.select(cols_to_keep).rename(rename_map)
    return df, new_names
