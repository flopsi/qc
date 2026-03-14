"""Polars schema definitions for the proteomics QC app.
All modules must validate data against these schemas."""
import polars as pl

# Expected columns after parsing and standardization
PROTEIN_MATRIX_SCHEMA = {
    "protein_id": pl.Utf8,
    "gene_name": pl.Utf8,
    "species": pl.Utf8,  # "human", "yeast", "ecoli"
}

METADATA_SCHEMA = {
    "sample": pl.Utf8,       # Column name in protein matrix
    "condition": pl.Utf8,    # e.g., "A" or "B"
    "replicate": pl.Int32,   # 1, 2, 3
}

# These are defaults — actual columns are determined dynamically from uploaded data
DEFAULT_INTENSITY_COLUMNS = ["A_1", "A_2", "A_3", "B_1", "B_2", "B_3"]
DEFAULT_CONDITIONS = ["A", "B"]
DEFAULT_REPLICATES = [1, 2, 3]
