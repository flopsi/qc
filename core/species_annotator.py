"""Assign species labels based on protein ID prefixes or FASTA headers."""
import polars as pl

# Default UniProt prefix mapping for three-proteome benchmark
SPECIES_MAP = {
    "_HUMAN": "human",
    "_YEAST": "yeast",
    "_ECOLI": "ecoli",
    "HUMAN": "human",
    "YEAST": "yeast",
    "ECOLI": "ecoli",
    # Common alternative identifiers
    "HOMO SAPIENS": "human",
    "SACCHAROMYCES": "yeast",
    "ESCHERICHIA": "ecoli",
    "OS=HOMO": "human",
    "OS=SACCHAROMYCES": "yeast",
    "OS=ESCHERICHIA": "ecoli",
    # UniProt organism IDs
    "_9606": "human",
    "_4932": "yeast",
    "_83333": "ecoli",
    "OX=9606": "human",
    "OX=4932": "yeast",
    "OX=83333": "ecoli",
}


def annotate_species(
    df: pl.DataFrame,
    id_column: str = "protein_id",
    species_map: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Add 'species' column based on protein ID / gene name string matching.

    Robust to:
    - Null/missing values in the detection column (falls back row-by-row)
    - Non-string column types (casts to string)
    - Protein groups with semicolons (checks whole string)
    - Data without gene_name column (falls back to protein_id)
    - Data without any recognisable species patterns (labels as 'unknown')

    When both gene_name and protein_id exist, the function coalesces them:
    it uses gene_name where available and falls back to protein_id for
    rows where gene_name is null/empty.
    """
    if species_map is None:
        species_map = SPECIES_MAP

    # Build a single detection string per row by coalescing gene_name → protein_id
    has_gene = "gene_name" in df.columns
    has_id = id_column in df.columns

    if has_gene and has_id:
        # Coalesce: prefer gene_name, fall back to protein_id when null/empty
        safe_col = (
            pl.when(
                pl.col("gene_name").is_not_null()
                & (pl.col("gene_name").cast(pl.Utf8, strict=False).str.len_chars() > 0)
            )
            .then(pl.col("gene_name").cast(pl.Utf8, strict=False))
            .otherwise(pl.col(id_column).cast(pl.Utf8, strict=False).fill_null(""))
        ).str.to_uppercase()
    elif has_gene:
        safe_col = (
            pl.col("gene_name")
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .str.to_uppercase()
        )
    elif has_id:
        safe_col = (
            pl.col(id_column)
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .str.to_uppercase()
        )
    else:
        return df.with_columns(pl.lit("unknown").alias("species"))

    # Build when/then chain — process longer patterns first to avoid partial matches
    species_expr = pl.lit("unknown")
    sorted_patterns = sorted(species_map.keys(), key=len, reverse=True)
    for pattern in sorted_patterns:
        species = species_map[pattern]
        species_expr = (
            pl.when(safe_col.str.contains(pattern.upper(), literal=True))
            .then(pl.lit(species))
            .otherwise(species_expr)
        )

    return df.with_columns(species_expr.alias("species"))
