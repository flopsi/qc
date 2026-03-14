"""PERMANOVA and dispersion tests using scikit-bio."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix, permanova, permdisp


@dataclass
class PermanovaResult:
    test_statistic: float
    p_value: float
    r_squared: float
    n_permutations: int
    sample_size: int
    n_groups: int
    min_achievable_p: float
    is_min_p: bool


@dataclass
class DispersionResult:
    test_statistic: float
    p_value: float
    n_permutations: int
    sample_size: int
    n_groups: int
    test: str


def _compute_min_p(permutations: int) -> float:
    # FIXED: in scikit-bio, the reported p-value floor is set by permutation count.
    if permutations <= 0:
        return float("nan")
    return 1.0 / (1.0 + permutations)


def _to_distance_matrix(intensity_matrix: np.ndarray, metric: str) -> DistanceMatrix:
    dist_condensed = pdist(intensity_matrix, metric=metric)
    dist_square = squareform(dist_condensed)
    n = intensity_matrix.shape[0]
    return DistanceMatrix(dist_square, ids=[str(i) for i in range(n)])


def run_permanova(
    intensity_matrix: np.ndarray,
    grouping: list[str],
    metric: str = "euclidean",
    permutations: int = 999,
    seed: int | None = 42,
) -> PermanovaResult:
    n = len(grouping)
    if intensity_matrix.shape[0] != n:
        raise ValueError("intensity_matrix rows must match len(grouping)")

    dm = _to_distance_matrix(intensity_matrix, metric)

    # FIXED: pass seed through so permutations are reproducible.
    result = permanova(dm, grouping, permutations=permutations, seed=seed)

    f_stat = float(result["test statistic"])
    p_val_raw = result["p-value"]
    p_val = float(p_val_raw) if not np.isnan(p_val_raw) else float("nan")
    k = len(set(grouping))

    if f_stat <= 0:
        r_squared = 0.0
    else:
        r_squared = 1.0 / (1.0 + ((n - k) / ((k - 1) * f_stat)))

    min_p = _compute_min_p(permutations)
    is_min_p = bool(
        not np.isnan(p_val)
        and not np.isnan(min_p)
        and np.isclose(p_val, min_p, rtol=0.0, atol=1e-12)
    )

    return PermanovaResult(
        test_statistic=f_stat,
        p_value=p_val,
        r_squared=r_squared,
        n_permutations=permutations,
        sample_size=n,
        n_groups=k,
        min_achievable_p=min_p,
        is_min_p=is_min_p,
    )


def run_permdisp(
    intensity_matrix: np.ndarray,
    grouping: list[str],
    metric: str = "euclidean",
    test: str = "median",
    permutations: int = 999,
    seed: int | None = 42,
) -> DispersionResult:
    n = len(grouping)
    if intensity_matrix.shape[0] != n:
        raise ValueError("intensity_matrix rows must match len(grouping)")

    dm = _to_distance_matrix(intensity_matrix, metric)
    result = permdisp(dm, grouping, test=test, permutations=permutations, seed=seed)

    return DispersionResult(
        test_statistic=float(result["test statistic"]),
        p_value=float(result["p-value"]) if not np.isnan(result["p-value"]) else float("nan"),
        n_permutations=permutations,
        sample_size=n,
        n_groups=len(set(grouping)),
        test=test,
    )


def interpret_permanova(
    result: PermanovaResult,
    context: str = "global",
    dispersion: DispersionResult | None = None,
    alpha: float = 0.05,
) -> tuple[str, str]:
    p = result.p_value
    r2 = result.r_squared
    parts: list[str] = []

    if np.isnan(p):
        severity = "info"
        parts.append(
            f"PERMANOVA computed pseudo-F={result.test_statistic:.3f} and R²={r2:.3f}, "
            f"but the p-value is unavailable because permutations=0."
        )
    elif p <= alpha:
        severity = "success"
        parts.append(
            f"PERMANOVA indicates a group effect (pseudo-F={result.test_statistic:.3f}, "
            f"p={p:.3f}, R²={r2:.3f})."
        )
        if result.is_min_p:
            parts.append(
                f"The reported p-value is at the resolution limit for "
                f"{result.n_permutations} permutations; increase permutations for finer precision."
            )
    else:
        severity = "warning"
        parts.append(
            f"PERMANOVA does not reach alpha={alpha:.2f} (pseudo-F={result.test_statistic:.3f}, "
            f"p={p:.3f}, R²={r2:.3f})."
        )
        if result.is_min_p:
            parts.append(
                f"This p-value is at the resolution limit for {result.n_permutations} permutations, "
                f"so more permutations are needed before making a stronger claim."
            )

    if context == "species_stable":
        if not np.isnan(p) and p <= alpha:
            severity = "warning"
            parts.append(
                "For a context expected to be stable, this suggests an unexpected group effect."
            )
        elif not np.isnan(p):
            severity = "success"
            parts.append(
                "For a context expected to be stable, this is consistent with no detectable group effect."
            )
    elif context == "species_variable":
        if not np.isnan(p) and p <= alpha:
            severity = "success"
            parts.append(
                "For a context expected to vary, this is consistent with a detectable group effect."
            )
        elif not np.isnan(p):
            severity = "warning"
            parts.append(
                "For a context expected to vary, this does not provide evidence for the expected group effect."
            )
    elif context == "intensity_bin":
        if not np.isnan(p) and p <= alpha:
            parts.append(
                "Within this intensity bin, the grouping factor is detectable at the selected alpha."
            )
        elif not np.isnan(p):
            parts.append(
                "Within this intensity bin, the grouping factor is not detected at the selected alpha."
            )

    if dispersion is not None:
        if np.isnan(dispersion.p_value):
            parts.append(
                f"PERMDISP computed {dispersion.test} dispersion differences, but its p-value is unavailable because permutations=0."
            )
        elif dispersion.p_value <= alpha:
            parts.append(
                f"PERMDISP is also significant (p={dispersion.p_value:.3f}, test={dispersion.test}), "
                "so group dispersion differs and the PERMANOVA result should be interpreted cautiously."
            )
        else:
            parts.append(
                f"PERMDISP is not significant (p={dispersion.p_value:.3f}, test={dispersion.test}), "
                "which is consistent with similar within-group dispersion across groups."
            )

    return severity, " ".join(parts)
