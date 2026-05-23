"""PCA reduction for Deepdraw embedding NPZ files."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

LAPACK_INT32_MAX = np.iinfo(np.int32).max
SELECTION_METHODS = {"target_variance", "elbow", "kneedle", "l_method"}


@dataclass(frozen=True)
class EmbeddingData:
    """Embeddings loaded from a Deepdraw-compatible NPZ file."""

    embeddings: np.ndarray
    ids: np.ndarray
    lengths: np.ndarray | None = None


@dataclass(frozen=True)
class PCASummary:
    """Summary of a PCA reduction."""

    n_components: int
    cumulative_explained_variance: float
    fitted_components: int
    selection_method: str
    explained_variance_ratio: np.ndarray


def load_embeddings(input_file: str | Path) -> EmbeddingData:
    """Load embeddings, IDs, and optional sequence lengths from an NPZ file."""

    input_path = Path(input_file).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input embeddings NPZ does not exist: {input_path}")

    data = np.load(input_path, allow_pickle=True)
    if "embeddings" not in data:
        raise ValueError(
            f"'embeddings' array not found in {input_path}. "
            f"Available keys: {list(data.keys())}"
        )
    ids = _extract_ids(data, input_path)
    embeddings = np.asarray(data["embeddings"])
    if embeddings.shape[0] != len(ids):
        raise ValueError(
            f"Embedding rows ({embeddings.shape[0]}) do not match ids ({len(ids)})."
        )

    lengths = _extract_embedding_lengths(data)
    if lengths is not None and len(lengths) != len(ids):
        raise ValueError(
            f"Embedding lengths ({len(lengths)}) do not match ids ({len(ids)})."
        )

    logger.info("Loaded %d embeddings from %s", len(ids), input_path)
    return EmbeddingData(embeddings=embeddings, ids=ids, lengths=lengths)


def flatten_embeddings(
    embeddings: np.ndarray,
    *,
    lengths: np.ndarray | None = None,
    use_mean_pooling: bool = False,
) -> np.ndarray:
    """Convert 2D, 3D, or ragged embeddings to a 2D PCA input matrix."""

    if embeddings.dtype == object and embeddings.ndim == 1:
        return _flatten_ragged_embeddings(
            embeddings,
            use_mean_pooling=use_mean_pooling,
        )

    if embeddings.ndim == 2:
        if use_mean_pooling:
            logger.warning("Embeddings are already 2D; no mean pooling was applied.")
        return np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 3:
        raise ValueError(
            "Embeddings must be 2D, 3D, or a 1D object array of per-sequence "
            f"2D arrays; got shape {embeddings.shape}."
        )

    if use_mean_pooling:
        pooled = _mean_pool_padded_embeddings(embeddings, lengths=lengths)
        logger.info(
            "Mean pooled embeddings from %s to %s", embeddings.shape, pooled.shape
        )
        return pooled

    n_samples, seq_len, dim = embeddings.shape
    flattened = np.asarray(embeddings, dtype=np.float32).reshape(
        n_samples,
        seq_len * dim,
    )
    logger.info("Flattened embeddings from %s to %s", embeddings.shape, flattened.shape)
    return flattened


def reduce_dimensionality(
    embeddings: np.ndarray,
    *,
    n_components: int | None = None,
    target_variance: float = 0.95,
    exact_n_components: bool = False,
    power_of_two: bool = True,
    selection_method: str = "kneedle",
) -> tuple[np.ndarray, PCASummary]:
    """Fit PCA and select the output dimensionality."""

    if embeddings.ndim != 2:
        raise ValueError(f"PCA input must be 2D; got shape {embeddings.shape}.")
    n_samples, n_features = embeddings.shape
    if n_samples < 2:
        raise ValueError("PCA requires at least two samples.")

    max_rank = min(n_samples, n_features)
    if n_components is None:
        n_components = max_rank
    if n_components <= 0:
        raise ValueError(f"n_components must be positive; got {n_components}.")
    if n_components > max_rank:
        raise ValueError(
            f"n_components ({n_components}) cannot exceed "
            f"min(n_samples, n_features) ({max_rank})."
        )
    if not 0.0 < target_variance <= 1.0:
        raise ValueError(
            f"target_variance must be in the interval (0, 1]; got {target_variance}."
        )
    if selection_method not in SELECTION_METHODS:
        raise ValueError(
            f"selection_method must be one of {sorted(SELECTION_METHODS)}; "
            f"got {selection_method!r}."
        )

    matrix_elements = int(n_samples) * int(n_features)
    if matrix_elements > LAPACK_INT32_MAX and n_components >= max_rank:
        raise ValueError(
            "Full-rank CPU PCA would exceed LAPACK integer indexing limits for "
            f"matrix shape {embeddings.shape}. Set --n-components below {max_rank}."
        )

    svd_solver = "randomized" if n_components < max_rank else "auto"
    logger.info(
        "Fitting PCA with %d components on matrix shape %s using %s solver.",
        n_components,
        embeddings.shape,
        svd_solver,
    )
    pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=42)
    transformed = pca.fit_transform(np.asarray(embeddings, dtype=np.float32))
    evr = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)

    if exact_n_components:
        chosen_k = n_components
    elif selection_method == "target_variance":
        chosen_k = _select_target_variance_k(
            evr,
            target_variance=target_variance,
            power_of_two=power_of_two,
        )
    elif selection_method == "elbow":
        chosen_k = _select_elbow_k(np.cumsum(evr))
    elif selection_method == "kneedle":
        chosen_k = _select_kneedle_k(evr)
    else:
        chosen_k = _select_l_method_k(evr)

    chosen_k = max(1, min(int(chosen_k), len(evr)))
    cumulative = float(np.cumsum(evr)[chosen_k - 1])
    logger.info(
        "Keeping %d PCs selected by %s (%.2f%% cumulative explained variance).",
        chosen_k,
        "exact_n_components" if exact_n_components else selection_method,
        cumulative * 100.0,
    )
    return transformed[:, :chosen_k], PCASummary(
        n_components=chosen_k,
        cumulative_explained_variance=cumulative,
        fitted_components=n_components,
        selection_method="exact_n_components"
        if exact_n_components
        else selection_method,
        explained_variance_ratio=evr,
    )


def reduce_embeddings_pca(
    input_file: str | Path,
    output_file: str | Path,
    *,
    n_components: int | None = None,
    target_variance: float = 0.95,
    use_mean_pooling: bool = False,
    exact_n_components: bool | None = None,
    power_of_two: bool = True,
    selection_method: str = "kneedle",
) -> PCASummary:
    """Reduce an embedding NPZ file and save a Deepdraw-compatible 2D NPZ."""

    data = load_embeddings(input_file)
    matrix = flatten_embeddings(
        data.embeddings,
        lengths=data.lengths,
        use_mean_pooling=use_mean_pooling,
    )
    if exact_n_components is None:
        exact_n_components = n_components is not None

    reduced, summary = reduce_dimensionality(
        matrix,
        n_components=n_components,
        target_variance=target_variance,
        exact_n_components=exact_n_components,
        power_of_two=power_of_two,
        selection_method=selection_method,
    )
    save_reduced_embeddings(reduced, data.ids, output_file, summary)
    return summary


def save_reduced_embeddings(
    embeddings: np.ndarray,
    ids: np.ndarray,
    output_file: str | Path,
    summary: PCASummary,
) -> None:
    """Save reduced embeddings with IDs and PCA metadata."""

    output_path = Path(output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        ids=ids,
        pca_n_components=np.asarray(summary.n_components, dtype=np.int32),
        pca_fitted_components=np.asarray(summary.fitted_components, dtype=np.int32),
        pca_cumulative_explained_variance=np.asarray(
            summary.cumulative_explained_variance,
            dtype=np.float64,
        ),
        pca_selection_method=np.asarray(summary.selection_method),
        pca_explained_variance_ratio=summary.explained_variance_ratio,
    )
    logger.info(
        "Saved %d reduced embeddings with shape %s to %s",
        len(ids),
        embeddings.shape,
        output_path,
    )


def _extract_ids(data: Any, input_path: Path) -> np.ndarray:
    if "ids" in data:
        return np.asarray(data["ids"])
    if "sample_ids" in data:
        return np.asarray(data["sample_ids"])
    raise ValueError(
        f"{input_path} must contain an 'ids' or 'sample_ids' array. "
        f"Available keys: {list(data.keys())}"
    )


def _extract_embedding_lengths(data: Any) -> np.ndarray | None:
    if "embedding_lengths" in data:
        return np.asarray(data["embedding_lengths"], dtype=np.int64)
    if "lengths" in data:
        return np.asarray(data["lengths"], dtype=np.int64)
    if "sequence_lengths_bp" not in data:
        return None

    sequence_lengths = np.asarray(data["sequence_lengths_bp"], dtype=np.int64)
    resolution = 1
    if "resolution" in data:
        resolution = int(np.asarray(data["resolution"]).item())
    if resolution <= 0:
        raise ValueError(f"resolution must be positive; got {resolution}.")
    return np.ceil(sequence_lengths / resolution).astype(np.int64)


def _flatten_ragged_embeddings(
    embeddings: np.ndarray,
    *,
    use_mean_pooling: bool,
) -> np.ndarray:
    arrays = [np.asarray(item, dtype=np.float32) for item in embeddings.tolist()]
    if not arrays:
        raise ValueError("Embeddings array cannot be empty.")
    for idx, arr in enumerate(arrays):
        if arr.ndim != 2:
            raise ValueError(
                "Ragged embeddings must contain per-sequence 2D arrays; "
                f"item {idx} has shape {arr.shape}."
            )

    if use_mean_pooling:
        return np.stack([arr.mean(axis=0) for arr in arrays]).astype(np.float32)

    first_shape = arrays[0].shape
    if any(arr.shape != first_shape for arr in arrays):
        raise ValueError(
            "Ragged embeddings have variable sequence lengths. Use "
            "--use-mean-pooling before PCA, or regenerate padded embeddings."
        )
    stacked = np.stack(arrays)
    return stacked.reshape(stacked.shape[0], stacked.shape[1] * stacked.shape[2])


def _mean_pool_padded_embeddings(
    embeddings: np.ndarray,
    *,
    lengths: np.ndarray | None,
) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if lengths is None:
        return embeddings.mean(axis=1)

    n_samples, seq_len, dim = embeddings.shape
    pooled = np.empty((n_samples, dim), dtype=np.float32)
    for idx, length in enumerate(np.asarray(lengths, dtype=np.int64)):
        if length <= 0 or length > seq_len:
            raise ValueError(
                f"Invalid embedding length {length} for row {idx}; "
                f"expected a value in [1, {seq_len}]."
            )
        pooled[idx] = embeddings[idx, : int(length), :].mean(axis=0)
    return pooled


def _select_target_variance_k(
    evr: np.ndarray,
    *,
    target_variance: float,
    power_of_two: bool,
) -> int:
    cumulative = np.cumsum(evr)
    if not power_of_two:
        return int(np.searchsorted(cumulative, target_variance, side="left") + 1)

    candidates: list[int] = []
    value = 1
    while value <= len(evr):
        candidates.append(value)
        value *= 2
    for candidate in candidates:
        if cumulative[candidate - 1] >= target_variance:
            return candidate
    return candidates[-1]


def _select_elbow_k(cumulative_variance: np.ndarray) -> int:
    n_points = len(cumulative_variance)
    if n_points <= 2:
        return n_points

    x = np.arange(1, n_points + 1, dtype=np.float64)
    y = cumulative_variance.astype(np.float64)
    start = np.array([x[0], y[0]])
    end = np.array([x[-1], y[-1]])
    line_vec = end - start
    line_norm = float(np.linalg.norm(line_vec))
    if line_norm == 0.0:
        return 1

    points = np.column_stack((x, y))
    vec_from_start = points - start
    distances = (
        np.abs(line_vec[0] * vec_from_start[:, 1] - line_vec[1] * vec_from_start[:, 0])
        / line_norm
    )
    distances[0] = -np.inf
    distances[-1] = -np.inf
    return int(np.argmax(distances) + 1)


def _select_kneedle_k(evr: np.ndarray) -> int:
    n_points = len(evr)
    if n_points <= 2:
        return n_points

    x = np.linspace(0.0, 1.0, n_points, dtype=np.float64)
    y = evr.astype(np.float64)
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if y_max <= y_min:
        return 1

    y_norm = (y - y_min) / (y_max - y_min)
    y_increasing = 1.0 - y_norm
    distances = y_increasing - x
    distances[0] = -np.inf
    distances[-1] = -np.inf
    return int(np.argmax(distances) + 1)


def _select_l_method_k(evr: np.ndarray) -> int:
    n_points = len(evr)
    if n_points <= 2:
        return n_points

    x = np.arange(1, n_points + 1, dtype=np.float64)
    y = np.log(np.clip(evr.astype(np.float64), 1e-12, None))
    best_k = 2
    best_score = np.inf
    for k in range(2, n_points):
        left_score = _linear_sse(x[:k], y[:k])
        right_score = _linear_sse(x[k - 1 :], y[k - 1 :])
        score = (left_score + right_score) / n_points
        if score < best_score:
            best_score = score
            best_k = k
    return int(best_k)


def _linear_sse(x: np.ndarray, y: np.ndarray) -> float:
    if x.size <= 1:
        return 0.0
    slope, intercept = np.polyfit(x, y, deg=1)
    residuals = y - (slope * x + intercept)
    return float(np.sum(residuals * residuals))
