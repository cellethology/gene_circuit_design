"""
Query context utilities for selection strategies.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from experiments.core.experiment import ActiveLearningExperiment


@dataclass
class QueryContext:
    """
    Snapshot of the experiment state passed to selection strategies.
    """

    round_idx: int
    batch_size: int
    pool_indices: List[int]
    train_indices: List[int]
    model: Any
    sequence_labels: np.ndarray
    log_likelihoods: np.ndarray
    sequences: List[str]
    selected_variants: List[Dict[str, Any]]
    round_history: List[Dict[str, Any]]
    random_seed: int
    _encode_fn: Callable[[List[int]], np.ndarray]

    def encode(self, indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Encode sequences for the provided indices. Defaults to the full pool.
        """
        target_indices = indices if indices is not None else self.pool_indices
        return self._encode_fn(target_indices)


def build_query_context(
    experiment: ActiveLearningExperiment, round_idx: int
) -> QueryContext:
    """
    Construct a QueryContext from the current experiment state.
    """

    pool_indices = list(experiment.data_split.unlabeled_indices)

    return QueryContext(
        round_idx=round_idx,
        batch_size=experiment.batch_size,
        pool_indices=pool_indices,
        train_indices=list(experiment.data_split.train_indices),
        model=experiment.model,
        sequence_labels=experiment.dataset.sequence_labels,
        log_likelihoods=experiment.dataset.log_likelihoods,
        sequences=experiment.dataset.sequences,
        selected_variants=experiment.variant_tracker.get_all_variants(),
        round_history=copy.deepcopy(experiment.results),
        random_seed=experiment.random_seed,
        _encode_fn=lambda indices: experiment._encode_sequences(indices),
    )
