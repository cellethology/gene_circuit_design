"""
Query strategy implementations.

This module provides a clean, extensible way to implement different
selection strategies for active learning experiments.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class QueryStrategyBase(ABC):
    """
    Abstract base class for query strategies.

    Each concrete strategy implements the `_select_batch` method to choose
    the next batch of samples based on its specific criteria.
    """

    requires_model: bool = True

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    def select(self, experiment: Any) -> List[int]:
        """
        Select the next batch of samples (template method).

        This method handles bounds checking and delegates to _select_batch()
        for the actual selection logic.

        Returns:
            List of selected indices from unlabeled_indices
        """
        unlabeled_pool = experiment.unlabeled_indices
        batch_size = experiment.batch_size

        # Early return if pool is smaller than batch size
        if len(unlabeled_pool) < batch_size:
            return unlabeled_pool

        # Delegate to strategy-specific selection logic
        selected_indices = self._select_batch(experiment, unlabeled_pool, batch_size)
        self._log_round(selected_indices)
        return selected_indices

    @abstractmethod
    def _select_batch(
        self, experiment: Any, unlabeled_pool: List[int], batch_size: int
    ) -> List[int]:
        """
        Strategy-specific batch selection logic.

        Args:
            experiment: The active learning experiment
            unlabeled_pool: List of unlabeled indices (guaranteed to have length >= batch_size)
            batch_size: Number of samples to select (guaranteed to be <= len(unlabeled_pool))

        Returns:
            List of selected indices from unlabeled_pool
        """
        pass

    def _log_round(
        self,
        selected_indices: List[int],
        extra_info: Optional[str] = None,
    ) -> None:
        """
        Log information about the round.

        Args:
            selected_indices: Indices that were selected
            extra_info: Extra information to include in the log message
        """
        log_msg = f"Selected indices: {selected_indices}"
        if extra_info:
            log_msg += f" {extra_info}"
        logger.info(log_msg)


class Random(QueryStrategyBase):
    """Strategy that selects samples randomly."""

    def __init__(self, seed: int) -> None:
        super().__init__("RANDOM")
        self.seed = seed
        self.requires_model = False

    def _select_batch(
        self, experiment: Any, unlabeled_pool: List[int], batch_size: int
    ) -> List[int]:
        rng = np.random.default_rng(self.seed)
        selected_indices = rng.choice(
            unlabeled_pool, batch_size, replace=False
        ).tolist()
        return selected_indices


class TopPredictions(QueryStrategyBase):
    """Selects samples with highest k predicted label values."""

    def __init__(self) -> None:
        super().__init__("TOP_K_PRED")

    def _select_batch(
        self, experiment: Any, unlabeled_pool: List[int], batch_size: int
    ) -> List[int]:
        preds = experiment.trainer.predict(
            experiment.dataset.embeddings[unlabeled_pool, :]
        )

        # get indices of top k predictions (descending)
        top_k_local = np.argpartition(-preds, batch_size - 1)[:batch_size]

        # map to original indices
        selected_indices = [unlabeled_pool[i] for i in top_k_local]
        return selected_indices


class PredStdHybrid(QueryStrategyBase):
    """Selects samples with highest combined prediction and uncertainty."""

    def __init__(self, alpha: float) -> None:
        super().__init__("PRED_STD_HYBRID")
        self.alpha = alpha

    def _select_batch(
        self, experiment: Any, unlabeled_pool: List[int], batch_size: int
    ) -> List[int]:
        # get weighted sum of prediction and standard deviation of prediction
        preds, stds = experiment.trainer.predict(
            experiment.dataset.embeddings[unlabeled_pool, :],
            return_std=True,
        )
        weights = np.array([self.alpha, 1 - self.alpha])
        weighted_preds = preds * weights[0] + stds * weights[1]
        top_k_local = np.argpartition(-weighted_preds, batch_size - 1)[:batch_size]
        selected_indices = [unlabeled_pool[i] for i in top_k_local]
        return selected_indices


class BoTorchAcquisition(QueryStrategyBase):
    """Select samples by maximizing a BoTorch acquisition over the unlabeled pool."""

    def __init__(
        self,
        acquisition: str = "ei",
        beta: float = 2.0,
        maximize: bool = True,
    ) -> None:
        super().__init__(f"BOTORCH_{acquisition.upper()}")
        self.acquisition = acquisition.lower()
        self.beta = beta
        self.maximize = maximize

    def _select_batch(
        self, experiment: Any, unlabeled_pool: List[int], batch_size: int
    ) -> List[int]:
        torch, acq_class = self._resolve_acquisition()
        model, feature_transformer, target_transformer = self._unwrap_estimator(
            experiment.trainer.get_model()
        )

        botorch_model = self._as_botorch_model(model)

        X_pool = experiment.dataset.embeddings[unlabeled_pool, :]
        if feature_transformer is not None:
            X_pool = feature_transformer.transform(X_pool)
        X_pool = np.asarray(X_pool)

        train_labels = experiment.dataset.labels[experiment.train_indices]
        train_labels = self._transform_targets(train_labels, target_transformer)
        best_f = train_labels.max() if self.maximize else train_labels.min()

        acq = acq_class(botorch_model, best_f=float(best_f))
        X_tensor = self._to_model_tensor(torch, botorch_model, X_pool)
        with torch.no_grad():
            scores = acq(X_tensor).detach().cpu().numpy().reshape(-1)

        top_k_local = np.argpartition(-scores, batch_size - 1)[:batch_size]
        selected_indices = [unlabeled_pool[i] for i in top_k_local]
        return selected_indices

    def _resolve_acquisition(self):
        import torch
        from botorch.acquisition.analytic import (
            LogExpectedImprovement,
            ProbabilityOfImprovement,
            UpperConfidenceBound,
        )

        if self.acquisition == "log_ei":
            return torch, lambda model, best_f: LogExpectedImprovement(
                model=model, best_f=best_f, maximize=self.maximize
            )
        if self.acquisition == "pi":
            return torch, lambda model, best_f: ProbabilityOfImprovement(
                model=model, best_f=best_f, maximize=self.maximize
            )
        if self.acquisition == "ucb":
            return torch, lambda model, best_f: UpperConfidenceBound(
                model=model, beta=self.beta, maximize=self.maximize
            )
        raise ValueError(
            f"Unsupported acquisition '{self.acquisition}'. Use 'ei', 'pi', or 'ucb'."
        )

    def _unwrap_estimator(self, estimator: Any):
        if estimator is None:
            raise ValueError("PredictorTrainer.train must be called before select.")

        target_transformer = None
        if isinstance(estimator, TransformedTargetRegressor):
            target_transformer = estimator.transformer_
            estimator = estimator.regressor_

        feature_transformer = None
        if isinstance(estimator, Pipeline):
            if len(estimator.steps) == 0:
                raise ValueError("Pipeline has no steps to unwrap.")
            if len(estimator.steps) > 1:
                feature_transformer = Pipeline(estimator.steps[:-1])
            estimator = estimator.steps[-1][1]

        return estimator, feature_transformer, target_transformer

    def _as_botorch_model(self, estimator: Any):
        if hasattr(estimator, "get_botorch_model"):
            return estimator.get_botorch_model()
        if hasattr(estimator, "posterior"):
            return estimator
        raise ValueError(
            "Predictor does not expose a BoTorch model. "
            "Use BoTorchGPRegressor or a compatible BoTorch model."
        )

    def _to_model_tensor(self, torch, model: Any, X: np.ndarray):
        train_input = None
        if hasattr(model, "train_inputs") and model.train_inputs:
            train_input = model.train_inputs[0]
        device = train_input.device if train_input is not None else torch.device("cpu")
        dtype = train_input.dtype if train_input is not None else torch.double
        X_tensor = torch.as_tensor(X, device=device, dtype=dtype)
        if X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(-2)
        return X_tensor

    def _transform_targets(self, targets: np.ndarray, transformer: Any) -> np.ndarray:
        if transformer is None:
            return np.asarray(targets)
        values = np.asarray(targets).reshape(-1, 1)
        transformed = transformer.transform(values)
        return np.asarray(transformed).reshape(-1)
