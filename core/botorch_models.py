"""
BoTorch model wrappers for sklearn-style integration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.base import RegressorMixin


class BoTorchGPRegressor(RegressorMixin):
    """
    Sklearn-compatible wrapper around a BoTorch SingleTaskGP.
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "float64",
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.model_: SingleTaskGP | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "device": self.device,
            "dtype": self.dtype,
        }

    def set_params(self, **params: Any) -> BoTorchGPRegressor:
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> BoTorchGPRegressor:
        X_tensor = self._to_tensor(X)
        y_tensor = self._to_tensor(y).view(-1, 1)

        self.model_ = SingleTaskGP(
            X_tensor,
            y_tensor,
        )
        self.model_.train()
        mll = ExactMarginalLogLikelihood(self.model_.likelihood, self.model_)
        fit_gpytorch_mll(mll)
        self.model_.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_with_std(X)
        return mean

    def predict_with_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()
        X_tensor = self._to_tensor(X)
        with torch.no_grad():
            posterior = self.model_.posterior(X_tensor)
            mean = posterior.mean.squeeze(-1).cpu().numpy()
            std = posterior.variance.clamp_min(0.0).sqrt().squeeze(-1).cpu().numpy()
        return mean, std

    def posterior(self, X: np.ndarray) -> Any:
        self._check_is_fitted()
        X_tensor = self._to_tensor(X)
        return self.model_.posterior(X_tensor)

    def get_botorch_model(self) -> SingleTaskGP:
        self._check_is_fitted()
        return self.model_

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _resolve_dtype(self) -> torch.dtype:
        if isinstance(self.dtype, torch.dtype):
            return self.dtype
        text = str(self.dtype).lower()
        if text in {"float64", "double", "torch.float64"}:
            return torch.double
        if text in {"float32", "float", "torch.float32"}:
            return torch.float
        raise ValueError(f"Unsupported dtype '{self.dtype}'.")

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        device = self._resolve_device()
        dtype = self._resolve_dtype()
        return torch.as_tensor(array, device=device, dtype=dtype)

    def _check_is_fitted(self) -> None:
        if self.model_ is None:
            raise ValueError("BoTorchGPRegressor.fit must be called before predict.")
