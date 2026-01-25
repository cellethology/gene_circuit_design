"""
BoTorch model wrappers for sklearn-style integration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from sklearn.base import RegressorMixin

_DEFAULT_OUTCOME_TRANSFORM = object()


def _resolve_rrp_model_class():
    try:
        from botorch.models import RobustRelevancePursuitSingleTaskGP

        return RobustRelevancePursuitSingleTaskGP
    except Exception:
        try:
            from botorch.models.robust_relevance_pursuit_model import (
                RobustRelevancePursuitSingleTaskGP,
            )

            return RobustRelevancePursuitSingleTaskGP
        except Exception as nested_exc:  # pragma: no cover - import guard
            raise ImportError(
                "RobustRelevancePursuitSingleTaskGP is not available in this "
                "BoTorch install. Upgrade botorch to a version that provides "
                "robust_relevance_pursuit_model."
            ) from nested_exc


class _BoTorchBaseRegressor(RegressorMixin):
    """
    Shared utilities for BoTorch sklearn-style wrappers.
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "float64",
        ard: bool = False,
        kernel: str = "rbf",
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.ard = ard
        self.kernel = kernel
        self.model_: Any | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "device": self.device,
            "dtype": self.dtype,
            "ard": self.ard,
            "kernel": self.kernel,
        }

    def set_params(self, **params: Any):
        for key, value in params.items():
            setattr(self, key, value)
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

    def get_botorch_model(self) -> Any:
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

    def _build_kernel(self, X_tensor: torch.Tensor) -> ScaleKernel:
        ard_dims = X_tensor.shape[-1] if self.ard else None
        kernel_key = str(self.kernel).lower()
        if kernel_key in {"rbf", "se", "squared_exponential"}:
            base_kernel = RBFKernel(ard_num_dims=ard_dims)
        elif kernel_key in {"matern_12", "matern12", "matern_1/2", "matern-1/2"}:
            base_kernel = MaternKernel(nu=0.5, ard_num_dims=ard_dims)
        elif kernel_key in {"matern_32", "matern32", "matern_3/2", "matern-3/2"}:
            base_kernel = MaternKernel(nu=1.5, ard_num_dims=ard_dims)
        elif kernel_key in {"matern_52", "matern52", "matern_5/2", "matern-5/2"}:
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=ard_dims)
        else:
            raise ValueError(
                "Unsupported kernel "
                f"'{self.kernel}'. Use rbf, matern_12, matern_32, or matern_52."
            )
        return ScaleKernel(base_kernel)

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        device = self._resolve_device()
        dtype = self._resolve_dtype()
        return torch.as_tensor(array, device=device, dtype=dtype)

    def _check_is_fitted(self) -> None:
        if self.model_ is None:
            raise ValueError(
                f"{self.__class__.__name__}.fit must be called before predict."
            )


class BoTorchGPRegressor(_BoTorchBaseRegressor):
    """
    Sklearn-compatible wrapper around a BoTorch SingleTaskGP.
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "float64",
        ard: bool = False,
        kernel: str = "rbf",
        noise_constraint: float | None = None,
        noise_prior: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype, ard=ard, kernel=kernel)
        self.noise_constraint = noise_constraint
        self.noise_prior = noise_prior

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = super().get_params(deep=deep)
        params.update(
            {
                "noise_constraint": self.noise_constraint,
                "noise_prior": self.noise_prior,
            }
        )
        return params

    def fit(self, X: np.ndarray, y: np.ndarray) -> BoTorchGPRegressor:
        X_tensor = self._to_tensor(X)
        y_tensor = self._to_tensor(y).view(-1, 1)

        covar_module = self._build_kernel(X_tensor)
        likelihood = self._build_likelihood()

        self.model_ = SingleTaskGP(
            X_tensor,
            y_tensor,
            covar_module=covar_module.to(device=X_tensor.device, dtype=X_tensor.dtype),
            likelihood=likelihood.to(device=X_tensor.device, dtype=X_tensor.dtype),
        )
        self.model_.train()
        mll = ExactMarginalLogLikelihood(self.model_.likelihood, self.model_)
        fit_gpytorch_mll(mll)
        self.model_.eval()
        return self

    def _build_likelihood(self) -> GaussianLikelihood:
        constraint = (
            GreaterThan(self.noise_constraint)
            if self.noise_constraint is not None
            else None
        )
        likelihood = (
            GaussianLikelihood(noise_constraint=constraint)
            if constraint is not None
            else GaussianLikelihood()
        )
        if self.noise_prior is not None:
            concentration, rate = self.noise_prior
            likelihood.noise_covar.register_prior(
                "noise_prior",
                GammaPrior(concentration, rate),
                "noise",
            )
        return likelihood


class BoTorchRobustRelevancePursuitGPRegressor(_BoTorchBaseRegressor):
    """
    Sklearn-compatible wrapper around RobustRelevancePursuitSingleTaskGP.
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "float64",
        ard: bool = False,
        kernel: str = "rbf",
        convex_parameterization: bool = True,
        prior_mean_of_support: float | None = None,
        cache_model_trace: bool = False,
        outcome_transform: Any = _DEFAULT_OUTCOME_TRANSFORM,
        input_transform: Any | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype, ard=ard, kernel=kernel)
        self.convex_parameterization = convex_parameterization
        self.prior_mean_of_support = prior_mean_of_support
        self.cache_model_trace = cache_model_trace
        self.outcome_transform = outcome_transform
        self.input_transform = input_transform
        self.model_kwargs = model_kwargs or {}

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = super().get_params(deep=deep)
        params.update(
            {
                "convex_parameterization": self.convex_parameterization,
                "prior_mean_of_support": self.prior_mean_of_support,
                "cache_model_trace": self.cache_model_trace,
                "outcome_transform": self.outcome_transform,
                "input_transform": self.input_transform,
                "model_kwargs": self.model_kwargs,
            }
        )
        return params

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> BoTorchRobustRelevancePursuitGPRegressor:
        X_tensor = self._to_tensor(X)
        y_tensor = self._to_tensor(y).view(-1, 1)

        covar_module = self._build_kernel(X_tensor)
        model_cls = _resolve_rrp_model_class()

        model_kwargs: dict[str, Any] = dict(self.model_kwargs)
        model_kwargs.update(
            {
                "covar_module": covar_module.to(
                    device=X_tensor.device, dtype=X_tensor.dtype
                ),
                "convex_parameterization": self.convex_parameterization,
                "prior_mean_of_support": self.prior_mean_of_support,
                "cache_model_trace": self.cache_model_trace,
            }
        )
        if self.input_transform is not None:
            model_kwargs["input_transform"] = self.input_transform
        if self.outcome_transform is not _DEFAULT_OUTCOME_TRANSFORM:
            model_kwargs["outcome_transform"] = self.outcome_transform
        self.model_ = model_cls(X_tensor, y_tensor, **model_kwargs)
        self.model_.train()
        mll = ExactMarginalLogLikelihood(self.model_.likelihood, self.model_)
        fit_gpytorch_mll(mll)
        self.model_.eval()
        return self
