"""
Unit tests for the UncertaintyWrapper helper.
"""

import numpy as np
import pytest
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from core.uncertainty import UncertaintyWrapper


def _make_linear_data(n_samples: int = 8):
    X = np.linspace(0.0, 1.0, n_samples).reshape(-1, 1)
    y = np.sin(2 * np.pi * X).ravel()
    return X, y


class _DummyMember:
    def __init__(self, value: float):
        self.value = value

    def predict(self, X):
        return np.full(X.shape[0], self.value, dtype=float)


class _DummyBagging(RegressorMixin):
    def __init__(self, member_values=None):
        if member_values is None:
            member_values = ()
        self.member_values = member_values
        self.estimators_ = [_DummyMember(v) for v in self.member_values]

    def get_params(self, deep=True):
        return {"member_values": self.member_values}

    def set_params(self, **params):
        if "member_values" in params:
            self.member_values = params["member_values"]
            self.estimators_ = [_DummyMember(v) for v in self.member_values]
        return self

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.estimators_], axis=0)


def test_unwrap_pipeline_with_target_transformer():
    X, y = _make_linear_data()
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=2, random_state=0)),
        ]
    )
    ttr = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
    ttr.fit(X, y)

    wrapper = UncertaintyWrapper(ttr)
    base_estimator, feature_pipe, target_tx = wrapper._unwrap_estimator()

    assert isinstance(base_estimator, RandomForestRegressor)
    assert isinstance(feature_pipe, Pipeline)
    assert isinstance(target_tx, StandardScaler)


def test_compute_std_random_forest_returns_variance():
    X, y = _make_linear_data()
    rf = RandomForestRegressor(n_estimators=4, random_state=0)
    rf.fit(X, y)

    wrapper = UncertaintyWrapper(rf)
    std = wrapper.compute_std(X)

    assert std.shape == (len(X),)
    assert np.all(std >= 0.0)
    assert np.any(std > 0.0)


def test_compute_std_gradient_boosting_uses_stage_predictions():
    X, y = _make_linear_data()
    gbr = GradientBoostingRegressor(n_estimators=5, random_state=0)
    gbr.fit(X, y)

    wrapper = UncertaintyWrapper(gbr)
    std = wrapper.compute_std(X)

    assert std.shape == (len(X),)
    assert np.all(std >= 0.0)


def test_compute_std_gaussian_process_with_transformer():
    X, y = _make_linear_data()
    gpr = GaussianProcessRegressor(kernel=RBF(length_scale=0.5), alpha=1e-2).fit(X, y)
    transformer = FunctionTransformer(
        func=lambda arr: arr - 1.0,
        inverse_func=lambda arr: arr + 1.0,
        validate=False,
        check_inverse=False,
    )
    ttr = TransformedTargetRegressor(regressor=gpr, transformer=transformer)
    ttr.fit(X, y)

    wrapper = UncertaintyWrapper(ttr)
    std = wrapper.compute_std(X)

    assert std.shape == (len(X),)
    assert np.all(std >= 0.0)


def test_generic_bagging_std_uses_target_inverse_transform():
    X, y = _make_linear_data()
    bagging = _DummyBagging(member_values=[0.0, 1.0, 2.0])
    transformer = FunctionTransformer(
        func=lambda arr: arr - 1.0,
        inverse_func=lambda arr: arr + 1.0,
        validate=False,
        check_inverse=False,
    )
    ttr = TransformedTargetRegressor(regressor=bagging, transformer=transformer)
    ttr.fit(X, y)

    wrapper = UncertaintyWrapper(ttr)
    std = wrapper.compute_std(X)

    stacked = np.vstack([np.full(len(X), val + 1.0) for val in [0.0, 1.0, 2.0]])
    expected = np.std(stacked, axis=0)
    assert np.allclose(std, expected)


def test_std_from_member_estimators_raises_without_valid_members():
    X, y = _make_linear_data()
    bagging = _DummyBagging(member_values=[])
    bagging.estimators_ = [None]
    wrapper = UncertaintyWrapper(bagging)

    with pytest.raises(ValueError, match="No valid base estimators"):
        wrapper.compute_std(X)


def test_unwrap_pipeline_without_steps_raises():
    empty_pipeline = Pipeline([])
    wrapper = UncertaintyWrapper(empty_pipeline)

    with pytest.raises(ValueError, match="Pipeline has no steps"):
        wrapper._unwrap_estimator()
