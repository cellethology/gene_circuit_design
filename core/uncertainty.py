import numpy as np
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline


class UncertaintyWrapper(RegressorMixin):
    """
    A universal wrapper that adds compute_std() to any sklearn regressor.
    """

    def __init__(self, estimator: RegressorMixin):
        self.estimator = estimator

    def compute_std(self, X):
        base_estimator, feature_pipeline, target_transformer = self._unwrap_estimator()
        X_transformed = self._transform_features(feature_pipeline, X)

        # ----- RandomForest and bagging-style ensembles -----
        if isinstance(base_estimator, RandomForestRegressor):
            return self._std_from_member_estimators(
                base_estimator.estimators_, X_transformed, target_transformer
            )

        # ----- GradientBoostingRegressor: use stage trees -----
        if isinstance(base_estimator, GradientBoostingRegressor):
            stage_preds = [
                stage[0].predict(X_transformed) for stage in base_estimator.estimators_
            ]
            all_preds = np.stack(stage_preds, axis=0)
            all_preds = self._inverse_transform_targets(all_preds, target_transformer)
            return all_preds.std(axis=0)

        # ----- GaussianProcessRegressor: uses built-in std -----
        if isinstance(base_estimator, GaussianProcessRegressor):
            means, std = base_estimator.predict(X_transformed, return_std=True)
            if target_transformer is None:
                return std
            approx_bounds = np.stack([means - std, means + std], axis=0)
            approx_bounds = self._inverse_transform_targets(
                approx_bounds,
                target_transformer,
            )
            return 0.5 * np.abs(approx_bounds[1] - approx_bounds[0])

        # ----- Bagging-style ensembles -----
        if hasattr(base_estimator, "estimators_"):
            return self._std_from_member_estimators(
                base_estimator.estimators_, X_transformed, target_transformer
            )

        raise NotImplementedError(
            "Uncertainty computation not implemented for estimator type "
            f"{base_estimator.__class__.__name__}"
        )

    def _unwrap_estimator(self):
        est = self.estimator
        target_transformer = None

        if isinstance(est, TransformedTargetRegressor):
            target_transformer = est.transformer_
            est = est.regressor_

        feature_transformer = None
        if isinstance(est, Pipeline):
            if len(est.steps) == 0:
                raise ValueError("Pipeline has no steps to unwrap.")
            if len(est.steps) > 1:
                feature_transformer = Pipeline(est.steps[:-1])
            est = est.steps[-1][1]

        return est, feature_transformer, target_transformer

    def _transform_features(self, transformer, X):
        if transformer is None:
            return X
        return transformer.transform(X)

    def _inverse_transform_targets(self, preds, transformer):
        if transformer is None:
            return preds
        preds = np.asarray(preds)
        original_shape = preds.shape
        flattened = preds.reshape(-1, 1)
        inversed = transformer.inverse_transform(flattened)
        return np.asarray(inversed).reshape(original_shape)

    def _std_from_member_estimators(self, estimators, X, target_transformer):
        flat_estimators = np.asarray(estimators, dtype=object).ravel()
        member_preds = []
        for member in flat_estimators:
            if member is None or not hasattr(member, "predict"):
                continue
            member_preds.append(member.predict(X))

        if not member_preds:
            raise ValueError(
                "No valid base estimators available to compute uncertainty."
            )

        stacked_preds = np.stack(member_preds, axis=0)
        stacked_preds = self._inverse_transform_targets(
            stacked_preds, target_transformer
        )
        return stacked_preds.std(axis=0)
