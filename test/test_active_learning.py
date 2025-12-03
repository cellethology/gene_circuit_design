"""
Tests for experiments.active_learning.run_single_experiment.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from experiments.active_learning import run_single_experiment
from experiments.core.initial_selection_strategies import RandomInitialSelection
from experiments.core.query_strategies import Random as RandomStrategy


def _make_dataset(tmp_path, n_samples=8, dim=4):
    embeddings = np.random.randn(n_samples, dim).astype(np.float32)
    ids = np.arange(n_samples, dtype=np.int32)
    emb_path = tmp_path / "embeddings.npz"
    np.savez_compressed(emb_path, embeddings=embeddings, ids=ids)

    df = pd.DataFrame({"Expression": np.linspace(0, 1, n_samples)})
    csv_path = tmp_path / "metadata.csv"
    df.to_csv(csv_path, index=False)
    return str(emb_path), str(csv_path)


def _build_cfg(tmp_path):
    emb_path, csv_path = _make_dataset(tmp_path)
    al_settings = {
        "batch_size": 2,
        "starting_batch_size": 2,
        "max_rounds": 1,
        "normalize_features": False,
        "normalize_labels": False,
        "output_dir": str(tmp_path / "outputs"),
        "label_key": "Expression",
    }
    return SimpleNamespace(
        predictor=LinearRegression(),
        query_strategy=RandomStrategy(seed=0),
        initial_selection_strategy=RandomInitialSelection(
            seed=0, starting_batch_size=2
        ),
        seed=0,
        embedding_path=emb_path,
        metadata_path=csv_path,
        al_settings=al_settings,
    )


def test_run_single_experiment_creates_summary(tmp_path, monkeypatch):
    cfg = _build_cfg(tmp_path)
    monkeypatch.setattr("experiments.active_learning.instantiate", lambda obj: obj)
    summary = run_single_experiment(cfg)

    out_dir = Path(cfg.al_settings["output_dir"])
    summary_path = out_dir / "summary.json"
    results_path = out_dir / "results.csv"

    assert results_path.exists()
    assert summary_path.exists()
    saved_summary = json.loads(summary_path.read_text())
    assert saved_summary["query_strategy"] == summary["query_strategy"]
    assert "auc_normalized_true" in saved_summary


def test_run_single_experiment_requires_output_dir(tmp_path, monkeypatch):
    cfg = _build_cfg(tmp_path)
    cfg.al_settings.pop("output_dir")
    monkeypatch.setattr("experiments.active_learning.instantiate", lambda obj: obj)
    with pytest.raises(ValueError):
        run_single_experiment(cfg)
