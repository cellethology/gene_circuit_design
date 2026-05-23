from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import deepdraw.embeddings.alphagenome as alphagenome
from deepdraw.embeddings.alphagenome import (
    embed_design_pool,
    normalize_dna_sequence,
    read_design_pool_sequences,
    read_fasta_sequences,
)


def test_read_design_pool_sequences_uses_stable_ids(tmp_path):
    pool_path = tmp_path / "designs.csv"
    pd.DataFrame(
        {
            "variant_id": ["v1", "v2"],
            "sequence": ["atgc", "NNNN"],
        }
    ).to_csv(pool_path, index=False)

    records = read_design_pool_sequences(
        pool_path,
        sequence_column="sequence",
        id_column="variant_id",
    )

    assert [record.sample_id for record in records] == ["v1", "v2"]
    assert [record.sequence for record in records] == ["ATGC", "NNNN"]


def test_read_fasta_sequences_parses_multiline_records(tmp_path):
    fasta_path = tmp_path / "designs.fasta"
    fasta_path.write_text(">v1 first design\nATG\nC\n>v2\nNNNN\n")

    records = read_fasta_sequences(fasta_path)

    assert [record.sample_id for record in records] == ["v1", "v2"]
    assert [record.sequence for record in records] == ["ATGC", "NNNN"]


def test_normalize_dna_sequence_rejects_invalid_characters():
    with pytest.raises(ValueError, match="Invalid DNA characters"):
        normalize_dna_sequence("ATGX", sample_id="bad")


def test_embed_design_pool_writes_npz_with_fake_runtime(monkeypatch, tmp_path):
    pool_path = tmp_path / "designs.csv"
    output_path = tmp_path / "embeddings.npz"
    pd.DataFrame(
        {
            "variant_id": ["v1", "v2"],
            "sequence": ["ATGCATGC", "ATGCNNNN"],
        }
    ).to_csv(pool_path, index=False)

    class FakeEncoder:
        def encode(self, sequence):
            return np.ones((len(sequence), 4), dtype=np.float32)

    class FakeForward:
        def apply(self, params, state, rng, onehot, organism_index, metadata):
            batch_size, padded_len, _ = onehot.shape
            embeddings = SimpleNamespace(
                embeddings_1bp=np.ones((batch_size, padded_len, 3), dtype=np.float32),
                embeddings_128bp=np.ones(
                    (batch_size, padded_len // 128, 3),
                    dtype=np.float32,
                ),
            )
            return embeddings, {}

    fake_model = SimpleNamespace(
        _one_hot_encoder=FakeEncoder(),
        _params={},
        _state={},
        _metadata={},
    )
    monkeypatch.setattr(
        alphagenome,
        "_load_alphagenome_runtime",
        lambda: SimpleNamespace(jnp=np),
    )
    monkeypatch.setattr(alphagenome, "_auto_device", lambda runtime, device: None)
    monkeypatch.setattr(
        alphagenome,
        "_create_model_from_huggingface",
        lambda runtime, model_version, device: fake_model,
    )
    monkeypatch.setattr(
        alphagenome,
        "_build_forward",
        lambda runtime, **kwargs: FakeForward(),
    )

    embed_design_pool(
        pool_csv=pool_path,
        output_path=output_path,
        sequence_column="sequence",
        id_column="variant_id",
        resolution=128,
        pooling="mean",
    )

    data = np.load(output_path, allow_pickle=True)
    assert data["ids"].tolist() == ["v1", "v2"]
    assert data["embeddings"].shape == (2, 3)
    assert data["pooling"].item() == "mean"
    assert data["resolution"].item() == 128
