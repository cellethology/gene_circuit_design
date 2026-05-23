import numpy as np
import pytest

from deepdraw.embeddings.pca import flatten_embeddings, reduce_embeddings_pca


def test_reduce_embeddings_pca_kneedle_writes_deepdraw_npz(tmp_path):
    rng = np.random.default_rng(123)
    latent = rng.normal(size=(16, 3))
    weights = rng.normal(size=(3, 6))
    embeddings = (latent @ weights + 0.01 * rng.normal(size=(16, 6))).astype(np.float32)
    ids = np.asarray([f"variant_{idx}" for idx in range(16)], dtype=object)
    input_path = tmp_path / "raw_embeddings.npz"
    output_path = tmp_path / "pca_embeddings.npz"
    np.savez_compressed(input_path, embeddings=embeddings, ids=ids)

    summary = reduce_embeddings_pca(
        input_path,
        output_path,
        selection_method="kneedle",
    )

    reduced = np.load(output_path, allow_pickle=True)
    assert reduced["embeddings"].shape == (16, summary.n_components)
    assert reduced["embeddings"].dtype == np.float32
    assert reduced["ids"].tolist() == ids.tolist()
    assert reduced["pca_selection_method"].item() == "kneedle"
    assert 1 <= summary.n_components <= min(embeddings.shape)


def test_flatten_embeddings_mean_pools_with_lengths():
    embeddings = np.asarray(
        [
            [[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]],
            [[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]],
        ],
        dtype=np.float32,
    )
    lengths = np.asarray([2, 3])

    pooled = flatten_embeddings(
        embeddings,
        lengths=lengths,
        use_mean_pooling=True,
    )

    np.testing.assert_allclose(pooled, [[2.0, 4.0], [4.0, 6.0]])


def test_ragged_embeddings_require_mean_pooling_when_lengths_vary():
    embeddings = np.asarray(
        [
            np.ones((2, 3), dtype=np.float32),
            np.ones((4, 3), dtype=np.float32),
        ],
        dtype=object,
    )

    with pytest.raises(ValueError, match="variable sequence lengths"):
        flatten_embeddings(embeddings, use_mean_pooling=False)
