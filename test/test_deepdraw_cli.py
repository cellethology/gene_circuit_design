import numpy as np
import pytest

from deepdraw import cli


def test_cli_validation_error_omits_traceback(monkeypatch, capsys):
    def fail_suggest(**kwargs):
        raise ValueError("Measurements are missing labels.")

    monkeypatch.setattr(cli, "suggest_next_batch", fail_suggest)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "suggest",
                "--run-dir",
                "deepdraw_run",
                "--measurements",
                "measurements.csv",
                "--label-column",
                "Expression",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert captured.out == ""
    assert captured.err == "Error: Measurements are missing labels.\n"
    assert "Traceback" not in captured.err


def test_cli_debug_log_level_reraises_validation_error(monkeypatch):
    def fail_suggest(**kwargs):
        raise ValueError("debug me")

    monkeypatch.setattr(cli, "suggest_next_batch", fail_suggest)

    with pytest.raises(ValueError, match="debug me"):
        cli.main(
            [
                "suggest",
                "--log-level",
                "DEBUG",
                "--run-dir",
                "deepdraw_run",
                "--measurements",
                "measurements.csv",
                "--label-column",
                "Expression",
            ]
        )


def test_cli_pca_command_writes_output(tmp_path, capsys):
    rng = np.random.default_rng(44)
    input_path = tmp_path / "embeddings.npz"
    output_path = tmp_path / "embeddings_pca.npz"
    np.savez_compressed(
        input_path,
        embeddings=rng.normal(size=(8, 4)).astype(np.float32),
        ids=np.asarray([f"v{idx}" for idx in range(8)], dtype=object),
    )

    cli.main(
        [
            "pca",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--selection-method",
            "kneedle",
        ]
    )

    captured = capsys.readouterr()
    assert "Wrote PCA embeddings" in captured.out
    assert output_path.exists()


def test_cli_embed_alphagenome_delegates_pool_csv(monkeypatch, tmp_path, capsys):
    calls = {}

    def fake_embed_design_pool(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(cli, "embed_design_pool", fake_embed_design_pool)
    pool_path = tmp_path / "designs.csv"
    output_path = tmp_path / "embeddings.npz"

    cli.main(
        [
            "embed-alphagenome",
            "--pool-csv",
            str(pool_path),
            "--sequence-column",
            "sequence",
            "--id-column",
            "variant_id",
            "--output",
            str(output_path),
            "--resolution",
            "1",
            "--device",
            "cpu",
        ]
    )

    captured = capsys.readouterr()
    assert "Wrote AlphaGenome embeddings" in captured.out
    assert calls["pool_csv"] == pool_path
    assert calls["output_path"] == output_path
    assert calls["sequence_column"] == "sequence"
    assert calls["id_column"] == "variant_id"
    assert calls["resolution"] == 1
    assert calls["device"] == "cpu"
