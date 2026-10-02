"""Traffic converter: PeMS 12/12 defaults and train-only statistics."""

import numpy as np

from tsflab.cli.commands.convert_traffic import split_windows, train_range_stats


def test_stats_ignore_validation_and_test_rows():
    t = 200
    values = np.zeros((t, 3, 1), np.float32)
    values[150:] = 1000.0  # lies beyond every training window
    train, _, _ = split_windows(t, 12, 12, (0.7, 0.1, 0.2))
    mean, std, train_end = train_range_stats(values, train, 12)
    assert train_end <= 150
    assert float(mean[0]) == 0.0 and float(std[0]) == 0.0


def test_default_horizon_is_pems_12(monkeypatch, tmp_path):
    import sys
    from tsflab.cli.commands import convert_traffic

    np.save(tmp_path / "v.npy", np.random.rand(100, 4).astype(np.float32))
    monkeypatch.setattr(sys, "argv", ["x", "--values", str(tmp_path / "v.npy"),
                                      "--output-dir", str(tmp_path / "out")])
    convert_traffic.main()
    bundle = np.load(tmp_path / "out" / "his.npz")
    assert int(bundle["seq_len"]) == 12 and int(bundle["pred_len"]) == 12
