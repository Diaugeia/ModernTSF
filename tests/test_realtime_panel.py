"""Real-time tracks served as static datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsflab.data.datasets.realtime_panel import (
    Dataset_RealtimePanel_ST,
    Dataset_RealtimePanel_TS,
    load_panel,
)
from tsflab.realtime.store import PanelStore

TRACK = "traffic_pems_sb"  # a shipped track config (hourly), the store itself is synthetic


def _store(tmp_path: Path, hours: int = 24 * 40) -> Path:
    index = pd.date_range("2024-01-01", periods=hours, freq="h")
    t = np.arange(hours)[:, None]
    panel = pd.DataFrame(50 + 5 * np.sin(2 * np.pi * t / 24) + np.arange(3)[None, :], index=index,
                         columns=["a", "b", "c"])
    panel.iloc[5:8, 1] = np.nan  # a gap the loader must fill causally
    panel.iloc[10:20, 2] = np.nan
    PanelStore(TRACK, tmp_path).append(panel.iloc[: hours // 2])
    store = PanelStore(TRACK, tmp_path)
    store.append(panel.iloc[hours // 2:])
    manifest = store.manifest()
    manifest["releases"][0]["version"] = "2024.01.01-0000"  # same-minute releases would collide
    store._write_manifest(manifest)
    return tmp_path / TRACK


def _kw(path: Path, **extra):
    return dict(root_path=str(path), data_path="", size=(24, 0, 6), track=TRACK, **extra)


def test_time_series_layout_split_and_scaling(tmp_path: Path) -> None:
    path = _store(tmp_path)
    train, val, test = (Dataset_RealtimePanel_TS(flag=f, **_kw(path)) for f in ("train", "val", "test"))
    x, y, xm, ym = train[0]
    assert x.shape == (24, 3) and y.shape == (6, 3) and xm.shape == (24, 6) and ym.shape == (6, 6)
    total = 24 * 40
    assert len(train) == int(0.7 * total) - 24 - 6 + 1
    val_end = int((0.7 + 0.1) * total)  # the repository's border arithmetic
    assert len(val) == (val_end - int(0.7 * total)) - 6 + 1
    assert len(test) == (total - val_end) - 6 + 1
    # Scaling uses the training rows only.
    raw = load_panel(str(path), TRACK).to_numpy(np.float32)[: int(0.7 * total)]
    assert train.value_mean == pytest.approx(float(raw.mean()), rel=1e-5)
    assert np.allclose(train.inverse_transform(train.values[:5]), load_panel(str(path), TRACK).to_numpy()[:5], atol=1e-3)
    # The first validation target is the first validation row.
    first = int(0.7 * total)
    _, target, _, _ = val[0]
    assert np.allclose(target, val.values[first:first + 6])
    assert not np.isnan(train.values).any()


def test_gaps_are_filled_causally(tmp_path: Path) -> None:
    panel = load_panel(str(_store(tmp_path)), TRACK)
    assert (panel["b"].iloc[5:8] == panel["b"].iloc[4]).all()
    assert (panel["c"].iloc[10:20] == panel["c"].iloc[9]).all()


def test_spatiotemporal_layout_matches_the_export_bundle_covariates(tmp_path: Path) -> None:
    path = _store(tmp_path)
    st = Dataset_RealtimePanel_ST(flag="train", **_kw(path))
    value, future, cov, cov_future = st[3]
    assert value.shape == (24, 3) and future.shape == (6, 3)
    assert cov.shape == (24, 3, 2) and cov_future.shape == (6, 3, 2)
    h0 = 3 + 24 - 24  # window centre 23 + 3, history starts at row 3
    assert cov[0, 0, 0] == pytest.approx(((h0 * 60) % (24 * 60)) / (24 * 60))
    assert Dataset_RealtimePanel_ST(flag="train", calendar=False, **_kw(path))[0][2].shape == (24, 3, 0)


def test_release_version_freezes_the_data(tmp_path: Path) -> None:
    path = _store(tmp_path)
    store = PanelStore(TRACK, tmp_path)
    first = store.manifest()["releases"][0]
    frozen = load_panel(str(path), TRACK, version=first["version"])
    assert frozen.index.max() == pd.Timestamp(first["last_timestamp"])
    assert len(frozen) == 24 * 20
    assert len(load_panel(str(path), TRACK)) == 24 * 40
    with pytest.raises(ValueError, match="no release"):
        load_panel(str(path), TRACK, version="nope")
    with pytest.raises(FileNotFoundError):
        load_panel(str(tmp_path / "empty" / TRACK), TRACK)


def test_hub_revision_pulls_into_a_cache(tmp_path: Path, monkeypatch) -> None:
    source = _store(tmp_path / "src")
    calls = []

    def fake_pull(store, repo_id, revision):
        calls.append((repo_id, revision))
        store.directory.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copytree(source, store.directory)
        return True

    monkeypatch.setattr("tsflab.realtime.publish.pull_track", fake_pull)
    target = tmp_path / "data" / TRACK
    panel = load_panel(str(target), TRACK, revision="abcdef1234567890")
    assert len(panel) == 24 * 40 and calls[0][1] == "abcdef1234567890"
    load_panel(str(target), TRACK, revision="abcdef1234567890")
    assert len(calls) == 1  # cached


def test_local_traffic_store_if_present() -> None:
    root = Path("dataset/realtime")
    if not (root / "traffic_pems_sb" / "manifest.json").is_file():
        pytest.skip("no local traffic_pems_sb store")
    ds = Dataset_RealtimePanel_ST(root_path=str(root / "traffic_pems_sb"), data_path="", size=(168, 0, 24),
                                  flag="test", track="traffic_pems_sb", start="2023-11-01", max_windows=4)
    value, future, cov, _ = ds[0]
    assert value.shape[0] == 168 and future.shape[0] == 24 and cov.shape[-1] == 2
    assert np.isfinite(value).all()
