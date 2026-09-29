"""Tests for pinned dataset publishing/download selection (no network)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from moderntsf.hub import datasets as hd
from moderntsf.tsf_core.paths import repository_root

ROOT = repository_root()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_selection_covers_file_directory_and_ultratraffic_presets() -> None:
    assert hd.selection("etth1").matches("ETT-small/ETTh1.csv")
    assert not hd.selection("etth1").matches("ETT-small/ETTh2.csv")
    pems = hd.selection("pems08")
    assert pems.matches("pems08/his.npz") and not pems.matches("pems08x/his.npz")
    ba = hd.selection("ultratraffic_ba_st")
    assert ba.matches("ultratraffic/manifest.json")
    assert ba.matches("ultratraffic/PEMS_BA/static/2023.parquet")
    assert ba.matches("ultratraffic/PEMS_BA/static/2023.json")
    assert ba.matches("ultratraffic/PEMS_BA/static_sensor_changes_log.txt")
    assert not ba.matches("ultratraffic/PEMS_BA/static/2022.parquet")
    assert not ba.matches("ultratraffic/PEMS_BA/cl/2023_added.parquet")
    assert not ba.matches("ultratraffic/PEMS_LA/static/2023.parquet")
    assert hd.selection("ultratraffic_sb_cl").matches("ultratraffic/PEMS_SB/cl/2023_added.parquet")
    with pytest.raises(ValueError):
        hd.selection("synthetic_st")


def test_fetch_preset_downloads_only_pinned_files_and_verifies(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "configs" / "datasets", root / "configs" / "datasets")
    payload = b"date,OT\n2020-01-01,1\n"
    manifest = {"schema_version": 1, "repo": "o/r", "files": {
        "ETT-small/ETTh1.csv": {"revision": "abc", "sha256": hashlib.sha256(payload).hexdigest(),
                                "size": len(payload)},
        "ETT-small/ETTh2.csv": {"revision": "abc", "sha256": "0" * 64, "size": 1},
    }}
    _write(root / hd.MANIFEST_RELATIVE, json.dumps(manifest).encode())
    calls = []

    def fake_download(uri: str, destination: Path, sha256: str | None = None) -> Path:
        calls.append(uri)
        _write(destination, payload)
        return destination

    monkeypatch.setattr(hd, "download", fake_download)
    written = hd.fetch_preset("etth1", tmp_path / "dataset", root)
    assert calls == ["hf://datasets/o/r@abc/ETT-small/ETTh1.csv"]
    assert written[0].read_bytes() == payload
    assert set(hd.available_presets(root)) == {"etth1", "etth2"}
    with pytest.raises(FileNotFoundError):
        hd.fetch_preset("weather", tmp_path / "dataset", root)


def test_local_files_respects_selection(tmp_path) -> None:
    data = tmp_path / "dataset"
    for name in ("ultratraffic/manifest.json", "ultratraffic/PEMS_BA/static/2023.parquet",
                 "ultratraffic/PEMS_BA/static/2019.parquet", "ultratraffic/PEMS_LA/static/2023.parquet"):
        _write(data / name, b"x")
    assert hd.local_files("ultratraffic_ba_st", data) == [
        "ultratraffic/PEMS_BA/static/2023.parquet", "ultratraffic/manifest.json"]
    whole = hd.Selection(preset="ultratraffic", base="ultratraffic")
    assert len(hd.local_files("ultratraffic", data, chosen=whole)) == 4


def test_packaged_manifest_is_well_formed() -> None:
    manifest = hd.load_manifest()
    assert manifest["repo"] == hd.DEFAULT_STATIC_REPO
    for name, entry in manifest["files"].items():
        assert not name.startswith("/") and set(entry) == {"revision", "sha256", "size"}
