"""tsflab.release: hf:// addressing, weights bundles, and pinned dataset publishing (no network)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from pydantic import BaseModel

from tsflab.catalog.model_artifacts import fetch_artifact
from tsflab.catalog.registry.models import ModelArtifact, ModelSpec
from tsflab.core.paths import repository_root
from tsflab.release import hub
from tsflab.release.hub import datasets as hd

# ---------------------------------------------------------------------------
# hf:// URIs, artifacts, and weights bundles
# ---------------------------------------------------------------------------


REPO = "Diaugeia/TSFLab-Weights"


REV = "0123abcd"


def test_parse_round_trips_every_repo_type() -> None:
    for text, repo_type in (
        (f"hf://{REPO}@{REV}/weather/DLinear/run/model.safetensors", "model"),
        ("hf://datasets/Diaugeia/TSFLab-Static@main/ett/ETTh1.csv", "dataset"),
        ("hf://spaces/Diaugeia/TSFLab@v1/index.html", "space"),
    ):
        uri = hub.parse(text)
        assert uri.repo_type == repo_type
        assert str(uri) == text
    uri = hub.parse("hf://datasets/Diaugeia/TSFLab-Static@main/ett/ETTh1.csv")
    assert uri.resolve_url() == (
        "https://huggingface.co/datasets/Diaugeia/TSFLab-Static/resolve/main/ett/ETTh1.csv"
    )


@pytest.mark.parametrize(
    "text",
    [
        f"hf://{REPO}/model.safetensors",  # unpinned
        f"hf://{REPO}@{REV}/../secret",  # traversal
        f"hf://{REPO}@{REV}/a//b",  # empty segment
        f"https://huggingface.co/{REPO}",  # wrong scheme
    ],
)
def test_parse_rejects_unpinned_or_unsafe_uris(text: str) -> None:
    with pytest.raises(ValueError):
        hub.parse(text)


def _serve(root: Path, path: str, payload: bytes) -> None:
    """Lay out ``payload`` as a file:// endpoint would serve an hf:// URI."""
    target = root / REPO / "resolve" / REV / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


class _Params(BaseModel):
    width: int = 1


def test_model_artifacts_accept_pinned_hf_uris(tmp_path: Path, monkeypatch) -> None:
    payload = b"foundation weights"
    _serve(tmp_path / "hub", "fm/weights.bin", payload)
    monkeypatch.setenv("HF_ENDPOINT", (tmp_path / "hub").as_uri())
    artifact = ModelArtifact(
        name="weights",
        url=f"hf://{REPO}@{REV}/fm/weights.bin",
        revision=REV,
        sha256=hashlib.sha256(payload).hexdigest(),
        filename="weights.bin",
        required=True,
    )
    spec = ModelSpec(
        name="Fixture",
        module="tsflab.models.fixture",
        model_class=nn.Identity,
        factory=lambda cfg, params: nn.Identity(),
        params_schema=_Params,
        capabilities=frozenset({"time-series"}),
        artifacts=(artifact,),
    )
    assert fetch_artifact(spec, artifact, tmp_path / "cache").read_bytes() == payload
    with pytest.raises(ValueError, match="pinned to its revision"):
        ModelArtifact(
            name="weights",
            url=f"hf://{REPO}@other/fm/weights.bin",
            revision=REV,
            sha256=artifact.sha256,
            filename="weights.bin",
        )


def _fake_run(work: Path) -> str:
    run_id = "DLinear_weather_sl96_pl12_seed42_1"
    model_dir = work / "work_dirs" / "weather" / "DLinear"
    (model_dir / "records").mkdir(parents=True)
    (model_dir / "checkpoints" / run_id).mkdir(parents=True)
    torch.manual_seed(0)
    torch.save(nn.Linear(96, 12).state_dict(), model_dir / "checkpoints" / run_id / "best_checkpoint.pth")
    record = {
        "model": "DLinear",
        "dataset_id": "weather",
        "track": "time_series",
        "seed": 42,
        "config": {"seq_len": 96, "label_len": 0, "pred_len": 12, "features": "M"},
        "env": {"framework_version": "0.8.0", "git_sha": "abc"},
        "results": [{"horizon": 12, "run_id": run_id, "metrics": {"mse": 0.5, "mae": None}}],
    }
    (model_dir / "records" / f"{run_id}.json").write_text(json.dumps(record))
    return run_id


def test_pack_then_load_state_dict_round_trips(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("safetensors", reason="needs the `hub` extra")
    run_id = _fake_run(tmp_path)
    record, checkpoint = hub.find_run(run_id, tmp_path)
    bundle = tmp_path / "bundle"
    manifest = hub.pack(record, checkpoint, bundle)
    assert manifest["path"] == f"weather/DLinear/{run_id}"
    assert manifest["metrics"] == {"mse": 0.5}
    assert (bundle / "README.md").read_text().startswith("---\nlibrary_name: tsflab")

    served = tmp_path / "hub"
    for file in bundle.iterdir():
        _serve(served, f"{manifest['path']}/{file.name}", file.read_bytes())
    monkeypatch.setenv("HF_ENDPOINT", served.as_uri())
    uri = f"hf://{REPO}@{REV}/{manifest['path']}"
    state_dict, loaded = hub.load_state_dict(uri, cache_root=tmp_path / "cache")
    expected = torch.load(checkpoint, weights_only=True)
    assert loaded == manifest
    assert all(torch.equal(state_dict[key], expected[key]) for key in expected)

    weights = served / REPO / "resolve" / REV / manifest["path"] / "model.safetensors"
    weights.write_bytes(weights.read_bytes()[:-1] + b"\0")
    with pytest.raises(ValueError, match="checksum mismatch"):
        hub.load_state_dict(uri, cache_root=tmp_path / "fresh-cache")


def test_hub_owner_comes_from_the_environment(monkeypatch) -> None:
    import importlib

    from tsflab.release.hub import uri

    monkeypatch.setenv("TSFLAB_HUB_OWNER", "someone")
    try:
        assert importlib.reload(uri).default_repo("TSFLab-Static") == "someone/TSFLab-Static"
    finally:
        monkeypatch.delenv("TSFLAB_HUB_OWNER")
        importlib.reload(uri)
    assert uri.default_repo("TSFLab-Static") == "Diaugeia/TSFLab-Static"


# ---------------------------------------------------------------------------
# Dataset publishing and download selection
# ---------------------------------------------------------------------------


ROOT = repository_root()


def _write_bytes(path: Path, data: bytes) -> None:
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
    with pytest.raises(FileNotFoundError):
        hd.selection("synthetic_st")  # a test fixture, not a catalog dataset


def test_fetch_preset_downloads_only_pinned_files_and_verifies(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "configs" / "datasets", root / "configs" / "datasets")
    payload = b"date,OT\n2020-01-01,1\n"
    manifest = {"schema_version": 1, "repo": "o/r", "files": {
        "ETT-small/ETTh1.csv": {"revision": "abc", "sha256": hashlib.sha256(payload).hexdigest(),
                                "size": len(payload)},
        "ETT-small/ETTh2.csv": {"revision": "abc", "sha256": "0" * 64, "size": 1},
    }}
    _write_bytes(root / hd.MANIFEST_RELATIVE, json.dumps(manifest).encode())
    calls = []

    def fake_download(uri: str, destination: Path, sha256: str | None = None) -> Path:
        calls.append(uri)
        _write_bytes(destination, payload)
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
        _write_bytes(data / name, b"x")
    assert hd.local_files("ultratraffic_ba_st", data) == [
        "ultratraffic/PEMS_BA/static/2023.parquet", "ultratraffic/manifest.json"]
    whole = hd.Selection(preset="ultratraffic", base="ultratraffic")
    assert len(hd.local_files("ultratraffic", data, chosen=whole)) == 4


def test_packaged_manifest_is_well_formed() -> None:
    manifest = hd.load_manifest()
    assert manifest["repo"] == hd.DEFAULT_STATIC_REPO
    for name, entry in manifest["files"].items():
        assert not name.startswith("/") and set(entry) == {"revision", "sha256", "size"}
