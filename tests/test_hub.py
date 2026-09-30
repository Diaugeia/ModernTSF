"""Tests for hf:// addressing, pinned downloads, weights bundles, and scaffolding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import BaseModel
import torch
import torch.nn as nn

from moderntsf import hub
from moderntsf.benchmark.model_artifacts import fetch_artifact
from moderntsf.benchmark.registry.models import ModelArtifact, ModelSpec
from moderntsf.scaffold import init_project

REPO = "Diaugeia/ModernTSF-Weights"
REV = "0123abcd"


def test_parse_round_trips_every_repo_type() -> None:
    for text, repo_type in (
        (f"hf://{REPO}@{REV}/weather/DLinear/run/model.safetensors", "model"),
        (f"hf://datasets/Diaugeia/ModernTSF-Static@main/ett/ETTh1.csv", "dataset"),
        (f"hf://spaces/Diaugeia/ModernTSF@v1/index.html", "space"),
    ):
        uri = hub.parse(text)
        assert uri.repo_type == repo_type
        assert str(uri) == text
    uri = hub.parse(f"hf://datasets/Diaugeia/ModernTSF-Static@main/ett/ETTh1.csv")
    assert uri.resolve_url() == (
        "https://huggingface.co/datasets/Diaugeia/ModernTSF-Static/resolve/main/ett/ETTh1.csv"
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
        module="moderntsf.models.fixture",
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


def _run(work: Path) -> str:
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
    run_id = _run(tmp_path)
    record, checkpoint = hub.find_run(run_id, tmp_path)
    bundle = tmp_path / "bundle"
    manifest = hub.pack(record, checkpoint, bundle)
    assert manifest["path"] == f"weather/DLinear/{run_id}"
    assert manifest["metrics"] == {"mse": 0.5}
    assert (bundle / "README.md").read_text().startswith("---\nlibrary_name: moderntsf")

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


def test_init_project_scaffolds_and_inherits_package_configs(tmp_path: Path) -> None:
    from moderntsf.benchmark.config.loader import _resolve_extends

    written = init_project(tmp_path / "proj")
    assert (tmp_path / "proj" / "configs" / "runs" / "example.toml") in written
    with pytest.raises(FileExistsError):
        init_project(tmp_path / "proj")
    import tomllib

    run = tmp_path / "proj" / "configs" / "runs" / "example.toml"
    merged = _resolve_extends(tomllib.loads(run.read_text()), str(run.parent))
    assert merged["dataset"]["name"] == "ETTh1"
    assert merged["experiment"]["description"] == "proj: DLinear on ETTh1"


def test_hub_owner_comes_from_the_environment(monkeypatch) -> None:
    import importlib

    from moderntsf.hub import uri

    monkeypatch.setenv("MODERNTSF_HUB_OWNER", "someone")
    try:
        assert importlib.reload(uri).default_repo("ModernTSF-Static") == "someone/ModernTSF-Static"
    finally:
        monkeypatch.delenv("MODERNTSF_HUB_OWNER")
        importlib.reload(uri)
    assert uri.default_repo("ModernTSF-Static") == "Diaugeia/ModernTSF-Static"
