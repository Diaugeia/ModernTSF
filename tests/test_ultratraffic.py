"""UltraTraffic store, converter, and datasets."""

from __future__ import annotations

import io
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

from moderntsf.data.calendar import node_calendar
from moderntsf.data.datasets.ultratraffic import Dataset_UltraTraffic_ST, Dataset_UltraTraffic_TS
from moderntsf.data.prepare.ultratraffic import convert
from moderntsf.data.ultratraffic_store import load_panel


def _year_csv(year: int, stations: list[str]) -> bytes:
    index = pd.date_range(f"{year}-01-01", periods=24 * 20, freq="h")
    t = np.arange(len(index))[:, None]
    frame = pd.DataFrame(100 + 10 * np.sin(2 * np.pi * t / 24) + np.arange(len(stations))[None, :],
                         index=index, columns=stations)
    frame.index.name = "date"
    return frame.to_csv().encode()


def _archive(tmp_path: Path) -> Path:
    path = tmp_path / "UltraTraffic_CL.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for region, short in (("PEMS_SB", "SB"), ("PEMS_NC", "NC")):
            zf.writestr(f"UltraTraffic_CL/{region}/{short}_Static/2022.csv", _year_csv(2022, ["801", "802"]))
            zf.writestr(f"UltraTraffic_CL/{region}/{short}_Static/2023.csv", _year_csv(2023, ["801", "803"]))
            zf.writestr(f"UltraTraffic_CL/{region}/{short}_CL/2023.csv", _year_csv(2023, ["801", "803"]))
            zf.writestr(f"UltraTraffic_CL/{region}/{short}_CL/2023_added.csv", _year_csv(2023, ["803"]))
    return path


def test_converter_skips_duplicates_and_redundant_cl_files(tmp_path: Path) -> None:
    manifest = convert(_archive(tmp_path), tmp_path / "store")
    assert set(manifest["regions"]) == {"PEMS_SB"}  # PEMS_NC duplicates PEMS_SAC and is skipped
    files = manifest["regions"]["PEMS_SB"]["files"]
    assert set(files) == {"static/2022.parquet", "static/2023.parquet", "cl/2023_added.parquet"}
    assert files["static/2023.parquet"]["stations"] == 2


def test_panel_station_policies(tmp_path: Path) -> None:
    convert(_archive(tmp_path), tmp_path / "store")
    root = str(tmp_path / "store")
    last = load_panel(root, "PEMS_SB", [2022, 2023], "static", "last")
    inter = load_panel(root, "PEMS_SB", [2022, 2023], "static", "intersection")
    assert list(last.columns) == ["801", "803"] and last["803"].iloc[:10].isna().all()
    assert list(inter.columns) == ["801"]


def test_datasets_follow_the_item_contracts(tmp_path: Path) -> None:
    convert(_archive(tmp_path), tmp_path / "store")
    kwargs = dict(root_path=str(tmp_path / "store"), data_path="", size=(24, 0, 12), region="PEMS_SB", years=[2023])
    st = Dataset_UltraTraffic_ST(flag="train", **kwargs)
    value_hist, value_fut, cov_hist, cov_fut = st[0]
    assert value_hist.shape == (24, 2) and value_fut.shape == (12, 2)
    assert cov_hist.shape == (24, 2, 2) and cov_fut.shape == (12, 2, 2)
    assert st.num_nodes == 2 and st.adj_mx is None
    test = Dataset_UltraTraffic_ST(flag="test", **kwargs)
    assert test.idx.min() > st.idx.max()  # chronological, non-overlapping targets
    ts = Dataset_UltraTraffic_TS(flag="val", **kwargs)
    assert ts[0][2].shape == (24, 6)
    restored = st.inverse_transform(value_hist)
    assert abs(restored.mean() - 100.5) < 10


def test_calendar_covariates() -> None:
    index = pd.date_range("2026-01-05", periods=3, freq="6h")  # a Monday
    features = node_calendar(index, 4)
    assert features.shape == (3, 4, 2)
    assert features[1, 0, 0] == 0.25 and features[0, 0, 1] == 0.0


def test_store_reader_is_torch_free() -> None:
    import subprocess

    code = "import sys, moderntsf.data.ultratraffic_store, moderntsf.data.calendar; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"
