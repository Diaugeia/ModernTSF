"""Generated preparation text names the right command per loader family."""

from pathlib import Path

from tsflab.catalog.cards.datasets import _preparation, dataset_records

ROOT = Path(__file__).resolve().parents[1]


def _text(name):
    record = next(r for r in dataset_records(ROOT) if r.name == name)
    return _preparation(record)


def test_commands_per_family():
    assert "--from ultratraffic --archive" in _text("ultratraffic_sb_ts")
    assert "--from traffic" in _text("pems04")
    assert "--from gift --datasets jena_weather/H" in _text("gift_eval/jena_weather_H")
    assert "tsf data download" not in _text("etth1")
