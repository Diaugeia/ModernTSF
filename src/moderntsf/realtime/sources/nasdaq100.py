"""NASDAQ-100 constituent daily log returns (free, no key).

Constituents come from the Nasdaq list endpoint. Closes are forward-adjusted
(``qfq``) histories from Sina through AKShare; the Nasdaq historical endpoint is
the fallback and is split- but not dividend-adjusted, so a fallback day can
differ by the dividend yield. Symbols are stored as Nasdaq tickers with ``.``
replaced by ``-`` (e.g. ``BRK-B``).
"""

from __future__ import annotations

import pandas as pd

from moderntsf.realtime.sources.equity import fetch_panel, with_fallback
from moderntsf.realtime.tracks import TrackSpec

_NASDAQ = "https://api.nasdaq.com/api"
_HEADERS = {"User-Agent": "Mozilla/5.0 (ModernTSF real-time track)", "Accept": "application/json"}


def _get(path: str, **params) -> dict:
    import requests

    response = requests.get(f"{_NASDAQ}/{path}", params=params, headers=_HEADERS, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("data"):
        raise RuntimeError(f"Nasdaq API returned no data for {path}: {payload.get('status')}")
    return payload["data"]


def constituents() -> list[str]:
    rows = _get("quote/list-type/nasdaq100")["data"]["rows"]
    return sorted({row["symbol"].strip().upper().replace(".", "-") for row in rows})


def _sina(symbol: str, start: pd.Timestamp, end: pd.Timestamp, adjust: str) -> pd.Series:
    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("the stock track needs `pip install akshare`") from exc
    history = ak.stock_us_daily(symbol=symbol.replace("-", "."), adjust=adjust)
    close = pd.Series(history["close"].to_numpy(dtype=float), index=pd.to_datetime(history["date"]))
    return close.loc[start:end]


def _nasdaq(symbol: str, start: pd.Timestamp, end: pd.Timestamp, adjust: str) -> pd.Series:
    data = _get(f"quote/{symbol.replace('-', '.')}/historical", assetclass="stocks",
                fromdate=f"{start:%Y-%m-%d}", todate=f"{end:%Y-%m-%d}", limit=9999)
    rows = (data.get("tradesTable") or {}).get("rows") or []
    values = [float(str(row["close"]).replace("$", "").replace(",", "")) for row in rows]
    return pd.Series(values, index=pd.to_datetime([row["date"] for row in rows], format="%m/%d/%Y"))


_PREFERRED = [_sina, _nasdaq]


def daily_close(symbol: str, start: pd.Timestamp, end: pd.Timestamp, adjust: str = "qfq") -> pd.Series:
    return with_fallback(_PREFERRED, symbol, start, end, adjust, label=symbol)


def fetch(track: TrackSpec, start: pd.Timestamp, end: pd.Timestamp, channels: list[str] | None) -> pd.DataFrame:
    adjust = track.source.get("adjust", "qfq")
    return fetch_panel(track, start, end, channels or constituents(),
                       lambda symbol, lo, hi: daily_close(symbol, lo, hi, adjust))
