"""CSI-300 constituent daily log returns through AKShare (free, no key).

The panel stores ``log(close_t / close_{t-1})`` of forward-adjusted (``qfq``)
closes, so dividends and splits do not appear as jumps and values are
comparable across symbols. AKShare throttles large historical requests, so the
fetcher pulls only the requested range per symbol and pauses between symbols.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from moderntsf.realtime.tracks import TrackSpec


def _akshare():
    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("the stock track needs `pip install akshare`") from exc
    return ak


def constituents() -> list[str]:
    ak = _akshare()
    frame = ak.index_stock_cons_csindex(symbol="000300")
    return sorted({str(code).zfill(6) for code in frame["成分券代码"]})


def _eastmoney(ak, symbol: str, start: pd.Timestamp, end: pd.Timestamp, adjust: str) -> pd.Series:
    history = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start.strftime("%Y%m%d"),
                                 end_date=end.strftime("%Y%m%d"), adjust=adjust)
    return pd.Series(history["收盘"].to_numpy(dtype=float), index=pd.to_datetime(history["日期"]))


def _sina(ak, symbol: str, start: pd.Timestamp, end: pd.Timestamp, adjust: str) -> pd.Series:
    prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
    history = ak.stock_zh_a_daily(symbol=prefix + symbol, start_date=start.strftime("%Y%m%d"),
                                  end_date=end.strftime("%Y%m%d"), adjust=adjust)
    return pd.Series(history["close"].to_numpy(dtype=float), index=pd.to_datetime(history["date"]))


_PREFERRED = [_eastmoney, _sina]


def daily_close(symbol: str, start: pd.Timestamp, end: pd.Timestamp, adjust: str = "qfq",
                attempts: int = 4) -> pd.Series:
    """Adjusted daily closes, retrying with backoff and falling back across vendors.

    The vendor that last succeeded is tried first, so a blocked vendor costs one
    failed request per run rather than one per symbol.
    """
    ak = _akshare()
    last_error: Exception | None = None
    for attempt in range(attempts):
        for provider in list(_PREFERRED):
            try:
                series = provider(ak, symbol, start, end, adjust)
            except Exception as exc:  # vendors raise heterogeneous network/parse errors
                last_error = exc
                continue
            if _PREFERRED[0] is not provider:
                _PREFERRED.remove(provider)
                _PREFERRED.insert(0, provider)
            return series
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not fetch {symbol}: {last_error}")


def fetch(track: TrackSpec, start: pd.Timestamp, end: pd.Timestamp, channels: list[str] | None) -> pd.DataFrame:
    symbols = channels or constituents()
    adjust = track.source.get("adjust", "qfq")
    series = {}
    for number, symbol in enumerate(symbols, start=1):
        if number % 25 == 0:
            print(f"  fetched {number}/{len(symbols)} symbols", flush=True)
        # one extra week so the first requested day has a previous close
        close = daily_close(symbol, start - pd.Timedelta(days=7), end, adjust)
        if len(close):
            if track.source.get("transform", "log_return") == "log_return":
                close = np.log(close).diff()
            series[symbol] = close.loc[start:end]
        time.sleep(float(track.source.get("pause_seconds", 0.5)))
    return pd.DataFrame(series).sort_index()
