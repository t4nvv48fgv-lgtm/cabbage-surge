"""Univariate time-series baselines that mirror the two KREI studies.

  chronos2    : Amazon Chronos-2 zero-shot (KREI 세미나자료는 미세조정판; 여기선 제로샷)
  mstl_arima  : MSTL 분해 + AutoARIMA 추세 (KREI 연구보고의 STL-조합 계열)
  mstl_ets    : MSTL 분해 + AutoETS 추세

All use only the Garak daily price series up to (and including) the origin.
Monthly origins -> monthly-mean series, h=1 (익월 평균).  Weekly origins -> weekly-mean
series, h=4, mean of the 4 steps.  Returned value is the predicted mean price (원/10kg).
"""
from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
QUANTILES = (0.1, 0.5, 0.9)


@lru_cache(maxsize=1)
def load_price() -> pd.Series:
    p = pd.read_csv(RAW / "clean_daily.csv", parse_dates=["date"]).set_index("date")["price"]
    return p[p > 0].sort_index()


def history(origin: pd.Timestamp, freq: str) -> tuple[pd.Series, int]:
    """Regular series ending at the origin + forecast horizon covering the next 28 days."""
    p = load_price()
    p = p[p.index <= origin]
    if freq == "M":
        s, h = p.resample("MS").mean(), 1
    else:  # weekly origins fall on Sundays
        s, h = p.resample("W-SUN").mean(), 4
    s = s.interpolate(limit_direction="both")
    return s, h


# ---------------------------------------------------------------- Chronos-2
@lru_cache(maxsize=1)
def _chronos():
    import torch
    from chronos import BaseChronosPipeline
    torch.set_num_threads(max(1, torch.get_num_threads() // 2))
    return BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu", torch_dtype=torch.float32)


def chronos2(origin: pd.Timestamp, freq: str) -> dict[str, float]:
    import torch
    s, h = history(origin, freq)
    ctx = torch.tensor(s.values, dtype=torch.float32).reshape(1, 1, -1)
    q, _ = _chronos().predict_quantiles(ctx, prediction_length=h, quantile_levels=list(QUANTILES))
    q = np.asarray(q[0] if isinstance(q, list) else q)[0]       # (h, n_q)
    out = {"chronos2": float(q[:, 1].mean())}
    out["chronos2_q10"] = float(q[:, 0].mean())
    out["chronos2_q90"] = float(q[:, 2].mean())
    return out


# ---------------------------------------------------------------- MSTL (statsforecast)
def mstl(origin: pd.Timestamp, freq: str) -> dict[str, float]:
    from statsforecast import StatsForecast
    from statsforecast.models import MSTL, AutoARIMA, AutoETS
    s, h = history(origin, freq)
    season = 12 if freq == "M" else 52
    df = pd.DataFrame({"unique_id": "garak", "ds": s.index, "y": np.log(s.values)})
    models = [
        MSTL(season_length=season, trend_forecaster=AutoARIMA(), alias="mstl_arima"),
        MSTL(season_length=season, trend_forecaster=AutoETS(model="ZZN"), alias="mstl_ets"),
    ]
    sf = StatsForecast(models=models, freq="MS" if freq == "M" else "W-SUN", n_jobs=1)
    fc = sf.forecast(df=df, h=h)
    return {m: float(np.exp(fc[m].values).mean()) for m in ("mstl_arima", "mstl_ets")}


if __name__ == "__main__":
    import sys, time
    o = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2024-08-31")
    f = sys.argv[2] if len(sys.argv) > 2 else "M"
    for fn in (chronos2, mstl):
        t = time.time(); r = fn(o, f)
        print(fn.__name__, {k: round(v) for k, v in r.items()}, f"{time.time()-t:.1f}s")
