"""Seasonal-anomaly model for summer surge forecasting.

Idea: model the deviation of next-28-day log price from its *seasonal normal*
(mean of the same window in the previous 5 years) as a sparse linear function of
  - current price anomaly (persistence)
  - highland weather anomalies (heat / rain) over the last 30~60 days
  - inflow YoY, government release
Linear + weak regularisation so an unprecedented heat anomaly (2024-08) can be
extrapolated instead of being clipped like a tree model.

Trained on summer origins only (origin month 6~9), expanding window, no leakage.
Compared with the same structure fitted on all months.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
FIRST_ORIGIN = "2016-01-31"
NORM_YEARS = 5

CORE = [
    "cur_anom7", "cur_anom28", "dlp7",
    "H_hot30_sum30", "H_hot30_sum60", "H_h_tmax_anom_mean30", "H_h_tmax_anom_mean60",
    "H_h_tmin_anom_mean30", "H_tropical_sum30",
    "H_h_rain_anom_mean30", "H_h_rain_anom_mean60", "H_heavy50_sum30", "H_heavy50_sum60",
    "qyoy28", "qmom7_28",
]
SPARSE = [
    "cur_anom7", "dlp7",
    "H_h_tmax_anom_mean30", "H_hot30_sum60",
    "H_h_rain_anom_mean30", "H_heavy50_sum60",
    "qyoy28",
]
# 2026-10-13(case27 승격, 사용자 결정 "고랭지 기상은 출하기에만 쓰는 걸로"): 고랭지 기상 4개는 여름 작형이 출하되는 때에만 켠다.
#   켜지는 정도 w(d) = 앞으로 28일(d+1~d+28) 가운데 여름 작형의 수확 기간에 드는 날의 비율(세부 작형별로 구해 가장 큰 값).
#   수확 기간은 사용자 업무 파일(작형별 정식·출하 시기 2026ver)의 선을 순 단위로 읽은 것: 준고랭지 1기작 6/11~7/20, 고랭지 7/11~9/30, 준고랭지 2기작 9/21~10/31.
#   그 밖의 날(w = 0)에는 고랭지 기상 재료가 0 → 봄·가을·겨울 가격을 예측할 때 평창·태백 날씨를 보지 않는다.
SUMMER_HARVEST = [((6, 11), (7, 20)), ((7, 11), (9, 30)), ((9, 21), (10, 31))]
HL_WX = ["H_h_tmax_anom_mean30", "H_hot30_sum60", "H_h_rain_anom_mean30", "H_heavy50_sum60"]
SPARSE_SHIP = ["cur_anom7", "dlp7"] + [c + "_ship" for c in HL_WX] + ["qyoy28"]
# 2026-10-08: gov30(정부 방출 30일 합) 제외 — stock.csv 방출은 2022~만 있어 학습구간이 0으로 채워지고,
# 2022-08 첫 등장 때 외삽 사고(an_ridge_all 2022-09 −52% → 제외 시 −12%, MAPE 16.3→15.6%). 상세는 backtest.py 주석·README.


def summer_ship_weight(idx: pd.DatetimeIndex, horizon: int = 28) -> pd.Series:
    """앞으로 horizon 일 가운데 여름 작형(준고랭지 1·2기작, 고랭지) 수확 기간에 드는 날의 비율. 세부 작형 중 최댓값(0~1)."""
    d = idx.values.astype("datetime64[D]")
    best = np.zeros(len(idx))
    for y in range(idx.year.min(), idx.year.max() + 2):
        for (m0, d0), (m1, d1) in SUMMER_HARVEST:
            lo = np.maximum(d + np.timedelta64(1, "D"), np.datetime64(pd.Timestamp(y, m0, d0), "D"))
            hi = np.minimum(d + np.timedelta64(horizon, "D"), np.datetime64(pd.Timestamp(y, m1, d1), "D"))
            best = np.maximum(best, np.clip((hi - lo) / np.timedelta64(1, "D") + 1, 0, horizon) / horizon)
    return pd.Series(best, index=idx)


def add_ship_weather(df: pd.DataFrame) -> pd.DataFrame:
    """고랭지 기상 4개 × 여름 작형 출하기 가중(SPARSE_SHIP 용 열)."""
    w = summer_ship_weight(df.index)
    df["w_summer_ship"] = w
    for c in HL_WX:
        df[c + "_ship"] = df[c] * w
    return df


def add_normals(df: pd.DataFrame) -> pd.DataFrame:
    lp7, lp28, yf = df["lp7"], df["lp28"], df["y_lp_fwd28"]
    n7 = pd.concat([lp7.shift(364 * k) for k in range(1, NORM_YEARS + 1)], axis=1).mean(axis=1)
    n28 = pd.concat([lp28.shift(364 * k) for k in range(1, NORM_YEARS + 1)], axis=1).mean(axis=1)
    # normal for the forward window: same forward window in prior years (fully observed at origin)
    nf = pd.concat([yf.shift(364 * k) for k in range(1, NORM_YEARS + 1)], axis=1).mean(axis=1)
    df["norm7"], df["norm28"], df["norm_fwd"] = n7, n28, nf
    df["cur_anom7"] = lp7 - n7
    df["cur_anom28"] = lp28 - n28
    df["y_anom"] = yf - nf
    return df


def run(freq: str = "M") -> pd.DataFrame:
    df = pd.read_csv(PROC / "daily_features.csv", parse_dates=["date"]).set_index("date")
    df = add_ship_weather(add_normals(df))
    pool = df[(df.is_trading == 1) & df.y_anom.notna()].dropna(subset=CORE).iloc[::3]

    if freq == "M":
        origins = pd.DatetimeIndex(sorted(set(df.index.to_period("M").to_timestamp("M"))))
    else:
        origins = df.index[df.index.weekday == 6]
    origins = origins[(origins >= FIRST_ORIGIN) & (origins <= df.index.max())]

    specs = {
        # 대표 사양(2026-10-09, 사용자 결정): experiments/case03 에서 2016~2023 원점만으로 고른 사양(SPARSE·전체 학습·alpha 10).
        # 사전 선택이라 R-1(사후 설계) 방어가 서고, 학습 풀 정의·추출 시작점·alpha 1~30 에 결론이 안정(2024-09 −9.5~−12.4%, 2025-08 +20~+22%).
        "an_sparse_all":   (SPARSE, False, lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        # 2026-10-13 case27 승격: 같은 사양에서 고랭지 기상만 여름 작형 출하기에 켠 것. gate_model 이 여름 원점 밖의 대표 모형으로 쓴다.
        "an_sparse_ship":  (SPARSE_SHIP, False, lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        "an_ridge_all":    (CORE,   False, lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        "an_ridge_summer": (CORE,   True,  lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        "an_sparse_summer": (SPARSE, True, lambda: make_pipeline(StandardScaler(), Ridge(alpha=0.3))),
        "an_huber_summer": (SPARSE, True,  lambda: make_pipeline(StandardScaler(), HuberRegressor(alpha=0.001, epsilon=1.5, max_iter=500))),
    }
    rows = []
    for o in origins:
        if o not in df.index:
            continue
        te = df.loc[[o]]
        if te.y_anom.isna().iloc[0] or te[CORE].isna().any(axis=1).iloc[0]:
            continue
        tr = pool[pool.index + pd.Timedelta(days=28) <= o]
        rec = {"origin": o, "actual": te.y_price_fwd28_true.iloc[0], "summer": int(te.summer.iloc[0]),
               "normal_fwd": float(np.exp(te.norm_fwd.iloc[0]))}
        for name, (feats, summer_only, mk) in specs.items():
            t = tr[tr.summer == 1] if (summer_only and te.summer.iloc[0] == 1) else tr
            if len(t) < 120:
                rec[name] = np.nan
                continue
            m = mk().fit(t[feats], t["y_anom"])
            rec[name] = float(np.exp(te.norm_fwd.iloc[0] + m.predict(te[feats])[0]))
        rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    freq = sys.argv[1] if len(sys.argv) > 1 else "M"
    pred = run(freq)
    models = [c for c in pred.columns if c.startswith("an_")]
    pd.set_option("display.width", 250)
    for sub_name, mk in {"all_2021_2025": pred.origin.dt.year.between(2021, 2025),
                         "summer_origin_2021_2025": pred.origin.dt.year.between(2021, 2025) & (pred.summer == 1),
                         "summer_origin_2016_2025": pred.origin.dt.year.between(2016, 2025) & (pred.summer == 1)}.items():
        s = pred[mk]
        print(f"\n[{sub_name}] n={len(s)}")
        for m in models + ["normal_fwd"]:
            e = s[m] - s.actual
            print(f"  {m:18s} MAE={e.abs().mean():7.0f}  MAPE={100*(e.abs()/s.actual).mean():5.1f}%  bias={e.mean():7.0f}")
    keys = pd.to_datetime(["2020-08-31", "2020-09-30", "2022-08-31", "2022-09-30", "2024-07-31", "2024-08-31", "2024-09-30", "2024-10-31", "2025-08-31"]) if freq == "M" else pred.origin[(pred.origin >= "2024-07-01") & (pred.origin <= "2024-10-31")]
    sc = pred[pred.origin.isin(keys)].copy()
    for m in models:
        sc[f"{m}%"] = (sc[m] / sc.actual - 1) * 100
    print("\n=== SURGE CASES ===")
    print(sc[["origin", "actual", "normal_fwd"] + models + [f"{m}%" for m in models]].round(0).to_string(index=False))
    tag = "" if freq == "M" else f"_{freq}"
    pred.to_csv(OUT / f"anomaly_predictions{tag}.csv", index=False, float_format="%.1f")
