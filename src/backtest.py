"""Rolling-origin backtest: forecast mean price over the next 28 days.

Origins: every month-end (comparable to KREI 익월 전망) from 2016-01 to the last
complete window. Expanding training window; the training set only contains rows
whose 28-day target window ends on or before the origin (no leakage).

Models
  naive_last28     : next 28d = last 28d mean
  seasonal_naive   : next 28d = same window last year
  price_only_ridge : lags/seasonality only  (~ what KREI/Chronos-type models see)
  wx_ridge         : price + weather + inflow   (정부 방출 피처는 2026-10-08 제외, SUPPLY_FEATS 주석 참조)
  wx_hgb           : HistGradientBoosting on the same features
  wx_hgb_summer    : HGB trained on summer-origin rows only (6~9월), falls back to wx_hgb otherwise
  chronos2         : Chronos-2 zero-shot on the monthly(weekly)-mean price series (KREI 세미나자료 계열, ts_baselines.py)
  mstl_arima/_ets  : MSTL + AutoARIMA / AutoETS on log monthly(weekly) means (KREI STL-조합 연구 계열)
Outputs: outputs/backtest_predictions.csv, outputs/backtest_metrics.csv, outputs/surge_cases.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ts_baselines

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

PRICE_FEATS = [
    "lp7", "lp28", "lp90", "lp28_lag364", "lp28_lag728", "mom7_28", "mom28_90", "yoy28",
    "vol28", "dlp7", "dlp14", "sin1", "cos1", "sin2", "cos2",
]
# 2026-10-08: gov30/gov60(정부 방출 30/60일 합) 제외. stock.csv 방출은 2022~만 있고 수집 누락이 0으로 들어가
# 2016~2021 학습구간이 전부 0 → 2022-08 첫 등장 때 선형 모형 외삽 사고(기상 ridge 2022-09 −59%, 계절편차 ridge −52%).
# 제외 시 기상 ridge MAPE 17.2→15.0%, 2022-09 −31%; 2024-09는 변화 없음. 정부 방출은 experiments/case01 트리거로만 사용.
SUPPLY_FEATS = ["lq7", "lq28", "qyoy28", "qmom7_28"]
GOV_FEATS_EXCLUDED = ["gov30", "gov60"]
SPREAD_FEATS = ["spread_hm7", "spread_hm28", "dspread_hm", "spread_sh28", "spread_yoy"]  # 검증 결과 노이즈 → 기본 제외
WX_FEATS = [
    "H_hot30_sum14", "H_hot30_sum30", "H_hot30_sum60", "H_hot33_sum30",
    "H_heavy50_sum30", "H_heavy50_sum60", "H_tropical_sum30", "H_heat_sum30", "H_heat_sum60",
    "H_rain_sum14", "H_rain_sum30", "H_rain_sum60", "H_rain_day_sum30",
    "H_h_tmax_anom_mean14", "H_h_tmax_anom_mean30", "H_h_tmax_anom_mean60",
    "H_h_tmin_anom_mean30", "H_h_tavg_anom_mean30", "H_h_rain_anom_mean30", "H_h_rain_anom_mean60",
    "L_hot30_sum30", "L_heavy50_sum30", "L_rain_sum30", "L_l_tmax_anom_mean30", "L_l_rain_anom_mean30",
    "L_l_tmin_anom_mean30",
]
import os
BASE = os.environ.get("SURGE_BASE", "lp28")   # 예측 기준점: lp28(최근28일 평균) 또는 lp7(최근7일 평균)
TARGET = "y_chg"            # log(next28 mean) - BASE  (run()에서 재계산)
TS_BASELINES = {"chronos2": ts_baselines.chronos2, "mstl_arima": ts_baselines.mstl, "mstl_ets": ts_baselines.mstl}
FIRST_ORIGIN = "2016-01-31"


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(set(idx.to_period("M").to_timestamp("M"))))


def interact_summer(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = X.copy()
    for c in cols:
        X[f"{c}_x_summer"] = X[c] * X["summer"]
    return X


def fit_predict(name: str, tr: pd.DataFrame, te: pd.DataFrame) -> float:
    if name == "naive_last28":
        return float(te["lp28"].iloc[0] - te[BASE].iloc[0])
    if name == "seasonal_naive":
        return float(te["lp28_lag364"].iloc[0] - te[BASE].iloc[0]) if pd.notna(te["lp28_lag364"].iloc[0]) else 0.0
    if name == "price_only_ridge":
        feats = PRICE_FEATS
        Xtr, Xte = tr[feats], te[feats]
        m = make_pipeline(StandardScaler(), Ridge(alpha=3.0))
    elif name == "wx_ridge":
        feats = PRICE_FEATS + SUPPLY_FEATS + WX_FEATS + ["summer"]
        Xtr, Xte = interact_summer(tr[feats], WX_FEATS), interact_summer(te[feats], WX_FEATS)
        m = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    elif name in ("wx_hgb", "wx_hgb_summer", "wx_hgb_w"):
        feats = PRICE_FEATS + SUPPLY_FEATS + WX_FEATS + ["summer", "month"]
        if name == "wx_hgb_summer":
            if te["summer"].iloc[0] == 1:
                tr = tr[tr.summer == 1]
            # else: identical to wx_hgb
        Xtr, Xte = tr[feats], te[feats]
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.04, max_depth=4, min_samples_leaf=25,
            l2_regularization=1.0, random_state=0,
        )
        if name == "wx_hgb_w":
            # 여름 원점 + 큰 변동(|y|>0.3) 표본 가중 → 급등락 학습 강조
            wgt = 1.0 + 2.0 * tr["summer"].values + 2.0 * (tr[TARGET].abs().values > 0.3)
            m.fit(Xtr, tr[TARGET], sample_weight=wgt)
            return float(m.predict(Xte)[0])
    else:
        raise ValueError(name)
    m.fit(Xtr, tr[TARGET])
    return float(m.predict(Xte)[0])


def run(models: list[str], origin_freq: str = "M") -> pd.DataFrame:
    df = pd.read_csv(PROC / "daily_features.csv", parse_dates=["date"]).set_index("date")
    df[TARGET] = df["y_lp_fwd28"] - df[BASE]
    # training rows: trading days with complete target; thin to every 3rd day to cut autocorrelation/compute
    train_pool = df[(df.is_trading == 1) & df[TARGET].notna()].copy()
    train_pool = train_pool.dropna(subset=PRICE_FEATS + SUPPLY_FEATS + WX_FEATS)
    train_pool = train_pool.iloc[::3]

    if origin_freq == "M":
        origins = month_ends(df.index)
    else:
        origins = df.index[df.index.weekday == 6]  # Sundays
    origins = origins[(origins >= FIRST_ORIGIN) & (origins <= df.index.max())]

    rows = []
    for o in origins:
        if o not in df.index:
            continue
        te = df.loc[[o]]
        if te[TARGET].isna().iloc[0] or te[PRICE_FEATS + SUPPLY_FEATS + WX_FEATS].isna().any(axis=1).iloc[0]:
            continue
        # no leakage: target window of training rows must end <= origin
        tr = train_pool[train_pool.index + pd.Timedelta(days=28) <= o]
        if len(tr) < 300:
            continue
        rec = {"origin": o, "target_start": o + pd.Timedelta(days=1), "target_end": o + pd.Timedelta(days=28),
               "lp28": te.lp28.iloc[0], "actual": te.y_price_fwd28_true.iloc[0],
               "summer": int(te.summer.iloc[0])}
        yh = {}
        ts_done: set = set()
        for name in models:
            if name == "wx_ens":
                continue
            if name in TS_BASELINES:
                # univariate series models: predicted price directly (prices only up to origin)
                fn = TS_BASELINES[name]
                if fn not in ts_done:
                    rec.update(fn(o, origin_freq))
                    ts_done.add(fn)
                yh[name] = float(np.log(rec[name]) - te[BASE].iloc[0])
                continue
            yh[name] = fit_predict(name, tr, te)
            rec[name] = float(np.exp(te[BASE].iloc[0] + yh[name]))
        if "wx_ens" in models:
            parts = [yh[k] for k in ("wx_ridge", "wx_hgb", "wx_hgb_w") if k in yh]
            rec["wx_ens"] = float(np.exp(te[BASE].iloc[0] + np.mean(parts)))
        rows.append(rec)
        print(f"{o.date()}  actual={rec['actual']:8.0f}  " + "  ".join(f"{n}={rec[n]:8.0f}" for n in models), flush=True)
    return pd.DataFrame(rows)


def metrics(pred: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    out = []
    masks = {
        "all_2016_2025": pred.origin.dt.year.between(2016, 2025),
        "all_2021_2025": pred.origin.dt.year.between(2021, 2025),
        "summer_origin_2016_2025": pred.origin.dt.year.between(2016, 2025) & (pred.summer == 1),
        "summer_origin_2021_2025": pred.origin.dt.year.between(2021, 2025) & (pred.summer == 1),
    }
    for mname, mk in masks.items():
        sub = pred[mk]
        for m in models:
            e = sub[m] - sub.actual
            out.append({"subset": mname, "model": m, "n": len(sub),
                        "MAE": e.abs().mean(), "MAPE%": (e.abs() / sub.actual).mean() * 100,
                        "RMSE": np.sqrt((e ** 2).mean()), "bias": e.mean()})
    return pd.DataFrame(out)


def surge_cases(pred: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    keys = ["2020-08-31", "2020-09-30", "2022-08-31", "2022-09-30", "2024-07-31", "2024-08-31", "2024-09-30", "2024-10-31", "2025-08-31"]
    sub = pred[pred.origin.isin(pd.to_datetime(keys))].copy()
    for m in models:
        sub[f"{m}_err%"] = (sub[m] / sub.actual - 1) * 100
    return sub


if __name__ == "__main__":
    models = ["naive_last28", "seasonal_naive", "price_only_ridge", "chronos2", "mstl_arima", "mstl_ets",
              "wx_ridge", "wx_hgb", "wx_hgb_summer", "wx_hgb_w", "wx_ens"]
    freq = sys.argv[1] if len(sys.argv) > 1 else "M"
    OUT.mkdir(exist_ok=True)
    pred = run(models, freq)
    tag = ("" if freq == "M" else f"_{freq}") + ("" if BASE == "lp28" else f"_{BASE}")
    pred.to_csv(OUT / f"backtest_predictions{tag}.csv", index=False, float_format="%.1f")
    met = metrics(pred, models)
    met.to_csv(OUT / f"backtest_metrics{tag}.csv", index=False, float_format="%.2f")
    sc = surge_cases(pred, models)
    sc.to_csv(OUT / f"surge_cases{tag}.csv", index=False, float_format="%.1f")
    pd.set_option("display.width", 250)
    print("\n=== METRICS ===")
    print(met.round(1).to_string(index=False))
    print("\n=== SURGE CASES (origin -> next 28 days) ===")
    cols = ["origin", "actual"] + models + [f"{m}_err%" for m in models]
    print(sc[cols].round(0).to_string(index=False))
