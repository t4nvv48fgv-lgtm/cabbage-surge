"""All-available-data fit (2026-10-07 탐색).

가용 데이터를 전부 피처로 넣은 변형들을 **기존과 같은 원점 집합·같은 타깃**(향후 28일 평균가격)에서 평가한다.
평가 설계(원점·타깃·누수 차단·하이퍼파라미터)는 backtest.py / anomaly_model.py 와 동일하게 두고 피처만 늘린다.

추가된 정보 (build_features.py '[all-data 확장]' 블록)
  SPREAD  등급별 가격비(상/중, 특/상) — 1차 검증에서 노이즈로 제외됐던 것을 다시 포함
  EXTRA   정부 매입 30/60일 합(2021~), 전년 동일 작형 생산량(통계청, 365일 lag), 반입량 lq90·qyoy7,
          고랭지 90일 누적 폭염·강수·기온편차, 준고랭지(영월) 폭염·기온편차,
          전체 관측소 평균 폭염·기온·강수 편차
  사용 못 한 것: KREI 월보 단수·생산량 전망(2022~2025, 3·5·6·10·12월호만 → 학습 기간 미포함, 8월말 원점에 쓸 수 있는 최신호는 6월호),
                 guide.csv(2023~2025 수급조절 기준선, 값 오류 많음), 일조시간(스냅숏에서 거의 전부 0),
                 history.csv 정부매입(2021~만 값 있음 → at_gov_daily 와 동일)

Models
  all_ridge       PRICE+SUPPLY+WX+SPREAD+EXTRA (+summer 상호작용), Ridge alpha 10  (= wx_ridge 구조)
  all_hgb         같은 피처, HGB (= wx_hgb 하이퍼파라미터)
  all_lgbm        같은 피처, LightGBM
  all_anom_ridge  계절 정상치 편차 타깃, CORE + EXTRA, Ridge alpha 1  (= an_ridge_all 구조)
  all_anom_sparse 계절 정상치 편차 타깃, SPARSE + 핵심 EXTRA, Ridge alpha 0.3
  chronos2_cov    Chronos-2 제로샷 + 과거 공변량(월별 기상·반입·정부방출)

Usage (루트에서):
  python experiments/case02_all_data_fit/case_features.py        # 메인 daily_features.csv → 확장 피처
  python experiments/case02_all_data_fit/all_data_model.py [M|W]  # 메인 outputs/backtest_predictions·anomaly_predictions 와 병합 비교
Outputs: experiments/case02_all_data_fit/outputs/alldata_predictions{tag}.csv, alldata_metrics{tag}.csv, alldata_surge_cases{tag}.csv
"""
from __future__ import annotations

import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 독립 케이스(experiments/case02_all_data_fit). 메인 src/는 import만 하고 수정하지 않는다.
CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import backtest as bt          # noqa: E402  피처 목록·interact_summer·metrics·surge_cases 재사용
import anomaly_model as am     # noqa: E402  add_normals·CORE·SPARSE 재사용
import ts_baselines            # noqa: E402  Chronos-2 파이프라인 재사용

warnings.filterwarnings("ignore")
MAIN_OUT = ROOT / "outputs"                              # 메인 backtest_predictions / anomaly_predictions (비교용, 읽기만)
OUT = CASE_DIR / "outputs"                               # 케이스 산출물
FEATS_EXT = OUT / "daily_features_ext.csv"               # case_features.py 산출물

EXTRA_FEATS = [
    "govpur30", "govpur60", "govpur_avail",
    "lprod_lag1y", "dprod_lag1y",
    "lq90", "qyoy7",
    "H_hot30_sum90", "H_heavy50_sum90", "H_h_tmax_anom_mean90", "H_h_rain_anom_mean90",
    "M_hot30_sum30", "M_hot30_sum60", "M_m_tmax_anom_mean30", "M_m_tmax_anom_mean60",
    "M_rain_sum30", "M_m_rain_anom_mean30",
    "A_hot30_sum30", "A_hot30_sum60", "A_heavy50_sum30",
    "A_a_tmax_anom_mean30", "A_a_tmin_anom_mean30", "A_a_rain_anom_mean30",
]
EXTRA_SPARSE = ["govpur30", "dprod_lag1y", "qyoy7", "H_h_tmax_anom_mean90",
                "M_m_tmax_anom_mean30", "A_a_tmax_anom_mean30"]
ALL_FEATS = bt.PRICE_FEATS + bt.SUPPLY_FEATS + bt.WX_FEATS + bt.SPREAD_FEATS + EXTRA_FEATS
WX_LIKE = bt.WX_FEATS + [c for c in EXTRA_FEATS if c[:2] in ("H_", "M_", "A_")]   # summer 상호작용 대상

# Chronos-2 과거 공변량: 원점 행의 30일 창 값 ≈ 월 집계
COV_COLS = ["H_hot30_sum30", "H_h_tmax_anom_mean30", "H_h_rain_anom_mean30", "H_heavy50_sum30",
            "qyoy28", "lq28", "gov30", "spread_hm28", "M_m_tmax_anom_mean30"]


def _ridge(alpha: float):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


def _hgb():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_depth=4, min_samples_leaf=25,
                                         l2_regularization=1.0, random_state=0)


def _lgbm():
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=25,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                             random_state=0, verbose=-1)


def fit_predict(name: str, tr: pd.DataFrame, te: pd.DataFrame) -> float:
    """Return predicted log change vs BASE (tabular) or log anomaly (anom models)."""
    if name == "all_ridge":
        feats = ALL_FEATS + ["summer"]
        Xtr, Xte = bt.interact_summer(tr[feats], WX_LIKE), bt.interact_summer(te[feats], WX_LIKE)
        return float(_ridge(10.0).fit(Xtr, tr[bt.TARGET]).predict(Xte)[0])
    if name in ("all_hgb", "all_lgbm"):
        feats = ALL_FEATS + ["summer", "month"]
        m = _hgb() if name == "all_hgb" else _lgbm()
        return float(m.fit(tr[feats], tr[bt.TARGET]).predict(te[feats])[0])
    if name == "all_anom_ridge":
        feats = am.CORE + EXTRA_FEATS
        return float(_ridge(1.0).fit(tr[feats], tr["y_anom"]).predict(te[feats])[0])
    if name == "all_anom_sparse":
        feats = am.SPARSE + EXTRA_SPARSE
        return float(_ridge(0.3).fit(tr[feats], tr["y_anom"]).predict(te[feats])[0])
    raise ValueError(name)


# ---------------------------------------------------------------- Chronos-2 with past covariates
def chronos2_cov(df: pd.DataFrame, origin: pd.Timestamp, freq: str) -> dict[str, float]:
    s, h = ts_baselines.history(origin, freq)
    pipe = ts_baselines._chronos()
    # covariates at period ends (month-end rows for M, Sunday rows for W), aligned to the series index
    if freq == "M":
        ends = pd.DatetimeIndex([p.to_timestamp("M") for p in s.index.to_period("M")])
    else:
        ends = s.index
    ends = ends[ends <= origin]
    cov = df.reindex(ends)[COV_COLS].ffill().bfill()
    cov.index = s.index[: len(cov)]
    long = pd.DataFrame({"item_id": "garak", "timestamp": s.index, "target": s.values})
    for c in COV_COLS:
        long[c] = cov[c].reindex(s.index).ffill().bfill().values
    long = long.dropna()
    fc = pipe.predict_df(long, prediction_length=h, quantile_levels=[0.1, 0.5, 0.9],
                         freq="MS" if freq == "M" else "W-SUN")
    q50 = fc["0.5"].values if "0.5" in fc.columns else fc["predictions"].values
    out = {"chronos2_cov": float(np.mean(q50))}
    if "0.9" in fc.columns:
        out["chronos2_cov_q90"] = float(np.mean(fc["0.9"].values))
    return out


def run(models: list[str], freq: str = "M") -> pd.DataFrame:
    df = pd.read_csv(FEATS_EXT, parse_dates=["date"]).set_index("date")
    df[bt.TARGET] = df["y_lp_fwd28"] - df[bt.BASE]
    df = am.add_normals(df)
    core_req = bt.PRICE_FEATS + bt.SUPPLY_FEATS + bt.WX_FEATS            # 원점 포함 조건은 backtest.py 와 동일
    pool = df[(df.is_trading == 1) & df[bt.TARGET].notna() & df.y_anom.notna()].dropna(subset=core_req).iloc[::3]

    origins = bt.month_ends(df.index) if freq == "M" else df.index[df.index.weekday == 6]
    origins = origins[(origins >= bt.FIRST_ORIGIN) & (origins <= df.index.max())]

    rows = []
    for o in origins:
        if o not in df.index:
            continue
        te = df.loc[[o]]
        if te[bt.TARGET].isna().iloc[0] or te[core_req].isna().any(axis=1).iloc[0]:
            continue
        tr = pool[pool.index + pd.Timedelta(days=28) <= o]
        if len(tr) < 300:
            continue
        base = float(te[bt.BASE].iloc[0])
        rec = {"origin": o, "actual": float(te.y_price_fwd28_true.iloc[0]), "summer": int(te.summer.iloc[0]),
               "lp28": base}
        for name in models:
            if name == "chronos2_cov":
                rec.update(chronos2_cov(df, o, freq))
                continue
            yh = fit_predict(name, tr, te)
            anchor = float(te.norm_fwd.iloc[0]) if name.startswith("all_anom") else base
            rec[name] = float(np.exp(anchor + yh))
        rows.append(rec)
        print(f"{o.date()}  actual={rec['actual']:8.0f}  " + "  ".join(f"{n}={rec[n]:8.0f}" for n in models), flush=True)
    return pd.DataFrame(rows)


def compare_with_existing(pred: pd.DataFrame, models: list[str], freq: str) -> tuple[pd.DataFrame, list[str]]:
    """Join previously saved baseline predictions on the same origins."""
    tag = "" if freq == "M" else f"_{freq}"
    ref_cols = ["price_only_ridge", "chronos2", "mstl_ets", "wx_ridge", "wx_hgb"]
    out = pred.copy()
    try:
        b = pd.read_csv(MAIN_OUT / f"backtest_predictions{tag}.csv", parse_dates=["origin"])
        out = out.merge(b[["origin"] + [c for c in ref_cols if c in b.columns]], on="origin", how="left")
    except FileNotFoundError:
        ref_cols = []
    try:
        a = pd.read_csv(MAIN_OUT / f"anomaly_predictions{tag}.csv", parse_dates=["origin"])
        out = out.merge(a[["origin", "an_ridge_all"]], on="origin", how="left")
        ref_cols = ref_cols + ["an_ridge_all"]
    except FileNotFoundError:
        pass
    return out, [c for c in ref_cols if c in out.columns]


if __name__ == "__main__":
    freq = sys.argv[1] if len(sys.argv) > 1 else "M"
    models = ["all_ridge", "all_hgb", "all_lgbm", "all_anom_ridge", "all_anom_sparse", "chronos2_cov"]
    OUT.mkdir(exist_ok=True)
    pred = run(models, freq)
    tag = "" if freq == "M" else f"_{freq}"
    pred.to_csv(OUT / f"alldata_predictions{tag}.csv", index=False, float_format="%.1f")

    full, refs = compare_with_existing(pred, models, freq)
    allm = refs + models
    full = full.dropna(subset=allm)
    met = bt.metrics(full, allm)
    met.to_csv(OUT / f"alldata_metrics{tag}.csv", index=False, float_format="%.2f")
    sc = bt.surge_cases(full, allm)
    sc.to_csv(OUT / f"alldata_surge_cases{tag}.csv", index=False, float_format="%.1f")
    pd.set_option("display.width", 300)
    print(f"\n=== METRICS (common origins n={len(full)}) ===")
    print(met[met.subset.isin(["all_2021_2025", "summer_origin_2021_2025"])].round(1).to_string(index=False))
    print("\n=== SURGE CASES  err% = pred/actual - 1 ===")
    cols = ["origin", "actual"] + [f"{m}_err%" for m in allm]
    print(sc[cols].round(0).to_string(index=False))
