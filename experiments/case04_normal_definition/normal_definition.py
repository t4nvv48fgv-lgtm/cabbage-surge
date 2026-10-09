"""Case 04 — 계절편차 모형의 '평년치' 정의 비교.

배경: 2025-08-31 원점에서 대표 모형(an_sparse_all)이 +21% 과대 예측한 원인의 절반 이상이 평년치(직전 5년 같은 창 평균)에
급등 해 셋(2020·2022·2024)이 들어가 18,243원으로 부풀어 있었기 때문(2026-10-09 분해). 평년치 정의만 바꿔 같은 원점·같은 사양에서 비교한다.

후보 (실행 전 고정)
  mean5    직전 5년 평균 (현재 메인)
  median5  직전 5년 중앙값
  olympic5 직전 5년 최대·최소 제외 평균   ← KREI 관측월보의 '평년' 정의(최근 5개년 최대·최소 제외 평균)와 같음 → 사전 근거 있는 후보
  mean7    직전 7년 평균
  mean10   직전 10년 평균 (자료 시작 2010이라 초기 원점은 가용 연도만, 최소 5년)
평년치는 norm7/norm28(가격 편차 피처 cur_anom7/28)과 norm_fwd(타깃 y_anom 기준)에 **동일하게** 적용한다.
모형 사양은 메인 대표 사양(SPARSE 8피처·전체 원점 학습·ridge alpha 10)으로 고정. 원점·학습 표본·누수 차단은 메인 anomaly_model.py와 동일.

보고: 2020-08·2022-08·2024-08·2025-08 오차, 2021~2025 MAPE(전체·여름), 설계 구간(2016~2023) MAPE(사전 선택 가능성 확인).
Usage: python experiments/case04_normal_definition/normal_definition.py
Outputs: outputs/normal_predictions.csv, normal_scores.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import anomaly_model as am  # noqa: E402  SPARSE·FIRST_ORIGIN 재사용

warnings.filterwarnings("ignore")
PROC = ROOT / "data" / "processed"
OUT = CASE_DIR / "outputs"
ALPHA = 10.0
FEATS = am.SPARSE

DEFS = {
    "mean5": ("mean", 5), "median5": ("median", 5), "olympic5": ("olympic", 5),
    "mean7": ("mean", 7), "mean10": ("mean", 10),
}


def normal(s: pd.Series, how: str, years: int) -> pd.Series:
    """메인 anomaly_model.add_normals 와 같은 관례: 가용한 과거 연도만으로 계산(초기 원점은 1~years년).
    olympic 은 최대·최소를 빼야 하므로 3년 이상 있을 때만, 그 전에는 mean 으로 대체."""
    lags = pd.concat([s.shift(364 * k) for k in range(1, years + 1)], axis=1)
    n = lags.notna().sum(axis=1)
    if how == "mean":
        out = lags.mean(axis=1)
    elif how == "median":
        out = lags.median(axis=1)
    elif how == "olympic":
        oly = (lags.sum(axis=1) - lags.max(axis=1) - lags.min(axis=1)) / (n - 2)
        out = oly.where(n >= 3, lags.mean(axis=1))
    else:
        raise ValueError(how)
    return out.where(n >= 1)


def add_normals(df: pd.DataFrame, how: str, years: int) -> pd.DataFrame:
    d = df.copy()
    d["norm7"] = normal(d["lp7"], how, years)
    d["norm28"] = normal(d["lp28"], how, years)
    d["norm_fwd"] = normal(d["y_lp_fwd28"], how, years)
    d["cur_anom7"] = d["lp7"] - d["norm7"]
    d["cur_anom28"] = d["lp28"] - d["norm28"]
    d["y_anom"] = d["y_lp_fwd28"] - d["norm_fwd"]
    return d


def run() -> pd.DataFrame:
    base = pd.read_csv(PROC / "daily_features.csv", parse_dates=["date"]).set_index("date")
    origins = pd.DatetimeIndex(sorted(set(base.index.to_period("M").to_timestamp("M"))))
    origins = origins[(origins >= am.FIRST_ORIGIN) & (origins <= base.index.max())]
    rows = {}
    for name, (how, years) in DEFS.items():
        df = add_normals(base, how, years)
        pool = df[(df.is_trading == 1) & df.y_anom.notna()].dropna(subset=am.CORE).iloc[::3]
        for o in origins:
            if o not in df.index:
                continue
            te = df.loc[[o]]
            if te.y_anom.isna().iloc[0] or te[am.CORE].isna().any(axis=1).iloc[0]:
                continue
            tr = pool[pool.index + pd.Timedelta(days=28) <= o]
            if len(tr) < 120:
                continue
            m = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)).fit(tr[FEATS], tr["y_anom"])
            rec = rows.setdefault(o, {"origin": o, "actual": float(te.y_price_fwd28_true.iloc[0]), "summer": int(te.summer.iloc[0])})
            rec[f"normal_{name}"] = float(np.exp(te.norm_fwd.iloc[0]))
            rec[f"pred_{name}"] = float(np.exp(te.norm_fwd.iloc[0] + m.predict(te[FEATS])[0]))
        print(f"{name}: done", flush=True)
    return pd.DataFrame(list(rows.values())).sort_values("origin")


def score(p: pd.DataFrame) -> pd.DataFrame:
    y = p.origin.dt.year
    masks = {"all_2021_2025": y.between(2021, 2025), "summer_2021_2025": y.between(2021, 2025) & (p.summer == 1),
             "design_2016_2023": y.between(2016, 2023), "design_summer_2016_2023": y.between(2016, 2023) & (p.summer == 1)}
    out = []
    for name in DEFS:
        c = f"pred_{name}"
        r = {"normal": name}
        for k, mk in masks.items():
            s = p[mk].dropna(subset=[c])
            e = s[c] - s.actual
            r[f"{k}_MAPE"] = (e.abs() / s.actual).mean() * 100
            r[f"{k}_bias"] = e.mean()
        for o in ("2020-08-31", "2022-08-31", "2023-08-31", "2024-08-31", "2024-09-30", "2025-08-31"):
            s = p[p.origin == o]
            r[f"err_{o[:7]}"] = float((s[c].iloc[0] / s.actual.iloc[0] - 1) * 100) if len(s) and pd.notna(s[c].iloc[0]) else np.nan
            if o in ("2024-08-31", "2025-08-31"):
                r[f"normal_{o[:7]}"] = float(s[f"normal_{name}"].iloc[0]) if len(s) else np.nan
        out.append(r)
    return pd.DataFrame(out)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    p = run()
    p.to_csv(OUT / "normal_predictions.csv", index=False, float_format="%.1f")
    sc = score(p)
    sc.to_csv(OUT / "normal_scores.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 300)
    cols = ["normal", "normal_2024-08", "normal_2025-08", "err_2020-08", "err_2022-08", "err_2023-08", "err_2024-08", "err_2024-09", "err_2025-08",
            "all_2021_2025_MAPE", "summer_2021_2025_MAPE", "all_2021_2025_bias", "design_2016_2023_MAPE", "design_summer_2016_2023_MAPE"]
    print("\n=== 평년치 정의별 결과 (대표 사양 고정: SPARSE·전체 학습·alpha 10) ===")
    print(sc[cols].round(1).to_string(index=False))
