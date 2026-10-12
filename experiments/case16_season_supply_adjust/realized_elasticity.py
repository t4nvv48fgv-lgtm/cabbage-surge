"""가격 신축성 β 의 경험적 추정 — 통계청 작형별 '실제' 생산량 vs 출하기 '실제' 가격 (2005~2024).

사용자 질문(2026-10-12) "β를 경험적으로 추정할 수 있나?" → 전망치가 아니라 실적으로 추정하는 길.
  Δlog(출하기 평균가격_t) = β × Δlog(생산량_t) + ε   (작형별, 연도 차분, 절편 유무 둘 다)
출하기: 겨울 = 익년 1~4월, 봄 = 5~6월, 여름 = 7~10월, 가을 = 11~12월. 가격 = 가락 상품(2005~2009 garak_daily price_high, 2010~ clean_daily).
생산량: `data/local/yupgeun/(4)배추 통계청 생산량.xlsx` 시트 '통계청'(비공개 경로) → 연도·작형별 생산량만 outputs/stat_production_by_season.csv 로 저장(통계청 공표치).

주의: 전망치에 쓸 β 는 실적 β 에 전망의 신뢰도(cov(전망,실적)/var(전망))를 곱한 값이어야 한다(전망 오차에 의한 감쇠).
Usage: python experiments/case16_season_supply_adjust/realized_elasticity.py
Outputs: outputs/realized_beta.csv, stat_production_by_season.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
OUT = CASE_DIR / "outputs"
XLSX = ROOT / "data" / "local" / "yupgeun" / "(4)배추 통계청 생산량.xlsx"
MON = {"겨울": (1, 2, 3, 4), "봄": (5, 6), "여름": (7, 8, 9, 10), "가을": (11, 12)}


def load_production() -> pd.DataFrame:
    csv = OUT / "stat_production_by_season.csv"
    if XLSX.exists():
        x = pd.read_excel(XLSX, sheet_name="통계청", header=None)
        top, sub = x.iloc[0].ffill(), x.iloc[1]
        cols = {str(top[j]).strip(): j for j in range(1, x.shape[1]) if str(sub[j]).strip() == "생산량"}
        yr = pd.to_numeric(x.iloc[2:, 0], errors="coerce")
        prod = pd.DataFrame({k: pd.to_numeric(x.iloc[2:, j], errors="coerce").values for k, j in cols.items()}, index=yr.values)
        prod = prod[prod.index.notna()]
        prod.index = prod.index.astype(int)
        prod = prod[~prod.index.duplicated(keep="first")].sort_index()[["봄", "여름", "가을", "겨울"]]
        prod.index.name = "year"
        prod.to_csv(csv, float_format="%.0f")
        return prod
    return pd.read_csv(csv, index_col="year")


def monthly_price() -> pd.Series:
    g = pd.read_csv(ROOT / "data" / "raw" / "garak_daily.csv", parse_dates=["date"]).set_index("date").price_high
    c = pd.read_csv(ROOT / "data" / "raw" / "clean_daily.csv", parse_dates=["date"])
    c = c[c.price > 0].set_index("date").price
    p = pd.concat([g[g.index < "2010-01-01"], c]).sort_index()
    return p.groupby(p.index.to_period("M")).mean()


def season_price(m: pd.Series, year: int, sea: str) -> float:
    yy = year + 1 if sea == "겨울" else year
    v = [m.get(pd.Period(f"{yy}-{mm:02d}")) for mm in MON[sea]]
    return np.nan if any(t is None or np.isnan(t) for t in v) else float(np.mean(v))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    prod, m = load_production(), monthly_price()
    rows = []
    for sea in ["봄", "여름", "가을", "겨울"]:
        q = prod[sea].dropna()
        q = q[(q.index >= 2004) & (q.index <= 2025)]
        pr = pd.Series({y: season_price(m, y, sea) for y in q.index}).dropna()
        idx = q.index.intersection(pr.index)
        dq, dp = np.log(q.loc[idx]).diff().dropna(), np.log(pr.loc[idx]).diff().dropna()
        j = dq.index.intersection(dp.index)
        b0 = float((dq[j] * dp[j]).sum() / (dq[j] ** 2).sum())
        X = np.vstack([np.ones(len(j)), dq[j].values]).T
        b1 = float(np.linalg.lstsq(X, dp[j].values, rcond=None)[0][1])
        rows.append({"season": sea, "n": len(j), "years": f"{j.min()}~{j.max()}", "beta_no_intercept": round(b0, 2), "beta_intercept": round(b1, 2),
                     "corr": round(float(np.corrcoef(dq[j], dp[j])[0, 1]), 2), "std_dlogQ%": round(dq[j].std() * 100, 1), "std_dlogP%": round(dp[j].std() * 100, 1)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "realized_beta.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))
