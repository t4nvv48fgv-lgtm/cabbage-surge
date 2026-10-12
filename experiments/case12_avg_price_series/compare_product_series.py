"""Case 12 보충(2026-10-12) — 상품 계열(원/10kg)에서 현재 최종 모형을 기존 연구·관측월보 전망과 비교.

비교 대상
  STL 분해-예측조합 연구보고(2026-06): 2021~2025 표본외 월별 MAE 1,856·RMSE 2,457(원/10kg), 2024-09 약 −24%. 보고서 수치 인용(재현 아님), 실제값 정의·집계가 조금 달라 엄밀 비교 아님.
  관측월보 가격 전망치: data/local/yupgeun/(KREI)(21_26)_배추_기초자료_관측월보기준.xlsx 1)절(3~12월, 원/10kg). 로컬 전용 파일(미추적)이라 없으면 그 부분은 건너뜀.
  우리 재현 기준선: Chronos-2 제로샷, MSTL+ETS, MSTL+ARIMA, 가격만 ridge (outputs/backtest_predictions.csv).
실제값은 모두 메인의 actual(원점 뒤 28일 평균)로 맞춘다. 월보 전망은 달력 월 기준이라 며칠의 차이가 있다.

Usage: python experiments/case12_avg_price_series/compare_product_series.py
Outputs: outputs/prior_comparison_product.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
OUT = CASE_DIR / "outputs"
KREI_XLSX = ROOT / "data" / "local" / "yupgeun" / "(KREI)(21_26)_배추_기초자료_관측월보기준.xlsx"
STL = {"MAE": 1856, "RMSE": 2457}
warnings.filterwarnings("ignore")


def st(s: pd.DataFrame, c: str) -> dict:
    e = s[c] - s.actual
    ape = e.abs() / s.actual * 100
    return {"n": len(s), "MAE": e.abs().mean(), "RMSE": float(np.sqrt((e ** 2).mean())), "MAPE%": ape.mean(), "bias": e.mean(), "over30": int((ape > 30).sum()), "over40": int((ape > 40).sum())}


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    g = pd.read_csv(ROOT / "outputs" / "gate_predictions.csv", parse_dates=["origin"], encoding="utf-8-sig")
    b = pd.read_csv(ROOT / "outputs" / "backtest_predictions.csv", parse_dates=["origin"])[["origin", "mstl_ets", "mstl_arima"]]
    d = g.merge(b, on="origin")
    t = d.origin + pd.Timedelta(days=15)
    d["ty"], d["tmo"], d["oy"] = t.dt.year, t.dt.month, d.origin.dt.year
    rows = []
    cols = ["final", "final_base", "gate_hgb", "an_sparse_all", "price_only_ridge", "chronos2", "mstl_ets", "mstl_arima"]
    w = d[d.oy.between(2021, 2025)]
    for c in cols:
        rows.append({"subset": "product_2021_2025_all_months", "model": c, **st(w, c)})
    rows.append({"subset": "product_2021_2025_all_months", "model": "STL 연구보고(인용)", "n": np.nan, **STL})
    if KREI_XLSX.exists():
        x = pd.read_excel(KREI_XLSX, header=None)
        k = pd.DataFrame([(int(x.iloc[i, 0]), m, float(x.iloc[i, j])) for i in range(4, 15) for j, m in enumerate(range(3, 13), start=1) if pd.notna(x.iloc[i, j])], columns=["ty", "tmo", "krei_bulletin"])
        d = d.merge(k, on=["ty", "tmo"], how="left")
        for nm, lo in (("bulletin_months_2021_2025", 2021), ("bulletin_months_2016_2025", 2016)):
            s = d[d.ty.between(lo, 2025) & d.krei_bulletin.notna()]
            for c in ["krei_bulletin", "final", "gate_hgb", "price_only_ridge", "chronos2", "mstl_ets"]:
                rows.append({"subset": nm, "model": c, **st(s, c)})
            if lo == 2021:
                closer = int(((s.final / s.actual - 1).abs() < (s.krei_bulletin / s.actual - 1).abs()).sum())
                print(f"관측월보보다 최종 모형이 가까운 달: {closer} / {len(s)}")
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "prior_comparison_product.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    for sub in R.subset.unique():
        print(f"\n[{sub}]")
        print(R[R.subset == sub].drop(columns="subset").round(1).to_string(index=False))
