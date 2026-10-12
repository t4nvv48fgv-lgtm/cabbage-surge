"""Case 12 — 타깃을 '가락 전 등급 평균가격(원/kg)'으로 바꿔 메인 파이프라인을 그대로 돌린다 (KREI Chronos-2 세미나와 같은 계열에서 비교).

배경: KREI 세미나자료(2026-09-30) 그림 3-7의 계열은 가락 상품이 아니라 aT 전 등급 평균(원/kg)이다(2026-10-12 확인: 2024-09 실제 점 ≈1,852원/kg
= 농넷 패널 가락 전 등급 일별 kg가격 월평균). 우리 메인은 상품 등급(원/10kg)이라 오차율·MAE를 직접 비교할 수 없다. 같은 계열에서 돌려 비교한다.

타깃 계열 (원/kg, 일별)
  2018-01~2026-10  농넷 패널 가락 전체(모든 품종·등급) 일별 평균가격 = 총거래금액 ÷ 총거래물량 (`data/ext/nongnet/market_daily_total.csv`, price_kg).
                   월평균을 내면 2024-09 1,852원/kg으로 세미나 그림의 실제 점과 일치.
  2010-01~2017-12  농넷 가락 자료는 2011~2017 구간에 단위 오류 잔존(상품 대비 0.45~0.55, case08 보정 후에도) → 쓰지 않고,
                   CABIS 등급별 일별 가격(`data/raw/garak_daily.csv` 특·상·중·하, 원/10kg)의 가중합으로 **복원**. 가중치는 2018~2026에서
                   농넷 평균가격에 최소제곱 적합(절편 없음, 4등급). 적합 통계는 outputs/series_fit.txt.
  반입량(피처용)은 메인과 같은 clean_daily.csv의 quantity(가락 반입 톤). 날짜는 두 자료의 교집합.
메인 모듈(build_features·backtest·anomaly_model·gate_model·ts_baselines)을 import 해 경로·가격 로더만 바꿔 실행 → 산출물은 이 케이스 outputs/ 에만.
평가: 메인 지표 창(2021~2025 월말) + **세미나 창(원점 2023-12-31~2025-11-30 = 대상 2024-01~2025-12, 24개)** MAE(원/kg)·MAPE·WMAPE.

Usage: python experiments/case12_avg_price_series/run_case.py [M]
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
OUT = CASE_DIR / "outputs"
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import ts_baselines  # noqa: E402
import build_features as bf  # noqa: E402
import backtest as bt  # noqa: E402
import anomaly_model as am  # noqa: E402
import gate_model as gm  # noqa: E402

MODELS = ["naive_last28", "seasonal_naive", "price_only_ridge", "chronos2", "mstl_ets", "wx_ridge", "wx_hgb", "wx_hgb_summer"]
GRADES = ["price_special", "price_high", "price_mid", "price_low"]


# ---------------------------------------------------------------- 1. 타깃 계열
def build_series() -> pd.DataFrame:
    m = pd.read_csv(ROOT / "data" / "ext" / "nongnet" / "market_daily_total.csv", parse_dates=["date"])
    nn = m[m.market.str.contains("가락")][["date", "price_kg"]].sort_values("date")
    nn = nn[(nn.price_kg > 0) & (nn.date >= "2018-01-01")]
    g = pd.read_csv(ROOT / "data" / "raw" / "garak_daily.csv", parse_dates=["date"]).dropna(subset=GRADES)
    fit = g.merge(nn, on="date")
    X, y = fit[GRADES].values / 10.0, fit.price_kg.values
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ w
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    fit = fit.assign(pred=pred, year=fit.date.dt.year)
    by = fit.groupby("year").apply(lambda x: pd.Series({"ratio": (x.pred / x.price_kg).mean(), "mape": (abs(x.pred - x.price_kg) / x.price_kg).mean() * 100}), include_groups=False)
    txt = [f"가중치(특·상·중·하, 원/10kg→원/kg): {dict(zip(GRADES, w.round(4)))}  합 {w.sum():.3f}",
           f"적합 2018~2026 n={len(fit)} R2={r2:.3f} MAPE={(abs(y - pred) / y).mean() * 100:.1f}%", by.round(3).to_string()]
    (OUT / "series_fit.txt").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))
    rec = g[(g.date < "2018-01-01") & (g.date >= "2010-01-01")].copy()
    rec["price_kg"] = rec[GRADES].values / 10.0 @ w
    rec["source"] = "reconstructed_garak_grades"
    nn = nn.assign(source="nongnet_all_grade")
    s = pd.concat([rec[["date", "price_kg", "source"]], nn[["date", "price_kg", "source"]]]).sort_values("date")
    c = pd.read_csv(ROOT / "data" / "raw" / "clean_daily.csv", parse_dates=["date"])
    c = c[c.price > 0][["date", "quantity", "price"]].rename(columns={"price": "price_sangpum_10kg"})
    s = s.merge(c, on="date", how="inner")
    s = s.rename(columns={"price_kg": "price"})
    s.to_csv(OUT / "avg_price_daily.csv", index=False, float_format="%.2f")
    print(f"계열 {s.date.min().date()}~{s.date.max().date()} n={len(s)} (복원 {int((s.source != 'nongnet_all_grade').sum())}, 농넷 {int((s.source == 'nongnet_all_grade').sum())})")
    return s


# ---------------------------------------------------------------- 2. 메인 모듈 경로·로더 교체
def patch(series: pd.DataFrame) -> None:
    p = series[["date", "price", "quantity"]].rename(columns={"quantity": "qty"}).set_index("date")
    bf.load_price = lambda: p
    bf.PROC = OUT
    ps = p["price"].sort_index()
    ts_baselines.load_price = lambda: ps
    bt.PROC, bt.OUT = OUT, OUT
    am.PROC, am.OUT = OUT, OUT
    gm.OUT = OUT


def run_pipeline(freq: str) -> pd.DataFrame:
    tag = "" if freq == "M" else f"_{freq}"
    feats = OUT / "daily_features.csv"
    if not feats.exists():
        df = bf.build()
        df.to_csv(feats, float_format="%.5f")
        print(f"features {df.shape}")
    pred = bt.run(MODELS, freq)
    pred.to_csv(OUT / f"backtest_predictions{tag}.csv", index=False, float_format="%.2f")
    bt.metrics(pred, MODELS).to_csv(OUT / f"backtest_metrics{tag}.csv", index=False, float_format="%.2f")
    an = am.run(freq)
    an.to_csv(OUT / f"anomaly_predictions{tag}.csv", index=False, float_format="%.2f")
    g = gm.run(freq)
    g.to_csv(OUT / f"gate_predictions{tag}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    return g.merge(pred[["origin", "mstl_ets", "wx_ridge", "wx_hgb", "chronos2_q90"]], on="origin")


# ---------------------------------------------------------------- 3. 평가
def metrics(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    y = d.origin.dt.year
    masks = {"all_2021_2025": y.between(2021, 2025), "summer_origin_2021_2025": y.between(2021, 2025) & (d.summer == 1),
             "summer_origin_2021_2026": y.between(2021, 2026) & (d.summer == 1),
             "seminar_2024_2025": (d.origin >= "2023-12-31") & (d.origin <= "2025-11-30")}
    out = []
    for name, mk in masks.items():
        s = d[mk]
        for c in cols:
            e = s[c] - s.actual
            out.append({"subset": name, "model": c, "n": len(s), "MAE": e.abs().mean(), "MAPE%": (e.abs() / s.actual).mean() * 100,
                        "WMAPE%": e.abs().sum() / s.actual.sum() * 100, "bias": e.mean()})
    return pd.DataFrame(out)


if __name__ == "__main__":
    freq = sys.argv[1] if len(sys.argv) > 1 else "M"
    OUT.mkdir(parents=True, exist_ok=True)
    series = build_series()
    patch(series)
    d = run_pipeline(freq)
    cols = ["price_only_ridge", "chronos2", "mstl_ets", "wx_ridge", "wx_hgb", "wx_hgb_summer", "an_sparse_all", "gate_ridge", "gate_hgb"]
    met = metrics(d, cols)
    tag = "" if freq == "M" else f"_{freq}"
    met.to_csv(OUT / f"case12_metrics{tag}.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 300)
    print("\n=== MAE(원/kg) / MAPE / WMAPE ===")
    for sub in ["all_2021_2025", "seminar_2024_2025", "summer_origin_2021_2025"]:
        print(f"\n[{sub}]")
        print(met[met.subset == sub].set_index("model")[["n", "MAE", "MAPE%", "WMAPE%", "bias"]].loc[cols].round(1).to_string())
    keys = pd.to_datetime([f"{y}-0{m}-{dd}" for y in range(2020, 2027) for m, dd in ((8, "31"), (9, "30"))])
    sc = d[d.origin.isin(keys)].copy()
    for c in cols:
        sc[c + "%"] = ((sc[c] / sc.actual - 1) * 100).round(0)
    print("\n=== 주요 원점 (원/kg) 오차% ===")
    print(sc[["origin", "actual", "S1", "gate_calm"] + [c + "%" for c in cols]].round(0).to_string(index=False))
