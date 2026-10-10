"""Case 06 승격 조건 1~3 점검.

1. 2017~2020 여름(KREI 판독표 범위 밖): 잔량 게이트 G_S만으로 방향이 맞는지 — 2020 급등·2017~2019 비급등.
2. 평시 모형 분리 선택: 후보 {price_only_ridge, wx_ridge, wx_hgb, wx_hgb_summer, chronos2, mstl_ets}를
   2016~2023 월말 원점 MAPE(기준 A 전체 / B 여름)로 고르고, 그 평시 모형으로 G_K를 2024~2026에서 평가.
3. 주간 원점(일요일): S1 = 원점까지 발행된 최신호(월 1회 갱신), S2 = 원점 시점 잔량. G_K·G_S·G_both 주간 지표와 2024·2025 추적.

Usage: python experiments/case06_supply_gate/promotion_checks.py
Outputs: outputs/check1_2017_2020.csv, check2_calm_selection.csv, check3_weekly_metrics.csv, check3_weekly_2024_2025.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
sys.path.insert(0, str(CASE_DIR))
sys.path.insert(0, str(ROOT / "experiments" / "case05_gov_stock_krei"))
import supply_gate as sg      # noqa: E402
import gov_stock_krei as c5   # noqa: E402

OUT = CASE_DIR / "outputs"
MAIN_OUT = ROOT / "outputs"
HEAD = sg.HEAD
CANDS = ["price_only_ridge", "wx_ridge", "wx_hgb", "wx_hgb_summer", "chronos2", "mstl_ets"]


def load_monthly() -> pd.DataFrame:
    a = pd.read_csv(MAIN_OUT / "anomaly_predictions.csv", parse_dates=["origin"])
    b = pd.read_csv(MAIN_OUT / "backtest_predictions.csv", parse_dates=["origin"])
    return b.merge(a[["origin", HEAD]], on="origin")


def err(d, c):
    return (d[c] / d.actual - 1) * 100


def mape(d, c):
    return float(((d[c] - d.actual).abs() / d.actual).mean() * 100)


# ---------------------------------------------------------------- 1
def check1(m: pd.DataFrame, lots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in range(2017, 2021):
        for mo, dd in ((7, 31), (8, 31), (9, 30)):
            o = pd.Timestamp(year=y, month=mo, day=dd)
            r = m[m.origin == o]
            if r.empty:
                continue
            r = r.iloc[0]
            st = c5.stock_at(lots, o)["stock_est"]
            calm = st >= sg.STOCK_T
            gs = r["price_only_ridge"] if calm else r[HEAD]
            rows.append({"origin": o.date(), "actual": round(r.actual), "S2_stock": round(st), "calm_S": int(calm),
                         "err_price_only%": round((r.price_only_ridge / r.actual - 1) * 100), "err_head%": round((r[HEAD] / r.actual - 1) * 100),
                         "err_G_S%": round((gs / r.actual - 1) * 100)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2
def check2(m: pd.DataFrame, sig: pd.DataFrame, lots: pd.DataFrame) -> pd.DataFrame:
    y = m.origin.dt.year
    des, des_s = y.between(2016, 2023), y.between(2016, 2023) & (m.summer == 1)
    rows = []
    for c in CANDS:
        rows.append({"calm_model": c, "design_all_MAPE": mape(m[des], c), "design_summer_MAPE": mape(m[des_s], c)})
    sel = pd.DataFrame(rows)
    bestA = sel.loc[sel.design_all_MAPE.idxmin(), "calm_model"]
    bestB = sel.loc[sel.design_summer_MAPE.idxmin(), "calm_model"]
    # G_K with each candidate on 2021~2026
    s1 = {o: sg.latest_signal(sig, o)["S1"] for o in m.origin}
    m = m.copy()
    m["S1"] = m.origin.map(s1)
    calmK = (m.S1 > 0) & (m.summer == 1)
    val = m.origin.dt.year.between(2021, 2025)
    out = []
    for c in CANDS:
        g = np.where(calmK, m[c], m[HEAD])
        d = m.assign(G=g)
        e = lambda o: float((d[d.origin == o].G.iloc[0] / d[d.origin == o].actual.iloc[0] - 1) * 100)
        out.append({"calm_model": c, "selected_by": ("A " if c == bestA else "") + ("B" if c == bestB else ""),
                    "design_all_MAPE": sel.set_index("calm_model").loc[c, "design_all_MAPE"],
                    "design_summer_MAPE": sel.set_index("calm_model").loc[c, "design_summer_MAPE"],
                    "GK_MAPE_2021_25": mape(d[val], "G"), "GK_summer_MAPE_2021_25": mape(d[val & (d.summer == 1)], "G"),
                    "GK_2024_08": e("2024-08-31"), "GK_2025_08": e("2025-08-31"), "GK_2026_08": e("2026-08-31"),
                    "GK_2021_08": e("2021-08-31"), "GK_2023_08": e("2023-08-31")})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 3
def check3(sig: pd.DataFrame, lots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.read_csv(MAIN_OUT / "anomaly_predictions_W.csv", parse_dates=["origin"])
    b = pd.read_csv(MAIN_OUT / "backtest_predictions_W.csv", parse_dates=["origin"])
    w = b.merge(a[["origin", HEAD]], on="origin")
    w = w[w.origin.dt.year >= 2017].copy()
    w["S1"] = [sg.latest_signal(sig, o)["S1"] if s == 1 else np.nan for o, s in zip(w.origin, w.summer)]
    w["S2"] = [c5.stock_at(lots, o)["stock_est"] for o in w.origin]
    hasK = w.S1.notna()
    calmK, calmS = w.S1 > 0, (w.S2 >= sg.STOCK_T) & (w.summer == 1)
    w["G_K"] = np.where(hasK & calmK, w.price_only_ridge, w[HEAD])
    w["G_S"] = np.where(calmS, w.price_only_ridge, w[HEAD])
    w["G_both"] = np.where(hasK & calmK & calmS, w.price_only_ridge, w[HEAD])
    cols = ["price_only_ridge", "chronos2", "wx_hgb", HEAD, "G_K", "G_S", "G_both"]
    y = w.origin.dt.year
    met = []
    for name, mk in {"W_all_2021_2025": y.between(2021, 2025), "W_summer_2021_2025": y.between(2021, 2025) & (w.summer == 1),
                     "W_summer_2021_2026": y.between(2021, 2026) & (w.summer == 1)}.items():
        for c in cols:
            s = w[mk]
            e = s[c] - s.actual
            met.append({"subset": name, "model": c, "n": len(s), "MAE": e.abs().mean(), "MAPE%": (e.abs() / s.actual).mean() * 100, "bias": e.mean()})
    trk = w[(w.origin.dt.month.isin([8, 9, 10])) & (w.origin.dt.year.isin([2024, 2025, 2026]))].copy()
    for c in ["price_only_ridge", HEAD, "G_K", "G_S"]:
        trk[c + "%"] = err(trk, c).round(0)
    return pd.DataFrame(met), trk[["origin", "actual", "S1", "S2", "price_only_ridge%", HEAD + "%", "G_K%", "G_S%"]]


if __name__ == "__main__":
    pd.set_option("display.width", 300)
    m = load_monthly()
    lots = c5.load_lots()
    sig = sg.krei_summer_signal()
    c1 = check1(m, lots); c1.to_csv(OUT / "check1_2017_2020.csv", index=False)
    print("=== 조건 1: 2017~2020 여름, 잔량 게이트만 ===\n", c1.to_string(index=False))
    c2 = check2(m, sig, lots); c2.to_csv(OUT / "check2_calm_selection.csv", index=False, float_format="%.2f")
    print("\n=== 조건 2: 평시 모형 분리 선택(2016~2023) → G_K 검증(2021~2026) ===\n", c2.round(1).to_string(index=False))
    met, trk = check3(sig, lots); met.to_csv(OUT / "check3_weekly_metrics.csv", index=False, float_format="%.2f"); trk.to_csv(OUT / "check3_weekly_2024_2026.csv", index=False, float_format="%.1f")
    print("\n=== 조건 3: 주간 원점 지표 ===\n", met.round(1).to_string(index=False))
    print("\n=== 조건 3: 주간 추적 2024~2026 (8~10월 원점) ===\n", trk.round(0).to_string(index=False))
