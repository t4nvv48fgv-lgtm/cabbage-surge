"""Case 06 확장 — 봄배추 저장 소진율(S4)을 세 번째 게이트 신호로 (case07 결론 반영, 2026-10-10).

S4 = (7/1 ~ 원점) 전국 34개 도매시장 저장 방출(사용자 정의: 강원 산지·중국산 수입 제외 전부) ÷ 그해 봄배추 저장량(월보 7월호·김치협회)
  저장량: 2021 59,000 / 2022 25,000 / 2023 35,450(역산) / 2024 39,900(역산) / 2025 42,672 / 2026 46,983 t. 2020 이전은 없음 → S4 없음.
  적용 원점: 8/15 이후(월말은 8/31·9/30, 주간은 8/15 이후 일요일). 그 전에는 소진이 자연히 낮아 의미 없음.
규칙(실행 전 고정. 임계 40%는 2023 40.5%가 경계임을 알고 정한 사후 값)
  risk_K = S1 < 0  (KREI 최신호 여름배추 생산 전망 전년비 음수)
  risk_D = S4 >= 0.40
  G_D     : risk_D 면 대표, 아니면 평시(가격만 ridge)
  G_K_or_D: risk_K or risk_D 면 대표, 아니면 평시
  G_K_and_D: risk_K and risk_D 면 대표, 아니면 평시
  (비교) G_K: case06 원 규칙. S4 없는 원점은 G_K 와 동일.
Usage: python experiments/case06_supply_gate/supply_gate_s4.py
Outputs: outputs/s4_signals.csv, s4_metrics.csv, s4_surge_table.csv, s4_weekly_2024_2026.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
sys.path.insert(0, str(CASE_DIR))
import supply_gate as sg  # noqa: E402

OUT = CASE_DIR / "outputs"
MAIN_OUT = ROOT / "outputs"
PANEL = ROOT / "data" / "ext" / "nongnet" / "nongnet_cabbage_panel_2011_2026.parquet"
LEVEL = {2021: 59000, 2022: 25000, 2023: 35450, 2024: 39900, 2025: 42672, 2026: 46983}
THR = 0.40
HEAD, CALM = sg.HEAD, sg.CALM


def storage_release_daily() -> pd.Series:
    d = pd.read_parquet(PANEL, columns=["date", "origin_sido", "variety_std", "qty_kg", "market"])
    imp = d.origin_sido.astype(str).str.contains("중국") | (d.variety_std == "수입")
    s = d[(d.date.dt.month >= 7) & ~d.origin_sido.astype(str).str.contains("강원") & ~imp]
    return s.groupby("date").qty_kg.sum().div(1000)


def depletion_at(rel: pd.Series, o: pd.Timestamp) -> float:
    if o.year not in LEVEL or o < pd.Timestamp(year=o.year, month=8, day=15) or o.month > 9:
        return np.nan
    start = pd.Timestamp(year=o.year, month=7, day=1)
    return float(rel[(rel.index >= start) & (rel.index <= o)].sum() / LEVEL[o.year])


def attach(df: pd.DataFrame, rel: pd.Series, sig: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["S1"] = [sg.latest_signal(sig, o)["S1"] if s == 1 else np.nan for o, s in zip(df.origin, df.summer)]
    df["S4"] = [depletion_at(rel, o) for o in df.origin]
    rk, rd = df.S1 < 0, df.S4 >= THR
    hasK, hasD = df.S1.notna(), df.S4.notna()
    df["G_K"] = np.where(hasK & ~rk, df[CALM], df[HEAD])
    df["G_D"] = np.where(hasD & ~rd, df[CALM], df[HEAD])
    df["G_K_or_D"] = np.where((hasK | hasD) & ~(rk.fillna(False) | rd.fillna(False)), df[CALM], df[HEAD])
    df["G_K_and_D"] = np.where(hasK & hasD & ~(rk & rd), df[CALM], np.where(hasK & ~hasD & ~rk, df[CALM], df[HEAD]))
    return df


def metrics(d: pd.DataFrame, cols: list[str], label: str) -> pd.DataFrame:
    y = d.origin.dt.year
    masks = {f"{label}_all_2021_2025": y.between(2021, 2025), f"{label}_summer_2021_2025": y.between(2021, 2025) & (d.summer == 1),
             f"{label}_summer_2021_2026": y.between(2021, 2026) & (d.summer == 1)}
    out = []
    for name, mk in masks.items():
        s = d[mk]
        for c in cols:
            e = s[c] - s.actual
            out.append({"subset": name, "model": c, "n": len(s), "MAE": e.abs().mean(), "MAPE%": (e.abs() / s.actual).mean() * 100, "bias": e.mean()})
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 300)
    rel = storage_release_daily()
    sig = sg.krei_summer_signal()
    cols = [CALM, HEAD, "G_K", "G_D", "G_K_or_D", "G_K_and_D"]
    # 월말
    a = pd.read_csv(MAIN_OUT / "anomaly_predictions.csv", parse_dates=["origin"]); b = pd.read_csv(MAIN_OUT / "backtest_predictions.csv", parse_dates=["origin"])
    m = attach(b[["origin", "actual", "summer", CALM]].merge(a[["origin", HEAD]], on="origin"), rel, sig)
    m[["origin", "actual", "S1", "S4"]].to_csv(OUT / "s4_signals.csv", index=False, float_format="%.3f")
    keys = pd.to_datetime([f"{y}-08-31" for y in range(2021, 2027)] + [f"{y}-09-30" for y in range(2021, 2027)])
    sc = m[m.origin.isin(keys)].copy()
    for c in cols:
        sc[c + "%"] = (sc[c] / sc.actual - 1) * 100
    sc = sc.sort_values("origin"); sc.to_csv(OUT / "s4_surge_table.csv", index=False, float_format="%.1f")
    print("=== 월말 8/31·9/30 원점: 신호와 오차% ===\n", sc[["origin", "actual", "S1", "S4"] + [c + "%" for c in cols]].round(2).to_string(index=False))
    met_m = metrics(m, cols, "M")
    # 주간
    aw = pd.read_csv(MAIN_OUT / "anomaly_predictions_W.csv", parse_dates=["origin"]); bw = pd.read_csv(MAIN_OUT / "backtest_predictions_W.csv", parse_dates=["origin"])
    w = attach(bw[["origin", "actual", "summer", CALM]].merge(aw[["origin", HEAD]], on="origin"), rel, sig)
    met_w = metrics(w, cols, "W")
    met = pd.concat([met_m, met_w]); met.to_csv(OUT / "s4_metrics.csv", index=False, float_format="%.2f")
    print("\n=== 지표 ===\n", met.round(1).to_string(index=False))
    trk = w[w.origin.dt.month.isin([8, 9]) & w.origin.dt.year.isin([2023, 2024, 2025, 2026])].copy()
    for c in cols:
        trk[c + "%"] = (trk[c] / trk.actual - 1) * 100
    trk[["origin", "actual", "S1", "S4"] + [c + "%" for c in cols]].round(2).to_csv(OUT / "s4_weekly_2024_2026.csv", index=False)
    print("\n=== 주간 8~9월 2023~2026 ===\n", trk[["origin", "actual", "S1", "S4"] + [c + "%" for c in cols]].round(1).to_string(index=False))
