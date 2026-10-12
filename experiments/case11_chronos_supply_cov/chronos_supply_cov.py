"""Case 11 — Chronos-2 제로샷에 공급 신호를 공변량으로 넣으면 효과가 있는가.

질문(사용자, 2026-10-12): "크로노스2에 공급신호를 넣었을 때 효과를 볼 수 있나?"
배경: 메인 최종 모형은 공급 신호(KREI 월보 생산 전망 S1)를 '스위치'로 써서 모형을 통째로 바꾼다. Chronos-2는 공변량을 받을 수 있으므로
같은 신호를 조건화 입력으로 주면 급등 추종·거짓 경보가 어떻게 되는지 본다. case02(기상·반입·정부매입 과거 공변량)는 효과가 없었다.

입력 (모두 원점까지의 정보, 시계열 인덱스 = 월초(MS, h=1) 또는 일요일(W-SUN, h=4), 메인 ts_baselines.history 와 동일)
  S1       KREI 월보 최신호 여름배추 생산량 전망 전년비(%). 각 기간 말 시점에 알려진 값(src/gate_model.latest_signal). 없으면 0.
  s1_avail 그 값이 있었는지(1/0). 2021년 이전·1~2월은 0.
  alarm    여름(6~9월) & S1 ≤ 0 이면 1 (메인 게이트의 경보 상태 그대로), 아니면 0.
  S2       정부비축 잔량 추정(천 톤, case05 stock_at). 2017년 이전 0, s2_avail 0.
미래 공변량: S1·s1_avail·alarm 은 원점 시점 최신 값을 예측 구간에 그대로 이월(원점에서 알려진 값). S2 는 과거 공변량만.

변형 (실행 전 고정)
  V0 chronos2          메인 제로샷(비교, outputs/backtest_predictions 에서 읽음)
  V1 cov_s1            + S1, s1_avail (과거+미래)
  V2 cov_s1_s2         + S1, s1_avail, S2, s2_avail
  V3 cov_alarm         + alarm 만 (과거+미래)
  V4 cov_s1_summer     + S1 을 여름 기간에만(그 외 0), s1_avail 도 여름에만
비교: 최종 게이트 gate_hgb(outputs/gate_predictions), 대표 an_sparse_all. 각 변형의 q50 과 q90 저장.
기대(실행 전 기록): 제로샷 공변량 효과는 문맥 창 안에서 즉석 학습되는데 S1 은 2021~(급등 2회)뿐이라 효과가 작을 것. 거짓 경보(2025) 쪽은 줄 수 있음.

Usage: python experiments/case11_chronos_supply_cov/chronos_supply_cov.py [M|W]
Outputs: experiments/case11_chronos_supply_cov/outputs/cov_predictions{tag}.csv, cov_metrics{tag}.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "case05_gov_stock_krei"))
import gate_model as gm  # noqa: E402  latest_signal, load_issues
import gov_stock_krei as c5  # noqa: E402  stock_at
import ts_baselines  # noqa: E402

warnings.filterwarnings("ignore")
OUT = CASE_DIR / "outputs"
MAIN_OUT = ROOT / "outputs"
QL = [0.1, 0.5, 0.9]
FIRST_ORIGIN = "2016-01-31"
VARIANTS = {"cov_s1": (["S1", "s1_avail"], ["S1", "s1_avail"]),
            "cov_s1_s2": (["S1", "s1_avail", "S2", "s2_avail"], ["S1", "s1_avail"]),
            "cov_alarm": (["alarm"], ["alarm"]),
            "cov_s1_summer": (["S1_summer", "s1_avail_summer"], ["S1_summer", "s1_avail_summer"])}


def signal_table(dates: pd.DatetimeIndex, issues: pd.DataFrame, lots: pd.DataFrame) -> pd.DataFrame:
    """기간 말 날짜별 신호 값."""
    rows = []
    for d in dates:
        _, s1 = gm.latest_signal(issues, d)
        avail = float(np.isfinite(s1))
        s1v = float(s1) if avail else 0.0
        summer = float(d.month in (6, 7, 8, 9))
        st = c5.stock_at(lots, d)["stock_est"] if d.year >= 2017 else np.nan
        rows.append({"date": d, "S1": s1v, "s1_avail": avail, "alarm": float(summer and avail and s1v <= 0),
                     "S1_summer": s1v * summer, "s1_avail_summer": avail * summer,
                     "S2": (st / 1000.0) if np.isfinite(st) else 0.0, "s2_avail": float(np.isfinite(st))})
    return pd.DataFrame(rows).set_index("date")


def period_ends(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([p.to_timestamp("M") for p in idx.to_period("M")]) if freq == "M" else idx


def predict(pipe, s: pd.Series, h: int, cov: pd.DataFrame, past_keys: list[str], fut_keys: list[str]) -> np.ndarray:
    import torch
    past = {k: torch.tensor(cov[k].values, dtype=torch.float32) for k in past_keys}
    fut = {k: torch.full((h,), float(cov[k].iloc[-1]), dtype=torch.float32) for k in fut_keys}
    inp = [{"target": torch.tensor(s.values, dtype=torch.float32), "past_covariates": past, "future_covariates": fut}]
    q, _ = pipe.predict_quantiles(inp, prediction_length=h, quantile_levels=QL)
    q = np.asarray(q[0])[0]            # (h, n_q)
    return q.mean(axis=0)


def run(freq: str) -> pd.DataFrame:
    tag = "" if freq == "M" else f"_{freq}"
    b = pd.read_csv(MAIN_OUT / f"backtest_predictions{tag}.csv", parse_dates=["origin"])
    g = pd.read_csv(MAIN_OUT / f"gate_predictions{tag}.csv", parse_dates=["origin"], encoding="utf-8-sig")
    d = b[["origin", "actual", "summer", "price_only_ridge", "chronos2", "chronos2_q90"]].merge(
        g[["origin", "an_sparse_all", "gate_hgb", "S1", "gate_calm"]], on="origin")
    d = d[d.origin >= FIRST_ORIGIN].reset_index(drop=True)
    issues, lots = gm.load_issues(), c5.load_lots()
    full_s, _ = ts_baselines.history(d.origin.max(), freq)
    sig = signal_table(period_ends(full_s.index, freq), issues, lots)
    sig.index = full_s.index                      # 시계열 인덱스(월초/일요일)에 맞춤
    pipe = ts_baselines._chronos()
    rows = []
    for i, r in d.iterrows():
        s, h = ts_baselines.history(r.origin, freq)
        cov = sig.reindex(s.index).fillna(0.0)
        rec = {"origin": r.origin}
        for name, (pk, fk) in VARIANTS.items():
            qv = predict(pipe, s, h, cov, pk, fk)
            rec[name], rec[name + "_q90"] = float(qv[1]), float(qv[2])
        rows.append(rec)
        if i % 10 == 0:
            print(f"  {r.origin.date()} actual={r.actual:7.0f} chronos2={r.chronos2:7.0f} " + " ".join(f"{k}={rec[k]:7.0f}" for k in VARIANTS), flush=True)
    return d.merge(pd.DataFrame(rows), on="origin")


def metrics(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    y = d.origin.dt.year
    masks = {"all_2021_2025": y.between(2021, 2025), "summer_origin_2021_2025": y.between(2021, 2025) & (d.summer == 1),
             "summer_origin_2021_2026": y.between(2021, 2026) & (d.summer == 1), "all_2016_2020": y.between(2016, 2020)}
    out = []
    for name, mk in masks.items():
        s = d[mk]
        for c in cols:
            e = s[c] - s.actual
            out.append({"subset": name, "model": c, "n": len(s), "MAE": e.abs().mean(), "MAPE%": (e.abs() / s.actual).mean() * 100, "bias": e.mean()})
    return pd.DataFrame(out)


if __name__ == "__main__":
    freq = sys.argv[1] if len(sys.argv) > 1 else "M"
    tag = "" if freq == "M" else f"_{freq}"
    OUT.mkdir(parents=True, exist_ok=True)
    d = run(freq)
    cols = ["price_only_ridge", "chronos2", "an_sparse_all", "gate_hgb"] + list(VARIANTS) + [v + "_q90" for v in VARIANTS] + ["chronos2_q90"]
    d.to_csv(OUT / f"cov_predictions{tag}.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    met = metrics(d, cols)
    met.to_csv(OUT / f"cov_metrics{tag}.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 300)
    print("\n=== MAPE ===")
    print(met.pivot(index="model", columns="subset", values="MAPE%").loc[cols].round(1).to_string())
    if freq == "M":
        keys = pd.to_datetime([f"{y}-0{m}-{dd}" for y in range(2020, 2027) for m, dd in ((8, "31"), (9, "30"))])
    else:
        keys = d.origin[(d.origin.dt.year >= 2020) & d.origin.dt.month.isin([8, 9]) & ((d.origin.dt.day <= 7) | (d.origin.dt.day >= 25))]
    sc = d[d.origin.isin(keys)].copy()
    for c in cols:
        sc[c + "%"] = ((sc[c] / sc.actual - 1) * 100).round(0)
    print("\n=== 주요 원점 오차% ===")
    print(sc[["origin", "actual", "S1", "gate_calm"] + [c + "%" for c in cols]].to_string(index=False))
