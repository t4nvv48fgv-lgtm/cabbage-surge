"""Chronos-2 분위수 전환 트리거 (정부 비축 방출 소진 신호).

아이디어: 가격 이력만 보는 Chronos-2 제로샷은 2024-09 급등을 중위수로 −18% 과소하고, 0.9 분위수는 +6% 였다.
"가격이 오르는데 정부 방출이 급감한다"(비축 여력 소진 → 평균회귀 붕괴)를 원점 당일까지의 정보로 판정해,
그 심각도에 따라 점예측으로 쓰는 분위수 수준을 0.5 → 0.9 로 **연속적으로** 올린다.

트리거 입력 (모두 원점 d 까지의 정보)
  rel7   = 최근 7일 정부 방출 합 (aT at_gov_daily, 2021~)
  prior  = 그 직전 2주의 주평균 방출 (d-21 ~ d-8)
  dlp7   = 7일 전 대비 로그가격 변화 (lp7 - lp7.shift(7))
심각도 (0~1)
  act  = clip(prior / 300, 0, 1)          방출이 활발했는가 (300 t/주 이상이면 1)
  drop = clip((1 - rel7/prior) / 0.5, 0, 1) 방출이 얼마나 급감했는가 (50% 이상 감소면 1)
  up   = clip(dlp7 / 0.10, 0, 1)          가격이 오르고 있는가 (7일 +10% 이상이면 1)
  sev  = act * drop * up
분위수 수준  q = 0.5 + 0.4 * sev  (저장된 0.1~0.9 분위수를 선형보간)
이진 변형    prior>300 & rel7/prior<0.6 & dlp7>0 이면 0.9, 아니면 0.5  (2026-10-07 1차 검증 규칙)

주의: 규칙 형태와 상수는 2024-08 사례를 본 뒤 정했다(사후 설계, risk-audit R-1). 발동 사례 수와 임계 민감도를 함께 보고한다.
aT 자료는 2021~ 이므로 그 이전 원점은 sev=0 (= 중위수).

Usage: python experiments/case01_release_trigger/trigger_quantile.py [M|W]   (메인 outputs/backtest_predictions{tag}.csv 필요)
Outputs: experiments/case01_release_trigger/outputs/trigger_quantile_predictions{tag}.csv, _metrics{tag}.csv, _sensitivity{tag}.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# 독립 케이스(experiments/case01_release_trigger). 메인 src/는 읽기만 하고 수정하지 않는다.
CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ts_baselines  # noqa: E402  (src/ts_baselines.py 재사용)

warnings.filterwarnings("ignore")
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MAIN_OUT = ROOT / "outputs"            # 메인 backtest_predictions{tag}.csv 의 원점 집합을 읽음
OUT = CASE_DIR / "outputs"             # 케이스 산출물은 여기에만 쓴다
QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FIRST_ORIGIN = "2021-01-31"          # aT 방출 자료 시작 이후만
ACT_T, DROP_T, UP_T = 300.0, 0.5, 0.10


def load_release() -> pd.Series:
    a = pd.read_csv(RAW / "at_gov_daily.csv", parse_dates=["date"]).set_index("date")["gov_release"]
    cal = pd.date_range(a.index.min(), a.index.max())
    return a.reindex(cal).fillna(0.0)


def trigger_inputs(rel: pd.Series, feats: pd.DataFrame, o: pd.Timestamp) -> dict:
    if o not in rel.index:
        return {"rel7": np.nan, "prior": np.nan, "ratio": np.nan, "dlp7": float(feats.loc[o, "dlp7"])}
    rel7 = float(rel.loc[o - pd.Timedelta(days=6): o].sum())
    prior = float(rel.loc[o - pd.Timedelta(days=20): o - pd.Timedelta(days=7)].sum() / 2.0)
    ratio = rel7 / prior if prior > 0 else np.nan
    return {"rel7": rel7, "prior": prior, "ratio": ratio, "dlp7": float(feats.loc[o, "dlp7"])}


def severity(t: dict, act_t=ACT_T, drop_t=DROP_T, up_t=UP_T) -> float:
    if not np.isfinite(t["prior"]) or t["prior"] <= 0:
        return 0.0
    act = min(max(t["prior"] / act_t, 0.0), 1.0)
    drop = min(max((1.0 - t["ratio"]) / drop_t, 0.0), 1.0)
    up = min(max(t["dlp7"] / up_t, 0.0), 1.0)
    return float(act * drop * up)


def binary(t: dict) -> int:
    return int(np.isfinite(t["prior"]) and t["prior"] > 300 and np.isfinite(t["ratio"]) and t["ratio"] < 0.6 and t["dlp7"] > 0)


def chronos_quantiles(o: pd.Timestamp, freq: str) -> np.ndarray:
    import torch
    s, h = ts_baselines.history(o, freq)
    ctx = torch.tensor(s.values, dtype=torch.float32).reshape(1, 1, -1)
    q, _ = ts_baselines._chronos().predict_quantiles(ctx, prediction_length=h, quantile_levels=QL)
    q = np.asarray(q[0] if isinstance(q, list) else q)[0]       # (h, n_q)
    return q.mean(axis=0)                                         # 향후 h스텝 평균의 분위수(근사)


def interp_q(qv: np.ndarray, level: float) -> float:
    return float(np.interp(level, QL, qv))


def run(freq: str) -> pd.DataFrame:
    feats = pd.read_csv(PROC / "daily_features.csv", parse_dates=["date"]).set_index("date")
    rel = load_release()
    tag = "" if freq == "M" else f"_{freq}"
    base = pd.read_csv(MAIN_OUT / f"backtest_predictions{tag}.csv", parse_dates=["origin"])
    base = base[base.origin >= FIRST_ORIGIN]
    rows = []
    for _, r in base.iterrows():
        o = r.origin
        qv = chronos_quantiles(o, freq)
        t = trigger_inputs(rel, feats, o)
        sev = severity(t)
        lvl = 0.5 + 0.4 * sev
        rec = {"origin": o, "actual": r.actual, "summer": r.summer, **t, "sev": sev, "q_level": lvl,
               "chronos2_q50": qv[QL.index(0.5)], "chronos2_q90": qv[QL.index(0.9)],
               "trig_cont": interp_q(qv, lvl),
               "trig_bin": qv[QL.index(0.9)] if binary(t) else qv[QL.index(0.5)],
               "trig_bin_on": binary(t)}
        for lv, v in zip(QL, qv):
            rec[f"q{int(lv*100)}"] = v
        rows.append(rec)
        print(f"{o.date()} actual={r.actual:7.0f} q50={rec['chronos2_q50']:7.0f} sev={sev:.2f} lvl={lvl:.2f} "
              f"cont={rec['trig_cont']:7.0f} bin={rec['trig_bin']:7.0f}", flush=True)
    return pd.DataFrame(rows)


def metrics(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = []
    for name, mk in {"all_2021_2025": d.origin.dt.year.between(2021, 2025),
                     "summer_origin_2021_2025": d.origin.dt.year.between(2021, 2025) & (d.summer == 1)}.items():
        s = d[mk]
        for c in cols:
            e = s[c] - s.actual
            out.append({"subset": name, "model": c, "n": len(s), "MAE": e.abs().mean(),
                        "MAPE%": (e.abs() / s.actual).mean() * 100, "bias": e.mean()})
    return pd.DataFrame(out)


def sensitivity(d: pd.DataFrame) -> pd.DataFrame:
    """상수(act_t, drop_t, up_t)를 바꿔 발동 수·MAPE·2024-08 오차가 어떻게 움직이는지."""
    rows = []
    # 월말 원점이면 2024-08-31 / 2025-08-31, 주간 원점이면 그 직후 일요일(2024-09-01 / 2025-08-31)
    o24 = d.origin == d.origin[d.origin >= "2024-08-31"].min()
    o25 = d.origin == d.origin[d.origin >= "2025-08-31"].min()
    for act_t in (150, 300, 600):
        for drop_t in (0.3, 0.5, 0.7):
            for up_t in (0.05, 0.10, 0.20):
                sev = d.apply(lambda r: severity(r.to_dict(), act_t, drop_t, up_t), axis=1)
                lvl = 0.5 + 0.4 * sev
                pred = np.array([interp_q(r[[f"q{int(l*100)}" for l in QL]].values.astype(float), lv)
                                 for (_, r), lv in zip(d.iterrows(), lvl)])
                e = pred - d.actual
                rows.append({"act_t": act_t, "drop_t": drop_t, "up_t": up_t,
                             "n_sev>0.2": int((sev > 0.2).sum()), "n_sev>0.5": int((sev > 0.5).sum()),
                             "MAPE%": (e.abs() / d.actual).mean() * 100,
                             "err_2024_08%": float((pred[o24.values][0] / d.actual[o24].iloc[0] - 1) * 100),
                             "err_2025_08%": float((pred[o25.values][0] / d.actual[o25].iloc[0] - 1) * 100)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    freq = sys.argv[1] if len(sys.argv) > 1 else "M"
    tag = "" if freq == "M" else f"_{freq}"
    d = run(freq)
    d.to_csv(OUT / f"trigger_quantile_predictions{tag}.csv", index=False, float_format="%.3f")
    cols = ["chronos2_q50", "trig_cont", "trig_bin", "chronos2_q90"]
    met = metrics(d, cols)
    met.to_csv(OUT / f"trigger_quantile_metrics{tag}.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 250)
    print("\n=== METRICS ===")
    print(met.round(1).to_string(index=False))
    print(f"\n발동: sev>0 {int((d.sev > 0).sum())}건, sev>0.2 {int((d.sev > 0.2).sum())}건, sev>0.5 {int((d.sev > 0.5).sum())}건, 이진 {int(d.trig_bin_on.sum())}건")
    on = d[d.sev > 0].copy()
    for c in cols:
        on[f"{c}%"] = (on[c] / on.actual - 1) * 100
    print("\n=== sev>0 원점 ===")
    print(on[["origin", "actual", "prior", "ratio", "dlp7", "sev", "q_level"] + [f"{c}%" for c in cols]].round(2).to_string(index=False))
    keys = pd.to_datetime(["2021-08-31", "2022-08-31", "2023-08-31", "2024-07-31", "2024-08-31", "2024-09-30", "2024-10-31", "2025-08-31"])
    sc = d[d.origin.isin(keys)].copy()
    for c in cols:
        sc[f"{c}%"] = (sc[c] / sc.actual - 1) * 100
    print("\n=== 주요 원점 ===")
    print(sc[["origin", "actual", "sev", "q_level"] + [f"{c}%" for c in cols]].round(2).to_string(index=False))
    sens = sensitivity(d)
    sens.to_csv(OUT / f"trigger_quantile_sensitivity{tag}.csv", index=False, float_format="%.2f")
    print("\n=== 상수 민감도 ===")
    print(sens.round(1).to_string(index=False))
