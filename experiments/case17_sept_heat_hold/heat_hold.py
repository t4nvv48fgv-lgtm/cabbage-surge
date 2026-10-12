"""Case 17 — 9월 가을배추 주산지 고온이 10월 하락을 늦추는가 (10월 대상 예측의 잔차 조정).

계기(2026-10-12): 최종 모형의 30% 초과 달 2024-10(−34%)과 20% 초과 달 2023-10(−23%)은 둘 다 9월이 더웠던 해다.
가을배추 주산지 9곳(build_features.LOWLAND_AUTUMN)의 9월 30°C 이상 일수 H 가 크면 가을배추 정식·활착이 늦어져 10월 출하가 밀리고,
평년처럼 내린다고 본 모형이 10월을 과소 예측한다는 가설.

신호(원점 o, 대상월 = (o+15일)의 달 = 10월인 원점에만):
  H(o)  = 그해 9월 1일 ~ min(o, 9월 30일) 의 30°C 이상 일수(9곳 평균 지시자의 합). 9/30 원점에서는 daily_features 의 L_hot30_sum30 과 같다.
  F(o)  = 원점에서 알 수 있는 최신호 가을배추 생산 전망 전년비(없으면 면적), 규약 B. 메인 gate_model 의 로더 사용(2021~).
조정:  pred' = min( final × exp(γ × H),  max(final, 최근 7일 가격) )      # 하락을 늦출 뿐 최근 가격 위로 올리지는 않는다(사전 상한)
게이트: F > 0(가을 공급 증가 전망)이면 조정하지 않는다. F 가 없으면(2021 이전) 연다.

γ 를 정하는 네 방식(잔차 회귀 log(실제/final) ~ γ·H, 절편 없음, 음수면 0; 10월 대상 주간 원점으로 추정):
  prereg   2016~2020 에서 한 번 추정해 고정 → 2021~ 검증 (case16 과 같은 절차)
  expand   해 y 의 γ 는 2016~(y−1) 원점만으로 추정(확장 창, 실시간 모사)
  posthoc  2021~2025 에서 추정(표본내 상한, 근거로 쓰지 않음)
  각 방식에 게이트 유/무.
부가: (A) 해별 신호표 2011~2026, (B) 구조 회귀 log(10월/9월) ~ 가격 편차 + H (2011~2025, n=15, LOYO).

Usage: python experiments/case17_sept_heat_hold/heat_hold.py
Outputs: outputs/heat_signal_by_year.csv, structure_loyo.csv, heat_hold_predictions.csv, heat_hold_predictions_W.csv, heat_hold_metrics.csv, gamma.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
OUT = CASE_DIR / "outputs"
sys.path.insert(0, str(ROOT / "src"))
import build_features as bf  # noqa: E402
import gate_model as gm  # noqa: E402

TARGET_MONTH = 10
VARIANTS = ["prereg", "prereg_gate", "expand", "expand_gate", "posthoc", "posthoc_gate", "prereg_gate_sep", "expand_gate_sep"]


def lowland_daily() -> pd.DataFrame:
    L = bf.region_daily(bf.load_weather(), bf.LOWLAND_AUTUMN)
    L = L.reindex(pd.date_range(L.index.min(), L.index.max(), freq="D"))
    return bf.add_anomalies(L, ["tmax"], "l_")


def sept_signal(L: pd.DataFrame, o: pd.Timestamp) -> tuple[float, float]:
    a, b = pd.Timestamp(o.year, 9, 1), min(o, pd.Timestamp(o.year, 9, 30))
    if b < a:
        return 0.0, np.nan
    s = L.loc[a:b]
    return float(s.hot30.sum()), float(s.l_tmax_anom.mean())


def load(freq: str, df: pd.DataFrame, L: pd.DataFrame, fall: pd.DataFrame) -> pd.DataFrame:
    tag = "" if freq == "M" else f"_{freq}"
    g = pd.read_csv(ROOT / "outputs" / f"gate_predictions{tag}.csv", parse_dates=["origin"], encoding="utf-8-sig")
    b = pd.read_csv(ROOT / "outputs" / f"backtest_predictions{tag}.csv", parse_dates=["origin"])
    d = g[["origin", "actual", "tmonth", "gate_calm", "S1", "price_only_ridge", "gate_hgb", "final"]].merge(b[["origin", "wx_ridge"]], on="origin")
    d["oct"] = (d.tmonth == TARGET_MONTH).astype(int)
    sig = [sept_signal(L, o) if k else (0.0, np.nan) for o, k in zip(d.origin, d.oct)]
    d["H"] = [h for h, _ in sig]
    d["tmax_anom_sep"] = [t for _, t in sig]
    d["F"] = [gm.adjust_signal(fall, o, o.year, "crop_year")[1] if k else np.nan for o, k in zip(d.origin, d.oct)]
    d["p7"] = np.exp(df.lp7.reindex(d.origin).values)
    d["resid"] = np.log(d.actual / d.final)
    return d


def gamma_of(s: pd.DataFrame) -> float:
    s = s[(s.oct == 1) & s.resid.notna()]
    den = float((s.H * s.H).sum())
    return max(0.0, float((s.H * s.resid).sum() / den)) if den > 0 else 0.0


def apply(d: pd.DataFrame, gam: pd.Series, gate: bool, sep_only: bool = False) -> np.ndarray:
    """gam: 원점 연도 → γ. 상한 = max(final, 최근 7일 가격)."""
    g = d.origin.dt.year.map(gam).fillna(0.0)
    on = (d.oct == 1) & ~((d.F > 0) if gate else pd.Series(False, index=d.index))
    if sep_only:
        on &= d.origin.dt.month == 9
    raw = d.final * np.exp(g * d.H)
    cap = np.maximum(d.final, d.p7)
    return np.where(on, np.minimum(raw, cap), d.final)


def mape(s: pd.DataFrame, c: str) -> float:
    return float(((s[c] - s.actual).abs() / s.actual).mean() * 100) if len(s) else np.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 320)
    df = pd.read_csv(ROOT / "data" / "processed" / "daily_features.csv", parse_dates=["date"]).set_index("date")
    L = lowland_daily()
    fall = gm.load_fall_issues()

    # ---------------- (A) 해별 신호표: 9/30 원점
    rows = []
    for y in range(2011, 2027):
        o = pd.Timestamp(y, 9, 30)
        if o not in df.index:
            continue
        r = df.loc[o]
        h, t = sept_signal(L, o)
        prev = [df.lp28.get(pd.Timestamp(yy, 9, 30), np.nan) for yy in range(2010, y)]
        rows.append({"year": y, "H_sep": round(h, 1), "L_hot30_sum30": round(float(r.L_hot30_sum30), 1), "tmax_anom_sep": round(t, 2),
                     "sep_price": round(float(np.exp(r.lp28))), "oct_price": round(float(np.exp(r.y_lp_fwd28))) if pd.notna(r.y_lp_fwd28) else np.nan,
                     "log_ratio": float(r.y_lp_fwd28 - r.lp28) if pd.notna(r.y_lp_fwd28) else np.nan,
                     "price_anom": float(r.lp28 - np.nanmean(prev)), "fall_forecast": gm.adjust_signal(fall, o, y, "crop_year")[1]})
    A = pd.DataFrame(rows)
    A["oct_over_sep"] = np.exp(A.log_ratio).round(2)
    A.to_csv(OUT / "heat_signal_by_year.csv", index=False, float_format="%.3f", encoding="utf-8-sig")
    print("=== (A) 9/30 원점 해별 신호 ===")
    print(A[["year", "H_sep", "L_hot30_sum30", "tmax_anom_sep", "sep_price", "oct_price", "oct_over_sep", "price_anom", "fall_forecast"]].round(2).to_string(index=False))

    # ---------------- (B) 구조 회귀 (2011~2025): log(10월/9월) ~ 1 + 가격 편차 (+ H)
    S = A.dropna(subset=["log_ratio"]).reset_index(drop=True)
    def ols(X, y):
        Xd = np.column_stack([np.ones(len(X))] + [X[c].values for c in X.columns])
        beta, *_ = np.linalg.lstsq(Xd, y.values, rcond=None)
        return beta, Xd
    res = {}
    for name, cols in (("price_only", ["price_anom"]), ("price+H", ["price_anom", "H_sep"])):
        beta, Xd = ols(S[cols], S.log_ratio)
        fit = Xd @ beta
        r2 = 1 - ((S.log_ratio - fit) ** 2).sum() / ((S.log_ratio - S.log_ratio.mean()) ** 2).sum()
        loyo = []
        for i in range(len(S)):
            tr = S.drop(index=i)
            bb, _ = ols(tr[cols], tr.log_ratio)
            loyo.append(float(np.r_[1.0, S.loc[i, cols].values.astype(float)] @ bb))
        S[f"loyo_{name}"] = loyo
        res[name] = {"coef": dict(zip(["const"] + cols, np.round(beta, 3))), "R2": round(float(r2), 2),
                     "LOYO_MAE(log)": round(float((S.log_ratio - S[f"loyo_{name}"]).abs().mean()), 3)}
    S.to_csv(OUT / "structure_loyo.csv", index=False, float_format="%.3f", encoding="utf-8-sig")
    print(f"\n=== (B) 구조 회귀 log(10월/9월), n={len(S)} ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
    S["e_price_only"] = S.log_ratio - S["loyo_price_only"]
    S["e_price+H"] = S.log_ratio - S["loyo_price+H"]
    print(S[["year", "H_sep", "log_ratio", "loyo_price_only", "loyo_price+H", "e_price_only", "e_price+H"]].round(2).to_string(index=False))

    # ---------------- (C) 잔차 조정 규칙
    M, W = load("M", df, L, fall), load("W", df, L, fall)
    yW = W.origin.dt.year
    gam = {"prereg": pd.Series({y: gamma_of(W[yW.between(2016, 2020)]) for y in range(2016, 2027)}),
           "expand": pd.Series({y: gamma_of(W[(yW >= 2016) & (yW < y)]) for y in range(2016, 2027)}),
           "posthoc": pd.Series({y: gamma_of(W[yW.between(2021, 2025)]) for y in range(2016, 2027)})}
    G = pd.DataFrame(gam).round(4)
    G.index.name = "year"
    G.to_csv(OUT / "gamma.csv", encoding="utf-8-sig")
    print("\n=== (C) γ (30°C 이상 하루당 로그 조정) ===")
    print(G.to_string())
    c = W[(W.oct == 1) & W.resid.notna()].assign(year=yW, HR=lambda x: x.H * x.resid, HH=lambda x: x.H * x.H).groupby("year").agg(n=("H", "size"), H=("H", "mean"), resid=("resid", "mean"), HR=("HR", "sum"), HH=("HH", "sum")).round(3)
    c.to_csv(OUT / "gamma_contrib_by_year.csv", encoding="utf-8-sig")
    print("\n해별 기여 (γ = ΣHR / ΣHH, 10월 대상 주간 원점):")
    print(c.to_string())
    for d in (M, W):
        for k in gam:
            d[k] = apply(d, gam[k], gate=False)
            d[k + "_gate"] = apply(d, gam[k], gate=True)
        for k in ("prereg", "expand"):
            d[k + "_gate_sep"] = apply(d, gam[k], gate=True, sep_only=True)
    M.to_csv(OUT / "heat_hold_predictions.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    W.to_csv(OUT / "heat_hold_predictions_W.csv", index=False, float_format="%.2f", encoding="utf-8-sig")

    cols = ["final"] + VARIANTS
    met = []
    for freq, d in (("M", M), ("W", W)):
        y = d.origin.dt.year
        masks = {"oct_2016_2020": y.between(2016, 2020) & (d.oct == 1), "oct_2021_2025": y.between(2021, 2025) & (d.oct == 1),
                 "all_2016_2020": y.between(2016, 2020), "all_2021_2025": y.between(2021, 2025), "all_2021_2026": y.between(2021, 2026)}
        for nm, mk in masks.items():
            for c in cols:
                s = d[mk]
                e = (s[c] / s.actual - 1) * 100
                met.append({"freq": freq, "subset": nm, "model": c, "n": len(s), "MAPE%": mape(s, c), "over20": int((e.abs() > 20).sum()), "over30": int((e.abs() > 30).sum())})
    met = pd.DataFrame(met)
    met.to_csv(OUT / "heat_hold_metrics.csv", index=False, float_format="%.2f")
    for freq in ("M", "W"):
        print(f"\n=== MAPE [{freq}] ===")
        print(met[met.freq == freq].pivot(index="model", columns="subset", values="MAPE%").loc[cols].round(2).to_string())
    print("\n=== 30% 초과 개수 [M, 대상월 2021-01~] ===")
    rng = M.origin >= "2020-12-31"
    print({c: int((((M.loc[rng, c] / M.loc[rng, "actual"]) - 1).abs() > 0.30).sum()) for c in cols},
          "| 20% 초과", {c: int((((M.loc[rng, c] / M.loc[rng, "actual"]) - 1).abs() > 0.20).sum()) for c in cols})
    k = M[M.oct == 1].copy()
    for c in cols + ["wx_ridge", "price_only_ridge"]:
        k[c + "%"] = ((k[c] / k.actual - 1) * 100).round(0)
    print("\n=== 9월 말 원점 → 10월 (월말) ===")
    print(k[["origin", "actual", "H", "F", "p7"] + [c + "%" for c in cols + ["wx_ridge", "price_only_ridge"]]].round(1).to_string(index=False))
    kw = W[(W.oct == 1) & (W.origin.dt.year >= 2023)].copy()
    for c in ["final", "prereg_gate", "expand_gate", "posthoc_gate", "expand_gate_sep"]:
        kw[c + "%"] = ((kw[c] / kw.actual - 1) * 100).round(0)
    print("\n=== 10월 대상 주간 원점 2023~ ===")
    print(kw[["origin", "actual", "H", "F", "final%", "prereg_gate%", "expand_gate%", "posthoc_gate%", "expand_gate_sep%"]].round(1).to_string(index=False))
    # 2026 현재 신호
    o26 = pd.Timestamp(2026, 9, 30)
    if o26 in df.index:
        h, t = sept_signal(L, o26)
        print(f"\n[2026-09-30 실시간] H={h:.1f}일, 9월 최고기온 편차 {t:+.2f}°C, 가을 전망 {gm.adjust_signal(fall, o26, 2026, 'crop_year')}, γ(expand)={gam['expand'][2026]:.4f} → 조정 {np.exp(gam['expand'][2026] * h) - 1:+.1%} (상한 적용 전)")


if __name__ == "__main__":
    main()
