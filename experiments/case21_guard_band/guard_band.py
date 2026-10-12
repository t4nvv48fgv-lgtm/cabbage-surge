"""Case 21 — 오차 40% 이상을 막는 안전 범위(가드 밴드): 예측이 '최근 7일 가격'과 '대상 기간 평년치' 둘 다를 벗어나면 그 경계로 되돌린다.

사용자 지시(2026-10-12): "딴건 몰라도 40% 이상은 무조건 잡아야겠다."
40% 이상 6개 달(2016-02~2026-09, 최종 모형 final)의 진단:
  2018-10 −43%  예측 3,271 < 최근 7일 5,867·평년 4,565   (원점 직전 급락을 추세로 읽어 둘 다 아래로)
  2018-11 +53%  예측 7,746 > 최근 7일 5,792·평년 5,023   (평년 대비 편차 +0.45 지속 + 평년 경로 상승 → 둘 다 위로)
  2022-11 +41%  예측 7,884 > 최근 7일 7,372·평년 6,657   (급등 뒤 급락 중, 28일 평균 기준점이 높음)
  2017-05 +74%  2017-07 −56%  2019-09 −48%             (실제가 최근 가격·평년치 모두에서 멀리 벗어남: 봄 풍작·장마 급등·태풍. 예측은 범위 안)
→ 앞의 셋은 예측이 두 기준을 모두 벗어난 달이다. 뒤의 셋은 어떤 모형도 30% 안에 못 든 달이라 이 규칙의 대상이 아니다.

규칙(달별 보정이 아니라 모든 원점에 같은 식):
  하한 lo = min(최근 7일 가격, 평년치) × (1 − k)      항상 적용
  상한 hi = max(최근 7일 가격, 평년치) × (1 + k)      **여름 원점(6~9월)에는 적용하지 않음** — 급등기에는 두 기준을 모두 넘는 예측이 정당(2022-09·2024-09)
  guard = clip(final, lo, hi)
  평년치 = anomaly_model 의 normal_fwd(직전 5년 같은 대상 창 평균), 최근 7일 가격 = exp(lp7). 둘 다 원점 당일까지의 정보.
여유폭 k ∈ {0, 5, 10, 15, 20%}. **선택 기준(사전)**: 2016~2020 월말 원점에서 40% 초과 수 → 30% 초과 수 → MAPE 순으로 최소. 2021~2026으로 검증.
참고 변형: 하한만(lower), 비여름 상한만(upper_ns), 여름 포함 양쪽(both_all).
주의: 여름 상한 제외는 대칭 규칙이 급등 달을 깎는 것(2024-09 −6 → −17%)을 본 뒤 넣었다.

Usage: python experiments/case21_guard_band/guard_band.py
Outputs: outputs/guard_metrics.csv, guard_predictions.csv, guard_predictions_W.csv, guard_big_months.csv, run.log
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
OUT = CASE_DIR / "outputs"
KS = [0.0, 0.05, 0.10, 0.15, 0.20]
MODES = ["both_ns", "lower", "upper_ns", "both_all"]


def load(freq: str) -> pd.DataFrame:
    tag = "" if freq == "M" else f"_{freq}"
    lp7 = pd.read_csv(ROOT / "data" / "processed" / "daily_features.csv", parse_dates=["date"], usecols=["date", "lp7"]).set_index("date").lp7
    g = pd.read_csv(ROOT / "outputs" / f"gate_predictions{tag}.csv", parse_dates=["origin"], encoding="utf-8-sig")
    d = g[["origin", "actual", "summer", "tmonth", "normal_fwd", "gate_hgb", "final"]].copy()
    d["p7"] = np.exp(lp7.reindex(d.origin).values)
    d["year"] = d.origin.dt.year
    d["tm"] = (d.origin + pd.Timedelta(days=15)).dt.strftime("%Y-%m")
    return d


def guard(d: pd.DataFrame, k: float, mode: str) -> np.ndarray:
    lo = np.minimum(d.p7, d.normal_fwd) * (1 - k)
    hi = np.maximum(d.p7, d.normal_fwd) * (1 + k)
    p = d.final.values.copy()
    if mode in ("both_ns", "lower", "both_all"):
        p = np.maximum(p, lo)
    if mode in ("both_ns", "upper_ns"):
        p = np.where(d.summer == 0, np.minimum(p, hi), p)
    if mode == "both_all":
        p = np.minimum(p, hi)
    return p


def stats(d: pd.DataFrame, p, lo: int, hi: int) -> dict:
    mk = d.year.between(lo, hi)
    e = (np.asarray(p)[mk] / d.actual[mk] - 1) * 100
    return {"MAPE": float(e.abs().mean()), "over40": int((e.abs() > 40).sum()), "over30": int((e.abs() > 30).sum()), "over20": int((e.abs() > 20).sum()), "n": int(mk.sum())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 320)
    M, W = load("M"), load("W")
    rows = []
    for freq, d in (("M", M), ("W", W)):
        for mode in ["none"] + MODES:
            for k in ([np.nan] if mode == "none" else KS):
                p = d.final.values if mode == "none" else guard(d, k, mode)
                r = {"freq": freq, "mode": mode, "k": k, "adjusted%": 0.0 if mode == "none" else float((p != d.final.values).mean() * 100)}
                for lo, hi in ((2016, 2020), (2021, 2025), (2021, 2026)):
                    r.update({f"{kk}_{lo}_{hi}": v for kk, v in stats(d, p, lo, hi).items() if kk != "n"})
                rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "guard_metrics.csv", index=False, float_format="%.2f")
    show = ["mode", "k", "adjusted%", "MAPE_2016_2020", "over40_2016_2020", "over30_2016_2020", "MAPE_2021_2025", "over40_2021_2025", "over30_2021_2025", "MAPE_2021_2026", "over40_2021_2026", "over30_2021_2026"]
    for freq in ("M", "W"):
        print(f"\n=== [{freq}] ===")
        print(R[R.freq == freq][show].round(2).to_string(index=False))

    # 사전 선택: both_ns, 2016~2020 월말에서 over40 → over30 → MAPE
    cand = R[(R.freq == "M") & (R["mode"] == "both_ns")].sort_values(["over40_2016_2020", "over30_2016_2020", "MAPE_2016_2020"])
    k_star = float(cand.iloc[0].k)
    print(f"\n사전 선택 여유폭 k* = {k_star:.0%} (2016~2020 월말: 40% 초과 {int(cand.iloc[0].over40_2016_2020)}, 30% 초과 {int(cand.iloc[0].over30_2016_2020)}, MAPE {cand.iloc[0].MAPE_2016_2020:.2f})")
    for d in (M, W):
        d["guard"] = guard(d, k_star, "both_ns")
        d["guard_k5"] = guard(d, 0.05, "both_ns")
        for c in ("final", "guard", "guard_k5"):
            d["e_" + c] = (d[c] / d.actual - 1) * 100
    M.to_csv(OUT / "guard_predictions.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    W.to_csv(OUT / "guard_predictions_W.csv", index=False, float_format="%.2f", encoding="utf-8-sig")

    big = M[(M.e_final.abs() > 30) | (M.e_guard.abs() > 30) | (M.e_guard_k5.abs() > 30)]
    big[["tm", "actual", "p7", "normal_fwd", "final", "guard", "e_final", "e_guard", "e_guard_k5"]].to_csv(OUT / "guard_big_months.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    print("\n=== 30% 초과 달 전후 (월말) ===")
    print(big[["tm", "actual", "p7", "normal_fwd", "final", "guard", "e_final", "e_guard", "e_guard_k5"]].round(0).to_string(index=False))
    for nm, c in (("k*", "guard"), ("k 5%", "guard_k5")):
        ch = M[M[c] != M.final]
        better, worse = int((ch["e_" + c].abs() < ch.e_final.abs()).sum()), int((ch["e_" + c].abs() > ch.e_final.abs()).sum())
        dlt = (ch["e_" + c].abs() - ch.e_final.abs())
        print(f"\n[{nm}] 조정된 월말 원점 {len(ch)}개 / {len(M)}: 개선 {better}, 악화 {worse} | 개선 폭 평균 {-dlt[dlt < 0].mean():.1f}%p, 악화 폭 평균 {dlt[dlt > 0].mean():.1f}%p, 5%p 넘게 악화 {int((dlt > 5).sum())}개")
        w5 = ch[dlt > 5]
        if len(w5):
            print("   5%p 넘게 악화:", ", ".join(f"{t} {a:+.0f}→{b:+.0f}" for t, a, b in zip(w5.tm, w5.e_final, w5["e_" + c])))
    key = M[M.tm.isin(["2020-09", "2022-09", "2024-09", "2025-09", "2026-09", "2024-10"])]
    print("\n급등·거짓 경보 점검 달(변화 없어야 함):")
    print(key[["tm", "e_final", "e_guard", "e_guard_k5"]].round(1).to_string(index=False))
    tr = M.assign(season=np.where(M.tmonth.isin([1, 2, 3, 4]), "겨울 1~4", np.where(M.tmonth.isin([5, 6]), "봄 5~6", np.where(M.tmonth.isin([7, 8, 9, 10]), "여름 7~10", "가을 11~12"))))
    s = tr[tr.year.between(2021, 2026)].groupby("season")[["e_final", "e_guard", "e_guard_k5"]].agg(lambda x: x.abs().mean()).round(1)
    print("\n작형별 MAPE 2021~2026 (월말):")
    print(s.to_string())


if __name__ == "__main__":
    main()
