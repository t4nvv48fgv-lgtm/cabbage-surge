"""Supply-gated model (2026-10-12 메인 편입, experiments/case06 → case10).

최종 예측 = 공급 게이트가 "조용함"이면 평시 모형, 아니면 대표 모형.

  대표 모형   an_sparse_all   (anomaly_model.py: 평년치 기준점 + 현재 가격 편차 지속 + 고랭지 기상 편차, 희소 ridge)
  평시 모형   wx_hgb_summer   (backtest.py: 기상 HGB 여름특화) — case10에서 선택(사용자 결정 2026-10-12)
  게이트 S1   KREI 농업관측 월보 여름배추(당해년산) 생산량 전망 전년비 — 원점까지 발행된 최신호 값.
              여름 원점(6~9월)에서 S1 > 0 (공급이 전년보다 늘 것이라는 전망) 이면 calm → 평시 모형,
              S1 ≤ 0 이거나 신호가 없으면(비여름·2021 이전) 대표 모형.
              호별 값은 data/ext/krei_summer_issues.csv (2021~, 발행일은 호 월 5일로 보수적 가정; 8월호 실제 게시 8/1~8/5).
              생산 전년비가 없는 호(4~5월호)는 면적 전년비로 대체.

규칙·임계(0)는 case06에서 2024·2025를 본 뒤 정한 사후 설계이며 신호는 6개 해(급등 2회)뿐이다. 평시 모형 선택은 case10(2016~2020 설계
구간에서는 가격만 ridge가 더 나았고, 2021~ 검증 구간에서 HGB가 나음). 서술 상한은 README "성공 기준 점검" 참조.

Usage: python src/gate_model.py [M|W]   (outputs/backtest_predictions{tag}.csv, anomaly_predictions{tag}.csv 가 먼저 있어야 함)
Outputs: outputs/gate_predictions{tag}.csv  (origin, actual, summer, krei_issue, S1, gate_calm, 대표·평시·gate_hgb, 참고 gate_ridge)
         outputs/gate_metrics{tag}.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
KREI_ISSUES = ROOT / "data" / "ext" / "krei_summer_issues.csv"
HEAD, CALM, CALM_REF = "an_sparse_all", "wx_hgb_summer", "price_only_ridge"
PUB_DAY = 5


def load_issues() -> pd.DataFrame:
    d = pd.read_csv(KREI_ISSUES, encoding="utf-8-sig")
    d["pub_date"] = pd.to_datetime(dict(year=d.year, month=d.issue_month, day=PUB_DAY))
    return d.sort_values("pub_date").reset_index(drop=True)


def latest_signal(issues: pd.DataFrame, origin: pd.Timestamp) -> tuple[str | float, float]:
    """원점 연도의 호 중 발행일 ≤ 원점인 최신호. (호 표기, S1). 없으면 (nan, nan)."""
    s = issues[(issues.year == origin.year) & (issues.pub_date <= origin)]
    if s.empty:
        return np.nan, np.nan
    r = s.iloc[-1]
    s1 = r.prod_yoy if pd.notna(r.prod_yoy) else r.area_yoy
    return f"{int(r.year)}-{int(r.issue_month):02d}", float(s1)


def run(freq: str = "M") -> pd.DataFrame:
    tag = "" if freq == "M" else f"_{freq}"
    b = pd.read_csv(OUT / f"backtest_predictions{tag}.csv", parse_dates=["origin"])
    a = pd.read_csv(OUT / f"anomaly_predictions{tag}.csv", parse_dates=["origin"])
    d = b[["origin", "actual", "summer", CALM_REF, CALM, "chronos2"]].merge(a[["origin", "normal_fwd", HEAD]], on="origin")
    issues = load_issues()
    sig = [latest_signal(issues, o) if s == 1 else (np.nan, np.nan) for o, s in zip(d.origin, d.summer)]
    d["krei_issue"] = [k for k, _ in sig]
    d["S1"] = [v for _, v in sig]
    d["gate_calm"] = (d.S1.notna() & (d.S1 > 0)).astype(int)
    d["gate_hgb"] = np.where(d.gate_calm == 1, d[CALM], d[HEAD])        # 최종 모형
    d["gate_ridge"] = np.where(d.gate_calm == 1, d[CALM_REF], d[HEAD])  # 참고: 평시 = 가격만 ridge (case06 G_K)
    return d


def metrics(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    y = d.origin.dt.year
    masks = {"all_2021_2025": y.between(2021, 2025), "summer_origin_2021_2025": y.between(2021, 2025) & (d.summer == 1),
             "summer_origin_2021_2026": y.between(2021, 2026) & (d.summer == 1), "calm_origins_2021_2026": y.between(2021, 2026) & (d.gate_calm == 1)}
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
    d = run(freq)
    cols = [CALM_REF, "chronos2", CALM, HEAD, "gate_ridge", "gate_hgb"]
    d.to_csv(OUT / f"gate_predictions{tag}.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    met = metrics(d, cols)
    met.to_csv(OUT / f"gate_metrics{tag}.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 250)
    print(f"[{freq}] 원점 {len(d)}개, calm 원점 {int(d.gate_calm.sum())}개")
    print(met.pivot(index="model", columns="subset", values="MAPE%").loc[cols].round(2).to_string())
    if freq == "M":
        keys = pd.to_datetime([f"{y}-0{m}-{dd}" for y in range(2020, 2027) for m, dd in ((7, "31"), (8, "31"), (9, "30"))])
    else:
        keys = d.origin[(d.origin.dt.year >= 2020) & d.origin.dt.month.isin([8, 9])]
    sc = d[d.origin.isin(keys)].copy()
    for c in cols:
        sc[c + "%"] = ((sc[c] / sc.actual - 1) * 100).round(0)
    print(sc[["origin", "actual", "krei_issue", "S1", "gate_calm"] + [c + "%" for c in cols]].to_string(index=False))
