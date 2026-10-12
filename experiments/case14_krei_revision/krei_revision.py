"""Case 14 — 월보 면적·단수·생산량 전망의 수준과 호간 수정폭을 게이트 신호로 (가격 전망·출하량·포전가격은 쓰지 않음).

사용자 결정(2026-10-12): 월보 가격 전망치·출하량 전망은 담당자 판단의 산물이라 순환이고, 포전거래가격은 수집 데이터라 제외.
남는 것은 관측 원자료에 가까운 **재배면적·단수·생산량 전망**이며, 그 (a) 수준과 (b) 호간 수정폭(작황 악화가 진행 중인가)을 쓴다.
가용 시점은 규약 B(담당자는 m월호 내용을 (m−1)월 말에 안다): 원점 m월 말에서 (m+1)월호까지 가용. 수정폭 Δ = S(m+1월호) − S(m월호).

신호(여름배추/고랭지배추 전년비, %, 호별)
  P  생산량 전년비 (메인 S1과 동일 정의)      A  재배면적 전년비      Y  단수 전년비 ≈ (1+P)/(1+A) − 1 (판독표 단수 열이 대부분 비어 있어 유도값 사용)
  ΔP, ΔA, ΔY  직전 호 대비 수정폭           ΔP_cum  그 해 첫 여름 호(6월호) 대비 누적 수정폭
자료: 2021~2026 `data/ext/krei_summer_issues.csv`(판독표), 2016~2020 `experiments/case13_krei_backfill/krei_summer_issues_2010_2020.csv`(case13 판독).

규칙(실행 전 고정; 경보 → 대표 an_sparse_all, 평시 → wx_hgb_summer. 메인 최종 모형의 평시 선택 그대로)
  R0 level      P ≤ 0                                   (메인 게이트, 규약 B)
  R1 level&rev  P ≤ 0 and ΔP ≤ 0                        (공급 감소 전망이고 호를 거듭해도 좋아지지 않음)
  R2 rev        ΔP ≤ −2                                 (수정폭만)
  R3 level|rev  P ≤ 0 or ΔP ≤ −3                        (둘 중 하나)
  R4 yield      Y ≤ 0                                   (단수 채널만: 작황)
  R5 area       A ≤ 0                                   (면적 채널만)
  R6 level&cum  P ≤ 0 and ΔP_cum ≤ 0                    (6월호 이후 누적 악화)
  R7 soft       w_경보 = clip(−P/10, 0, 1), 로그 가중 혼합 (수준의 크기 사용)
  R8 soft_rev   w_경보 = clip((−P − ΔP)/10, 0, 1)
선택 절차: 2016~2020 여름 원점(20개)에서 규칙을 고르고 2021~2026에서 검증. 2016~2020에는 메인 예측이 있고 신호는 case13 판독.

Usage: python experiments/case14_krei_revision/krei_revision.py
Outputs: outputs/issues_ext_2016_2026.csv, rule_predictions.csv, rule_metrics.csv, aug_origin_table.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
OUT = CASE_DIR / "outputs"
HEAD, CALM = "an_sparse_all", "wx_hgb_summer"


def load_issues() -> pd.DataFrame:
    new = pd.read_csv(ROOT / "data" / "ext" / "krei_summer_issues.csv", encoding="utf-8-sig")
    old = pd.read_csv(ROOT / "experiments" / "case13_krei_backfill" / "krei_summer_issues_2010_2020.csv")
    cols = ["year", "issue_month", "prod_yoy", "area_yoy"]
    d = pd.concat([old[cols], new[cols]]).sort_values(["year", "issue_month"]).reset_index(drop=True)
    d = d[d.year >= 2016]
    d["P"] = d.prod_yoy
    d["A"] = d.area_yoy
    d["Y"] = ((1 + d.P / 100) / (1 + d.A / 100) - 1) * 100
    # 생산 전년비가 없는 호(4·5월호, 2017-06)는 면적으로 대체(메인 latest_signal과 동일)
    d["S"] = d.P.where(d.P.notna(), d.A)
    d = d.sort_values(["year", "issue_month"])
    g = d.groupby("year")
    d["dP"] = g.S.diff()
    d["dA"] = g.A.diff()
    d["dY"] = g.Y.diff()
    first6 = d[d.issue_month >= 6].groupby("year").S.first()
    d["dP_cum"] = d.S - d.year.map(first6)
    return d.reset_index(drop=True)


def signal_at(iss: pd.DataFrame, origin: pd.Timestamp) -> dict:
    """규약 B: 원점 m월 말 → (m+1)월호까지 가용. 그 해의 호 중 issue_month ≤ m+1 인 최신호."""
    s = iss[(iss.year == origin.year) & (iss.issue_month <= origin.month + 1)]
    if s.empty:
        return {}
    r = s.iloc[-1]
    return {"issue": f"{int(r.year)}-{int(r.issue_month):02d}", "P": r.S, "A": r.A, "Y": r.Y, "dP": r.dP, "dA": r.dA, "dY": r.dY, "dP_cum": r.dP_cum}


def apply_rules(d: pd.DataFrame) -> pd.DataFrame:
    P, A, Y, dP, dPc = d.P, d.A, d.Y, d.dP, d.dP_cum
    has = d.P.notna()
    hard = {
        "R0_level": P <= 0,
        "R1_level_rev": (P <= 0) & (dP.fillna(0) <= 0),
        "R2_rev": dP.fillna(0) <= -2,
        "R3_level_or_rev": (P <= 0) | (dP.fillna(0) <= -3),
        "R4_yield": Y.fillna(0) <= 0,
        "R5_area": A.fillna(0) <= 0,
        "R6_level_cum": (P <= 0) & (dPc.fillna(0) <= 0),
    }
    lg_h, lg_c = np.log(d[HEAD]), np.log(d[CALM])
    for name, alarm in hard.items():
        a = (has & alarm) | (~has)                      # 신호 없는 원점(비여름·판독 전)은 대표
        d[f"alarm_{name}"] = a.astype(int)
        d[name] = np.where(a, d[HEAD], d[CALM])
    for name, w in {"R7_soft": np.clip(-P / 10, 0, 1), "R8_soft_rev": np.clip((-P - dP.fillna(0)) / 10, 0, 1)}.items():
        w = np.where(has, w, 1.0)
        d[f"alarm_{name}"] = np.round(w, 2)
        d[name] = np.exp(w * lg_h + (1 - w) * lg_c)
    return d


def metrics(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    y = d.origin.dt.year
    masks = {"summer_2016_2020(select)": y.between(2016, 2020) & (d.summer == 1), "summer_2021_2025(valid)": y.between(2021, 2025) & (d.summer == 1),
             "summer_2021_2026": y.between(2021, 2026) & (d.summer == 1), "summer_2016_2026": y.between(2016, 2026) & (d.summer == 1),
             "all_2021_2025": y.between(2021, 2025), "all_2016_2025": y.between(2016, 2025)}
    out = []
    for name, mk in masks.items():
        s = d[mk]
        for c in cols:
            e = s[c] - s.actual
            out.append({"subset": name, "model": c, "n": len(s), "MAE": e.abs().mean(), "MAPE%": (e.abs() / s.actual).mean() * 100, "bias": e.mean()})
    return pd.DataFrame(out)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    iss = load_issues()
    iss.to_csv(OUT / "issues_ext_2016_2026.csv", index=False, float_format="%.2f")
    g = pd.read_csv(ROOT / "outputs" / "gate_predictions.csv", parse_dates=["origin"], encoding="utf-8-sig")
    d = g[["origin", "actual", "summer", "price_only_ridge", CALM, HEAD, "gate_hgb"]].copy()
    sig = [signal_at(iss, o) if s == 1 else {} for o, s in zip(d.origin, d.summer)]
    for k in ("issue", "P", "A", "Y", "dP", "dA", "dY", "dP_cum"):
        d[k] = [x.get(k, np.nan) for x in sig]
    d = apply_rules(d)
    rules = ["R0_level", "R1_level_rev", "R2_rev", "R3_level_or_rev", "R4_yield", "R5_area", "R6_level_cum", "R7_soft", "R8_soft_rev"]
    cols = [HEAD, CALM, "gate_hgb"] + rules
    d.to_csv(OUT / "rule_predictions.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    met = metrics(d, cols)
    met.to_csv(OUT / "rule_metrics.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 320)
    print("=== 여름 원점 신호 (규약 B, 2016~2026) ===")
    s = d[(d.summer == 1) & (d.origin.dt.year >= 2016)]
    print(s[["origin", "actual", "issue", "P", "A", "Y", "dP", "dP_cum"]].round(1).to_string(index=False))
    print("\n=== MAPE ===")
    print(met.pivot(index="model", columns="subset", values="MAPE%").loc[cols].round(1).to_string())
    print("\n=== 8월 말 원점 오차% (경보 상태) ===")
    k = d[(d.origin.dt.month == 8) & (d.origin.dt.year >= 2016)].copy()
    for c in cols:
        k[c + "%"] = ((k[c] / k.actual - 1) * 100).round(0)
    show = ["origin", "actual", "P", "dP"] + [c + "%" for c in cols] + [f"alarm_{r}" for r in rules[:7]]
    print(k[show].to_string(index=False))
    k[show].to_csv(OUT / "aug_origin_table.csv", index=False, encoding="utf-8-sig")
