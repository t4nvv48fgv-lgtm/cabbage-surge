"""사전 설계 β — 2015~2020 월보 전망(판독)과 통계청 실적으로 작형별 전망 신뢰도 r 을 추정하고, β_k = KASMO_k × r_k 로 고정한 뒤 2021~2026 을 검증.

사용자 결정(2026-10-12) "그렇게 하자": β를 결과를 보고 맞추지 않고, 전망↔실적 관계가 있는 과거 구간에서 미리 정한다.
  r_k = cov(f, a) / var(f)   f = 그 작형 출하 첫 달 원점에서 쓰는 호의 생산 전망 전년비, a = 통계청 실적 전년비 (2015~2020 작형연도)
  β_k = KASMO_k × r_k  (KASMO 가격 신축성: 겨울 −4.3, 봄 −1.5, 여름 −5.6, 가을 −3.7; 사용자 제공)  — 비교용으로 실적 회귀 β_실적_k × r_k 도.
  겨울 r 이 0 근처면 겨울은 조정하지 않는다(사전 규칙: r_k ≤ 0 → β_k = 0).
적용: case16 과 같은 수준 조정 pred × (1 + β_k × S/100), M2 매핑(1~3월 겨울·4~6월 봄·7~10월 여름·11~12월 가을), 규약 B. 여름은 메인 게이트 경보 달에만 S1 로.
검증: 2021~2025 월말 60개(메인 지표 창) + 2021~2026. 참고로 2016~2020(r 추정 구간, 표본내)도 표시.

Usage: python experiments/case16_season_supply_adjust/prereg_beta.py
Outputs: outputs/prereg_reliability.csv, prereg_metrics.csv, prereg_predictions.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
OUT = CASE_DIR / "outputs"
sys.argv.append("--april-spring")          # M2 매핑으로 season_supply_adjust 를 import
sys.path.insert(0, str(CASE_DIR))
import season_supply_adjust as ssa  # noqa: E402

KASMO = {"겨울": -4.3, "봄": -1.5, "여름": -5.6, "가을": -3.7}
FIRST_MONTH = {"가을": 11, "봄": 5, "겨울": 12, "여름": 8}   # 출하 첫 달 원점에서 쓰는 호(규약 B): 가을 11월호(10월 말), 봄 5월호(4월 말), 겨울 12월호(11월 말), 여름 8월호(7월 말)
REL_YEARS = range(2015, 2021)


def load_all_issues() -> pd.DataFrame:
    old = pd.read_csv(CASE_DIR / "krei_nonsummer_issues_2015_2020.csv")
    new = pd.read_csv(OUT / "issues_nonsummer_2021_2026.csv", encoding="utf-8-sig")
    cols = ["season", "sub", "crop_year", "year", "issue_month", "prod_yoy", "area_yoy"]
    d = pd.concat([old[cols], new[cols]]).sort_values(["season", "crop_year", "year", "issue_month"]).reset_index(drop=True)
    d["label"] = ""
    return d


def summer_issues() -> pd.DataFrame:
    old = pd.read_csv(ROOT / "experiments" / "case13_krei_backfill" / "krei_summer_issues_2010_2020.csv")
    new = pd.read_csv(ROOT / "data" / "ext" / "krei_summer_issues.csv", encoding="utf-8-sig")
    d = pd.concat([old[["year", "issue_month", "prod_yoy", "area_yoy"]], new[["year", "issue_month", "prod_yoy", "area_yoy"]]])
    d["season"], d["sub"], d["crop_year"] = "여름", "여름", d.year
    return d


def forecast_first_month(iss: pd.DataFrame, sea: str, crop: int) -> float:
    """작형 출하 첫 달 원점에서 쓰는 호의 생산 전년비(없으면 면적). 겨울은 crop 연도 12월호, 가을 11월호, 봄 5월호, 여름 8월호."""
    m = FIRST_MONTH[sea]
    s = iss[(iss.season == sea) & (iss.crop_year == crop) & (iss.year == crop) & (iss.issue_month == m)]
    if sea == "봄":
        pri = {"전체": 0, "노지": 1, "시설": 2}
        s = s.assign(p=s["sub"].map(pri)).sort_values("p")
        s = s[s.prod_yoy.notna()] if s.prod_yoy.notna().any() else s
    if s.empty:
        return np.nan
    r = s.iloc[0]
    return float(r.prod_yoy) if pd.notna(r.prod_yoy) else float(r.area_yoy)


def reliability(iss: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sea in ("겨울", "봄", "여름", "가을"):
        pts = []
        for y in REL_YEARS:
            f = forecast_first_month(iss, sea, y)
            if sea in realized and y in realized.index and (y - 1) in realized.index and pd.notna(f):
                a = (realized.loc[y, sea] / realized.loc[y - 1, sea] - 1) * 100
                if pd.notna(a):
                    pts.append((y, f, a))
        if len(pts) < 3:
            rows.append({"season": sea, "n": len(pts), "r": np.nan})
            continue
        F, A = np.array([p[1] for p in pts]), np.array([p[2] for p in pts])
        r = float(((F - F.mean()) * (A - A.mean())).sum() / ((F - F.mean()) ** 2).sum())
        r0 = float((F * A).sum() / (F * F).sum())
        rows.append({"season": sea, "n": len(pts), "years": ",".join(str(p[0]) for p in pts), "r(centered)": round(r, 2), "r(no_intercept)": round(r0, 2),
                     "corr": round(float(np.corrcoef(F, A)[0, 1]), 2), "forecasts": ";".join(f"{p[1]:+.1f}" for p in pts), "realized": ";".join(f"{p[2]:+.1f}" for p in pts)})
    return pd.DataFrame(rows)


def main() -> None:
    iss = load_all_issues()
    sumi = summer_issues()
    realized = pd.read_csv(OUT / "stat_production_by_season.csv", index_col="year")
    rel = reliability(pd.concat([iss, sumi]), realized)
    rel.to_csv(OUT / "prereg_reliability.csv", index=False, encoding="utf-8-sig")
    beta_real = pd.read_csv(OUT / "realized_beta.csv").set_index("season")["beta_no_intercept"]
    pd.set_option("display.width", 320)
    print("=== 작형별 전망 신뢰도 r (2015~2020, 출하 첫 달 호 vs 통계청 실적) ===")
    print(rel.to_string(index=False))
    rr = rel.set_index("season")["r(no_intercept)"] if "r(no_intercept)" in rel else rel.set_index("season")["r"]
    beta = {}
    for sea in ("겨울", "봄", "여름", "가을"):
        r = rr.get(sea, np.nan)
        beta[sea] = {"r": r, "beta_kasmo": 0.0 if (pd.isna(r) or r <= 0) else KASMO[sea] * r, "beta_realized": 0.0 if (pd.isna(r) or r <= 0) else float(beta_real[sea]) * r}
    print("\n=== 사전 고정 β ===")
    print(pd.DataFrame(beta).T.round(2).to_string())
    # 신호: 2016~2026 원점, M2 매핑, 규약 B
    g = pd.read_csv(ROOT / "outputs" / "gate_predictions.csv", parse_dates=["origin"], encoding="utf-8-sig")
    d = g[g.origin >= "2015-12-31"].copy()
    sig = [ssa.signal_at(iss, o) for o in d.origin]
    d["season"] = [x["season"] for x in sig]
    d["S_season"] = [x.get("S", np.nan) for x in sig]
    # 여름 S1: 메인(2021~) + case13(2016~2020), 규약 B(다음 호)
    sum_tab = sumi.rename(columns={"year": "y"})
    def s1_at(o):
        s = sum_tab[(sum_tab.y == o.year) & (sum_tab.issue_month <= o.month + 1)]
        if s.empty:
            return np.nan
        r = s.sort_values("issue_month").iloc[-1]
        return float(r.prod_yoy) if pd.notna(r.prod_yoy) else float(r.area_yoy)
    d["S1_ext"] = [s1_at(o) if sea == "여름" else np.nan for o, sea in zip(d.origin, d.season)]
    alarm = (d.season == "여름") & (d.S1_ext.notna()) & (d.S1_ext <= 0)
    d["S_all"] = np.where(d.season == "여름", d.S1_ext, d.S_season)
    d["S_cap"] = d.S_all.clip(-ssa.CAP, ssa.CAP)
    # 잔차 β: 2016~2020 원점에서 log(actual/모형예측) ~ β_resid × S/100 (절편 없음). 모형이 이미 반영한 몫을 뺀 '남은' 반응 → 사전 설계의 올바른 β
    yd = d.origin.dt.year
    print("\n=== 잔차 β (2016~2020, log(실제/예측) ~ β·S) ===")
    for sea in ("겨울", "봄", "여름", "가을"):
        mk = yd.between(2016, 2020) & (d.season == sea) & d.S_cap.notna() & ((d.season != "여름") | alarm)
        s = d[mk]
        if len(s) >= 3:
            x = s.S_cap / 100; yy = np.log(s.actual / s.gate_hgb)
            br = float((x * yy).sum() / (x * x).sum())
            beta[sea]["beta_resid"] = min(br, 0.0)      # 사전 규칙: 양수(역반응)면 0
            print(f"  {sea}: β_resid = {br:+.2f} (n={len(s)}) → 사용 {beta[sea]['beta_resid']:+.2f}")
        else:
            beta[sea]["beta_resid"] = 0.0
    for name in ("beta_kasmo", "beta_realized", "beta_resid"):
        b = d.season.map({k: v[name] for k, v in beta.items()})
        f = 1 + b * d.S_cap / 100
        use = d.S_cap.notna() & ((d.season != "여름") | alarm)
        d[f"adj_{name}"] = np.where(use, d.gate_hgb * f, d.gate_hgb)
        d[f"adj_{name}_nonsummer"] = np.where(use & (d.season != "여름"), d.gate_hgb * f, d.gate_hgb)
    d.to_csv(OUT / "prereg_predictions.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    y = d.origin.dt.year
    masks = {"all_2016_2020(r추정구간)": y.between(2016, 2020), "all_2021_2025(검증)": y.between(2021, 2025), "all_2021_2026": y.between(2021, 2026)}
    for sea in ("겨울", "봄", "여름", "가을"):
        masks[f"{sea}_2021_2025"] = y.between(2021, 2025) & (d.season == sea)
    cols = ["gate_hgb", "adj_beta_kasmo", "adj_beta_kasmo_nonsummer", "adj_beta_realized", "adj_beta_realized_nonsummer", "adj_beta_resid", "adj_beta_resid_nonsummer"]
    met = []
    for nm, mk in masks.items():
        for c in cols:
            s = d[mk]
            e = s[c] - s.actual
            met.append({"subset": nm, "model": c, "n": len(s), "MAE": e.abs().mean(), "MAPE%": (e.abs() / s.actual).mean() * 100, "bias": e.mean()})
    met = pd.DataFrame(met)
    met.to_csv(OUT / "prereg_metrics.csv", index=False, float_format="%.2f")
    print("\n=== MAPE ===")
    print(met.pivot(index="model", columns="subset", values="MAPE%").loc[cols][list(masks)].round(1).to_string())
    big = d[(y >= 2021) & ((d.gate_hgb / d.actual - 1).abs() > 0.2)].copy()
    big["tm"] = (big.origin + pd.offsets.MonthBegin(1)).dt.strftime("%Y-%m")
    for c in ("gate_hgb", "adj_beta_kasmo", "adj_beta_resid"):
        big[c + "%"] = ((big[c] / big.actual - 1) * 100).round(0)
    print("\n=== 20% 초과 달 전후 ===")
    print(big[["tm", "season", "S_all", "gate_hgb%", "adj_beta_kasmo%", "adj_beta_resid%"]].to_string(index=False))
    pd.DataFrame(beta).T.round(3).to_csv(OUT / "prereg_beta.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
