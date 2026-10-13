"""Case 24 — 봄 급락 경보: 봄배추 생산 증가 전망일 때 5~6월 대상 예측을 낮추는 규칙.

사용자 지시(2026-10-13): "급등을 경보하는데 급락을 경보 가능한가?" → "봄 급락 경보 케이스로 시험해줘".
사전 확인(대화 중 계산): 급락(다음 28일 평균이 최근 7일 가격보다 25% 이상 낮음)은 128개 달 중 12개, 대상월 10월 5·5월 4.
10월 하락은 현행 모형이 이미 예측하고, 봄(5월)은 폭을 작게 본다. 생산 증가 전망이 급락을 가르는 작형은 봄뿐(13개 중 4 대 9개 중 1).

대상: 봄 대상월(원점 + 15일의 달이 5~6월, 여름 원점 제외). 신호 S = 봄배추 생산 전망 전년비(규약 B, 원점에서 알 수 있는 최신호; 없으면 면적).
  신호 원천: case16 판독표(2015~2020) + data/ext/krei_nonsummer_issues.csv(2021~). 상태: S > 임계 → '증가 전망'(급락 경보).
규칙(모두 2016~2020 에서 정하고 2021~ 에서 검증; 바탕 = 메인 final, 봄에는 대표 모형 그대로):
  beta      경보 원점에 final × (1 + β·S/100), β = 2016~2020 경보 원점의 log(실제/final) ~ β·S/100 (절편 없음, 양수면 0), |S| ≤ 20
  step      경보 원점에 final × exp(δ), δ = 2016~2020 경보 원점의 log(실제/final) 평균(양수면 0)
  beta_exp  β 를 그 해 이전 해들로 매해 다시 추정(확장 창, 경보 원점 3개 이상일 때)
  beta_sym  상태를 나누지 않고 모든 봄 원점에 대칭 β(case16 방식, 대조군)
민감도: 임계 0·5·10, 작형 맞춤 신호(5월 대상 = 시설봄, 6월 대상 = 노지봄 우선).
경보 자체의 적중: 경보 상태와 실제 급락(실제 ÷ 최근 7일 가격 − 1 ≤ −25%)의 교차표를 창별로.

Usage: python experiments/case24_spring_drop_alert/spring_drop.py [M|W]
Outputs: outputs/spring_metrics{tag}.csv, spring_alert_table{tag}.csv, spring_origins{tag}.csv, spring_picks{tag}.csv
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
sys.path.insert(0, str(ROOT / "experiments" / "case22_season_gate"))
import gate_model as gm  # noqa: E402
import season_gate as sg  # noqa: E402

WINDOWS = {"2016~2020": (2016, 2020), "2021~2025": (2021, 2025), "2021~2026": (2021, 2026)}
DROP = -25.0
CAP = gm.S_CAP


def crop_signal(iss: pd.DataFrame, origin: pd.Timestamp, ty: int, tmonth: int) -> float:
    """작형 맞춤 신호: 5월 대상은 시설봄, 6월 대상은 노지봄을 먼저 본다(없으면 전체, 그다음 다른 쪽)."""
    s = iss[(iss.season == "봄") & (iss.crop_year == ty) & (iss.avail_b <= origin)]
    if s.empty:
        return np.nan
    last = s[(s.year == s.year.iloc[-1]) & (s.issue_month == s.issue_month.iloc[-1])]
    pri = {"시설": 0, "전체": 1, "노지": 2} if tmonth == 5 else {"노지": 0, "전체": 1, "시설": 2}
    last = last.assign(p=last["sub"].map(pri)).sort_values("p")
    withp = last[last.prod_yoy.notna()]
    r = (withp if len(withp) else last).iloc[0]
    return float(r.prod_yoy if pd.notna(r.prod_yoy) else r.area_yoy)


def est_beta(s: pd.DataFrame, col: str = "S") -> float:
    if len(s) < 3:
        return 0.0
    x, y = s[col].clip(-CAP, CAP) / 100, np.log(s.actual / s.final)
    return min(0.0, float((x * y).sum() / (x ** 2).sum()))


def ape(s: pd.DataFrame, c: str) -> pd.Series:
    return (s[c] / s.actual - 1) * 100


def run(freq: str) -> None:
    tag = "" if freq == "M" else f"_{freq}"
    d = sg.load(freq)
    f = pd.read_csv(ROOT / "data" / "processed" / "daily_features.csv", parse_dates=["date"], usecols=["date", "lp7"]).set_index("date")
    d["p7"] = np.exp(f.lp7.reindex(d.origin).values)
    iss = sg.nonsummer_issues()
    ty = (d.origin + pd.Timedelta(days=15)).dt.year
    spring = (d.summer == 0) & d.tmonth.isin((5, 6))
    d["S"] = [sg.ns_signal(iss, o, "봄", y)[1] if sp else np.nan for o, y, sp in zip(d.origin, ty, spring)]
    d["S_crop"] = [crop_signal(iss, o, y, tm) if sp else np.nan for o, y, tm, sp in zip(d.origin, ty, d.tmonth, spring)]
    d["chg"] = (d.actual / d.p7 - 1) * 100
    d["drop"] = (d.chg <= DROP).astype(int)
    yr = d.origin.dt.year
    design = yr.between(2016, 2020)
    sp = d[spring & d.S.notna()]
    print(f"===== {freq}: 봄 대상 원점 {int(spring.sum())}개, 신호 있는 원점 {len(sp)}개 =====")

    # ----- 규칙 -----
    picks, rules = [], ["final"]

    def apply(name: str, thr: float, col: str, kind: str) -> None:
        al = spring & d[col].notna() & (d[col] > thr)
        k = d[al & design]
        fac = pd.Series(1.0, index=d.index)
        info = {"rule": name, "thr": thr, "signal": col, "n_alert_design": len(k)}
        if kind == "beta":
            b = est_beta(k, col)
            fac[al] = 1 + b * d.loc[al, col].clip(-CAP, CAP) / 100
            info["beta"] = b
        elif kind == "step":
            dl = min(0.0, float(np.log(k.actual / k.final).mean())) if len(k) >= 3 else 0.0
            fac[al] = float(np.exp(dl))
            info["delta"] = dl
        elif kind == "beta_exp":
            for y in sorted(yr[al].unique()):
                b = est_beta(d[al & (yr < y)], col)
                m = al & (yr == y)
                fac[m] = 1 + b * d.loc[m, col].clip(-CAP, CAP) / 100
                info[f"beta_{y}"] = round(b, 3)
        d[name] = d.final * fac
        rules.append(name)
        picks.append(info)

    apply("beta", 0.0, "S", "beta")
    apply("step", 0.0, "S", "step")
    apply("beta_exp", 0.0, "S", "beta_exp")
    apply("beta_thr5", 5.0, "S", "beta")
    apply("beta_thr10", 10.0, "S", "beta")
    apply("beta_crop", 0.0, "S_crop", "beta")
    allsp = spring & d.S.notna()
    b_sym = est_beta(d[allsp & design])
    d["beta_sym"] = np.where(allsp, d.final * (1 + b_sym * d.S.clip(-CAP, CAP).fillna(0) / 100), d.final)
    rules.append("beta_sym")
    picks.append({"rule": "beta_sym", "beta": b_sym, "n_alert_design": int((allsp & design).sum())})
    pk = pd.DataFrame(picks)
    pk.to_csv(OUT / f"spring_picks{tag}.csv", index=False, float_format="%.3f", encoding="utf-8-sig")
    print("\n[2016~2020 에서 정한 값]")
    print(pk.round(3).to_string(index=False))

    rows = []
    for wname, (a, b) in WINDOWS.items():
        for sub, mk in (("봄 대상", spring), ("전체", pd.Series(True, index=d.index))):
            s = d[yr.between(a, b) & mk]
            for r in rules:
                e = ape(s, r)
                rows.append({"window": wname, "subset": sub, "rule": r, "n": len(s), "MAPE%": e.abs().mean(), "bias%": e.mean(),
                             "n_gt20": int((e.abs() > 20).sum()), "n_gt30": int((e.abs() > 30).sum())})
    met = pd.DataFrame(rows)
    met.to_csv(OUT / f"spring_metrics{tag}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    for col in ("MAPE%", "bias%", "n_gt30"):
        print(f"\n[{col}]")
        print(met.pivot_table(index=["subset", "window"], columns="rule", values=col)[rules].round(2).to_string())

    # ----- 경보 자체의 적중 -----
    arows = []
    for wname, (a, b) in WINDOWS.items():
        for thr in (0.0, 5.0, 10.0):
            s = sp[sp.origin.dt.year.between(a, b)]
            up, dn = s[s.S > thr], s[s.S <= thr]
            arows.append({"window": wname, "thr": thr, "n_alert": len(up), "drop_in_alert": int(up["drop"].sum()), "n_noalert": len(dn),
                          "drop_in_noalert": int(dn["drop"].sum()), "chg_alert%": up.chg.mean(), "chg_noalert%": dn.chg.mean(),
                          "bias_alert%": ape(up, "final").mean(), "bias_noalert%": ape(dn, "final").mean()})
    at = pd.DataFrame(arows)
    at.to_csv(OUT / f"spring_alert_table{tag}.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    print(f"\n[경보(S > 임계)와 급락(실제 ÷ 최근 7일 − 1 ≤ {DROP:.0f}%)]")
    print(at.round(1).to_string(index=False))

    keep = ["origin", "actual", "tmonth", "p7", "chg", "drop", "S", "S_crop"] + rules
    o = d.loc[spring, keep].copy()
    for r in rules[:4]:
        o[r + "%"] = ape(o, r).round(1)
    o.to_csv(OUT / f"spring_origins{tag}.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    if freq == "M":
        print("\n[봄 대상 월말 원점]")
        print(o[["origin", "actual", "chg", "S", "S_crop", "final%", "beta%", "step%", "beta_exp%"]].round(1).to_string(index=False))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 330)
    pd.set_option("display.max_columns", 40)
    run(sys.argv[1] if len(sys.argv) > 1 else "M")
