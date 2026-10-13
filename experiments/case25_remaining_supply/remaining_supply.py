"""Case 25 — 월보 생산 전망 × 농넷 반입 실적으로 '남은 물량'을 추정해 급등·급락을 가를 수 있는가.

사용자 제안(2026-10-13): "월보에서 전망하는 생산량과 농넷 반입량을 통해 향후 출하를 예측하여 급등 혹은 급락을 확인할 수 없을까?
시장 반입량이 전체 생산량과 같지 않지만 추세 등은 패턴이 있을 것 같은데."

생각: 그 작형에서 나올 물량(월보 전망)은 정해져 있고, 지금까지 시장에 나온 물량(농넷 반입 누계)을 빼면 앞으로 나올 물량이 남는다.
  평년보다 빨리 나왔으면(조기 출하) 남은 물량이 적어 뒤에 오르고, 늦게 나오고 있으면(지연) 남은 물량이 몰려 뒤에 내린다.
  case23 원인 조사에서 30% 초과 달의 가장 흔한 원인이 이 '출하 시기 이동'이었다. case20(산지 반입 평년비)은 분모(나올 총량)가 없었다.

계산(원점 o, 대상 작형 k = 대상월 기준 겨울 1~4·봄 5~6·여름 7~10·가을 11~12월, 작형 시작일 s):
  q(d)   전국 도매시장 일별 반입량(농넷 원장, 쌈·절임·수입·부산물 제외)
  C      올해 s~o 누계,  A_prev = 작년 같은 작형 기간 총 반입량
  S      월보 그 작형 생산 전망 전년비(규약 B, 없으면 0)
  A_hat  올해 작형 기간 예상 총 반입량 = A_prev × (1 + S/100)          ← 시장 반입 비중이 작년과 같다고 가정
  p_bar  평년 진행률 = 직전 5년(최소 3년)의 같은 날 누계 ÷ 그 해 총량 평균
  X  = log( (1 − C/A_hat) / (1 − p_bar) )   남은 몫이 평년보다 많은가(+) 적은가(−)   ← 주 신호 '남은 물량'
  P0 = log( C / (A_prev × p_bar) )          누계 반입이 작년 기준 평년 속도보다 많은가   ← 월보를 쓰지 않는 대조 신호
  V  = log( (A_hat − C) / (A_prev − C_prev) ) 남은 물량의 전년비
  작형 시작 뒤 14일이 안 된 원점은 누계 정보가 없어 제외. X·V 는 ±0.7 에서 자른다.
검증: 메인 final 의 잔차 r = log(실제/final) 와 신호의 관계를 2016~2020 에서 추정(절편 없음, 기대 부호와 다르면 0) → 2021~ 에 적용.
  기대 부호: 남은 물량이 많으면 가격이 모형보다 낮다 → β < 0.
  규칙 X_all(전 작형 공통 β), X_sea(작형별 β), P0_all, V_all. 그리고 신호 구간별 급등·급락 빈도.

Usage: python experiments/case25_remaining_supply/remaining_supply.py [M|W]
Outputs: outputs/remain_signal{tag}.csv, remain_corr{tag}.csv, remain_metrics{tag}.csv, remain_events{tag}.csv, remain_picks{tag}.csv
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

PANEL = ROOT / "data" / "ext" / "nongnet" / "nongnet_cabbage_panel_2011_2026.parquet"
EXCLUDE = ["쌈", "절임", "수입", "기타부산물"]
SEASON = {1: "겨울", 2: "겨울", 3: "겨울", 4: "겨울", 5: "봄", 6: "봄", 7: "여름", 8: "여름", 9: "여름", 10: "여름", 11: "가을", 12: "가을"}
START = {"겨울": (1, 1), "봄": (5, 1), "여름": (7, 1), "가을": (11, 1)}
END = {"겨울": (4, 30), "봄": (6, 30), "여름": (10, 31), "가을": (12, 31)}
MIN_DAYS, CLIP, NORM_YEARS = 14, 0.7, 5
WINDOWS = {"2016~2020": (2016, 2020), "2021~2025": (2021, 2025), "2021~2026": (2021, 2026)}
MOVE = 25.0


def daily_arrivals() -> pd.Series:
    p = pd.read_parquet(PANEL, columns=["date", "variety_std", "qty_kg"])
    p = p[~p.variety_std.isin(EXCLUDE)]
    q = p.groupby("date").qty_kg.sum() / 1000.0
    return q.reindex(pd.date_range(q.index.min(), q.index.max(), freq="D")).fillna(0.0)


def cum(q: pd.Series, year: int, sea: str, n_days: int | None = None) -> float:
    s = pd.Timestamp(year, *START[sea])
    e = pd.Timestamp(year, *END[sea]) if n_days is None else s + pd.Timedelta(days=n_days)
    return float(q[s:e].sum())


def signal_row(q: pd.Series, sum_iss: pd.DataFrame, ns_iss: pd.DataFrame, o: pd.Timestamp) -> dict:
    t = o + pd.Timedelta(days=15)
    sea, ty = SEASON[t.month], t.year
    s = pd.Timestamp(ty, *START[sea])
    n = (o - s).days
    out = {"season": sea}
    if n < MIN_DAYS or o > q.index.max():
        return out
    if sea == "여름":
        S = gm.adjust_signal(sum_iss, o, ty)[1]
    else:
        S = sg.ns_signal(ns_iss, o, sea, ty)[1]
    a_prev, c, c_prev = cum(q, ty - 1, sea), cum(q, ty, sea, n), cum(q, ty - 1, sea, n)
    shares = [cum(q, j, sea, n) / cum(q, j, sea) for j in range(ty - NORM_YEARS, ty) if j >= 2011 and cum(q, j, sea) > 0]
    if len(shares) < 3 or a_prev <= 0:
        return out
    p_bar = float(np.mean(shares))
    a_hat = a_prev * (1 + (0.0 if pd.isna(S) else S) / 100.0)
    out.update({"n_days": n, "S": S, "p_bar": p_bar, "progress": c / a_hat,
                "X": float(np.clip(np.log(max(1 - c / a_hat, 0.02) / max(1 - p_bar, 0.02)), -CLIP, CLIP)),
                "P0": float(np.clip(np.log(c / (a_prev * p_bar)), -CLIP, CLIP)),
                "V": float(np.clip(np.log(max(a_hat - c, 0.02 * a_hat) / max(a_prev - c_prev, 0.02 * a_prev)), -CLIP, CLIP))})
    return out


def est(x: pd.Series, y: pd.Series, sign: int) -> float:
    """절편 없는 기울기. 기대 부호(sign)와 다르면 0."""
    if len(x) < 5 or float((x ** 2).sum()) == 0:
        return 0.0
    b = float((x * y).sum() / (x ** 2).sum())
    return b if np.sign(b) == sign else 0.0


def ape(s: pd.DataFrame, c: str) -> pd.Series:
    return (s[c] / s.actual - 1) * 100


def run(freq: str) -> None:
    tag = "" if freq == "M" else f"_{freq}"
    d = sg.load(freq)
    f = pd.read_csv(ROOT / "data" / "processed" / "daily_features.csv", parse_dates=["date"], usecols=["date", "lp7"]).set_index("date")
    d["p7"] = np.exp(f.lp7.reindex(d.origin).values)
    q = daily_arrivals()
    sum_iss, ns_iss = sg.summer_issues_ext(), sg.nonsummer_issues()
    sig = pd.DataFrame([signal_row(q, sum_iss, ns_iss, o) for o in d.origin], index=d.index)
    d = pd.concat([d, sig], axis=1)
    d["r"] = np.log(d.actual / d.final)
    d["chg"] = (d.actual / d.p7 - 1) * 100
    yr = d.origin.dt.year
    design = yr.between(2016, 2020)
    has = d.X.notna()
    print(f"===== {freq}: 원점 {len(d)}개, 신호 있는 원점 {int(has.sum())}개 (작형 시작 뒤 {MIN_DAYS}일 이상) =====")
    d[["origin", "actual", "final", "season", "n_days", "S", "p_bar", "progress", "X", "P0", "V", "r", "chg"]].to_csv(
        OUT / f"remain_signal{tag}.csv", index=False, float_format="%.3f", encoding="utf-8-sig")

    # ----- 신호와 잔차·실제 변화의 상관 -----
    crow = []
    for wname, (a, b) in list(WINDOWS.items())[:2]:
        for sea in ("전체", "겨울", "봄", "여름", "가을"):
            s = d[has & yr.between(a, b) & ((d.season == sea) if sea != "전체" else True)]
            if len(s) < 5:
                continue
            row = {"window": wname, "season": sea, "n": len(s)}
            for c in ("X", "P0", "V"):
                row[f"corr_r_{c}"] = float(np.corrcoef(s[c], s.r)[0, 1])
                row[f"corr_chg_{c}"] = float(np.corrcoef(s[c], s.chg)[0, 1])
            crow.append(row)
    corr = pd.DataFrame(crow)
    corr.to_csv(OUT / f"remain_corr{tag}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    print("\n[신호와 잔차(r)·실제 가격 변화(chg)의 상관]  기대: X·V 음(−), P0 양(+)")
    print(corr.round(2).to_string(index=False))

    # ----- 규칙 -----
    picks = {}
    k = d[has & design]
    for name, col, sign in (("X_all", "X", -1), ("P0_all", "P0", +1), ("V_all", "V", -1)):
        b = est(k[col], k.r, sign)
        picks[name] = b
        d[name] = np.where(has, d.final * np.exp(b * d[col].fillna(0)), d.final)
    fac = pd.Series(1.0, index=d.index)
    for sea in ("겨울", "봄", "여름", "가을"):
        ks = k[k.season == sea]
        b = est(ks.X, ks.r, -1)
        picks[f"X_{sea}"] = b
        m = has & (d.season == sea)
        fac[m] = np.exp(b * d.loc[m, "X"])
    d["X_sea"] = d.final * fac
    rules = ["final", "X_all", "X_sea", "P0_all", "V_all"]
    pd.DataFrame([picks]).to_csv(OUT / f"remain_picks{tag}.csv", index=False, float_format="%.3f", encoding="utf-8-sig")
    print("\n[2016~2020 에서 정한 계수]", {a: round(b, 3) for a, b in picks.items()})
    rows = []
    for wname, (a, b) in WINDOWS.items():
        for sub, mk in (("신호 있는 원점", has), ("전체", pd.Series(True, index=d.index))):
            s = d[yr.between(a, b) & mk]
            for r in rules:
                e = ape(s, r)
                rows.append({"window": wname, "subset": sub, "rule": r, "n": len(s), "MAPE%": e.abs().mean(), "n_gt30": int((e.abs() > 30).sum())})
    met = pd.DataFrame(rows)
    met.to_csv(OUT / f"remain_metrics{tag}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    for col in ("MAPE%", "n_gt30"):
        print(f"\n[{col}]")
        print(met.pivot_table(index=["subset", "window"], columns="rule", values=col)[rules].round(2).to_string())

    # ----- 급등·급락 빈도 -----
    erow = []
    for wname, (a, b) in list(WINDOWS.items())[:2]:
        s = d[has & yr.between(a, b)]
        for lab, mk in (("X ≤ −0.1 (남은 물량 적음)", s.X <= -0.1), ("−0.1 < X < 0.1", s.X.abs() < 0.1), ("X ≥ 0.1 (남은 물량 많음)", s.X >= 0.1)):
            g = s[mk]
            erow.append({"window": wname, "group": lab, "n": len(g), "surge(≥+25%)": int((g.chg >= MOVE).sum()), "drop(≤−25%)": int((g.chg <= -MOVE).sum()),
                         "chg_mean%": g.chg.mean(), "model_bias%": ape(g, "final").mean()})
    ev = pd.DataFrame(erow)
    ev.to_csv(OUT / f"remain_events{tag}.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    print("\n[남은 물량 신호 구간별 급등·급락]")
    print(ev.round(1).to_string(index=False))
    if freq == "M":
        big = d[has & (ape(d, "final").abs() > 30)]
        t = big.origin + pd.Timedelta(days=15)
        print("\n[30% 초과 달 가운데 신호가 있는 달]")
        print(pd.DataFrame({"대상월": t.dt.strftime("%Y-%m"), "작형": big.season, "S": big.S, "진행률": big.progress.round(2), "평년 진행률": big.p_bar.round(2),
                            "X": big.X.round(2), "P0": big.P0.round(2), "실제 변화%": big.chg.round(0), "final%": ape(big, "final").round(0),
                            "X_all%": ape(big, "X_all").round(0), "X_sea%": ape(big, "X_sea").round(0)}).to_string(index=False))
        key = d[has & (d.origin.dt.month == 8)]
        print("\n[8월 말 원점(9월 대상)]")
        print(pd.DataFrame({"origin": key.origin.dt.date, "S": key.S, "진행률": key.progress.round(2), "평년": key.p_bar.round(2), "X": key.X.round(2),
                            "실제 변화%": key.chg.round(0), "final%": ape(key, "final").round(0), "X_sea%": ape(key, "X_sea").round(0)}).to_string(index=False))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 330)
    pd.set_option("display.max_columns", 40)
    run(sys.argv[1] if len(sys.argv) > 1 else "M")
