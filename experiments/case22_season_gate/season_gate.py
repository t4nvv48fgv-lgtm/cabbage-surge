"""Case 22 — 작형별 생산 전망 평시·경보 게이트 + 여름 게이트·경보 조정의 2016~2020 소급.

사용자 지시(2026-10-13): "봄·가을·겨울도 생산량으로 평시·경보 나눠서 시험해주고 여름도 과거시점에 경보를 넣어줘".

메인(src/gate_model.py)은 여름 원점(6~9월)에서만 KREI 월보 여름배추 생산 전망 전년비 S1 로 평시(S1 > 0 → wx_hgb_summer)·경보(S1 ≤ 0 →
대표 an_sparse_all × (1 − 0.566·S/100))를 가르고, 신호는 2021년 이후만 쓴다. 이 케이스는 두 가지를 본다.

A. 여름 소급: case13 판독표(2016~2020 6~9월호)를 붙여 같은 규칙을 2016~2020 여름 원점에 적용.
   변형 = 평시 전환만 / 경보 조정만 / 둘 다. β −0.566 은 2016~2020 경보 원점에서 추정한 값이라 그 구간에서는 표본내 →
   그 해를 뺀 나머지 해로 다시 추정한 β(해 제외 추정)로도 본다. 9월 고온 층은 메인 함수로 다시 계산(γ 확장 창이 바뀔 수 있음).
B. 비여름 작형(대상월 겨울 1~4·봄 5~6·가을 11~12): 그 작형 생산 전망 전년비 S(규약 B, 원점에서 알 수 있는 최신호; 없으면 면적)로
   평시(S > 0)·경보(S ≤ 0)를 나눈다. 신호 = case16 판독표(2015~2020) + data/ext/krei_nonsummer_issues.csv(2021~).
   규칙(모두 2016~2020 에서 정하고 2021~ 에서 검증):
     fixed    여름과 같은 짝: 평시 → wx_hgb, 경보 → an_sparse_all
     sel      (작형, 상태)별로 2016~2020 MAPE 가 가장 낮은 후보 모형
     pooled   상태를 나누지 않고 작형별로 2016~2020 MAPE 가 가장 낮은 후보(대조군: 상태 분리의 몫만 보려는 것)
     uplift   현행 모형 × (1 + β·S/100) 을 경보 원점에만. β = 2016~2020 경보 원점 잔차 회귀(절편 없음, 양수면 0)
   현행 = 메인 final(겨울 = 겨울 작형 가격만 ridge, 봄 = 대표, 가을 = 대표 × 대칭 β −0.216[2021~만 신호 있음]).

Usage: python experiments/case22_season_gate/season_gate.py [M|W]
Outputs: outputs/summer_backfill{tag}.csv, summer_backfill_metrics{tag}.csv, season_cells{tag}.csv, season_rules{tag}.csv,
         season_predictions{tag}.csv
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
import gate_model as gm  # noqa: E402

HEAD, CALM = gm.HEAD, gm.CALM
SEASON = {1: "겨울", 2: "겨울", 3: "겨울", 4: "겨울", 5: "봄", 6: "봄", 11: "가을", 12: "가을"}
CANDS = ["price_only_ridge", "wx_ridge", "wx_hgb", "an_sparse_all", "chronos2", "mstl_ets", "winter_ridge"]
WINDOWS = {"2016~2020": (2016, 2020), "2021~2025": (2021, 2025), "2021~2026": (2021, 2026)}
SUMMER_OLD = ROOT / "experiments" / "case13_krei_backfill" / "krei_summer_issues_2010_2020.csv"
NS_OLD = ROOT / "experiments" / "case16_season_supply_adjust" / "krei_nonsummer_issues_2015_2020.csv"


# ---------- 신호 ----------
def summer_issues_ext() -> pd.DataFrame:
    new = gm.load_issues()
    old = pd.read_csv(SUMMER_OLD, parse_dates=["pub_date"])
    old = old[old.prod_yoy.notna() | old.area_yoy.notna()].copy()
    old["avail_b"] = pd.to_datetime(dict(year=old.year, month=old.issue_month, day=1)) - pd.Timedelta(days=1)
    cols = ["year", "issue_month", "pub_date", "avail_b", "prod_yoy", "area_yoy"]
    return pd.concat([old[cols], new[cols]]).sort_values("pub_date").reset_index(drop=True)


def nonsummer_issues() -> pd.DataFrame:
    cols = ["season", "sub", "crop_year", "year", "issue_month", "prod_yoy", "area_yoy"]
    d = pd.concat([pd.read_csv(NS_OLD)[cols], pd.read_csv(gm.SEASON_ISSUES, encoding="utf-8-sig")[cols]])
    d = d[d.prod_yoy.notna() | d.area_yoy.notna()].copy()
    d["avail_b"] = pd.to_datetime(dict(year=d.year, month=d.issue_month, day=1)) - pd.Timedelta(days=1)   # 규약 B
    return d.sort_values(["season", "crop_year", "year", "issue_month"]).reset_index(drop=True)


def ns_signal(iss: pd.DataFrame, origin: pd.Timestamp, sea: str, ty: int) -> tuple[str | float, float]:
    """규약 B로 원점에서 알 수 있는 그 작형 최신호의 생산 전망 전년비(없으면 면적). 겨울은 전년산(대상 연도 − 1)."""
    crop = ty - 1 if sea == "겨울" else ty
    s = iss[(iss.season == sea) & (iss.crop_year == crop) & (iss.avail_b <= origin)]
    if s.empty:
        return np.nan, np.nan
    last = s[(s.year == s.year.iloc[-1]) & (s.issue_month == s.issue_month.iloc[-1])]
    if sea == "봄":          # 같은 호 안에서 생산 전년비가 있는 행 우선, 전체 > 노지 > 시설
        last = last.assign(p=last["sub"].map({"전체": 0, "노지": 1, "시설": 2})).sort_values("p")
        withp = last[last.prod_yoy.notna()]
        last = withp if len(withp) else last
    r = last.iloc[0] if sea == "봄" else last.iloc[-1]
    v = r.prod_yoy if pd.notna(r.prod_yoy) else r.area_yoy
    return f"{int(r.year)}-{int(r.issue_month):02d}", float(v)


# ---------- A. 여름 소급 ----------
def main_like(d: pd.DataFrame, sum_iss: pd.DataFrame, fall: pd.DataFrame, monthly: bool, calm_switch: bool = True,
              alert_beta: bool = True, beta_by_year: dict | None = None) -> pd.DataFrame:
    """gm.run 의 2~4단계를 주어진 여름 호 표로 다시 계산. beta_by_year 가 있으면 그 해의 경보 β 로 쓴다."""
    d = d.copy()
    sig = [gm.latest_signal(sum_iss, o) if s == 1 else (np.nan, np.nan) for o, s in zip(d.origin, d.summer)]
    d["S1x"] = [v for _, v in sig]
    calm = d.S1x.notna() & (d.S1x > 0)
    gate = np.where(calm & calm_switch, d[CALM], d[HEAD])
    alert = (d.summer == 1) & d.S1x.notna() & ~calm
    ty = (d.origin + pd.Timedelta(days=15)).dt.year
    winter = d.tmonth.isin(gm.WINTER_MONTHS) & d.winter_ridge.notna()
    factor, s_used = [], []
    for o, y, tm, al in zip(d.origin, ty, d.tmonth, alert):
        s, beta = np.nan, 0.0
        if al and alert_beta:
            s = gm.adjust_signal(sum_iss, o, o.year)[1]
            b = gm.BETA_SUMMER_ALERT if beta_by_year is None else beta_by_year.get(o.year, gm.BETA_SUMMER_ALERT)
            beta = b if (pd.notna(s) and s <= 0) else 0.0
        elif tm in gm.FALL_MONTHS:
            s = gm.adjust_signal(fall, o, y, "crop_year")[1]
            beta = gm.BETA_FALL
        f = 1.0 if (pd.isna(s) or beta == 0.0) else 1.0 + beta * float(np.clip(s, -gm.S_CAP, gm.S_CAP)) / 100.0
        factor.append(f)
        s_used.append(s if f != 1.0 else np.nan)
    d["calm_x"], d["alert_x"], d["S_adj_x"], d["adj_pct_x"] = calm.astype(int), alert.astype(int), s_used, (np.array(factor) - 1) * 100
    d["gate_x"] = gate
    d["final_base"] = np.where(winter, d.winter_ridge, gate * np.array(factor))
    return gm.heat_hold(d, fall, monthly)


def loyo_beta(d: pd.DataFrame, sum_iss: pd.DataFrame, calm_switch: bool) -> tuple[float, dict]:
    """2016~2020 경보 원점의 log(실제/게이트 예측) ~ β·S/100 (절편 없음, 양수면 0). 전체 추정값과 '그 해를 뺀' 추정값."""
    sig = [gm.latest_signal(sum_iss, o) if s == 1 else (np.nan, np.nan) for o, s in zip(d.origin, d.summer)]
    s1 = pd.Series([v for _, v in sig], index=d.index)
    calm = s1.notna() & (s1 > 0)
    alert = (d.summer == 1) & s1.notna() & ~calm & d.origin.dt.year.between(2016, 2020)
    k = d[alert].copy()
    k["S"] = [gm.adjust_signal(sum_iss, o, o.year)[1] for o in k.origin]
    k = k[k.S.notna() & (k.S <= 0)]
    k["x"] = k.S.clip(-gm.S_CAP, gm.S_CAP) / 100
    k["y"] = np.log(k.actual / np.where(calm[k.index] & calm_switch, k[CALM], k[HEAD]))
    est = lambda s: min(0.0, float((s.x * s.y).sum() / (s.x ** 2).sum())) if len(s) and (s.x ** 2).sum() > 0 else 0.0
    return est(k), {y: est(k[k.origin.dt.year != y]) for y in range(2016, 2021)}


def mape(s: pd.DataFrame, c: str) -> float:
    return float(((s[c] - s.actual).abs() / s.actual).mean() * 100) if len(s) else np.nan


def bias(s: pd.DataFrame, c: str) -> float:
    return float(((s[c] / s.actual - 1)).mean() * 100) if len(s) else np.nan


def part_a(d: pd.DataFrame, monthly: bool, tag: str) -> pd.DataFrame:
    fall, new, ext = gm.load_fall_issues(), gm.load_issues(), summer_issues_ext()
    chk = main_like(d, new, fall, monthly)
    assert np.allclose(chk[gm.FINAL], d[gm.FINAL], rtol=0, atol=0.11), "메인 재현 실패"
    full, by_year = loyo_beta(d, ext, calm_switch=True)
    print(f"[A] 2016~2020 경보 원점 잔차 β(전체) {full:+.3f} (메인 고정값 {gm.BETA_SUMMER_ALERT:+.3f}), 해 제외 추정: "
          + ", ".join(f"{y} {b:+.2f}" for y, b in by_year.items()))
    var = {"현행": d,
           "평시전환만": main_like(d, ext, fall, monthly, calm_switch=True, alert_beta=False),
           "경보조정만": main_like(d, ext, fall, monthly, calm_switch=False, alert_beta=True),
           "둘다": main_like(d, ext, fall, monthly),
           "둘다(해제외β)": main_like(d, ext, fall, monthly, beta_by_year=by_year)}
    o = var["둘다"][["origin", "actual", "summer", "tmonth", "S1x", "calm_x", "alert_x", "S_adj_x", "adj_pct_x", HEAD, CALM]].copy()
    for k, v in var.items():
        o[k] = v[gm.FINAL].values
    o.to_csv(OUT / f"summer_backfill{tag}.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    y = o.origin.dt.year
    masks = {"여름원점 2016~2020": y.between(2016, 2020) & (o.summer == 1), "전체 2016~2020": y.between(2016, 2020),
             "여름원점 2021~2025": y.between(2021, 2025) & (o.summer == 1), "전체 2021~2025": y.between(2021, 2025),
             "전체 2016~2026": y.between(2016, 2026)}
    rows = []
    for name, mk in masks.items():
        s = o[mk]
        for k in var:
            e = (s[k] / s.actual - 1) * 100
            rows.append({"subset": name, "variant": k, "n": len(s), "MAPE%": e.abs().mean(), "bias%": e.mean(), "n_gt30": int((e.abs() > 30).sum())})
    met = pd.DataFrame(rows)
    met.to_csv(OUT / f"summer_backfill_metrics{tag}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    print("\n[A] MAPE% (괄호 밖), 30% 초과 수")
    print(met.pivot(index="variant", columns="subset", values="MAPE%").loc[list(var)].round(2).to_string())
    print(met.pivot(index="variant", columns="subset", values="n_gt30").loc[list(var)].to_string())
    if monthly:
        s = o[y.between(2016, 2020) & (o.summer == 1)].copy()
        for k in var:
            s[k + "%"] = ((s[k] / s.actual - 1) * 100).round(0)
        print("\n[A] 2016~2020 여름 원점")
        print(s[["origin", "actual", "S1x", "calm_x", "adj_pct_x"] + [k + "%" for k in var]].round(1).to_string(index=False))
        diff = o[(y >= 2021) & ((o["둘다"] - o["현행"]).abs() > 0.5)]
        print("\n[A] 소급으로 2021년 이후 값이 바뀌는 원점(9월 고온 층 γ 재추정):", "없음" if diff.empty else "")
        if not diff.empty:
            print(diff.assign(cur=lambda t: (t["현행"] / t.actual - 1) * 100, new=lambda t: (t["둘다"] / t.actual - 1) * 100)[["origin", "actual", "현행", "둘다", "cur", "new"]].round(1).to_string(index=False))
    return o


# ---------- B. 비여름 작형 ----------
def part_b(d: pd.DataFrame, tag: str) -> pd.DataFrame:
    iss = nonsummer_issues()
    d = d.copy()
    ty = (d.origin + pd.Timedelta(days=15)).dt.year
    d["season"] = np.where(d.summer == 1, "여름", d.tmonth.map(SEASON).fillna("기타"))
    sig = [ns_signal(iss, o, sea, y) if sea in ("겨울", "봄", "가을") else (np.nan, np.nan) for o, sea, y in zip(d.origin, d.season, ty)]
    d["ns_issue"], d["S"] = [k for k, _ in sig], [v for _, v in sig]
    d["state"] = np.where(d.S.isna(), "신호없음", np.where(d.S > 0, "평시", "경보"))
    d["cur"] = d[gm.FINAL]
    d["base"] = np.where(d.season == "가을", d.gate_hgb, d[gm.FINAL])     # 가을은 대칭 β 를 뺀 값이 조정의 바탕
    yr = d.origin.dt.year
    design = yr.between(2016, 2020)
    # 칸별 성적
    cells = []
    for sea in ("겨울", "봄", "가을"):
        for wname, (a, b) in WINDOWS.items():
            for st in ("평시", "경보"):
                s = d[(d.season == sea) & yr.between(a, b) & (d.state == st)]
                row = {"season": sea, "window": wname, "state": st, "n": len(s), "S_mean": s.S.mean()}
                for c in ["cur"] + CANDS:
                    if c == "winter_ridge" and sea != "겨울":
                        continue
                    row[c] = mape(s, c)
                    row[c + "_bias"] = bias(s, c)
                cells.append(row)
    cells = pd.DataFrame(cells)
    cells.to_csv(OUT / f"season_cells{tag}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    # 규칙
    d["fixed"], d["sel"], d["pooled"], d["uplift"] = d.cur, d.cur, d.cur, d.cur
    picks = []
    for sea in ("겨울", "봄", "가을"):
        cands = [c for c in CANDS if c != "winter_ridge" or sea == "겨울"]
        mk = (d.season == sea) & (d.state != "신호없음")
        d.loc[mk & (d.state == "평시"), "fixed"] = d.wx_hgb
        d.loc[mk & (d.state == "경보"), "fixed"] = d.an_sparse_all
        pool = d[mk & design]
        best_all = min(cands, key=lambda c: mape(pool, c)) if len(pool) else "cur"
        d.loc[mk, "pooled"] = d.loc[mk, best_all]
        row = {"season": sea, "pooled": best_all}
        for st in ("평시", "경보"):
            s = pool[pool.state == st]
            best = min(cands, key=lambda c: mape(s, c)) if len(s) >= 3 else best_all
            d.loc[mk & (d.state == st), "sel"] = d.loc[mk & (d.state == st), best]
            row[f"sel_{st}"], row[f"n_{st}"] = best, len(s)
        al = pool[pool.state == "경보"]
        x, yv = al.S.clip(-gm.S_CAP, gm.S_CAP) / 100, np.log(al.actual / al.base)
        beta = min(0.0, float((x * yv).sum() / (x ** 2).sum())) if len(al) >= 3 else 0.0
        a_mk = mk & (d.state == "경보")
        d.loc[mk, "uplift"] = d.loc[mk, "base"]
        d.loc[a_mk, "uplift"] = d.loc[a_mk, "base"] * (1 + beta * d.loc[a_mk, "S"].clip(-gm.S_CAP, gm.S_CAP) / 100)
        row["beta_alert"] = beta
        picks.append(row)
    picks = pd.DataFrame(picks)
    rules = ["cur", "fixed", "sel", "pooled", "uplift"]
    rows = []
    for wname, (a, b) in WINDOWS.items():
        for sea in ("겨울", "봄", "가을", "전체"):
            s = d[yr.between(a, b) & ((d.season == sea) if sea != "전체" else True)]
            for r in rules:
                e = (s[r] / s.actual - 1) * 100
                rows.append({"window": wname, "season": sea, "rule": r, "n": len(s), "MAPE%": e.abs().mean(), "bias%": e.mean(), "n_gt30": int((e.abs() > 30).sum())})
    met = pd.DataFrame(rows)
    met.to_csv(OUT / f"season_rules{tag}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    keep = ["origin", "actual", "tmonth", "season", "ns_issue", "S", "state"] + CANDS + rules
    d[keep].to_csv(OUT / f"season_predictions{tag}.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    print("\n[B] 작형·상태별 원점 수와 현행 모형의 MAPE·편향(%)")
    print(cells[["season", "window", "state", "n", "S_mean", "cur", "cur_bias"]].round(1).to_string(index=False))
    print("\n[B] 작형·상태별 후보 모형 MAPE%")
    show = cells[["season", "window", "state", "n"] + [c for c in CANDS]].round(1)
    print(show.to_string(index=False))
    print("\n[B] 2016~2020 에서 고른 것")
    print(picks.round(3).to_string(index=False))
    print("\n[B] 규칙별 MAPE%")
    print(met.pivot_table(index=["window", "season"], columns="rule", values="MAPE%")[rules].round(2).to_string())
    print("\n[B] 규칙별 30% 초과 수")
    print(met.pivot_table(index=["window", "season"], columns="rule", values="n_gt30")[rules].to_string())
    return d


def load(freq: str) -> pd.DataFrame:
    tag = "" if freq == "M" else f"_{freq}"
    g = pd.read_csv(ROOT / "outputs" / f"gate_predictions{tag}.csv", parse_dates=["origin"], encoding="utf-8-sig")
    b = pd.read_csv(ROOT / "outputs" / f"backtest_predictions{tag}.csv", parse_dates=["origin"])
    return g.merge(b[["origin", "wx_ridge", "wx_hgb", "mstl_ets"]], on="origin", how="left")


if __name__ == "__main__":
    freq = sys.argv[1] if len(sys.argv) > 1 else "M"
    tag = "" if freq == "M" else f"_{freq}"
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 330)
    pd.set_option("display.max_columns", 40)
    d = load(freq)
    print(f"===== {freq}: 원점 {len(d)}개 =====")
    part_a(d, monthly=(freq == "M"), tag=tag)
    part_b(d, tag=tag)
