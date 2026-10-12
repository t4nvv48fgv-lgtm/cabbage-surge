"""Case 19 — 작형 전환기 오차 대응 두 가지: (1) 산지 전환 진행률로 5·7월 예측 조정, (2) 월중 재예측의 효과 계량(10·12월 중심).

계기(2026-10-12, 사용자 "1번과 2번 묶어서 케이스로 시험해줘"): case18에서 생육기 기상·월보 생산 전망은 기각. 전환기 오차는 방향이 아니라 시점 문제.
사전 점검에서 가락 반입량의 **산지별 비중**이 5월·7월 전환의 진행 정도를 보여 주고 잔차와의 상관이 두 창에서 같은 부호였다
(5월 −0.38/−0.65, 7월 +0.33/+0.45; 월말 원점 11개). 10월·12월은 월말에 새 작형이 아직 안 들어와 신호가 없다 → 월중 갱신으로.

(1) 산지 전환 진행률  (자료: data/ext/nongnet/garak_daily_by_sigungu.csv, aT 농넷 공개, 2011~, 원점 당일까지)
  진행률 P(o) = 원점 직전 7일 가락 반입량 중 새 작형 산지 비중
     T5(5월 대상)  전남 외 산지(봄배추)      T7(7월)  강원(고랭지)      [참고 T10: 강원 외, T12: 전남]
  편차   A(o) = P(o) − 직전 5년 같은 시점(364일 간격) P 의 평균      # 계절 진행을 빼고 '평년보다 이른가'만 남김, 과거 해만 사용
  방향   d_T  = 그 전환기 평균 가격 변동의 부호(2011~2015 실적: 5월 −, 7월 +, 10월 −, 12월 −)
  가설   전환이 평년보다 이르면(A > 0) 가격이 계절 방향으로 모형보다 더 간다 → 잔차 log(실제/final) ~ γ · d_T · A,  γ > 0
  절차   γ 를 2016~2020 주간 원점(대상월 5·7)에서 추정해 고정(prereg) → 2021~2026 검증. 확장 창(expand)·전환기별(bytr)·네 전환기 합동(all4, 참고) 병기.
  조정   pred' = final × exp(γ · d_T · A)

(2) 월중 재예측  (새 자료 없음: 기존 주간 예측 outputs/gate_predictions_W.csv 의 final 과 실현 가격만 사용)
  월말 원점 o 의 대상 창 (o, o+28]. 갱신 시점 u1 = o 뒤 5~11일째 일요일, u2 = u1 + 7일.
  갱신 추정 = [지난 날의 실현 평균 × 지난 거래일 수 + 남은 날 예측 × 남은 거래일 수] / 전체 거래일 수
     남은 날 예측 후보:  hold = 월말 예측 F0 그대로 / reforecast = 시점 u 의 주간 예측 F_u (창이 (u, u+28] 이라 남은 기간과 정확히 같지는 않음, 근사)
  실현분만 반영한 효과(hold)와 재예측의 추가 효과(reforecast − hold)를 나눠 본다. 남은 기간만의 예측 오차도 따로 계산.

Usage: python experiments/case19_transition_progress/transition_progress.py
Outputs: outputs/progress_by_year.csv, coef.csv, predictions.csv, predictions_W.csv, metrics.csv, midmonth_update.csv, midmonth_metrics.csv, run.log
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

TR = {5: ("전라남도", True, -1), 7: ("강원도", False, +1), 10: ("강원도", True, -1), 12: ("전라남도", False, -1)}   # (기준 시도, 여집합 여부, d_T)
MAIN_TR = (5, 7)
EXCLUDE_SIDO = ["-", "중국", "서울특별시"]
VARIANTS = ["sh_prereg", "sh_expand", "sh_bytr_prereg", "sh_bytr_expand", "sh_all4_prereg"]


# ---------------------------------------------------------------- (1) 진행률
def sido_daily() -> pd.DataFrame:
    v = pd.read_csv(ROOT / "data" / "ext" / "nongnet" / "garak_daily_by_sigungu.csv", parse_dates=["date"], encoding="utf-8-sig")
    v = v[~v.origin_sido.isin(EXCLUDE_SIDO)]
    p = v.pivot_table(index="date", columns="origin_sido", values="qty_t", aggfunc="sum")
    return p.reindex(pd.date_range(p.index.min(), p.index.max(), freq="D")).fillna(0.0)


def progress(p: pd.DataFrame, t: int, o: pd.Timestamp) -> float:
    sido, inv, _ = TR[t]
    if o > p.index.max() or o - pd.Timedelta(days=6) < p.index.min():
        return np.nan
    w = p.loc[o - pd.Timedelta(days=6):o]
    tot = float(w.sum().sum())
    if tot <= 0:
        return np.nan
    s = float(w[sido].sum()) / tot
    return 1.0 - s if inv else s


def anomaly(p: pd.DataFrame, t: int, o: pd.Timestamp) -> tuple[float, float]:
    cur = progress(p, t, o)
    prev = [progress(p, t, o - pd.DateOffset(years=k)) for k in range(1, 6)]     # 같은 월·일(달력 기준). 364일 간격은 해마다 1.25일씩 밀려 4~5월처럼 비중이 급변하는 때 기준선을 왜곡
    prev = [x for x in prev if pd.notna(x)]
    if pd.isna(cur) or len(prev) < 3:
        return cur, np.nan
    return cur, cur - float(np.mean(prev))


def load(freq: str, p: pd.DataFrame) -> pd.DataFrame:
    tag = "" if freq == "M" else f"_{freq}"
    g = pd.read_csv(ROOT / "outputs" / f"gate_predictions{tag}.csv", parse_dates=["origin"], encoding="utf-8-sig")
    d = g[["origin", "actual", "tmonth", "gate_hgb", "final"]].copy()
    d["year"] = d.origin.dt.year
    d["tr"] = d.tmonth.where(d.tmonth.isin(list(TR)), 0).astype(int)
    P, A = [], []
    for o, t in zip(d.origin, d.tr):
        c, a = anomaly(p, t, o) if t else (np.nan, np.nan)
        P.append(c); A.append(a)
    d["P"], d["A"] = P, A
    d["x"] = [TR[t][2] * a if (t and pd.notna(a)) else 0.0 for t, a in zip(d.tr, d.A)]
    d["resid"] = np.log(d.actual / d.final)
    return d


def coef(s: pd.DataFrame) -> float:
    s = s[s.resid.notna() & s.A.notna()]
    den = float((s.x ** 2).sum())
    c = float((s.x * s.resid).sum() / den) if den > 0 else 0.0
    return max(c, 0.0)


def raw_coef(s: pd.DataFrame) -> float:
    s = s[s.resid.notna() & s.A.notna()]
    den = float((s.x ** 2).sum())
    return float((s.x * s.resid).sum() / den) if den > 0 else np.nan


def mape(s: pd.DataFrame, c: str) -> float:
    return float(((s[c] - s.actual).abs() / s.actual).mean() * 100) if len(s) else np.nan


# ---------------------------------------------------------------- (2) 월중 재예측
def midmonth(M: pd.DataFrame, W: pd.DataFrame) -> pd.DataFrame:
    price = bf.load_price().price
    wf = W.set_index("origin").final
    rows = []
    for _, r in M.iterrows():
        o = r.origin
        win = price.loc[o + pd.Timedelta(days=1):o + pd.Timedelta(days=28)]
        if len(win) < 10:
            continue
        rec = {"origin": o, "tmonth": int(r.tmonth), "year": o.year, "actual": float(r.actual), "actual_chk": float(win.mean()), "F0": float(r.final)}
        u1 = next((o + pd.Timedelta(days=k) for k in range(5, 12) if (o + pd.Timedelta(days=k)).weekday() == 6), None)
        for name, u in (("u1", u1), ("u2", u1 + pd.Timedelta(days=7) if u1 is not None else None)):
            if u is None or u not in wf.index:
                continue
            done, rest = win.loc[:u], win.loc[u + pd.Timedelta(days=1):]
            if len(done) == 0 or len(rest) == 0:
                continue
            n = len(win)
            comb = lambda f: float((done.sum() + f * len(rest)) / n)     # 산술 평균(메인의 actual = y_price_fwd28_true 정의와 같음)
            rec.update({f"{name}_days": (u - o).days, f"{name}_hold": comb(r.final), f"{name}_refc": comb(wf[u]), f"{name}_avg": (comb(r.final) + comb(wf[u])) / 2, f"{name}_Fu": float(wf[u]),
                        f"{name}_rest_actual": float(rest.mean())})
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 320)
    p = sido_daily()
    M, W = load("M", p), load("W", p)

    # ---------------- (1A) 전환기 × 해 진행률 표 (월말 원점)
    df = pd.read_csv(ROOT / "data" / "processed" / "daily_features.csv", parse_dates=["date"], usecols=["date", "lp28", "y_lp_fwd28"]).set_index("date")
    ends = {5: (4, 30), 7: (6, 30), 10: (9, 30), 12: (11, 30)}
    rows = []
    for t, (mo, dd) in ends.items():
        for y in range(2011, 2027):
            o = pd.Timestamp(y, mo, dd)
            cur, a = anomaly(p, t, o)
            m = M[M.origin == o]
            rows.append({"tr": t, "year": y, "P%": cur * 100 if pd.notna(cur) else np.nan, "A%p": a * 100 if pd.notna(a) else np.nan,
                         "act_chg%": (np.exp(df.y_lp_fwd28.get(o, np.nan) - df.lp28.get(o, np.nan)) - 1) * 100,
                         "err%": float((m.final.iloc[0] / m.actual.iloc[0] - 1) * 100) if len(m) else np.nan, "resid": float(m.resid.iloc[0]) if len(m) else np.nan})
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "progress_by_year.csv", index=False, float_format="%.3f", encoding="utf-8-sig")
    print("=== (1A) 전환기별 산지 진행률 (월말 원점, P = 새 작형 산지 비중, A = 직전 5년 대비 편차) ===")
    print("2011~2015 평균 가격 변동 부호(d_T 근거):", {t: round(float(A[(A.tr == t) & (A.year <= 2015)]["act_chg%"].mean()), 1) for t in TR})
    for t in TR:
        a = A[A.tr == t]
        k = a.dropna(subset=["resid", "A%p"])
        des, val = k[k.year <= 2020], k[k.year >= 2021]
        cc = lambda s: round(float(s["A%p"].corr(s.resid)), 2) if len(s) >= 3 else np.nan
        print(f"\n-- T{t} ({'전남 외' if t == 5 else '강원' if t == 7 else '강원 외' if t == 10 else '전남'} 비중, d={TR[t][2]:+d})")
        print(a.round(1).to_string(index=False))
        print(f"   상관(A, 잔차): 2016~2020 {cc(des)} / 2021~ {cc(val)} / 전체 {cc(k)}   (가설: d<0 이면 음, d>0 이면 양)")

    # ---------------- (1B) 계수
    yW = W.year
    main_tr = W.tr.isin(MAIN_TR)
    design = W[yW.between(2016, 2020)]
    g_pre = coef(design[design.tr.isin(MAIN_TR)])
    g_all4 = coef(design[design.tr > 0])
    g_tr_pre = {t: coef(design[design.tr == t]) for t in TR}
    g_exp = {y: coef(W[(yW >= 2016) & (yW < y) & main_tr]) for y in range(2016, 2027)}
    g_tr_exp = {(y, t): coef(W[(yW >= 2016) & (yW < y) & (W.tr == t)]) for y in range(2016, 2027) for t in TR}
    val = W[yW.between(2021, 2026)]
    C = pd.DataFrame({"gamma_prereg_T5T7": g_pre, "gamma_expand_T5T7": pd.Series(g_exp), "gamma_all4_prereg": g_all4})
    for t in TR:
        C[f"gamma_T{t}_prereg"] = g_tr_pre[t]
        C[f"gamma_T{t}_expand"] = pd.Series({y: g_tr_exp[(y, t)] for y in range(2016, 2027)})
    C.index.name = "year"
    C.round(3).to_csv(OUT / "coef.csv", encoding="utf-8-sig")
    print("\n=== (1B) γ (진행률 편차 1 = 100%p 당 로그 조정) ===")
    print(C.round(2).to_string())
    print("전환기별 원시 γ (2016~2020, 2021~2026), 주간 원점 수:", {t: (round(raw_coef(design[design.tr == t]), 2), round(raw_coef(val[val.tr == t]), 2), int((design.tr == t).sum()), int((val.tr == t).sum())) for t in TR})
    print("T5+T7 합동 원시 γ (2016~2020, 2021~2026):", round(raw_coef(design[design.tr.isin(MAIN_TR)]), 2), round(raw_coef(val[val.tr.isin(MAIN_TR)]), 2))

    # ---------------- (1C) 적용
    for d in (M, W):
        on57 = d.tr.isin(MAIN_TR)
        ge = d.year.map(g_exp).fillna(0.0)
        gt_pre = d.tr.map(g_tr_pre).fillna(0.0)
        gt_exp = pd.Series([g_tr_exp.get((y, t), 0.0) for y, t in zip(d.year, d.tr)], index=d.index)
        d["sh_prereg"] = np.where(on57, d.final * np.exp(g_pre * d.x), d.final)
        d["sh_expand"] = np.where(on57, d.final * np.exp(ge * d.x), d.final)
        d["sh_bytr_prereg"] = np.where(on57, d.final * np.exp(gt_pre * d.x), d.final)
        d["sh_bytr_expand"] = np.where(on57, d.final * np.exp(gt_exp * d.x), d.final)
        d["sh_all4_prereg"] = np.where(d.tr > 0, d.final * np.exp(g_all4 * d.x), d.final)
    M.to_csv(OUT / "predictions.csv", index=False, float_format="%.3f", encoding="utf-8-sig")
    W.to_csv(OUT / "predictions_W.csv", index=False, float_format="%.3f", encoding="utf-8-sig")
    cols = ["final"] + VARIANTS
    met = []
    for freq, d in (("M", M), ("W", W)):
        masks = {"T5T7_2016_2020": d.year.between(2016, 2020) & d.tr.isin(MAIN_TR), "T5T7_2021_2026": d.year.between(2021, 2026) & d.tr.isin(MAIN_TR),
                 "T5_2021_2026": d.year.between(2021, 2026) & (d.tr == 5), "T7_2021_2026": d.year.between(2021, 2026) & (d.tr == 7),
                 "all_2016_2020": d.year.between(2016, 2020), "all_2021_2025": d.year.between(2021, 2025), "all_2021_2026": d.year.between(2021, 2026)}
        for nm, mk in masks.items():
            for c in cols:
                s = d[mk]
                e = (s[c] / s.actual - 1) * 100
                met.append({"freq": freq, "subset": nm, "model": c, "n": len(s), "MAPE%": mape(s, c), "over20": int((e.abs() > 20).sum()), "over30": int((e.abs() > 30).sum())})
    met = pd.DataFrame(met)
    met.to_csv(OUT / "metrics.csv", index=False, float_format="%.2f")
    for freq in ("M", "W"):
        print(f"\n=== (1C) MAPE [{freq}] ===")
        print(met[met.freq == freq].pivot(index="model", columns="subset", values="MAPE%").loc[cols].round(2).to_string())
    full = M.origin >= "2016-01-31"
    rng = M.origin >= "2020-12-31"
    cnt = lambda mk, th: {c: int((((M.loc[mk, c] / M.loc[mk, "actual"]) - 1).abs() > th).sum()) for c in cols}
    print("\n30% 초과 [M, 대상월 2021-01~]:", cnt(rng, 0.30), "\n30% 초과 [M, 2016-02~ 전체]:", cnt(full, 0.30), "\n20% 초과 [M, 2021-01~]:", cnt(rng, 0.20))
    k = M[M.tr.isin(MAIN_TR)].copy()
    for c in cols:
        k[c + "%"] = ((k[c] / k.actual - 1) * 100).round(0)
    print("\n=== 5·7월 대상 월말 원점 (오차 %) ===")
    print(k.assign(**{"P%": (k.P * 100).round(1), "A%p": (k.A * 100).round(1)})[["origin", "tr", "P%", "A%p"] + [c + "%" for c in cols]].to_string(index=False))

    # ---------------- (2) 월중 재예측
    U = midmonth(M, W)
    print("\n=== (2) 월중 재예측 ===")
    print("대상 창 실제 재계산 일치 여부(산술 평균): 최대 차이 %.3f%%" % float(((U.actual_chk / U.actual - 1).abs() * 100).max()))
    U.to_csv(OUT / "midmonth_update.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    ape = lambda s, c, ref="actual": float(((s[c] / s[ref] - 1).abs() * 100).mean())
    rows = []
    groups = {"전환 5·7·10·12": U.tmonth.isin([5, 7, 10, 12]), "10·12월": U.tmonth.isin([10, 12]), "5·7월": U.tmonth.isin([5, 7]), "그 외 8개 달": ~U.tmonth.isin([5, 7, 10, 12]), "전체": U.tmonth > 0}
    for lo, hi in ((2016, 2020), (2021, 2026)):
        for nm, mk in groups.items():
            s = U[mk & U.year.between(lo, hi)].dropna(subset=["u1_hold", "u2_hold"])
            if not len(s):
                continue
            rows.append({"window": f"{lo}~{hi}", "group": nm, "n": len(s), "월말 예측": ape(s, "F0"),
                         "1주 뒤 실현분만": ape(s, "u1_hold"), "1주 뒤 재예측": ape(s, "u1_refc"), "1주 뒤 둘의 평균": ape(s, "u1_avg"), "2주 뒤 실현분만": ape(s, "u2_hold"), "2주 뒤 재예측": ape(s, "u2_refc"), "2주 뒤 둘의 평균": ape(s, "u2_avg"),
                         "남은기간 오차: 월말 예측(1주)": ape(s, "F0", "u1_rest_actual"), "남은기간 오차: 재예측(1주)": ape(s, "u1_Fu", "u1_rest_actual"),
                         "남은기간 오차: 월말 예측(2주)": ape(s, "F0", "u2_rest_actual"), "남은기간 오차: 재예측(2주)": ape(s, "u2_Fu", "u2_rest_actual"),
                         "30%초과: 월말": int(((s.F0 / s.actual - 1).abs() > 0.3).sum()), "30%초과: 1주 재예측": int(((s.u1_refc / s.actual - 1).abs() > 0.3).sum()),
                         "30%초과: 2주 재예측": int(((s.u2_refc / s.actual - 1).abs() > 0.3).sum()), "30%초과: 1주 평균": int(((s.u1_avg / s.actual - 1).abs() > 0.3).sum()), "30%초과: 2주 평균": int(((s.u2_avg / s.actual - 1).abs() > 0.3).sum()), "평균 경과일(1주)": float(s.u1_days.mean())})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "midmonth_metrics.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    print(R.round(1).T.to_string())
    big = U[((U.F0 / U.actual - 1).abs() > 0.30)].copy()
    big["tm"] = (big.origin + pd.Timedelta(days=15)).dt.strftime("%Y-%m")
    for c in ("F0", "u1_hold", "u1_refc", "u1_avg", "u2_hold", "u2_refc", "u2_avg"):
        big[c + "%"] = ((big[c] / big.actual - 1) * 100).round(0)
    print("\n30% 초과 달의 갱신 전후 (대상 창 전체 평균 기준 오차 %):")
    print(big[["tm", "F0%", "u1_hold%", "u1_refc%", "u1_avg%", "u2_hold%", "u2_refc%", "u2_avg%"]].to_string(index=False))


if __name__ == "__main__":
    main()
