"""Case 18 — 작형 전환기(대상월 5·7·10·12월)의 예측을, 새로 들어오는 작형의 생육기 기상과 월보 관측정보로 조정할 수 있는가.

계기(2026-10-12, 사용자): "작형이 변하는 시기에 가격 변동을 조정할 수 있을까? 5·7·10·12월." → "전환기 생육 기상 케이스로 시험해줘. 혹은 사전적으로 얻을 수 있는 관측정보도 활용해봐."
진단: 5·7·10월은 MAPE 20~22%로 가장 크고, 그 달 안에서 예측 변동과 실제 변동의 상관이 0.2~0.4(다른 달 0.6~0.9). 평균 방향·크기는 맞고 해별 편차를 못 본다.
과거 가격·모형 출력만으로 변동폭을 재보정하면 악화(12.7→17.8~19.0%) → 새 작형에 대한 사전 정보가 필요하다.

전환기 정의(대상월 = 원점+15일의 달). 새 작형 산지·생육 창·스트레스 부호는 **결과를 보기 전에 재배력으로 정한다**(단 T10은 case17에서 이미 봄).
  T5  5월 ← 봄배추(시설·조기 노지)   관측소 예산·아산·해남·의령군·고창   창 3/1~4/30   스트레스 = −최고기온 편차(저온 → 생육 지연)
  T7  7월 ← 고랭지 여름배추           평창·태백·정선군·강릉               창 5/1~6/30   스트레스 = +최고기온 편차(고온 → 정식·초기 생육 부진)
  T10 10월 ← 가을배추                 build_features.LOWLAND_AUTUMN 9곳   창 9/1~9/30   스트레스 = +최고기온 편차(고온 → 활착 지연)
  T12 12월 ← 가을 끝물·월동배추       해남·진도군·고창                    창 10/1~11/30 스트레스 = −최고기온 편차(저온 → 생육·출하 부진)
  z = 스트레스 / σ_T (σ_T = 2011~2020 해별 창 평균 편차의 표준편차), |z| ≤ 2.5. 원점이 창 끝 이전이면 원점까지의 부분 창.
가설: z > 0 이면 새 작형이 늦거나 적어 모형보다 가격이 높다 → 잔차 log(실제/final) ~ γ·z, γ > 0.

월보 관측정보(규약 B, 원점에서 아는 최신호): 새 작형의 생산량 전망 전년비 S(없으면 면적), 단수 전년비 Y = (1+생산)/(1+면적) − 1 (둘 다 있을 때).
  T5 봄(전체>노지>시설) / T7 여름 / T10 가을 / T12 겨울(그해 정식분). 2016~2020은 판독표가 있는 호만(T10 가을 10월호 없음).
  가격 전망·출하량 전망·포전가격은 쓰지 않는다(사용자 결정: 순환·수집 자료).
가설: S > 0 이면 공급이 늘어 모형보다 가격이 낮다 → 잔차 ~ β·S/100, β < 0.

절차(case16·17과 같음): 계수를 2016~2020 전환기 원점(주간)에서 추정해 고정(prereg) → 2021~2026 검증. 확장 창(expand: 해 y는 y−1까지)도 병기.
조정: pred' = final × exp(γ·z + β·S/100). 변형 wx(기상만)·k(월보만)·both, 각각 prereg/expand, 기상은 전환기별 γ(bytr)도.

Usage: python experiments/case18_transition_growth/transition_growth.py
Outputs: outputs/signals_by_year.csv, coef.csv, predictions.csv, predictions_W.csv, metrics.csv, run.log
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

TR = {
    5:  {"regions": ["예산", "아산", "해남", "의령군", "고창"], "start": (3, 1), "end": (4, 30), "sign": -1, "season": "봄"},
    7:  {"regions": bf.HIGHLAND, "start": (5, 1), "end": (6, 30), "sign": +1, "season": "여름"},
    10: {"regions": bf.LOWLAND_AUTUMN, "start": (9, 1), "end": (9, 30), "sign": +1, "season": "가을"},
    12: {"regions": ["해남", "진도군", "고창"], "start": (10, 1), "end": (11, 30), "sign": -1, "season": "겨울"},
}
ZCAP = 2.5
VARIANTS = ["wx_prereg", "wx_expand", "wx_bytr_expand", "k_prereg", "k_expand", "k_bytr_prereg", "k_bytr_expand", "both_prereg", "both_expand", "wxdet_prereg", "wxdet_expand"]


# ---------------------------------------------------------------- 기상
def region_frames() -> dict[int, pd.DataFrame]:
    w = bf.load_weather()
    out = {}
    for t, c in TR.items():
        L = bf.region_daily(w, c["regions"])
        L = L.reindex(pd.date_range(L.index.min(), L.index.max(), freq="D"))
        out[t] = bf.add_anomalies(L, ["tmax", "tmin", "rain"], "a_")
    return out


def window_stats(L: pd.DataFrame, t: int, o: pd.Timestamp) -> dict:
    c = TR[t]
    a, b = pd.Timestamp(o.year, *c["start"]), min(o, pd.Timestamp(o.year, *c["end"]))
    if b < a:
        return {"dT": np.nan, "dTmin": np.nan, "dR": np.nan, "hot30": np.nan}
    s = L.loc[a:b]
    return {"dT": float(s.a_tmax_anom.mean()), "dTmin": float(s.a_tmin_anom.mean()), "dR": float(s.a_rain_anom.mean()), "hot30": float(s.hot30.sum())}


def sigma_by_transition(LS: dict) -> dict[int, float]:
    """2011~2020 해별 전체 창 평균 최고기온 편차의 표준편차(검증 창 이전 자료만)."""
    return {t: float(np.nanstd([window_stats(LS[t], t, pd.Timestamp(y, *TR[t]["end"]))["dT"] for y in range(2011, 2021)], ddof=1)) for t in TR}


# ---------------------------------------------------------------- 월보
def load_issues() -> pd.DataFrame:
    cols = ["season", "sub", "crop_year", "year", "issue_month", "prod_yoy", "area_yoy"]
    ns_old = pd.read_csv(ROOT / "experiments" / "case16_season_supply_adjust" / "krei_nonsummer_issues_2015_2020.csv")[cols]
    ns_new = pd.read_csv(ROOT / "data" / "ext" / "krei_nonsummer_issues.csv", encoding="utf-8-sig")[cols]
    su_old = pd.read_csv(ROOT / "experiments" / "case13_krei_backfill" / "krei_summer_issues_2010_2020.csv")[["year", "issue_month", "prod_yoy", "area_yoy"]]
    su_new = pd.read_csv(ROOT / "data" / "ext" / "krei_summer_issues.csv", encoding="utf-8-sig")[["year", "issue_month", "prod_yoy", "area_yoy"]]
    su = pd.concat([su_old, su_new]).assign(season="여름", sub="여름", crop_year=lambda x: x.year)[cols]
    d = pd.concat([ns_old, ns_new, su]).drop_duplicates(["season", "sub", "crop_year", "year", "issue_month"], keep="last")
    d["avail"] = pd.to_datetime(dict(year=d.year, month=d.issue_month, day=1)) - pd.Timedelta(days=1)   # 규약 B
    return d.sort_values(["season", "crop_year", "year", "issue_month"]).reset_index(drop=True)


def krei_signal(iss: pd.DataFrame, t: int, o: pd.Timestamp) -> tuple[float, float]:
    """(S 생산 전년비 또는 면적, Y 단수 전년비). 새 작형의 원점 연도 작형, 규약 B 최신호."""
    sea = TR[t]["season"]
    s = iss[(iss.season == sea) & (iss.crop_year == o.year) & (iss.avail <= o) & (iss.prod_yoy.notna() | iss.area_yoy.notna())]
    if s.empty:
        return np.nan, np.nan
    last = s[(s.year == s.year.iloc[-1]) & (s.issue_month == s.issue_month.iloc[-1])]
    if sea == "봄":
        pri = {"전체": 0, "노지": 1, "시설": 2}
        last = last.assign(p=last["sub"].map(pri)).sort_values("p")
        withp = last[last.prod_yoy.notna()]
        r = withp.iloc[0] if len(withp) else last.iloc[0]
    else:
        r = last.iloc[-1]
    S = float(r.prod_yoy) if pd.notna(r.prod_yoy) else float(r.area_yoy)
    Y = ((1 + r.prod_yoy / 100) / (1 + r.area_yoy / 100) - 1) * 100 if (pd.notna(r.prod_yoy) and pd.notna(r.area_yoy)) else np.nan
    return float(np.clip(S, -20, 20)), (float(Y) if pd.notna(Y) else np.nan)


# ---------------------------------------------------------------- 자료 결합
def load(freq: str, LS: dict, sig: dict, iss: pd.DataFrame) -> pd.DataFrame:
    tag = "" if freq == "M" else f"_{freq}"
    g = pd.read_csv(ROOT / "outputs" / f"gate_predictions{tag}.csv", parse_dates=["origin"], encoding="utf-8-sig")
    b = pd.read_csv(ROOT / "outputs" / f"backtest_predictions{tag}.csv", parse_dates=["origin"])
    d = g[["origin", "actual", "tmonth", "gate_hgb", "final"]].merge(b[["origin", "lp28"]], on="origin")
    d["year"] = d.origin.dt.year
    d["tr"] = d.tmonth.where(d.tmonth.isin(list(TR)), 0).astype(int)
    rec = []
    for o, t in zip(d.origin, d.tr):
        if t == 0:
            rec.append({"dT": np.nan, "dTmin": np.nan, "dR": np.nan, "hot30": np.nan, "z": 0.0, "zdet": 0.0, "S": np.nan, "Y": np.nan})
            continue
        ws = window_stats(LS[t], t, o)
        z = float(np.clip(TR[t]["sign"] * ws["dT"] / sig[t], -ZCAP, ZCAP)) if pd.notna(ws["dT"]) else 0.0
        S, Y = krei_signal(iss, t, o)
        prev = [window_stats(LS[t], t, o - pd.DateOffset(years=k))["dT"] for k in range(1, 6) if o.year - k >= 2010]
        base = float(np.nanmean(prev)) if len(prev) and not np.all(np.isnan(prev)) else 0.0
        zdet = float(np.clip(TR[t]["sign"] * (ws["dT"] - base) / sig[t], -3.0, 3.0)) if pd.notna(ws["dT"]) else 0.0   # 민감도: 직전 5년 평균 대비(기후 추세 제거), 결과를 본 뒤 추가
        rec.append({**ws, "z": z, "zdet": zdet, "S": S, "Y": Y})
    d = pd.concat([d, pd.DataFrame(rec)], axis=1)
    d["Sx"] = d.S.fillna(0.0) / 100
    d["resid"] = np.log(d.actual / d.final)
    d["dact"] = np.log(d.actual) - d.lp28
    return d


def coef(s: pd.DataFrame, x: str, sign: int) -> float:
    """절편 없는 잔차 회귀 계수. 가설 부호와 다르면 0."""
    s = s[(s.tr > 0) & s.resid.notna()]
    den = float((s[x] ** 2).sum())
    c = float((s[x] * s.resid).sum() / den) if den > 0 else 0.0
    return c if c * sign > 0 else 0.0


def mape(s: pd.DataFrame, c: str) -> float:
    return float(((s[c] - s.actual).abs() / s.actual).mean() * 100) if len(s) else np.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 320)
    LS = region_frames()
    sig = sigma_by_transition(LS)
    iss = load_issues()
    M, W = load("M", LS, sig, iss), load("W", LS, sig, iss)
    print("σ_T (2011~2020 창 평균 최고기온 편차의 표준편차, °C):", {t: round(v, 2) for t, v in sig.items()})

    # ---------------- (A) 전환기 × 해 신호표 (월말 원점)
    df = pd.read_csv(ROOT / "data" / "processed" / "daily_features.csv", parse_dates=["date"], usecols=["date", "lp28", "y_lp_fwd28"]).set_index("date")
    rows = []
    for t, c in TR.items():
        for y in range(2011, 2027):
            o = pd.Timestamp(y, *c["end"])
            if o not in df.index:
                continue
            ws = window_stats(LS[t], t, o)
            S, Y = krei_signal(iss, t, o)
            m = M[M.origin == o]
            rows.append({"tr": t, "year": y, "dT": ws["dT"], "dR": ws["dR"], "hot30": ws["hot30"], "z": np.clip(c["sign"] * ws["dT"] / sig[t], -ZCAP, ZCAP),
                         "S": S, "Y": Y, "act_chg%": (np.exp(df.y_lp_fwd28.get(o, np.nan) - df.lp28.get(o, np.nan)) - 1) * 100,
                         "err%": float((m.final.iloc[0] / m.actual.iloc[0] - 1) * 100) if len(m) else np.nan, "resid": float(m.resid.iloc[0]) if len(m) else np.nan})
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "signals_by_year.csv", index=False, float_format="%.3f", encoding="utf-8-sig")
    print("\n=== (A) 전환기별 해별 신호 (월말 원점; z>0 = 새 작형 생육 스트레스) ===")
    for t in TR:
        a = A[A.tr == t]
        k = a.dropna(subset=["resid"])
        des, val = k[k.year <= 2020], k[k.year >= 2021]
        print(f"\n-- T{t} ({TR[t]['season']}, {'·'.join(TR[t]['regions'][:5])}{'…' if len(TR[t]['regions']) > 5 else ''}, {TR[t]['start'][0]}/{TR[t]['start'][1]}~{TR[t]['end'][0]}/{TR[t]['end'][1]})")
        print(a[["year", "dT", "dR", "hot30", "z", "S", "Y", "act_chg%", "err%", "resid"]].round(2).to_string(index=False))
        cc = lambda s, x: round(float(s[x].corr(s.resid)), 2) if s[x].notna().sum() >= 3 else np.nan
        print(f"   상관(z, 잔차): 2016~2020 {cc(des, 'z')} / 2021~ {cc(val, 'z')} / 전체 {cc(k, 'z')} | 상관(S, 잔차): 2016~2020 {cc(des, 'S')} / 2021~ {cc(val, 'S')} | 상관(Y, 잔차) 전체 {cc(k, 'Y')}"
              f" | 상관(z, 실제 변동) 2011~ {round(float(a.z.corr(a['act_chg%'])), 2)}")

    # ---------------- (B) 계수: prereg(2016~2020 주간 전환기 원점), expand, 전환기별
    yW = W.year
    design = W[yW.between(2016, 2020)]
    g_pre, b_pre = coef(design, "z", +1), coef(design, "Sx", -1)
    g_exp = {y: coef(W[(yW >= 2016) & (yW < y)], "z", +1) for y in range(2016, 2027)}
    b_exp = {y: coef(W[(yW >= 2016) & (yW < y)], "Sx", -1) for y in range(2016, 2027)}
    g_tr = {(y, t): coef(W[(yW >= 2016) & (yW < y) & (W.tr == t)], "z", +1) for y in range(2016, 2027) for t in TR}
    gd_pre = coef(design, "zdet", +1)
    gd_exp = {y: coef(W[(yW >= 2016) & (yW < y)], "zdet", +1) for y in range(2016, 2027)}
    b_tr_pre = {t: coef(design[design.tr == t], "Sx", -1) for t in TR}
    b_tr = {(y, t): coef(W[(yW >= 2016) & (yW < y) & (W.tr == t)], "Sx", -1) for y in range(2016, 2027) for t in TR}
    C = pd.DataFrame({"gamma_prereg": g_pre, "beta_prereg": b_pre, "gamma_expand": pd.Series(g_exp), "beta_expand": pd.Series(b_exp)})
    for t in TR:
        C[f"gamma_T{t}_expand"] = pd.Series({y: g_tr[(y, t)] for y in range(2016, 2027)})
        C[f"beta_T{t}_expand"] = pd.Series({y: b_tr[(y, t)] for y in range(2016, 2027)})
    C.index.name = "year"
    C.round(4).to_csv(OUT / "coef.csv", encoding="utf-8-sig")
    print("\n=== (B) 계수 (γ: z 1 표준편차당 로그 조정, β: 생산 전망 1 = 100% 당) ===")
    print(C.round(3).to_string())
    raw = {t: (round(float((design[design.tr == t].z * design[design.tr == t].resid).sum() / max((design[design.tr == t].z ** 2).sum(), 1e-9)), 3), int((design.tr == t).sum())) for t in TR}
    val = W[yW.between(2021, 2026)]
    rawb = {t: tuple(round(float((x[x.tr == t].Sx * x[x.tr == t].resid).sum() / max((x[x.tr == t].Sx ** 2).sum(), 1e-9)), 2) for x in (design, val)) for t in TR}
    rawg = {t: tuple(round(float((x[x.tr == t].z * x[x.tr == t].resid).sum() / max((x[x.tr == t].z ** 2).sum(), 1e-9)), 3) for x in (design, val)) for t in TR}
    print("전환기별 원시 β (2016~2020, 2021~2026):", rawb, "| 사전 고정 β_T:", {t: round(v, 2) for t, v in b_tr_pre.items()})
    print("전환기별 원시 γ (2016~2020, 2021~2026):", rawg)
    print("민감도(직전 5년 대비 편차 zdet): γ_prereg", round(gd_pre, 3), "| expand", {y: round(v, 3) for y, v in gd_exp.items() if y >= 2021},
          "| 상관(zdet, 잔차) 전환기별 (2016~2020, 2021~2026):", {t: tuple(round(float(x[x.tr == t].zdet.corr(x[x.tr == t].resid)), 2) for x in (design, val)) for t in TR})
    print("2016~2020 전환기별 원시 γ(부호 제한 전, n):", raw, "| 전체 표본내 2016~2026 γ:", round(coef(W, "z", +1), 3), "β:", round(coef(W, "Sx", -1), 3))

    # ---------------- (C) 적용
    for d in (M, W):
        on = d.tr > 0
        ge, be = d.year.map(g_exp).fillna(0.0), d.year.map(b_exp).fillna(0.0)
        gt = pd.Series([g_tr.get((y, t), 0.0) for y, t in zip(d.year, d.tr)], index=d.index)
        bt_pre = d.tr.map(b_tr_pre).fillna(0.0)
        bt_exp = pd.Series([b_tr.get((y, t), 0.0) for y, t in zip(d.year, d.tr)], index=d.index)
        adj = {"wxdet_prereg": gd_pre * d.zdet, "wxdet_expand": d.year.map(gd_exp).fillna(0.0) * d.zdet, "k_bytr_prereg": bt_pre * d.Sx, "k_bytr_expand": bt_exp * d.Sx, "wx_prereg": g_pre * d.z, "wx_expand": ge * d.z, "wx_bytr_expand": gt * d.z, "k_prereg": b_pre * d.Sx, "k_expand": be * d.Sx,
               "both_prereg": g_pre * d.z + b_pre * d.Sx, "both_expand": ge * d.z + be * d.Sx}
        for k, a in adj.items():
            d[k] = np.where(on, d.final * np.exp(a), d.final)
    M.to_csv(OUT / "predictions.csv", index=False, float_format="%.3f", encoding="utf-8-sig")
    W.to_csv(OUT / "predictions_W.csv", index=False, float_format="%.3f", encoding="utf-8-sig")

    cols = ["final"] + VARIANTS
    met = []
    for freq, d in (("M", M), ("W", W)):
        masks = {"tr_2016_2020": d.year.between(2016, 2020) & (d.tr > 0), "tr_2021_2026": d.year.between(2021, 2026) & (d.tr > 0),
                 "all_2021_2025": d.year.between(2021, 2025), "all_2021_2026": d.year.between(2021, 2026)}
        for t in TR:
            masks[f"T{t}_2021_2026"] = d.year.between(2021, 2026) & (d.tr == t)
        for nm, mk in masks.items():
            for c in cols:
                s = d[mk]
                e = (s[c] / s.actual - 1) * 100
                met.append({"freq": freq, "subset": nm, "model": c, "n": len(s), "MAPE%": mape(s, c), "over20": int((e.abs() > 20).sum()), "over30": int((e.abs() > 30).sum())})
    met = pd.DataFrame(met)
    met.to_csv(OUT / "metrics.csv", index=False, float_format="%.2f")
    for freq in ("M", "W"):
        print(f"\n=== MAPE [{freq}] ===")
        print(met[met.freq == freq].pivot(index="model", columns="subset", values="MAPE%").loc[cols].round(2).to_string())
    rng = M.origin >= "2020-12-31"
    print("\n30% 초과 [M, 대상월 2021-01~]:", {c: int((((M.loc[rng, c] / M.loc[rng, "actual"]) - 1).abs() > 0.30).sum()) for c in cols},
          "\n20% 초과:", {c: int((((M.loc[rng, c] / M.loc[rng, "actual"]) - 1).abs() > 0.20).sum()) for c in cols})
    k = M[(M.tr > 0) & (M.year >= 2021)].copy()
    for c in cols:
        k[c + "%"] = ((k[c] / k.actual - 1) * 100).round(0)
    print("\n=== 전환기 월말 원점 2021~ (오차 %) ===")
    print(k[["origin", "tr", "z", "zdet", "S"] + [c + "%" for c in cols]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
