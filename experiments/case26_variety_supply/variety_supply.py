"""Case 26 — 가락시장 품종(작형)별 반입량으로 '앞으로 28일 출하량'을 예측하고, 그것이 급등·급락·모형 오차를 가르는지 본다.

사용자 지시(2026-10-13): "반입량을 품종별로 구분해서 살펴볼래? 저장배추, 봄배추, 여름배추 등 카테고리별로 나눠서". case25(전국 합계·달력 구분)의 후속.

자료: data/ext/nongnet/garak_daily_by_variety.csv (가락시장 일별 품종별 반입, 2011~). 전국 원장은 '기타배추' 표시가 25%이고 시장마다 다르지만
      가락시장은 기타가 2~3%뿐이고 전국 반입의 절반가량을 차지한다. 가격(가락 상품)과도 같은 시장이다.
작형 묶음(품종 표시 → 작형, 출하 창):
  겨울  월동배추 + 저장배추(12~5월 반입분)      창 11/1 ~ 이듬해 5/31   전망 = 겨울배추(전년산)
  봄    봄배추 + 저장배추(6~9월 반입분)          창 4/1 ~ 9/30            전망 = 봄배추
  여름  여름배추 + 고랭지배추                    창 6/1 ~ 11/30           전망 = 여름배추
  가을  김장(가을)배추                           창 10/1 ~ 이듬해 1/31    전망 = 가을배추
작형 c, 원점 o (작형 창 시작 뒤 n 일; 아직 시작 전이면 n < 0, 누계 0):
  A_hat = 작년 그 작형 창의 총 반입량 × (1 + 월보 생산 전망 전년비 S)     (S 없으면 0)
  C     = 올해 창 시작 ~ o 누계,   p_bar(n) = 직전 5개 창(최소 3)의 평균 누적 비율
  Q_hat_c  = (A_hat − C) × [p_bar(n+28) − p_bar(n)] / [1 − p_bar(n)]       남은 물량이 평년 속도로 나올 때의 앞으로 28일 반입량
  Q_norm_c = A_hat × [p_bar(n+28) − p_bar(n)]                               진행이 평년과 같았다면
  Q_ly_c   = 작년 같은 28일의 반입량
  네 작형을 더해 Q_hat, Q_norm, Q_ly.  실제 Q_act = 원점 뒤 28일 반입량(네 작형 합).
신호:  Z = log(Q_hat / Q_ly)   앞으로 28일 예상 출하량의 전년비
       T = log(Q_hat / Q_norm) 그중 출하 시기(진행률) 몫만
       O = log(Q_act / Q_ly)   실제 출하량 전년비 — 미래 값이므로 쓸 수 없는 '정답'. 출하량을 완벽히 맞혔을 때 얻을 수 있는 한계를 재는 용도
확인: (1) 품종별 연·월 반입량 표  (2) 출하량 예측 정확도(Q_hat 대 작년 값·최근 28일 값)
      (3) 신호와 가격 변화·모형 잔차의 상관(2016~2020, 2021~2025)  (4) 잔차 조정 규칙  (5) 신호 구간별 급등·급락 빈도  (6) 작형별 진행률 신호

Usage: python experiments/case26_variety_supply/variety_supply.py [M|W]
Outputs: outputs/variety_yearly.csv, variety_monthly_profile.csv, vs_signal{tag}.csv, vs_shipment_forecast{tag}.csv, vs_corr{tag}.csv,
         vs_metrics{tag}.csv, vs_events{tag}.csv, vs_crop_corr{tag}.csv
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

GARAK = ROOT / "data" / "ext" / "nongnet" / "garak_daily_by_variety.csv"
# 작형: (창 시작 월, 일, 창 길이 일수, 창 시작이 전년도인가)
CROPS = {"겨울": (11, 1, 212), "봄": (4, 1, 183), "여름": (6, 1, 183), "가을": (10, 1, 123)}
NORM, H, CLIP, MOVE = 5, 28, 0.7, 25.0
WINDOWS = {"2016~2020": (2016, 2020), "2021~2025": (2021, 2025), "2021~2026": (2021, 2026)}
TSEASON = {1: "겨울", 2: "겨울", 3: "겨울", 4: "겨울", 5: "봄", 6: "봄", 7: "여름", 8: "여름", 9: "여름", 10: "여름", 11: "가을", 12: "가을"}


def crop_series() -> tuple[pd.DataFrame, pd.DataFrame]:
    g = pd.read_csv(GARAK, parse_dates=["date"])
    w = g.pivot_table(index="date", columns="variety_std", values="qty_t", aggfunc="sum").fillna(0.0)
    w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="D")).fillna(0.0)
    m = w.index.month
    raw = pd.DataFrame(index=w.index)
    raw["겨울"] = w["월동"] + w["저장"].where(m.isin([12, 1, 2, 3, 4, 5]), 0.0)
    raw["봄"] = w["봄"] + w["저장"].where(m.isin([6, 7, 8, 9]), 0.0)
    raw["여름"] = w["여름"] + w["고랭지"]
    raw["가을"] = w["가을"]
    # 농넷 가락 반입량은 2018년 이전에 덜 기록돼 있다(clean_daily 대비 2012 0.61·2013 0.53·2015 0.79·2017 0.62, 2018년 이후 0.93~0.98).
    # 그래서 수준은 가락 총 반입량(data/raw/clean_daily.csv)에서, 품종 구성(최근 7일 합 기준 비중)만 농넷에서 가져온다.
    r7 = raw.rolling(7, min_periods=1).sum()
    share = r7.div(r7.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    q = pd.read_csv(ROOT / "data" / "raw" / "clean_daily.csv", parse_dates=["date"]).set_index("date")["quantity"]
    c = share.mul(q.reindex(w.index).fillna(0.0), axis=0)
    return w, c


def window(crop: str, key: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """key = 창이 시작하는 해."""
    mo, dy, ln = CROPS[crop]
    s = pd.Timestamp(key, mo, dy)
    return s, s + pd.Timedelta(days=ln - 1)


def active_key(crop: str, o: pd.Timestamp) -> int | None:
    """원점 뒤 28일과 겹치는 그 작형 창(시작 연도). 없으면 None."""
    for key in (o.year, o.year - 1):
        s, e = window(crop, key)
        if o + pd.Timedelta(days=H) >= s and o < e:
            return key
    return None


def cumshare(c: pd.Series, crop: str, key: int, n: int) -> float:
    s, e = window(crop, key)
    tot = float(c[s:e].sum())
    if tot <= 0:
        return np.nan
    if n < 0:
        return 0.0
    return float(c[s:min(s + pd.Timedelta(days=n), e)].sum()) / tot


def crop_part(c: pd.Series, crop: str, o: pd.Timestamp, S: float, first_year: int) -> dict | None:
    key = active_key(crop, o)
    if key is None or key - 1 < first_year + 3:
        return None
    s, e = window(crop, key)
    n = (o - s).days
    ps, pe = window(crop, key - 1)
    a_prev = float(c[ps:pe].sum())
    prior = [k for k in range(key - NORM, key) if k >= first_year]
    p0 = np.nanmean([cumshare(c, crop, k, n) for k in prior])
    p1 = np.nanmean([cumshare(c, crop, k, n + H) for k in prior])
    if a_prev <= 0 or np.isnan(p0):
        return None
    a_hat = a_prev * (1 + (0.0 if pd.isna(S) else S) / 100.0)
    cum = float(c[s:o].sum()) if n >= 0 else 0.0
    rem = max(a_hat - cum, 0.02 * a_hat)
    q_hat = rem * (p1 - p0) / max(1 - p0, 0.02)
    q_norm = a_hat * (p1 - p0)
    q_ly = float(c[ps + pd.Timedelta(days=n + 1):min(ps + pd.Timedelta(days=n + H), pe)].sum()) if n + H >= 0 else 0.0
    x = float(np.clip(np.log(max(1 - cum / a_hat, 0.02) / max(1 - p0, 0.02)), -CLIP, CLIP)) if (n >= 14 and p0 < 0.95) else np.nan
    return {"q_hat": q_hat, "q_norm": q_norm, "q_ly": q_ly, "X": x, "progress": cum / a_hat, "p_bar": p0, "S": S, "n": n}


def crop_S(sum_iss: pd.DataFrame, ns_iss: pd.DataFrame, crop: str, o: pd.Timestamp) -> float:
    key = active_key(crop, o)
    if key is None:
        return np.nan
    if crop == "여름":
        return gm.adjust_signal(sum_iss, o, key)[1]
    s = ns_iss[(ns_iss.season == crop) & (ns_iss.crop_year == key) & (ns_iss.avail_b <= o)]
    if s.empty:
        return np.nan
    last = s[(s.year == s.year.iloc[-1]) & (s.issue_month == s.issue_month.iloc[-1])]
    if crop == "봄":
        last = last.assign(p=last["sub"].map({"전체": 0, "노지": 1, "시설": 2})).sort_values("p")
        withp = last[last.prod_yoy.notna()]
        last = withp if len(withp) else last
        r = last.iloc[0]
    else:
        r = last.iloc[-1]
    return float(r.prod_yoy if pd.notna(r.prod_yoy) else r.area_yoy)


def est(x: pd.Series, y: pd.Series, sign: int) -> float:
    if len(x) < 5 or float((x ** 2).sum()) == 0:
        return 0.0
    b = float((x * y).sum() / (x ** 2).sum())
    return b if np.sign(b) == sign else 0.0


def ape(s: pd.DataFrame, col: str) -> pd.Series:
    return (s[col] / s.actual - 1) * 100


def describe(w: pd.DataFrame, c: pd.DataFrame) -> None:
    yr = (w.groupby(w.index.year).sum() / 1000).round(1)
    yr.to_csv(OUT / "variety_yearly.csv", encoding="utf-8-sig")
    prof = (w[(w.index.year >= 2016) & (w.index.year <= 2025)].groupby(w[(w.index.year >= 2016) & (w.index.year <= 2025)].index.month).sum() / 10 / 1000).round(2).T
    prof.to_csv(OUT / "variety_monthly_profile.csv", encoding="utf-8-sig")
    cols = [k for k in ("월동", "저장", "봄", "여름", "고랭지", "가을", "기타", "쌈", "절임", "수입") if k in w.columns]
    print("[가락시장 품종별 연간 반입량(천 톤)]")
    print(yr.loc[2014:, cols].to_string())
    print("\n[가락시장 품종별 월평균 반입량(천 톤, 2016~2025)]")
    print(prof.loc[cols].to_string())
    cy = (c.groupby(c.index.year).sum() / 1000).round(1)
    print("\n[작형 묶음 연간 반입량(천 톤, 달력 연도)]")
    print(cy.loc[2014:].T.to_string())


def run(freq: str) -> None:
    tag = "" if freq == "M" else f"_{freq}"
    w, c = crop_series()
    first_year = int(c.index.year.min())
    if freq == "M":
        describe(w, c)
    d = sg.load(freq)
    f = pd.read_csv(ROOT / "data" / "processed" / "daily_features.csv", parse_dates=["date"], usecols=["date", "lp7"]).set_index("date")
    d["p7"] = np.exp(f.lp7.reindex(d.origin).values)
    sum_iss, ns_iss = sg.summer_issues_ext(), sg.nonsummer_issues()
    last_day = c.index.max()
    total = c.sum(axis=1)
    rows = []
    for o in d.origin:
        rec = {"q_hat": 0.0, "q_norm": 0.0, "q_ly": 0.0}
        for crop in CROPS:
            p = crop_part(c[crop], crop, o, crop_S(sum_iss, ns_iss, crop, o), first_year)
            if p is None:
                continue
            for k in ("q_hat", "q_norm", "q_ly"):
                rec[k] += p[k]
            rec[f"X_{crop}"], rec[f"S_{crop}"], rec[f"prog_{crop}"], rec[f"pbar_{crop}"] = p["X"], p["S"], p["progress"], p["p_bar"]
        rec["q_act"] = float(total[o + pd.Timedelta(days=1):o + pd.Timedelta(days=H)].sum()) if o + pd.Timedelta(days=H) <= last_day else np.nan
        rec["q_last"] = float(total[o - pd.Timedelta(days=H - 1):o].sum())
        rec["q_last_ly"] = float(total[o - pd.Timedelta(days=364 + H - 1):o - pd.Timedelta(days=364)].sum())
        rows.append(rec)
    s = pd.DataFrame(rows, index=d.index)
    d = pd.concat([d, s], axis=1)
    ok = (d.q_hat > 0) & (d.q_ly > 0) & (d.q_norm > 0) & d.q_act.notna() & (d.q_act > 0)
    d["Z"] = np.clip(np.log(d.q_hat / d.q_ly), -CLIP, CLIP).where(ok)
    d["T"] = np.clip(np.log(d.q_hat / d.q_norm), -CLIP, CLIP).where(ok)
    d["O"] = np.clip(np.log(d.q_act / d.q_ly), -CLIP, CLIP).where(ok)
    d["L"] = np.clip(np.log(d.q_last / d.q_last_ly.where(d.q_last_ly > 0)), -CLIP, CLIP).fillna(0.0).where(ok)   # 최근 28일 반입량의 전년비
    py = pd.read_csv(ROOT / "data" / "processed" / "daily_features.csv", parse_dates=["date"], usecols=["date", "y_price_fwd28_true"]).set_index("date")["y_price_fwd28_true"]
    d["pyoy"] = np.log(d.actual / py.reindex(d.origin - pd.Timedelta(days=364)).values)                          # 가격 전년비(점검용)
    d["r"] = np.log(d.actual / d.final)
    d["chg"] = (d.actual / d.p7 - 1) * 100
    d["tsea"] = d.tmonth.map(TSEASON)
    yr = d.origin.dt.year
    design = yr.between(2016, 2020)
    has = d.Z.notna()
    # 출하량 예측의 결합: 실제 전년비 O ~ a·L(최근 28일 전년비) + b·T(진행률에서 온 남은 물량 몫), 2016~2020 에서 추정
    kd = d[has & design]
    coef, *_ = np.linalg.lstsq(np.column_stack([kd.L, kd["T"]]), kd.O.values, rcond=None)
    d["B"] = np.clip(coef[0] * d.L + coef[1] * d["T"], -CLIP, CLIP).where(has)
    print(f"[출하량 결합식(2016~2020 추정)] 실제 전년비 = {coef[0]:+.2f} × 최근 28일 전년비 {coef[1]:+.2f} × 남은 물량 몫 T")
    print(f"\n===== {freq}: 원점 {len(d)}개, 신호 있는 원점 {int(has.sum())}개 =====")
    keep = ["origin", "actual", "final", "tmonth", "q_hat", "q_norm", "q_ly", "q_act", "q_last", "Z", "T", "L", "B", "O", "pyoy", "r", "chg"] + [k for k in d.columns if k[:2] in ("X_", "S_") or k[:5] in ("prog_", "pbar_")]
    d[keep].to_csv(OUT / f"vs_signal{tag}.csv", index=False, float_format="%.3f", encoding="utf-8-sig")

    # (2) 출하량 예측 정확도
    frow = []
    for wname, (a, b) in list(WINDOWS.items())[:2]:
        k = d[has & yr.between(a, b)]
        frow.append({"window": wname, "n": len(k),
                     "예측(Q_hat) 평균 절대 로그오차": float(np.log(k.q_hat / k.q_act).abs().mean()),
                     "평년 속도(Q_norm)": float(np.log(k.q_norm / k.q_act).abs().mean()),
                     "작년 같은 기간": float(np.log(k.q_ly / k.q_act).abs().mean()),
                     "최근 28일 그대로": float(np.log(k.q_last / k.q_act).abs().mean()),
                     "최근 전년비 지속(L)": float((k.L - k.O).abs().mean()), "결합(B)": float((k.B - k.O).abs().mean()),
                     "corr(Z,O)": float(np.corrcoef(k.Z, k.O)[0, 1]), "corr(T,O)": float(np.corrcoef(k["T"], k.O)[0, 1]),
                     "corr(L,O)": float(np.corrcoef(k.L, k.O)[0, 1]), "corr(B,O)": float(np.corrcoef(k.B, k.O)[0, 1]),
                     "corr(가격 전년비, O)": float(np.corrcoef(k.pyoy[k.pyoy.notna()], k.O[k.pyoy.notna()])[0, 1])})
    fc = pd.DataFrame(frow)
    fc.to_csv(OUT / f"vs_shipment_forecast{tag}.csv", index=False, float_format="%.3f", encoding="utf-8-sig")
    print("\n[앞으로 28일 가락 반입량 예측 정확도]")
    print(fc.round(3).to_string(index=False))

    # (3) 상관
    crow = []
    for wname, (a, b) in list(WINDOWS.items())[:2]:
        for sea in ("전체", "겨울", "봄", "여름", "가을"):
            k = d[has & yr.between(a, b) & ((d.tsea == sea) if sea != "전체" else True)]
            if len(k) < 5:
                continue
            row = {"window": wname, "target": sea, "n": len(k)}
            for col in ("T", "L", "B", "O"):
                row[f"r~{col}"] = float(np.corrcoef(k[col], k.r)[0, 1])
                row[f"chg~{col}"] = float(np.corrcoef(k[col], k.chg)[0, 1])
            crow.append(row)
    corr = pd.DataFrame(crow)
    corr.to_csv(OUT / f"vs_corr{tag}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    print("\n[신호와 모형 잔차(r)·가격 변화(chg)의 상관]  기대 부호: 모두 음(−). O 는 실제 출하량(정답)")
    print(corr.round(2).to_string(index=False))

    # (4) 조정 규칙
    k = d[has & design]
    picks, rules = {}, ["final"]
    for name, col in (("T_adj", "T"), ("L_adj", "L"), ("B_adj", "B"), ("O_adj(정답)", "O")):
        b = est(k[col], k.r, -1)
        picks[name] = b
        d[name] = np.where(has, d.final * np.exp(b * d[col].fillna(0)), d.final)
        rules.append(name)
    print("\n[2016~2020 에서 정한 계수]", {a: round(b, 3) for a, b in picks.items()})
    rows = []
    for wname, (a, b) in WINDOWS.items():
        ss = d[yr.between(a, b)]
        for r in rules:
            e = ape(ss, r)
            rows.append({"window": wname, "rule": r, "n": len(ss), "MAPE%": e.abs().mean(), "n_gt30": int((e.abs() > 30).sum())})
    met = pd.DataFrame(rows)
    met.to_csv(OUT / f"vs_metrics{tag}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    print(met.pivot_table(index="window", columns="rule", values="MAPE%")[rules].round(2).to_string())
    print(met.pivot_table(index="window", columns="rule", values="n_gt30")[rules].to_string())

    # (5) 급등·급락
    erow = []
    for wname, (a, b) in list(WINDOWS.items())[:2]:
        ss = d[has & yr.between(a, b)]
        for col, nm in (("T", "남은 물량 몫 T"), ("B", "예상 출하 전년비 B(결합)"), ("O", "실제 출하 전년비 O(정답)")):
            for lab, mk in (("≤ −0.1 (적음)", ss[col] <= -0.1), ("가운데", ss[col].abs() < 0.1), ("≥ +0.1 (많음)", ss[col] >= 0.1)):
                g = ss[mk]
                erow.append({"window": wname, "signal": nm, "group": lab, "n": len(g), "급등": int((g.chg >= MOVE).sum()), "급락": int((g.chg <= -MOVE).sum()),
                             "chg_mean%": g.chg.mean(), "model_bias%": ape(g, "final").mean()})
    ev = pd.DataFrame(erow)
    ev.to_csv(OUT / f"vs_events{tag}.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    print("\n[신호 구간별 급등·급락]")
    print(ev.round(1).to_string(index=False))

    # (6) 작형별 진행률 신호
    xrow = []
    for crop in CROPS:
        col = f"X_{crop}"
        if col not in d:
            continue
        for wname, (a, b) in list(WINDOWS.items())[:2]:
            k = d[d[col].notna() & yr.between(a, b)]
            if len(k) >= 5:
                xrow.append({"crop": crop, "window": wname, "n": len(k), "corr(r, X)": float(np.corrcoef(k[col], k.r)[0, 1]), "corr(chg, X)": float(np.corrcoef(k[col], k.chg)[0, 1])})
    xc = pd.DataFrame(xrow)
    xc.to_csv(OUT / f"vs_crop_corr{tag}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    print("\n[작형별 남은 물량 신호 X(남은 몫의 평년 대비)와 잔차·가격 변화의 상관]  기대 부호: 음(−)")
    print(xc.round(2).to_string(index=False))
    if freq == "M":
        key = d[d.origin.dt.month.isin([7, 8]) & (yr >= 2016)]
        print("\n[7·8월 말 원점: 봄(저장)·여름 작형 진행률과 신호]")
        print(pd.DataFrame({"origin": key.origin.dt.date, "S_여름": key.get("S_여름"), "여름 진행": key.get("prog_여름").round(2), "평년": key.get("pbar_여름").round(2),
                            "봄·저장 진행": key.get("prog_봄").round(2), "평년 ": key.get("pbar_봄").round(2), "T": key["T"].round(2), "Z": key.Z.round(2), "O": key.O.round(2),
                            "chg%": key.chg.round(0), "final%": ape(key, "final").round(0)}).to_string(index=False))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 330)
    pd.set_option("display.max_columns", 40)
    run(sys.argv[1] if len(sys.argv) > 1 else "M")
