"""Case 20 — 농넷 원장을 산지 시군 × 순기(10일) 단위로 쪼개 작형 전환의 시점 신호를 만들 수 있는가.

사용자 제안(2026-10-12): "농넷 자료 있잖아? 그걸로 순기별 혹은 필요로 하는 시기 단위별로 쪼개면 되지 않을까?"
배경: 순기별 정식·출하 예정 면적 조사는 여름배추 2021~2025뿐(검증 불가). case19의 산지 비중은 가락 한 곳·시도 단위·7일 비중이라 효과 미미.
여기서는 전국 34개 도매시장·산지 시군 단위·10일 물량으로 넓힌다. 원장은 2011~ 이라 2016~2020 추정 → 2021~2026 검증이 가능하다.

자료: data/ext/nongnet/nongnet_cabbage_panel_2011_2026.parquet (aT 농넷 공개). 쌈·절임·부산물·수입 제외, 산지 미상·중국·서울 제외.

산지 분류(2011~2015 월별 프로필만 사용 → 2016~ 평가에 누수 없음), 대상월 m 마다:
  새 산지 new_m  = m월 물량이 (m−1)월의 3배 초과이고 m월 전체의 2% 초과인 시군       (그 달에 출하가 시작되는 곳)
  빠지는 산지 old_m = (m−1)월 물량이 m월의 3배 초과이고 (m−1)월 전체의 2% 초과인 시군 (그 달에 출하가 끝나는 곳)
신호(원점 o, 대상월 m = (o+15일)의 달 ∈ {5,7,10,12}), 모두 원점 당일까지의 반입량만:
  f_new(o) = 새 산지의 직전 10일 전국 반입량 ÷ 직전 365일 전국 총반입량      # 원장의 수집 범위가 해마다 줄어(2012 446kt → 2025 290kt) 수준이 아니라 비율로
  N(o) = log( f_new(o) / 직전 5년 같은 월·일 f_new 평균 )                      # 새 산지가 평년보다 얼마나 많이 나오고 있는가
  O(o) = 빠지는 산지에 대해 같은 식                                           # 묵은 산지가 평년보다 얼마나 남아 있는가
절차: 전환기별 계수를 2016~2020 주간 원점의 잔차 회귀 log(실제/final) ~ γ_m·N (절편 없음, **부호 제한 없음**)로 추정해 고정(prereg) → 2021~2026 검증.
      부호를 미리 정하지 않는 이유: case18에서 재배력으로 정한 부호가 유지되지 않았다. 대신 설계 창의 부호가 검증 창에서 유지되는지를 본다.
      확장 창(expand), N+O 두 변수(no), 수준비 민감도(raw: 365일 총량으로 나누지 않음) 병기. 조정 폭은 ±0.4(로그)로 자른다.
주의: 산지 분류 규칙·10일 창·정규화는 사전 탐색(월말 원점 상관)을 본 뒤 정했다.

Usage: python experiments/case20_origin_sungi/origin_sungi.py
Outputs: outputs/origin_groups.csv, signals_by_year.csv, coef.csv, predictions.csv, predictions_W.csv, metrics.csv, run.log
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
OUT = CASE_DIR / "outputs"
TMONTHS = (5, 7, 10, 12)
EXCL_VAR = ["쌈", "절임", "기타부산물", "수입"]
EXCL_SIDO = ["-", "중국", "서울특별시"]
CLIP = 0.4
VARIANTS = ["n_prereg", "n_expand", "no_prereg", "raw_prereg", "n_prereg_monthend"]


def daily_by_sigungu() -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data" / "ext" / "nongnet" / "nongnet_cabbage_panel_2011_2026.parquet",
                        columns=["date", "variety_std", "origin_sido", "origin_sigungu", "qty_kg"])
    d = d[~d.variety_std.isin(EXCL_VAR) & ~d.origin_sido.isin(EXCL_SIDO) & d.origin_sigungu.notna()]
    d["sgg"] = d.origin_sido.str[:2] + "_" + d.origin_sigungu
    day = (d.assign(t=d.qty_kg / 1000).groupby(["date", "sgg"]).t.sum().unstack(fill_value=0.0))
    return day.reindex(pd.date_range(day.index.min(), day.index.max(), freq="D")).fillna(0.0)


def origin_groups(day: pd.DataFrame) -> dict[int, tuple[list[str], list[str]]]:
    base = day[(day.index.year >= 2011) & (day.index.year <= 2015)]
    prof = base.groupby(base.index.month).sum()
    prof = prof.loc[:, prof.sum() > 500]
    out = {}
    for m in TMONTHS:
        cur, prev = prof.loc[m], prof.loc[m - 1]
        out[m] = (cur[(cur > 3 * prev) & (cur > 0.02 * cur.sum())].index.tolist(), prev[(prev > 3 * cur) & (prev > 0.02 * prev.sum())].index.tolist())
    return out


class Signals:
    def __init__(self, day: pd.DataFrame, groups: dict):
        self.day, self.groups = day, groups
        self.tot365 = day.sum(axis=1).rolling(365, min_periods=330).sum()
        self.r10 = {(m, k): day[cols].sum(axis=1).rolling(10, min_periods=1).sum() for m, (new, old) in groups.items() for k, cols in (("new", new), ("old", old))}

    def frac(self, m: int, k: str, o: pd.Timestamp, raw: bool = False) -> float:
        if o not in self.day.index:
            return np.nan
        v = float(self.r10[(m, k)].get(o, np.nan))
        if raw:
            return v
        t = float(self.tot365.get(o, np.nan))
        return v / t if (pd.notna(t) and t > 0) else np.nan

    def anom(self, m: int, k: str, o: pd.Timestamp, raw: bool = False) -> float:
        cur = self.frac(m, k, o, raw)
        prev = [self.frac(m, k, o - pd.DateOffset(years=j), raw) for j in range(1, 6)]
        prev = [x for x in prev if pd.notna(x) and x > 0]
        if pd.isna(cur) or cur <= 0 or len(prev) < 3:
            return np.nan
        return float(np.log(cur / np.mean(prev)))


def load(freq: str, S: Signals) -> pd.DataFrame:
    tag = "" if freq == "M" else f"_{freq}"
    g = pd.read_csv(ROOT / "outputs" / f"gate_predictions{tag}.csv", parse_dates=["origin"], encoding="utf-8-sig")
    d = g[["origin", "actual", "tmonth", "final"]].copy()
    d["year"] = d.origin.dt.year
    d["tr"] = d.tmonth.where(d.tmonth.isin(TMONTHS), 0).astype(int)
    for col, k, raw in (("N", "new", False), ("O", "old", False), ("Nraw", "new", True)):
        d[col] = [S.anom(t, k, o, raw) if t else np.nan for o, t in zip(d.origin, d.tr)]
    d["resid"] = np.log(d.actual / d.final)
    return d


def fit(s: pd.DataFrame, xs: list[str], minn: int = 8) -> np.ndarray:
    s = s.dropna(subset=xs + ["resid"])
    if len(s) < minn:
        return np.zeros(len(xs))
    X = s[xs].values
    beta, *_ = np.linalg.lstsq(X, s.resid.values, rcond=None)
    return beta


def mape(s: pd.DataFrame, c: str) -> float:
    return float(((s[c] - s.actual).abs() / s.actual).mean() * 100) if len(s) else np.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 320)
    day = daily_by_sigungu()
    groups = origin_groups(day)
    S = Signals(day, groups)
    pd.DataFrame([{"tmonth": m, "kind": k, "sigungu": s} for m, (new, old) in groups.items() for k, lst in (("new", new), ("old", old)) for s in lst]).to_csv(OUT / "origin_groups.csv", index=False, encoding="utf-8-sig")
    print("시군 수", day.shape[1], "| 전국 연 반입량(kt):", (day.sum(axis=1).groupby(day.index.year).sum() / 1000).round(0).astype(int).to_dict())
    for m, (new, old) in groups.items():
        print(f"대상 {m}월  새 산지 {len(new)}: {', '.join(new)}\n          빠지는 산지 {len(old)}: {', '.join(old)}")
    M, W = load("M", S), load("W", S)

    # ---------------- (A) 월말 원점 해별 신호
    feat = pd.read_csv(ROOT / "data" / "processed" / "daily_features.csv", parse_dates=["date"], usecols=["date", "lp28", "y_lp_fwd28"]).set_index("date")
    ends = {5: (4, 30), 7: (6, 30), 10: (9, 30), 12: (11, 30)}
    rows = []
    for m, (mo, dd) in ends.items():
        for y in range(2012, 2027):
            o = pd.Timestamp(y, mo, dd)
            if o > day.index.max():
                continue
            k = M[M.origin == o]
            rows.append({"tr": m, "year": y, "new10_t": S.frac(m, "new", o, True), "old10_t": S.frac(m, "old", o, True), "N": S.anom(m, "new", o), "O": S.anom(m, "old", o),
                         "act_chg%": (np.exp(feat.y_lp_fwd28.get(o, np.nan) - feat.lp28.get(o, np.nan)) - 1) * 100,
                         "err%": float((k.final.iloc[0] / k.actual.iloc[0] - 1) * 100) if len(k) else np.nan, "resid": float(k.resid.iloc[0]) if len(k) else np.nan})
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "signals_by_year.csv", index=False, float_format="%.3f", encoding="utf-8-sig")
    print("\n=== (A) 월말 원점 해별 신호 (N: 새 산지 평년비, O: 빠지는 산지 평년비, 로그) ===")
    for m in TMONTHS:
        a = A[A.tr == m]
        k = a.dropna(subset=["resid"])
        cc = lambda s, x: round(float(s[x].corr(s.resid)), 2) if s[x].notna().sum() >= 3 else np.nan
        print(f"\n-- 대상 {m}월")
        print(a.round(2).to_string(index=False))
        b = a.dropna(subset=["N", "act_chg%"])
        print(f"   상관(N, 잔차) 2016~2020 {cc(k[k.year <= 2020], 'N')} / 2021~ {cc(k[k.year >= 2021], 'N')} | 상관(O, 잔차) {cc(k[k.year <= 2020], 'O')} / {cc(k[k.year >= 2021], 'O')}"
              f" | 상관(N, 실제 변동) 2014~ {round(float(b.N.corr(b['act_chg%'])), 2)} (n={len(b)})")

    # ---------------- (B) 계수: 전환기별, 주간 원점
    yW = W.year
    design, valid = W[yW.between(2016, 2020)], W[yW.between(2021, 2026)]
    C = []
    pre = {}
    for m in TMONTHS:
        gN, gNd = float(fit(design[design.tr == m], ["N"])[0]), float(fit(valid[valid.tr == m], ["N"])[0])
        gNO = fit(design[design.tr == m], ["N", "O"])
        gR, gRd = float(fit(design[design.tr == m], ["Nraw"])[0]), float(fit(valid[valid.tr == m], ["Nraw"])[0])
        pre[m] = {"n": gN, "no": gNO, "raw": gR}
        C.append({"tmonth": m, "n_design": int(design[design.tr == m].N.notna().sum()), "n_valid": int(valid[valid.tr == m].N.notna().sum()),
                  "gamma_N_2016_2020": gN, "gamma_N_2021_2026": gNd, "same_sign": bool(gN * gNd > 0), "gammaNO_N": float(gNO[0]), "gammaNO_O": float(gNO[1]),
                  "gamma_raw_2016_2020": gR, "gamma_raw_2021_2026": gRd,
                  "corr_N_resid_design": float(design[design.tr == m][["N", "resid"]].dropna().corr().iloc[0, 1]), "corr_N_resid_valid": float(valid[valid.tr == m][["N", "resid"]].dropna().corr().iloc[0, 1])})
    C = pd.DataFrame(C)
    C.to_csv(OUT / "coef.csv", index=False, float_format="%.3f", encoding="utf-8-sig")
    print("\n=== (B) 전환기별 계수 (주간 원점, 잔차 ~ γ·N) ===")
    print(C.round(2).to_string(index=False))
    exp = {(y, m): float(fit(W[(yW >= 2016) & (yW < y) & (W.tr == m)], ["N"])[0]) for y in range(2016, 2027) for m in TMONTHS}
    me = M[M.origin.dt.is_month_end]
    pre_me = {m: float(fit(me[me.year.between(2016, 2020) & (me.tr == m)], ["N"], minn=4)[0]) for m in TMONTHS}
    val_me = {m: float(fit(me[me.year.between(2021, 2026) & (me.tr == m)], ["N"], minn=4)[0]) for m in TMONTHS}
    print("month-end only gamma (2016-2020, 2021-2026):", {m: (round(pre_me[m], 2), round(val_me[m], 2)) for m in TMONTHS})

    # ---------------- (C) 적용
    for d in (M, W):
        on = d.tr > 0
        clip = lambda a: np.clip(np.nan_to_num(a, nan=0.0), -CLIP, CLIP)
        gn = d.tr.map({m: pre[m]["n"] for m in TMONTHS}).fillna(0.0)
        gr = d.tr.map({m: pre[m]["raw"] for m in TMONTHS}).fillna(0.0)
        gnoN = d.tr.map({m: pre[m]["no"][0] for m in TMONTHS}).fillna(0.0)
        gnoO = d.tr.map({m: pre[m]["no"][1] for m in TMONTHS}).fillna(0.0)
        ge = pd.Series([exp.get((y, t), 0.0) for y, t in zip(d.year, d.tr)], index=d.index)
        d["n_prereg"] = np.where(on, d.final * np.exp(clip(gn * d.N)), d.final)
        d["n_expand"] = np.where(on, d.final * np.exp(clip(ge * d.N)), d.final)
        d["no_prereg"] = np.where(on, d.final * np.exp(clip(gnoN * d.N.fillna(0) + gnoO * d.O.fillna(0))), d.final)
        d["raw_prereg"] = np.where(on, d.final * np.exp(clip(gr * d.Nraw)), d.final)
        d["n_prereg_monthend"] = np.where(on & d.origin.dt.is_month_end, d.final * np.exp(clip(d.tr.map(pre_me).fillna(0.0) * d.N)), d.final)
    M.to_csv(OUT / "predictions.csv", index=False, float_format="%.3f", encoding="utf-8-sig")
    W.to_csv(OUT / "predictions_W.csv", index=False, float_format="%.3f", encoding="utf-8-sig")
    cols = ["final"] + VARIANTS
    met = []
    for freq, d in (("M", M), ("W", W)):
        masks = {"tr_2016_2020": d.year.between(2016, 2020) & (d.tr > 0), "tr_2021_2026": d.year.between(2021, 2026) & (d.tr > 0),
                 "all_2021_2025": d.year.between(2021, 2025), "all_2021_2026": d.year.between(2021, 2026)}
        for m in TMONTHS:
            masks[f"T{m}_2021_2026"] = d.year.between(2021, 2026) & (d.tr == m)
        for nm, mk in masks.items():
            for c in cols:
                s = d[mk]
                e = (s[c] / s.actual - 1) * 100
                met.append({"freq": freq, "subset": nm, "model": c, "n": len(s), "MAPE%": mape(s, c), "over20": int((e.abs() > 20).sum()), "over30": int((e.abs() > 30).sum())})
    met = pd.DataFrame(met)
    met.to_csv(OUT / "metrics.csv", index=False, float_format="%.2f")
    for freq in ("M", "W"):
        print(f"\n=== (C) MAPE [{freq}] ===")
        print(met[met.freq == freq].pivot(index="model", columns="subset", values="MAPE%").loc[cols].round(2).to_string())
    rng = M.origin >= "2020-12-31"
    cnt = lambda mk, th: {c: int((((M.loc[mk, c] / M.loc[mk, "actual"]) - 1).abs() > th).sum()) for c in cols}
    print("\n30% 초과 [M, 대상월 2021-01~]:", cnt(rng, 0.30), "| 20% 초과:", cnt(rng, 0.20))
    k = M[(M.tr > 0) & (M.year >= 2021)].copy()
    for c in cols:
        k[c + "%"] = ((k[c] / k.actual - 1) * 100).round(0)
    print("\n=== 전환기 월말 원점 2021~ (오차 %) ===")
    print(k[["origin", "tr", "N", "O"] + [c + "%" for c in cols]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
