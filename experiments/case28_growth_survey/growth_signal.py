"""Case 28 — 생육·생산량 실측조사(2020~2025년산)로 가격 예측이 나아질 수 있는지 확인.

사용자 지시(2026-10-13): "생육의 실측데이터를 활용해서 개선될 수 있는지 확인해보자." (작업 폴더에 C:/Users/user/Desktop/생육 추가)
자료: 생육 프로젝트의 비식별 가공 패널(읽기 전용) data/processed/yield_farm.csv, growth.csv, farm.csv.
      농가 단위 자료는 이 저장소(공개)에 옮기지 않는다. 산출물은 data/local/case28/(미추적)에만 쓴다. 지역은 시도까지만 쓴다.

실시간으로 알 수 있는 것만 쓴다: 원점 o 에서 **조사일이 o 이전 30일 안**인 농가의 값.
  생산량 실측(수확기): 10a 수확량·정상포기율·개체 생중량의 로그를, 같은 칸(작형 × 정식월[봄은 시설/노지])의 **다른 해 평균**에서 뺀 값 → 평균
  생육조사(재배 중):   초장(1·2차)·구중·주중(3차 이후)의 로그를 '정식 후 일수' 2차식(다른 해로 적합)에서 뺀 값 → 평균
  ("다른 해 평균"은 그 해를 뺀 5개 해 기준이라 미래 해도 들어간다. 6개 해뿐이라 과거만으로는 기준을 못 만든다 — 정보가 있는지 보는 진단용.)
확인: (1) 신호와 메인 final 의 잔차(log 실제/예측)·가격 변화의 상관(전체·대상 작형별, 월말·주간)
      (2) 여름 원점 해별 표(급등 해 2020·2022·2024 대 평시 해 2021·2023·2025)
      (3) 그 해를 뺀 계수로 조정했을 때 2020~2025 MAPE(해 제외 교차검증)
      (4) 월보 생산 전망(전년비)과 실측 단수의 관계 — 실측이 월보에 이미 들어 있는 정보인가

Usage: python experiments/case28_growth_survey/growth_signal.py [M|W]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
LOCAL = ROOT / "data" / "local" / "case28"
SRC = Path(r"C:\Users\user\Desktop\생육\data\processed")
SEASON = {1: "겨울", 2: "겨울", 3: "겨울", 4: "겨울", 5: "봄", 6: "봄", 7: "여름", 8: "여름", 9: "여름", 10: "여름", 11: "가을", 12: "가을"}
WIN, MIN_N, MOVE = 30, 8, 25.0


def loyo_cell_dev(d: pd.DataFrame, var: str, cell: str) -> pd.Series:
    """같은 칸의 '다른 해' 평균에서 뺀 값."""
    g = d.groupby([cell, "year"])[var].agg(["sum", "count"]).reset_index()
    tot = g.groupby(cell)[["sum", "count"]].sum().rename(columns={"sum": "ts", "count": "tc"})
    g = g.join(tot, on=cell)
    g["other"] = (g.ts - g["sum"]) / (g.tc - g["count"]).where(g.tc - g["count"] >= 10)
    return d[var] - d.merge(g[[cell, "year", "other"]], on=[cell, "year"], how="left")["other"].values


def yield_signals() -> pd.DataFrame:
    y = pd.read_csv(SRC / "yield_farm.csv", parse_dates=["plant_date", "harvest_date", "survey_date", "end_date"])
    y = y[(y.y10a > 0) & (y.nrate > 0) & (y.w_mean > 0)].copy()
    y["date"] = y.survey_date.fillna(y.end_date).fillna(y.harvest_date)
    y = y[y.date.notna()].reset_index(drop=True)
    y["cell"] = np.where(y.crop == "봄", "봄_" + y.tag.astype(str), y.crop + "_" + y.pm.fillna(0).astype(int).astype(str))
    for v, src in (("dy", "y10a"), ("dr", "nrate"), ("dw", "w_mean")):
        y["_l"] = np.log(y[src])
        y[v] = loyo_cell_dev(y, "_l", "cell")
    return y[["year", "crop", "cell", "sido", "date", "dy", "dr", "dw"]]


def growth_signals() -> pd.DataFrame:
    g = pd.read_csv(SRC / "growth.csv", parse_dates=["date"], low_memory=False)
    f = pd.read_csv(SRC / "farm.csv", parse_dates=["plant_date"])[["fid", "plant_date"]]
    fr = g.groupby(["fid", "year", "crop", "round"]).agg(date=("date", "median"), height=("height", "mean"), head_wt=("head_wt", "mean"), plant_wt=("plant_wt", "mean")).reset_index()
    fr = fr.merge(f, on="fid", how="left")
    fr["dat"] = (fr.date - fr.plant_date).dt.days
    fr = fr[fr.date.notna() & fr.dat.between(5, 200)].reset_index(drop=True)
    out = fr[["year", "crop", "round", "date", "dat"]].copy()
    for v in ("height", "head_wt", "plant_wt"):
        z = pd.Series(np.nan, index=fr.index)
        ok = fr[v].notna() & (fr[v] > 0)
        for (crop, yr), idx in fr[ok].groupby(["crop", "year"]).groups.items():
            oth = fr[ok & (fr.crop == crop) & (fr.year != yr)]
            if len(oth) < 40:
                continue
            c = np.polyfit(oth.dat, np.log(oth[v]), 2)
            res_o = np.log(oth[v]) - np.polyval(c, oth.dat)
            z.loc[idx] = (np.log(fr.loc[idx, v]) - np.polyval(c, fr.loc[idx, "dat"])) / res_o.std()
        out["z_" + v] = z
    return out


def at_origin(tab: pd.DataFrame, cols: list[str], o: pd.Timestamp, crop: str | None = None) -> dict:
    s = tab[(tab.date > o - pd.Timedelta(days=WIN)) & (tab.date <= o)]
    if crop is not None:
        s = s[s.crop == crop]
    out = {}
    for c in cols:
        v = s[c].dropna()
        out[c] = float(v.mean()) if len(v) >= MIN_N else np.nan
        out["n_" + c] = len(v)
    return out


def run(freq: str) -> None:
    tag = "" if freq == "M" else f"_{freq}"
    LOCAL.mkdir(parents=True, exist_ok=True)
    g = pd.read_csv(ROOT / "outputs" / f"gate_predictions{tag}.csv", parse_dates=["origin"], encoding="utf-8-sig")
    f = pd.read_csv(ROOT / "data" / "processed" / "daily_features.csv", parse_dates=["date"], usecols=["date", "lp7"]).set_index("date")
    g["p7"] = np.exp(f.lp7.reindex(g.origin).values)
    g = g[g.origin.between("2020-04-01", "2026-03-31")].copy()
    g["tsea"] = g.tmonth.map(SEASON)
    g["r"] = np.log(g.actual / g.final)
    g["chg"] = (g.actual / g.p7 - 1) * 100
    Y, G = yield_signals(), growth_signals()
    rows = []
    for o in g.origin:
        rec = at_origin(Y, ["dy", "dr", "dw"], o)
        rec.update(at_origin(G, ["z_height", "z_head_wt", "z_plant_wt"], o))
        rows.append(rec)
    d = pd.concat([g.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    d["yr"] = d.origin.dt.year
    sig = ["dy", "dr", "dw", "z_height", "z_head_wt", "z_plant_wt"]
    name = {"dy": "실측 단수", "dr": "정상포기율", "dw": "개체 생중량", "z_height": "초장(생육)", "z_head_wt": "구중(생육)", "z_plant_wt": "주중(생육)"}
    d[["origin", "actual", "final", "tmonth", "tsea", "summer", "r", "chg"] + sig + ["n_" + s for s in sig]].to_csv(LOCAL / f"signal{tag}.csv", index=False, float_format="%.3f", encoding="utf-8-sig")
    print(f"===== {freq}: 2020-04~ 원점 {len(d)}개 =====")
    print("신호가 있는 원점 수:", {name[s]: int(d[s].notna().sum()) for s in sig})

    print("\n[신호와 모형 잔차(r)·가격 변화(chg)의 상관]  기대: 음(−) — 작황이 좋으면 가격이 모형보다 낮다")
    rows = []
    for sea in ("전체", "겨울", "봄", "여름", "가을"):
        for s in sig:
            k = d[d[s].notna() & ((d.tsea == sea) if sea != "전체" else True)]
            if len(k) >= 8:
                rows.append({"대상": sea, "신호": name[s], "n": len(k), "해 수": k.yr.nunique(), "corr(r)": np.corrcoef(k[s], k.r)[0, 1], "corr(chg)": np.corrcoef(k[s], k.chg)[0, 1]})
    c = pd.DataFrame(rows)
    c.to_csv(LOCAL / f"corr{tag}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    print(c.round(2).to_string(index=False))

    print("\n[해 제외 교차검증: 그 해를 뺀 나머지 해로 계수를 구해 final × exp(β·신호)]  기대 부호(음)가 아니면 0")
    rows = []
    for s in sig:
        for sea in ("전체", "여름", "가을", "겨울", "봄"):
            k = d[d[s].notna() & ((d.tsea == sea) if sea != "전체" else True)].copy()
            if len(k) < 12 or k.yr.nunique() < 4:
                continue
            adj = k.final.copy()
            betas = []
            for y in sorted(k.yr.unique()):
                tr = k[k.yr != y]
                b = float((tr[s] * tr.r).sum() / (tr[s] ** 2).sum())
                b = min(b, 0.0)
                betas.append(b)
                m = k.yr == y
                adj[m] = k.final[m] * np.exp(b * k.loc[m, s])
            e0, e1 = (k.final / k.actual - 1).abs() * 100, (adj / k.actual - 1).abs() * 100
            rows.append({"신호": name[s], "대상": sea, "n": len(k), "현행 MAPE": e0.mean(), "조정 MAPE": e1.mean(), "나아진 원점": int((e1 < e0 - 1e-9).sum()), "나빠진 원점": int((e1 > e0 + 1e-9).sum()),
                         "β 범위": f"{min(betas):.2f}~{max(betas):.2f}"})
    cv = pd.DataFrame(rows)
    cv.to_csv(LOCAL / f"loyo{tag}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    print(cv.round(2).to_string(index=False))

    if freq == "M":
        k = d[d.origin.dt.month.isin([7, 8, 9])]
        t = pd.DataFrame({"원점": k.origin.dt.strftime("%Y-%m-%d"), "가격 변화%": k.chg.round(0), "final 오차%": ((k.final / k.actual - 1) * 100).round(0),
                          "실측 단수": k.dy.round(2), "정상포기율": k.dr.round(2), "개체중": k.dw.round(2), "농가 수": k.n_dy, "구중(생육)": k.z_head_wt.round(2), "초장(생육)": k.z_height.round(2)})
        print("\n[7·8·9월 말 원점: 직전 30일에 조사된 농가의 실측값(다른 해 대비 로그 편차)과 생육(표준화 편차)]")
        print(t.to_string(index=False))
        # 작형·연도 수준: 실측 단수와 월보 생산 전망
        Y["cy"] = np.where((Y.crop == "겨울") & (Y.date.dt.month <= 6), Y.date.dt.year - 1, Y.date.dt.year)
        a = Y.groupby(["crop", "year"]).agg(n=("dy", "size"), 단수=("dy", "mean"), 정상포기율=("dr", "mean"), 개체중=("dw", "mean")).round(3)
        print("\n[작형·연도별 실측 편차(다른 해 대비 로그)]")
        print(a.unstack(0).round(2).to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 330)
    pd.set_option("display.max_columns", 40)
    run(sys.argv[1] if len(sys.argv) > 1 else "M")
