"""Case 27 — 고랭지 밖 작형(겨울·봄·가을)의 주산지 생육기·출하기 기상을 재료로 넣으면 나아지는가.

사용자 제안(2026-10-13): "지금 기상데이터 활용이 고랭지에만 집중된 것 같은데 타 작형도 생육 및 출하기 기상을 수집해서 사용해볼래?"
현재: 대표 모형(an_sparse_all)의 기상 재료 4개는 모두 고랭지 4곳 값이고 1년 내내 같은 재료를 쓴다. 겨울 모형은 기상을 쓰지 않는다.
      즉 1월 가격을 예측할 때도 평창·태백의 기온을 본다.

주산지(사용자 업무 파일 `작형별_정식_출하_시기_(2026ver).xlsx`의 '월별 주 출하지역'에서 옮김. CABIS 기상 스냅숏에 2010년부터 있는 관측소만):
  겨울(대상 1~4월)   해남            (진도·무안은 2014·2021년부터라 제외)
  봄  (대상 5~6월)   예산·의령군·아산·문경·영월·영양   (시설 예산 / 터널 의령·아산 / 노지 문경·영월·영양; 나주·의성·충주는 관측소 없음)
  여름(대상 7~10월)  평창·태백·정선군·강릉  (현행 고랭지)
  가을(대상 11~12월) 해남·진도군·고창·아산·괴산·제천·영양·문경·춘천  (현행 '가을 주산지' 9곳; 당진은 관측소 없음)
기상 재료(작형 주산지 평균, 평년 = 2010~2019 날짜별 평균을 15일 폭으로 고른 값):
  최고기온 편차 30일·90일, 최저기온 편차 30일, 강수 편차 30일·90일, 일조시간 편차 60일, 50mm 이상 비 온 날 60일,
  최저 −5°C 이하 일수 30일, 최고 30°C 이상 일수 60일
  → 원점에서 30일 창은 출하기 직전, 60~90일 창은 생육기에 해당한다. 대상월의 작형에 해당하는 주산지 값만 켠다(작형별로 가중치가 따로 붙는다).
모형(모두 메인과 같은 확장 창·누수 차단·사흘에 한 줄 솎기):
  an_nowx     가격 편차·최근 변화·반입 전년비만(기상 없음)                         — 기상의 몫을 재는 기준
  an_sparse_all  현행 대표(고랭지 기상 4개, 1년 내내)                              — outputs/anomaly_predictions 에서 읽음
  an_crop4    기상 없음 + 작형 주산지 기상 4개(최고 30일·최저 30일·강수 30일·큰비 60일)
  an_crop9    기상 없음 + 작형 주산지 기상 9개
  an_crop9_s  an_crop9 를 그 작형 대상월의 줄만으로 학습(120줄 미만이면 an_nowx)
  an_crop3_s  (2차, 결과를 본 뒤 추가) 그 작형 줄만으로 학습 + 주산지 기상 3개(최저 30일·강수 30일·강수 90일)
  an_nowx_s   (2차) 그 작형 줄만으로 학습, 기상 없음 — 작형별 학습의 몫과 기상의 몫을 가르는 기준
  wr_wx       겨울 모형(가격 15개, 겨울 대상 줄) + 해남 기상 9개
평가: 대상 작형별·구간별(2016~2020 설계 / 2021~2025 검증) MAPE. 조립: 메인 규칙(게이트·겨울 분기·수준 조정·9월 고온)은 그대로 두고
      여름 원점 밖의 대표 모형, 겨울 모형만 바꿔 넣은 최종값.

Usage: python experiments/case27_crop_weather/crop_weather.py [M|W]
Outputs: outputs/cw_predictions{tag}.csv, cw_season_metrics{tag}.csv, cw_final_metrics{tag}.csv, cw_corr{tag}.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
OUT = CASE_DIR / "outputs"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "case22_season_gate"))
import anomaly_model as am  # noqa: E402
import backtest as bt  # noqa: E402
import build_features as bf  # noqa: E402
import gate_model as gm  # noqa: E402
import season_gate as sg  # noqa: E402

GROUPS = {"겨울": ["해남"], "봄": ["예산", "의령군", "아산", "문경", "영월", "영양"],
          "여름": bf.HIGHLAND, "가을": bf.LOWLAND_AUTUMN}
SEASON = {1: "겨울", 2: "겨울", 3: "겨울", 4: "겨울", 5: "봄", 6: "봄", 7: "여름", 8: "여름", 9: "여름", 10: "여름", 11: "가을", 12: "가을"}
V9 = ["tmax30", "tmax90", "tmin30", "rain30", "rain90", "sun60", "heavy60", "cold30", "hot60"]
V4 = ["tmax30", "tmin30", "rain30", "heavy60"]
V3 = ["tmin30", "rain30", "rain90"]      # 2차(결과를 본 뒤 추가): 봄에서 두 구간 모두 같은 부호였던 셋
CORE3 = ["cur_anom7", "dlp7", "qyoy28"]
WINDOWS = {"2016~2020": (2016, 2020), "2021~2025": (2021, 2025), "2021~2026": (2021, 2026)}


def group_features(w: pd.DataFrame, regions: list[str], cal: pd.DatetimeIndex) -> pd.DataFrame:
    sub = w[w.region.isin(regions)]
    g = sub.groupby("date").agg(tmax=("max_temp", "mean"), tmin=("min_temp", "mean"), rain=("rainfall", "mean"), sun=("sunshine_hr", "mean")).reindex(cal)
    for name, cond in (("heavy", sub.rainfall >= 50), ("cold", sub.min_temp <= -5), ("hot", sub.max_temp >= 30)):
        g[name] = sub.assign(h=cond.astype(float)).groupby("date").h.mean().reindex(cal)
    clim = bf.climatology(g, ["tmax", "tmin", "rain", "sun"])
    doy = cal.dayofyear
    an = {c: g[c].values - clim.loc[doy, c].values for c in ("tmax", "tmin", "rain", "sun")}
    an = pd.DataFrame(an, index=cal)
    r = lambda s, n, how: (s.rolling(n, min_periods=max(3, n // 2)).mean() if how == "mean" else s.rolling(n, min_periods=max(3, n // 2)).sum())
    return pd.DataFrame({"tmax30": r(an.tmax, 30, "mean"), "tmax90": r(an.tmax, 90, "mean"), "tmin30": r(an.tmin, 30, "mean"),
                         "rain30": r(an.rain, 30, "mean"), "rain90": r(an.rain, 90, "mean"), "sun60": r(an.sun, 60, "mean"),
                         "heavy60": r(g.heavy, 60, "sum"), "cold30": r(g.cold, 30, "sum"), "hot60": r(g.hot, 60, "sum")}, index=cal)


def build_frame() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = pd.read_csv(bt.PROC / "daily_features.csv", parse_dates=["date"]).set_index("date")
    df[bt.TARGET] = df["y_lp_fwd28"] - df[bt.BASE]
    df = am.add_normals(df)
    w = bf.load_weather()
    sea = pd.Series((df.index + pd.Timedelta(days=15)).month.map(SEASON), index=df.index)
    df["tsea"] = sea
    cols = {"X4": [], "X9": [], "WIN9": []}
    for g, regions in GROUPS.items():
        gf = group_features(w, regions, df.index)
        on = (sea == g).astype(float)
        for v in V9:
            name = f"{g}_{v}"
            df[name] = gf[v] * on            # 그 작형이 대상인 줄에서만 켜짐 → 작형별 가중치
            cols["X9"].append(name)
            if v in V4:
                cols["X4"].append(name)
            if g == "겨울":
                df[f"W_{v}"] = gf[v]         # 겨울 모형용(겨울 대상 줄만 학습하므로 스위치 불필요)
                cols["WIN9"].append(f"W_{v}")
    return df, cols


def run(freq: str) -> None:
    tag = "" if freq == "M" else f"_{freq}"
    df, cols = build_frame()
    need = am.CORE + cols["X9"] + cols["WIN9"]
    pool = df[(df.is_trading == 1) & df.y_anom.notna()].dropna(subset=need).iloc[::3]
    wpool = df[(df.is_trading == 1) & df[bt.TARGET].notna()].dropna(subset=bt.PRICE_FEATS + bt.SUPPLY_FEATS + bt.WX_FEATS + am.CORE + ["y_anom"] + cols["WIN9"]).iloc[::3]
    wpool = wpool[wpool.tsea == "겨울"]
    g = sg.load(freq)
    ridge = lambda a: make_pipeline(StandardScaler(), Ridge(alpha=a))
    specs = {"an_nowx": CORE3, "an_crop4": CORE3 + cols["X4"], "an_crop9": CORE3 + cols["X9"]}
    rows = []
    for o in g.origin:
        if o not in df.index:
            continue
        te = df.loc[[o]]
        if te[need].isna().any(axis=1).iloc[0]:
            continue
        tr = pool[pool.index + pd.Timedelta(days=28) <= o]
        if len(tr) < 120:
            continue
        nf = float(te.norm_fwd.iloc[0])
        rec = {"origin": o}
        for name, feats in specs.items():
            m = ridge(10.0).fit(tr[feats], tr.y_anom)
            rec[name] = float(np.exp(nf + m.predict(te[feats])[0]))
        s = te.tsea.iloc[0]
        ts = tr[tr.tsea == s]
        sfe = CORE3 + [c for c in cols["X9"] if c.startswith(s + "_")]
        rec["an_crop9_s"] = float(np.exp(nf + ridge(10.0).fit(ts[sfe], ts.y_anom).predict(te[sfe])[0])) if len(ts) >= 120 else rec["an_nowx"]
        sf3 = CORE3 + [f"{s}_{v}" for v in V3]
        rec["an_crop3_s"] = float(np.exp(nf + ridge(10.0).fit(ts[sf3], ts.y_anom).predict(te[sf3])[0])) if len(ts) >= 120 else rec["an_nowx"]
        rec["an_nowx_s"] = float(np.exp(nf + ridge(10.0).fit(ts[CORE3], ts.y_anom).predict(te[CORE3])[0])) if len(ts) >= 120 else rec["an_nowx"]
        if s == "겨울":
            tw = wpool[wpool.index + pd.Timedelta(days=28) <= o]
            if len(tw) >= gm.MIN_TRAIN_WINTER:
                fe = bt.PRICE_FEATS + cols["WIN9"]
                rec["wr_wx"] = float(np.exp(float(te[bt.BASE].iloc[0]) + ridge(3.0).fit(tw[fe], tw[bt.TARGET]).predict(te[fe])[0]))
        for v in V9:
            rec[f"A_{v}"] = float(te[f"{s}_{v}"].iloc[0])
        rows.append(rec)
    p = pd.DataFrame(rows)
    d = g.merge(p, on="origin", how="left")
    d["tsea"] = d.tmonth.map(SEASON)
    d["r"] = np.log(d.actual / d.final)
    yr = d.origin.dt.year
    models = ["an_nowx", "an_sparse_all", "an_crop4", "an_crop9", "an_crop9_s", "an_crop3_s", "an_nowx_s"]
    d[["origin", "actual", "tmonth", "tsea", "summer", "final", "winter_ridge"] + models + ["wr_wx"] + [f"A_{v}" for v in V9]].to_csv(
        OUT / f"cw_predictions{tag}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    print(f"===== {freq}: 원점 {len(d)}개, 새 모형 예측 {int(d.an_crop9.notna().sum())}개 =====")

    ape = lambda s, c: ((s[c] / s.actual - 1).abs() * 100)
    rows = []
    for wname, (a, b) in WINDOWS.items():
        for sea in ("겨울", "봄", "여름", "가을", "전체"):
            s = d[yr.between(a, b) & ((d.tsea == sea) if sea != "전체" else True) & d.an_crop9.notna()]
            row = {"window": wname, "target": sea, "n": len(s)}
            for c in models + (["winter_ridge", "wr_wx"] if sea == "겨울" else []):
                row[c] = ape(s[s[c].notna()], c).mean()
            rows.append(row)
    sm = pd.DataFrame(rows)
    sm.to_csv(OUT / f"cw_season_metrics{tag}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    print("\n[대상 작형별 MAPE% — 편차 모형끼리(조정 전)]")
    print(sm.round(1).to_string(index=False))

    # 조립: 메인 규칙 그대로, 여름 원점 밖의 대표 모형과 겨울 모형만 교체
    new, fall = gm.load_issues(), gm.load_fall_issues()
    monthly = freq == "M"
    base = sg.main_like(d, new, fall, monthly)
    assert np.allclose(base[gm.FINAL], d[gm.FINAL], atol=0.11)
    variants = {"현행": d[gm.FINAL].values}
    for name in ("an_nowx", "an_crop4", "an_crop9", "an_crop9_s"):
        x = d.copy()
        x[gm.HEAD] = np.where((x.summer == 0) & x[name].notna(), x[name], x[gm.HEAD])
        variants[f"대표→{name}"] = sg.main_like(x, new, fall, monthly)[gm.FINAL].values
    for name in ("an_crop9_s", "an_crop3_s", "an_nowx_s"):          # 봄 대상월만 교체
        x = d.copy()
        x[gm.HEAD] = np.where((x.summer == 0) & (x.tsea == "봄") & x[name].notna(), x[name], x[gm.HEAD])
        variants[f"봄만→{name}"] = sg.main_like(x, new, fall, monthly)[gm.FINAL].values
    x = d.copy()                                                    # 여름 밖은 기상 없음, 봄은 봄 주산지 기상 3개
    x[gm.HEAD] = np.where((x.summer == 0) & x.an_nowx.notna(), np.where(x.tsea == "봄", x.an_crop3_s, x.an_nowx), x[gm.HEAD])
    variants["대표→an_nowx, 봄→an_crop3_s"] = sg.main_like(x, new, fall, monthly)[gm.FINAL].values
    x = d.copy()
    x["winter_ridge"] = np.where(x.wr_wx.notna(), x.wr_wx, x.winter_ridge)
    variants["겨울→wr_wx"] = sg.main_like(x, new, fall, monthly)[gm.FINAL].values
    x[gm.HEAD] = np.where((x.summer == 0) & x.an_crop9.notna(), x.an_crop9, x[gm.HEAD])
    variants["대표→an_crop9 + 겨울→wr_wx"] = sg.main_like(x, new, fall, monthly)[gm.FINAL].values
    rows = []
    for wname, (a, b) in WINDOWS.items():
        for sea in ("전체", "겨울", "봄", "가을"):
            mk = yr.between(a, b) & ((d.tsea == sea) if sea != "전체" else True)
            for k, v in variants.items():
                e = (pd.Series(v, index=d.index)[mk] / d.actual[mk] - 1) * 100
                rows.append({"window": wname, "target": sea, "variant": k, "n": int(mk.sum()), "MAPE%": e.abs().mean(), "n_gt30": int((e.abs() > 30).sum())})
    fm = pd.DataFrame(rows)
    fm.to_csv(OUT / f"cw_final_metrics{tag}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    print("\n[조립한 최종값 MAPE%]")
    print(fm.pivot_table(index=["window", "target"], columns="variant", values="MAPE%")[list(variants)].round(2).to_string())
    print("\n[조립한 최종값 30% 초과 수(전체)]")
    print(fm[fm.target == "전체"].pivot_table(index="window", columns="variant", values="n_gt30")[list(variants)].to_string())

    # 주산지 기상과 현행 모형 잔차의 상관
    rows = []
    for sea in ("겨울", "봄", "가을"):
        for wname, (a, b) in list(WINDOWS.items())[:2]:
            s = d[(d.tsea == sea) & yr.between(a, b) & d.A_tmax30.notna()]
            if len(s) < 6:
                continue
            row = {"target": sea, "window": wname, "n": len(s)}
            for v in V9:
                row[v] = float(np.corrcoef(s[f"A_{v}"], s.r)[0, 1]) if s[f"A_{v}"].std() > 0 else np.nan
            rows.append(row)
    cr = pd.DataFrame(rows)
    cr.to_csv(OUT / f"cw_corr{tag}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    print("\n[작형 주산지 기상과 현행 모형 잔차(실제 ÷ 예측의 로그)의 상관]")
    print(cr.round(2).to_string(index=False))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 330)
    pd.set_option("display.max_columns", 40)
    run(sys.argv[1] if len(sys.argv) > 1 else "M")
