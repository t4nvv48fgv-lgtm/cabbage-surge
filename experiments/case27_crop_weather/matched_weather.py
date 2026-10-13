"""Case 27 (2차) — 지역별 생산 시기에 그 지역 기상만 맞춰 쓰기.

사용자 지시(2026-10-13): "1년 내내 쓰는 게 아니라 지역별 생산시기에 지역의 기상을 매칭해줘. 그 지역에서 생산되지 않을 때는 기상을 참고하지 말아야 맞는 거지."
1차(crop_weather.py)는 대상월의 작형만 켰지만 '최근 30·60·90일'이라는 고정 길이 창이어서 정식 전 날씨가 섞이고 정식기 날씨가 빠졌다.

작형 달력: 사용자 업무 파일 `작형별_정식_출하_시기_(2026ver).xlsx` 배추 시트의 정식(빨강 점선)·수확(남색)·저장(하늘색) 선을 순(10일) 단위로 읽었다.
주산지: 같은 시트 '월별 주 출하지역'. CABIS 기상 스냅숏에 2010년부터 있는 관측소만 썼다(괄호 = 파일에는 있으나 관측소 없음).

  세부 작형     관측소                              정식            수확            저장
  봄 시설       예산 (나주)                         2/1~2/28        4/11~5/31
  봄 터널       의령군·아산 (의성)                  2/21~3/31       5/1~6/20
  봄 노지       문경·영월·영양 (충주)               3/11~4/30       5/21~7/10       (7/1~9/20, 물량이 작아 미사용)
  준고랭지 1기작 평창 (횡성 둔내)                   4/11~5/10       6/11~7/20
  고랭지        평창·태백·정선군·강릉 (삼척 하장)   5/1~7/20        7/11~9/30
  준고랭지 2기작 평창·강릉 (횡성)                   7/11~8/20       9/21~10/31
  가을          해남·진도군·고창·아산·괴산·제천·영양·문경·춘천 (당진)  8/21~9/30  10/11~12/31
  겨울          해남 (진도·무안은 기간이 짧아 제외)  9/11~9/30       12/21~이듬해 3/10   3/1~5/31

맞추는 규칙:
  (1) 어떤 세부 작형의 기상은 **정식 시작일부터 수확 종료일까지만** 센다. 그 밖의 날은 보지 않는다.
  (2) 원점 o 에서 그 작형이 '켜지는' 정도 w = 앞으로 28일(o+1~o+28) 가운데 그 작형의 출하 기간(수확 + 저장)에 드는 날의 비율(0~1).
      w = 0 이면 그 작형의 기상 재료는 0(참고하지 않음). 저장 기간에는 수확 종료일의 값으로 멈춘다.
  (3) 세부 작형 → 네 작형(겨울·봄·여름·가을)으로 w 가중 평균하고, 가장 큰 w 를 곱한다(전환기에는 두 작형이 함께 부분적으로 켜진다).
기상 요인(세부 작형 주산지 평균, 편차는 2010~2019 같은 날짜 평균 대비):
  gt_tmax·gt_tmin·gt_rain  정식 시작부터 지금까지의 최고·최저기온, 강수량 편차 평균(생육기 전체)
  r30_tmax·r30_rain        최근 30일(정식 시작 이후만) 최고기온·강수량 편차 평균(출하기 직전·출하기)
  pl_rain                  정식기 강수량 편차 평균(정식 지연)
  gt_heavy                 정식 뒤 50mm 이상 비 온 날 수
  ext                      정식 뒤 30°C 이상 일수(봄·여름·가을) / −5°C 이하 일수(겨울)
모형(메인과 같은 확장 창·누수 차단·사흘에 한 줄 솎기, 릿지 알파 10, 타깃 = 평년 대비 편차):
  an_sparse_all  현행 대표(고랭지 기상 4개를 1년 내내)
  an_nowx        기상 없음
  an_hl_on       현행 고랭지 기상 4개를 여름 작형이 켜질 때만(같은 재료에 '생산 시기에만' 원칙만 적용)
  an_match4      기상 없음 + 네 작형 × (gt_tmax, gt_rain, r30_tmax, r30_rain)
  an_match8      기상 없음 + 네 작형 × 8개 요인
  wr_match       겨울 모형(가격 15개, 겨울 대상 줄) + 겨울 작형 8개 요인
조립: 메인 규칙(게이트·겨울 분기·수준 조정·9월 고온)은 그대로. (가) 여름 원점 밖의 대표 모형만 교체, (나) 대표 모형을 전부 교체(여름 경보 원점 포함).

Usage: python experiments/case27_crop_weather/matched_weather.py [M|W]
Outputs: outputs/mw_predictions{tag}.csv, mw_season_metrics{tag}.csv, mw_final_metrics{tag}.csv, mw_aug{tag}.csv, mw_calendar.csv
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

# (작형, 관측소, 정식 시작, 정식 끝, 수확 시작, 수확 끝, 출하 끝[저장 포함], 수확·출하 끝이 이듬해인가)
SUB = {
    "봄시설": ("봄", ["예산"], (2, 1), (2, 28), (4, 11), (5, 31), (5, 31), False),
    "봄터널": ("봄", ["의령군", "아산"], (2, 21), (3, 31), (5, 1), (6, 20), (6, 20), False),
    "봄노지": ("봄", ["문경", "영월", "영양"], (3, 11), (4, 30), (5, 21), (7, 10), (7, 10), False),
    "준고랭지1": ("여름", ["평창"], (4, 11), (5, 10), (6, 11), (7, 20), (7, 20), False),
    "고랭지": ("여름", bf.HIGHLAND, (5, 1), (7, 20), (7, 11), (9, 30), (9, 30), False),
    "준고랭지2": ("여름", ["평창", "강릉"], (7, 11), (8, 20), (9, 21), (10, 31), (10, 31), False),
    "가을": ("가을", bf.LOWLAND_AUTUMN, (8, 21), (9, 30), (10, 11), (12, 31), (12, 31), False),
    "겨울": ("겨울", ["해남"], (9, 11), (9, 30), (12, 21), (3, 10), (5, 31), True),
}
CROPS = ["겨울", "봄", "여름", "가을"]
V4 = ["gt_tmax", "gt_rain", "r30_tmax", "r30_rain"]
V8 = V4 + ["gt_tmin", "gt_heavy", "pl_rain", "ext"]
CORE3 = ["cur_anom7", "dlp7", "qyoy28"]
HL = ["H_h_tmax_anom_mean30", "H_hot30_sum60", "H_h_rain_anom_mean30", "H_heavy50_sum60"]
SEASON = {1: "겨울", 2: "겨울", 3: "겨울", 4: "겨울", 5: "봄", 6: "봄", 7: "여름", 8: "여름", 9: "여름", 10: "여름", 11: "가을", 12: "가을"}
WINDOWS = {"2016~2020": (2016, 2020), "2021~2025": (2021, 2025), "2021~2026": (2021, 2026)}
H = 28


def station_base(w: pd.DataFrame, regions: list[str], cal: pd.DatetimeIndex) -> pd.DataFrame:
    sub = w[w.region.isin(regions)]
    g = sub.groupby("date").agg(tmax=("max_temp", "mean"), tmin=("min_temp", "mean"), rain=("rainfall", "mean")).reindex(cal)
    clim = bf.climatology(g, ["tmax", "tmin", "rain"])
    doy = cal.dayofyear
    b = pd.DataFrame({c + "_an": g[c].values - clim.loc[doy, c].values for c in ("tmax", "tmin", "rain")}, index=cal)
    for name, cond in (("heavy", sub.rainfall >= 50), ("hot", sub.max_temp >= 30), ("cold", sub.min_temp <= -5)):
        b[name] = sub.assign(h=cond.astype(float)).groupby("date").h.mean().reindex(cal).fillna(0.0)
    return b


def sub_features(base: pd.DataFrame, spec: tuple, cal: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series]:
    """세부 작형 한 개의 (기상 요약값, 켜지는 정도 w). 생산 기간 밖은 NaN·0."""
    crop, _, ps, pe, hs, he, se, cross = spec
    V = pd.DataFrame(np.nan, index=cal, columns=V8)
    wgt = pd.Series(0.0, index=cal)
    for y in range(cal.year.min(), cal.year.max() + 1):
        p0, p1, h0 = pd.Timestamp(y, *ps), pd.Timestamp(y, *pe), pd.Timestamp(y, *hs)
        h1, s1 = pd.Timestamp(y + int(cross), *he), pd.Timestamp(y + int(cross), *se)
        if p0 < cal.min() or p0 > cal.max():
            continue
        idx = pd.date_range(p0, min(s1, cal.max()))
        seg = base.loc[p0:min(h1, cal.max())]                       # 기상은 정식 시작 ~ 수확 종료만 센다
        f = pd.DataFrame(index=seg.index)
        em = seg[["tmax_an", "tmin_an", "rain_an"]].expanding(min_periods=5).mean()
        f["gt_tmax"], f["gt_tmin"], f["gt_rain"] = em.tmax_an, em.tmin_an, em.rain_an
        r30 = seg[["tmax_an", "rain_an"]].rolling(30, min_periods=5).mean()
        f["r30_tmax"], f["r30_rain"] = r30.tmax_an, r30.rain_an
        f["gt_heavy"] = seg.heavy.cumsum()
        f["ext"] = (seg.cold if crop == "겨울" else seg.hot).cumsum()
        f["pl_rain"] = seg.rain_an.where(seg.index <= p1).expanding(min_periods=5).mean().ffill()
        V.loc[idx, V8] = f.reindex(idx).ffill()[V8].values          # 저장 기간에는 수확 종료일 값으로 멈춤
        d = idx.values
        lo = np.maximum(d + np.timedelta64(1, "D"), np.datetime64(h0))
        hi = np.minimum(d + np.timedelta64(H, "D"), np.datetime64(s1))
        wgt.loc[idx] = np.clip((hi - lo) / np.timedelta64(1, "D") + 1, 0, H) / H
    return V, wgt


def build_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(bt.PROC / "daily_features.csv", parse_dates=["date"]).set_index("date")
    df[bt.TARGET] = df["y_lp_fwd28"] - df[bt.BASE]
    df = am.add_normals(df)
    w = bf.load_weather()
    cal = df.index
    parts = {name: sub_features(station_base(w, spec[1], cal), spec, cal) for name, spec in SUB.items()}
    act = pd.DataFrame(index=cal)
    for k in CROPS:
        subs = [n for n, s in SUB.items() if s[0] == k]
        W = pd.concat([parts[n][1] for n in subs], axis=1)
        tot = W.sum(axis=1)
        a = W.max(axis=1)
        act[k] = a
        for v in V8:
            num = sum(parts[n][0][v].fillna(0.0) * parts[n][1] for n in subs)
            df[f"M_{k}_{v}"] = (num / tot.where(tot > 0)).fillna(0.0) * a
        df[f"act_{k}"] = a
    for c in HL:
        df[c + "_on"] = df[c] * df["act_여름"]
    for n in SUB:
        act["sub_" + n] = parts[n][1]
    return df, act


def run(freq: str) -> None:
    tag = "" if freq == "M" else f"_{freq}"
    df, act = build_frame()
    if freq == "M":
        cal = act[act.index.year == 2023].resample("MS").mean().round(2)
        cal.index = cal.index.strftime("%m월")
        cal.to_csv(OUT / "mw_calendar.csv", encoding="utf-8-sig")
        print("[월별 평균 '켜지는 정도'(2023년 예): 그 달의 원점에서 앞으로 28일이 그 작형 출하 기간에 드는 비율]")
        print(cal.T.to_string())
    m4 = [f"M_{k}_{v}" for k in CROPS for v in V4]
    m8 = [f"M_{k}_{v}" for k in CROPS for v in V8]
    win8 = [f"M_겨울_{v}" for v in V8]
    specs = {"an_nowx": CORE3, "an_hl_on": CORE3 + [c + "_on" for c in HL], "an_match4": CORE3 + m4, "an_match8": CORE3 + m8}
    pool = df[(df.is_trading == 1) & df.y_anom.notna()].dropna(subset=am.CORE + m8).iloc[::3]
    tsea = pd.Series((df.index + pd.Timedelta(days=15)).month.map(SEASON), index=df.index)
    wpool = df[(df.is_trading == 1) & df[bt.TARGET].notna()].dropna(subset=bt.PRICE_FEATS + bt.SUPPLY_FEATS + bt.WX_FEATS + am.CORE + ["y_anom"]).iloc[::3]
    wpool = wpool[tsea.reindex(wpool.index) == "겨울"]
    g = sg.load(freq)
    ridge = lambda a: make_pipeline(StandardScaler(), Ridge(alpha=a))
    rows = []
    for o in g.origin:
        if o not in df.index:
            continue
        te = df.loc[[o]]
        if te[am.CORE + m8].isna().any(axis=1).iloc[0]:
            continue
        tr = pool[pool.index + pd.Timedelta(days=28) <= o]
        if len(tr) < 120:
            continue
        nf = float(te.norm_fwd.iloc[0])
        rec = {"origin": o}
        for name, feats in specs.items():
            rec[name] = float(np.exp(nf + ridge(10.0).fit(tr[feats], tr.y_anom).predict(te[feats])[0]))
        if tsea[o] == "겨울":
            tw = wpool[wpool.index + pd.Timedelta(days=28) <= o]
            if len(tw) >= gm.MIN_TRAIN_WINTER:
                fe = bt.PRICE_FEATS + win8
                rec["wr_match"] = float(np.exp(float(te[bt.BASE].iloc[0]) + ridge(3.0).fit(tw[fe], tw[bt.TARGET]).predict(te[fe])[0]))
        rows.append(rec)
    d = g.merge(pd.DataFrame(rows), on="origin", how="left")
    d["tsea"] = d.tmonth.map(SEASON)
    yr = d.origin.dt.year
    models = ["an_sparse_all", "an_nowx", "an_hl_on", "an_match4", "an_match8"]
    d[["origin", "actual", "tmonth", "tsea", "summer", "final", "winter_ridge"] + models + ["wr_match"]].to_csv(
        OUT / f"mw_predictions{tag}.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    print(f"\n===== {freq}: 원점 {len(d)}개, 새 모형 예측 {int(d.an_match8.notna().sum())}개 =====")
    ape = lambda s, c: ((s[c] / s.actual - 1).abs() * 100)
    rows = []
    for wname, (a, b) in WINDOWS.items():
        for sea in ("겨울", "봄", "여름", "가을", "전체"):
            s = d[yr.between(a, b) & ((d.tsea == sea) if sea != "전체" else True) & d.an_match8.notna()]
            row = {"window": wname, "target": sea, "n": len(s)}
            for c in models + (["winter_ridge", "wr_match"] if sea == "겨울" else []):
                row[c] = ape(s[s[c].notna()], c).mean()
            rows.append(row)
    sm = pd.DataFrame(rows)
    sm.to_csv(OUT / f"mw_season_metrics{tag}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    print("\n[대상 작형별 MAPE% — 편차 모형끼리(게이트·조정 전)]")
    print(sm.round(1).to_string(index=False))

    new, fall = gm.load_issues(), gm.load_fall_issues()
    monthly = freq == "M"
    base = sg.main_like(d, new, fall, monthly)
    assert np.allclose(base[gm.FINAL], d[gm.FINAL], atol=0.11)
    variants, frames = {"현행": d[gm.FINAL].values}, {}
    for name in ("an_nowx", "an_hl_on", "an_match4", "an_match8"):
        x = d.copy()
        x[gm.HEAD] = np.where((x.summer == 0) & x[name].notna(), x[name], x[gm.HEAD])
        variants[f"(가) 여름 밖만 {name}"] = sg.main_like(x, new, fall, monthly)[gm.FINAL].values
    for name in ("an_hl_on", "an_match4", "an_match8"):
        x = d.copy()
        x[gm.HEAD] = np.where(x[name].notna(), x[name], x[gm.HEAD])
        frames[name] = sg.main_like(x, new, fall, monthly)
        variants[f"(나) 전부 {name}"] = frames[name][gm.FINAL].values
    x = d.copy()
    x["winter_ridge"] = np.where(x.wr_match.notna(), x.wr_match, x.winter_ridge)
    variants["겨울 모형 → wr_match"] = sg.main_like(x, new, fall, monthly)[gm.FINAL].values
    rows = []
    for wname, (a, b) in WINDOWS.items():
        for sea in ("전체", "겨울", "봄", "여름", "가을"):
            mk = yr.between(a, b) & ((d.tsea == sea) if sea != "전체" else True)
            for k, v in variants.items():
                e = (pd.Series(v, index=d.index)[mk] / d.actual[mk] - 1) * 100
                rows.append({"window": wname, "target": sea, "variant": k, "n": int(mk.sum()), "MAPE%": e.abs().mean(), "n_gt30": int((e.abs() > 30).sum())})
    fm = pd.DataFrame(rows)
    fm.to_csv(OUT / f"mw_final_metrics{tag}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    print("\n[조립한 최종값 MAPE%]")
    print(fm.pivot_table(index=["window", "target"], columns="variant", values="MAPE%")[list(variants)].round(2).T.to_string())
    print("\n[조립한 최종값 30% 초과 수(전체)]")
    print(fm[fm.target == "전체"].pivot_table(index="window", columns="variant", values="n_gt30")[list(variants)].T.to_string())
    if monthly:
        key = d.origin.dt.month.eq(8) & (yr >= 2020)
        t = pd.DataFrame({"origin": d.origin[key].dt.date, **{k: ((pd.Series(v, index=d.index)[key] / d.actual[key] - 1) * 100).round(0) for k, v in variants.items() if k == "현행" or k.startswith("(나)")}})
        t.to_csv(OUT / "mw_aug.csv", index=False, encoding="utf-8-sig")
        print("\n[8월 말 원점(9월 대상) 오차율% — 대표 모형을 전부 바꿨을 때]")
        print(t.to_string(index=False))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 330)
    pd.set_option("display.max_columns", 40)
    run(sys.argv[1] if len(sys.argv) > 1 else "M")
