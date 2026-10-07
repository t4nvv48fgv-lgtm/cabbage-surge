"""Build a daily feature frame for cabbage surge forecasting.

Inputs (snapshot copies from CABIS, read-only):
  data/raw/clean_daily.csv          date, price(원/10kg 상품), quantity(톤)  2010~
  data/raw/weather_auto_daily.csv   station-level daily weather 2010~
  data/raw/stock.csv                date, gov_release, price, total_inflow 2005~

Output:
  data/processed/daily_features.parquet (or .csv)  one row per calendar day
  Every feature at row date d uses only information up to and including d.
  Target columns look forward and are NaN where the window is incomplete.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

HIGHLAND = ["평창", "태백", "정선군", "강릉"]          # 여름(고랭지) 주산지 관측소
LOWLAND_AUTUMN = ["해남", "진도군", "고창", "아산", "괴산", "제천", "영양", "문경", "춘천"]
SEMI_HIGHLAND = ["영월"]                               # 준고랭지(2기작) 대리 관측소
EXCLUDE_STATIONS = {"무안군", "서귀포", "예산군"}        # 2021-05~ 117일만 있는 관측소 → 전체 평균에서 제외
CLIM_YEARS = (2010, 2019)                              # 평년 기준 (anomaly 계산용)
TARGET_DAYS = 28                                       # 향후 28일 평균가격


def _daily_calendar(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="D")


def load_price() -> pd.DataFrame:
    p = pd.read_csv(RAW / "clean_daily.csv", parse_dates=["date"])
    p = p.rename(columns={"quantity": "qty"}).sort_values("date")
    p = p[p.price > 0]
    return p.set_index("date")


def load_gov() -> pd.Series:
    s = pd.read_csv(RAW / "stock.csv", parse_dates=["date"]).set_index("date")["gov_release"]
    return s.fillna(0.0)


def load_weather() -> pd.DataFrame:
    w = pd.read_csv(RAW / "weather_auto_daily.csv", parse_dates=["date"])
    return w


def region_daily(w: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    sub = w[w.region.isin(regions)]
    g = sub.groupby("date").agg(
        tmax=("max_temp", "mean"),
        tmin=("min_temp", "mean"),
        tavg=("avg_temp", "mean"),
        rain=("rainfall", "mean"),          # 지역 평균 강수(mm/일)
        heat=("heat_wave_day", "mean"),     # 폭염일 비율(0~1)
        rain_day=("rain_day", "mean"),
        sun=("sunshine_hr", "mean"),        # 일조시간(h/일)
    )
    g["hot30"] = (sub.assign(h=(sub.max_temp >= 30).astype(float)).groupby("date").h.mean())
    g["hot33"] = (sub.assign(h=(sub.max_temp >= 33).astype(float)).groupby("date").h.mean())
    g["heavy50"] = (sub.assign(h=(sub.rainfall >= 50).astype(float)).groupby("date").h.mean())
    g["tropical"] = (sub.assign(h=(sub.min_temp >= 25).astype(float)).groupby("date").h.mean())
    return g


def climatology(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Day-of-year climatology from CLIM_YEARS, smoothed with a 15-day circular window."""
    base = df[(df.index.year >= CLIM_YEARS[0]) & (df.index.year <= CLIM_YEARS[1])]
    doy = base.index.dayofyear
    clim = base[cols].groupby(doy).mean()
    clim = clim.reindex(range(1, 367)).interpolate(limit_direction="both")
    ext = pd.concat([clim.iloc[-7:], clim, clim.iloc[:7]])
    sm = ext.rolling(15, center=True, min_periods=1).mean().iloc[7:-7]
    sm.index = range(1, 367)
    return sm


def add_anomalies(df: pd.DataFrame, cols: list[str], prefix: str) -> pd.DataFrame:
    clim = climatology(df, cols)
    doy = df.index.dayofyear
    for c in cols:
        df[f"{prefix}{c}_anom"] = df[c].values - clim.loc[doy, c].values
    return df


def rolling_feats(s: pd.Series, name: str, windows=(7, 14, 30, 60), how="sum") -> pd.DataFrame:
    out = {}
    for w in windows:
        r = s.rolling(w, min_periods=max(3, w // 2))
        out[f"{name}_{how}{w}"] = r.sum() if how == "sum" else r.mean()
    return pd.DataFrame(out)


def build() -> pd.DataFrame:
    price = load_price()
    gov = load_gov()
    w = load_weather()

    start = "2010-01-01"
    end = max(price.index.max(), w.date.max()).strftime("%Y-%m-%d")
    cal = _daily_calendar(start, end)

    # ---------- price / quantity ----------
    P = price.reindex(cal)
    lp = np.log(P.price)
    lq = np.log(P.qty.where(P.qty > 0))
    f = pd.DataFrame(index=cal)
    f["is_trading"] = P.price.notna().astype(int)
    f["lp_last"] = lp.ffill()                                   # 최근 거래일 로그가격
    f["lp7"] = lp.rolling(7, min_periods=2).mean()
    f["lp28"] = lp.rolling(28, min_periods=8).mean()
    f["lp90"] = lp.rolling(90, min_periods=30).mean()
    f["lp28_lag364"] = f["lp28"].shift(364)
    f["lp28_lag728"] = f["lp28"].shift(728)
    f["mom7_28"] = f["lp7"] - f["lp28"]
    f["mom28_90"] = f["lp28"] - f["lp90"]
    f["yoy28"] = f["lp28"] - f["lp28_lag364"]
    f["vol28"] = lp.diff().rolling(28, min_periods=8).std()
    f["lq7"] = lq.rolling(7, min_periods=2).mean()
    f["lq28"] = lq.rolling(28, min_periods=8).mean()
    f["lq28_lag364"] = f["lq28"].shift(364)
    f["qyoy28"] = f["lq28"] - f["lq28_lag364"]
    f["qmom7_28"] = f["lq7"] - f["lq28"]
    # 7일 전 대비 가격 변화(급등 초기 포착)
    f["dlp7"] = f["lp7"] - f["lp7"].shift(7)
    f["dlp14"] = f["lp7"] - f["lp7"].shift(14)

    # ---------- Garak grade spread (품위 신호: 상품/중품 가격비) ----------
    g = pd.read_csv(RAW / "garak_daily.csv", parse_dates=["date"]).set_index("date").reindex(cal)
    spread = np.log(g.price_high / g.price_mid)
    spread_sp = np.log(g.price_special / g.price_high)
    f["spread_hm7"] = spread.rolling(7, min_periods=2).mean()
    f["spread_hm28"] = spread.rolling(28, min_periods=8).mean()
    f["dspread_hm"] = f["spread_hm7"] - f["spread_hm28"]
    f["spread_sh28"] = spread_sp.rolling(28, min_periods=8).mean()
    f["spread_hm28_lag364"] = f["spread_hm28"].shift(364)
    f["spread_yoy"] = f["spread_hm28"] - f["spread_hm28_lag364"]

    # ---------- government release ----------
    G = gov.reindex(cal).fillna(0.0)
    f["gov30"] = G.rolling(30, min_periods=1).sum()
    f["gov60"] = G.rolling(60, min_periods=1).sum()

    # ---------- weather: highland (여름) ----------
    H = region_daily(w, HIGHLAND).reindex(cal)
    H = add_anomalies(H, ["tmax", "tmin", "tavg", "rain"], "h_")
    for c, how in [("hot30", "sum"), ("hot33", "sum"), ("heavy50", "sum"), ("tropical", "sum"),
                   ("heat", "sum"), ("rain", "sum"), ("rain_day", "sum"),
                   ("h_tmax_anom", "mean"), ("h_tmin_anom", "mean"), ("h_tavg_anom", "mean"),
                   ("h_rain_anom", "mean")]:
        rf = rolling_feats(H[c], f"H_{c}", how=how)
        f = f.join(rf)

    # ---------- weather: lowland/autumn regions ----------
    L = region_daily(w, LOWLAND_AUTUMN).reindex(cal)
    L = add_anomalies(L, ["tmax", "tmin", "rain"], "l_")
    for c, how in [("hot30", "sum"), ("heavy50", "sum"), ("rain", "sum"),
                   ("l_tmax_anom", "mean"), ("l_tmin_anom", "mean"), ("l_rain_anom", "mean")]:
        rf = rolling_feats(L[c], f"L_{c}", how=how, windows=(14, 30, 60))
        f = f.join(rf)

    # ---------- [all-data 확장 2026-10-07] 고랭지 장기 누적(생육기 전체) ----------
    # (일조시간 sunshine_hr 은 스냅숏에서 거의 전부 0 → 피처로 쓰지 않음)
    for c, how in [("hot30", "sum"), ("heavy50", "sum"),
                   ("h_tmax_anom", "mean"), ("h_rain_anom", "mean")]:
        f = f.join(rolling_feats(H[c], f"H_{c}", how=how, windows=(90,)))

    # ---------- [all-data 확장] 준고랭지(영월) · 전체 관측소 평균 ----------
    M = region_daily(w, SEMI_HIGHLAND).reindex(cal)
    M = add_anomalies(M, ["tmax", "rain"], "m_")
    for c, how in [("hot30", "sum"), ("rain", "sum"), ("m_tmax_anom", "mean"), ("m_rain_anom", "mean")]:
        f = f.join(rolling_feats(M[c], f"M_{c}", how=how, windows=(14, 30, 60)))
    all_st = sorted(set(w.region) - EXCLUDE_STATIONS)
    A = region_daily(w, all_st).reindex(cal)
    A = add_anomalies(A, ["tmax", "tmin", "rain"], "a_")
    for c, how in [("hot30", "sum"), ("heavy50", "sum"),
                   ("a_tmax_anom", "mean"), ("a_tmin_anom", "mean"), ("a_rain_anom", "mean")]:
        f = f.join(rolling_feats(A[c], f"A_{c}", how=how, windows=(30, 60)))

    # ---------- [all-data 확장] 정부 매입 (at_gov_daily 2021~; 그 이전 자료 없음 → 0, govpur_avail 로 표시) ----------
    ag = pd.read_csv(RAW / "at_gov_daily.csv", parse_dates=["date"]).set_index("date")
    GP = ag["gov_purchase"].reindex(cal).fillna(0.0)
    f["govpur30"] = GP.rolling(30, min_periods=1).sum()
    f["govpur60"] = GP.rolling(60, min_periods=1).sum()
    f["govpur_avail"] = (cal >= pd.Timestamp("2021-01-01")).astype(int)

    # ---------- [all-data 확장] 작형별 연간 생산량 (produce.csv, 통계청) ----------
    # 당해년 값은 수확 후 공표되므로 원점 시점에 알 수 없음 → 전년(365일 전) 동일 작형 값만 사용 (누수 방지)
    pr = pd.read_csv(RAW / "produce.csv")
    pr["date"] = pd.to_datetime(pr.year.astype(str) + "-" + pr.month.astype(str) + "-01")
    prs = pr.set_index("date")["production_1000ton"].reindex(cal).ffill()
    lprod = np.log(prs)
    f["lprod_lag1y"] = lprod.shift(365)
    f["dprod_lag1y"] = lprod.shift(365) - lprod.shift(730)

    # ---------- [all-data 확장] 반입량 장기·단기 전년비 ----------
    f["lq90"] = lq.rolling(90, min_periods=30).mean()
    f["qyoy7"] = f["lq7"] - f["lq7"].shift(364)

    # ---------- calendar ----------
    doy = cal.dayofyear.values
    f["sin1"] = np.sin(2 * np.pi * doy / 365.25)
    f["cos1"] = np.cos(2 * np.pi * doy / 365.25)
    f["sin2"] = np.sin(4 * np.pi * doy / 365.25)
    f["cos2"] = np.cos(4 * np.pi * doy / 365.25)
    f["month"] = cal.month
    f["summer"] = cal.month.isin([6, 7, 8, 9]).astype(int)     # 원점이 6~9월이면 다음 28일은 7~10월 급등 위험 구간

    # ---------- targets (forward-looking) ----------
    fwd = lp.shift(-1)[::-1].rolling(TARGET_DAYS, min_periods=8).mean()[::-1]   # mean over d+1..d+28
    # rolling on reversed series: at position d (reversed) covers d+1..d+28 in original order
    f["y_lp_fwd28"] = fwd
    # ensure the whole window lies inside observed data
    last_obs = price.index.max()
    f.loc[f.index + pd.Timedelta(days=TARGET_DAYS) > last_obs, "y_lp_fwd28"] = np.nan
    f["y_chg"] = f["y_lp_fwd28"] - f["lp28"]                 # 로그 변화(모형 타깃)
    f["y_price_fwd28"] = np.exp(f["y_lp_fwd28"])
    f["y_price_fwd28_true"] = (
        P.price.shift(-1)[::-1].rolling(TARGET_DAYS, min_periods=8).mean()[::-1]
    )
    f.loc[f.index + pd.Timedelta(days=TARGET_DAYS) > last_obs, "y_price_fwd28_true"] = np.nan

    f.index.name = "date"
    return f


if __name__ == "__main__":
    PROC.mkdir(parents=True, exist_ok=True)
    df = build()
    out = PROC / "daily_features.csv"
    df.to_csv(out, float_format="%.5f")
    print(f"wrote {out}  shape={df.shape}  range={df.index.min().date()}..{df.index.max().date()}")
    chk = df.loc["2024-08-31", [c for c in df.columns if c.startswith("H_hot30") or c.startswith("H_h_tmax_anom") or c in ("lp28", "y_price_fwd28", "y_price_fwd28_true")]]
    print(chk.round(3).to_string())
