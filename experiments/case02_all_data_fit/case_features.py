"""Case 02 피처 확장 — 메인 daily_features.csv 에 '가용 데이터 전부' 피처를 덧붙인다.

메인 src/build_features.py 는 수정하지 않고 그 함수(load_weather, region_daily, add_anomalies, rolling_feats)만 재사용한다.
입력:  data/processed/daily_features.csv (메인 산출물), data/raw/*
출력:  experiments/case02_all_data_fit/outputs/daily_features_ext.csv  (메인 열 그대로 + 아래 확장 열)

확장 열
  정부 매입      govpur30/60 (at_gov_daily 2021~, 그 이전 자료 없음 → 0, govpur_avail 더미)
  작형별 생산량  lprod_lag1y, dprod_lag1y (produce.csv 통계청 연간값. 당해년 값은 수확 후 공표라 누수 → 365일 lag만)
  반입량        lq90, qyoy7
  고랭지 90일   H_hot30_sum90, H_heavy50_sum90, H_h_tmax_anom_mean90, H_h_rain_anom_mean90
  준고랭지(영월) M_hot30_sum{14,30,60}, M_rain_sum*, M_m_tmax_anom_mean*, M_m_rain_anom_mean*
  전체 관측소    A_hot30_sum{30,60}, A_heavy50_sum*, A_a_tmax_anom_mean*, A_a_tmin_anom_mean*, A_a_rain_anom_mean*
  (일조시간 sunshine_hr 은 스냅숏에서 거의 전부 0 → 제외)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_features as bf  # noqa: E402  (메인 함수 재사용, 메인 파일 무수정)

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = CASE_DIR / "outputs"

SEMI_HIGHLAND = ["영월"]                               # 준고랭지(2기작) 대리 관측소
EXCLUDE_STATIONS = {"무안군", "서귀포", "예산군"}        # 2021-05~ 117일만 있는 관측소 → 전체 평균에서 제외


def extend() -> pd.DataFrame:
    f = pd.read_csv(PROC / "daily_features.csv", parse_dates=["date"]).set_index("date")
    cal = f.index
    w = bf.load_weather()

    # 고랭지 장기 누적(생육기 전체)
    H = bf.region_daily(w, bf.HIGHLAND).reindex(cal)
    H = bf.add_anomalies(H, ["tmax", "tmin", "tavg", "rain"], "h_")
    for c, how in [("hot30", "sum"), ("heavy50", "sum"), ("h_tmax_anom", "mean"), ("h_rain_anom", "mean")]:
        f = f.join(bf.rolling_feats(H[c], f"H_{c}", how=how, windows=(90,)))

    # 준고랭지(영월) · 전체 관측소 평균
    M = bf.region_daily(w, SEMI_HIGHLAND).reindex(cal)
    M = bf.add_anomalies(M, ["tmax", "rain"], "m_")
    for c, how in [("hot30", "sum"), ("rain", "sum"), ("m_tmax_anom", "mean"), ("m_rain_anom", "mean")]:
        f = f.join(bf.rolling_feats(M[c], f"M_{c}", how=how, windows=(14, 30, 60)))
    all_st = sorted(set(w.region) - EXCLUDE_STATIONS)
    A = bf.region_daily(w, all_st).reindex(cal)
    A = bf.add_anomalies(A, ["tmax", "tmin", "rain"], "a_")
    for c, how in [("hot30", "sum"), ("heavy50", "sum"),
                   ("a_tmax_anom", "mean"), ("a_tmin_anom", "mean"), ("a_rain_anom", "mean")]:
        f = f.join(bf.rolling_feats(A[c], f"A_{c}", how=how, windows=(30, 60)))

    # 정부 매입 (at_gov_daily 2021~)
    ag = pd.read_csv(RAW / "at_gov_daily.csv", parse_dates=["date"]).set_index("date")
    GP = ag["gov_purchase"].reindex(cal).fillna(0.0)
    f["govpur30"] = GP.rolling(30, min_periods=1).sum()
    f["govpur60"] = GP.rolling(60, min_periods=1).sum()
    f["govpur_avail"] = (cal >= pd.Timestamp("2021-01-01")).astype(int)

    # 작형별 연간 생산량 — 전년 동일 작형만 (누수 방지)
    pr = pd.read_csv(RAW / "produce.csv")
    pr["date"] = pd.to_datetime(pr.year.astype(str) + "-" + pr.month.astype(str) + "-01")
    prs = pr.set_index("date")["production_1000ton"].reindex(cal).ffill()
    lprod = np.log(prs)
    f["lprod_lag1y"] = lprod.shift(365)
    f["dprod_lag1y"] = lprod.shift(365) - lprod.shift(730)

    # 반입량 장기·단기 전년비
    P = bf.load_price().reindex(cal)
    lq = np.log(P.qty.where(P.qty > 0))
    f["lq90"] = lq.rolling(90, min_periods=30).mean()
    f["qyoy7"] = f["lq7"] - f["lq7"].shift(364)
    return f


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    df = extend()
    out = OUT / "daily_features_ext.csv"
    df.to_csv(out, float_format="%.5f")
    print(f"wrote {out}  shape={df.shape}")
