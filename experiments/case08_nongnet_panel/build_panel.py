# -*- coding: utf-8 -*-
"""Case 08 — aT 농넷 배추 시장거래 자료(2011-01~2026-10) 정리·패널화.

입력 (data/local/nongnet/, 사용자 다운로드 2026-10-10, 공개 자료로 사용 문제 없음 확인)
  nongnet_cabbage_2011.xlsx, _2012_2016.xlsx, _2017_2021.xlsx, _2022_2026.xlsx
  열: DATE, 거래단위, 평균가격(거래단위당 원), 총거래물량(kg), 총거래금액(원), 도매시장, 품목, 품종, 산지-광역시도, 산지-시군구, 등급
  ※ 같은 플랫폼의 이전 추출본(W: `농넷_배추.xlsx`, 2020~2024)은 '도매법인' 열이 있었고 시장명 표기가 달랐다(서울가락도매 ↔ 서울가락).

출력 (data/ext/nongnet/ — 공개 가능, 추적)
  nongnet_cabbage_panel_2011_2026.csv.gz   전체 거래 행(정리본): date, market, variety, variety_std, origin_sido, origin_sigungu, grade, unit, qty_kg, amount_won, price_per_kg
  garak_daily_by_variety.csv               가락 일별 × 표준 품종: qty_t, amount, price_kg(가중)
  garak_daily_by_sigungu.csv               가락 일별 × 산지 시군구(상위 60개 시군): qty_t
  market_daily_total.csv                   시장별 일별 총물량·가중 kg가격
  storage_monthly_by_market.csv            '저장배추' 품종 월별 × 시장 (S3b 복원 검토용)
  variety_monthly_all.csv                  전국 월별 × 표준 품종 물량·가격
  qc_report.txt                            행 수·기간·결측·단위 분포·가락 상 등급 kg가격 vs clean_daily 대조

품종 표준화 (variety_std): 고랭지=고냉지배추·고랭지배추·고랭지 / 여름=여름배추 / 봄=봄배추 / 저장=저장배추 / 월동=월동배추·겨울배추 /
  가을=김장(가을)배추·가을배추·김장배추 / 쌈=쌈배추 / 절임=절임배추 / 수입=배추(수입) / 기타=기타배추·배추·나머지
Usage: python experiments/case08_nongnet_panel/build_panel.py   (xlsx 4개 읽기 수 분)
"""
from __future__ import annotations

import gzip
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "local" / "nongnet"
EXT = ROOT / "data" / "ext" / "nongnet"
CASE_OUT = Path(__file__).resolve().parent / "outputs"
FILES = ["nongnet_cabbage_2011.xlsx", "nongnet_cabbage_2012_2016.xlsx", "nongnet_cabbage_2017_2021.xlsx", "nongnet_cabbage_2022_2026.xlsx"]

VARIETY_MAP = [
    (r"고냉지|고랭지", "고랭지"), (r"여름", "여름"), (r"저장", "저장"), (r"봄", "봄"), (r"월동|겨울", "월동"),
    (r"김장|가을", "가을"), (r"쌈", "쌈"), (r"절임", "절임"), (r"수입", "수입"), (r"우거지|뿌리|배양|생채", "기타부산물"),
]


def std_variety(v: str) -> str:
    s = str(v)
    for pat, name in VARIETY_MAP:
        if re.search(pat, s):
            return name
    return "기타"


def load_all() -> pd.DataFrame:
    parts = []
    for f in FILES:
        t = time.time()
        d = pd.read_excel(SRC / f, engine="openpyxl")
        d.columns = ["date", "unit", "avg_price_unit", "qty_kg", "amount_won", "market", "item", "variety", "origin_sido", "origin_sigungu", "grade"][: len(d.columns)]
        parts.append(d)
        print(f"{f}: {len(d):,} rows, {time.time() - t:.0f}s", flush=True)
    d = pd.concat(parts, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ("qty_kg", "amount_won", "avg_price_unit"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d.date.notna() & d.qty_kg.gt(0)].copy()
    d["market"] = d["market"].astype(str).str.strip().str.replace("도매시장$|도매$", "", regex=True)
    d["variety_std"] = d["variety"].map(std_variety)
    # 거래단위 → kg. '10kg그물망'=10, '10kg그물망 3개'=30, '8ton트럭'=8000, '1kg상자'=1. 파싱 불가(예: '개', '포기', '단')는 NaN.
    u = d["unit"].astype(str).str.replace(" ", "")
    kg = u.str.extract(r"(\d+(?:\.\d+)?)\s*kg", flags=re.I)[0].astype(float)
    ton = u.str.extract(r"(\d+(?:\.\d+)?)\s*ton", flags=re.I)[0].astype(float) * 1000
    mult = u.str.extract(r"(\d+)개")[0].astype(float).fillna(1.0)
    d["unit_kg"] = kg.fillna(ton) * mult
    # kg당 가격 = 단위당 평균가격 ÷ 단위 kg (총거래금액/물량과 동일한 값이나 트럭 단위 등 혼입 행은 단위 파싱으로 걸러짐)
    d["price_per_kg"] = np.where(d["unit_kg"] > 0, d["avg_price_unit"] / d["unit_kg"], np.nan)
    # **가격 단위 보정(2026-10-10 QC)**: 2011-01-01~2013-12-31, 2014-09-01~2017-12-31 구간은 전 시장에서 평균가격·거래금액이
    # 실제의 약 10배로 기록됨(가락 '특' 10kg 그물망 vs clean_daily 비율 7.8~9.8, 보정 후 0.78~0.98; 구간 내 로그 상관 0.84~0.98).
    # 2014-01~08은 정상(0.85), 2014-09부터 ×10, 2018-01부터 정상. 해당 구간 가격에 0.1을 곱하고 flag 를 남긴다.
    scale_mask = ((d.date >= "2011-01-01") & (d.date <= "2013-12-31")) | ((d.date >= "2014-09-01") & (d.date <= "2017-12-31"))
    d["price_scale_applied"] = np.where(scale_mask, 0.1, 1.0)
    d["price_per_kg"] = d["price_per_kg"] * d["price_scale_applied"]
    d["amount_won"] = d["price_per_kg"] * d["qty_kg"]       # 보정된 kg가격 × 물량(가중 평균용). 원 금액 열은 버림
    d["grade"] = d["grade"].astype(str).str.strip()
    d = d.sort_values(["date", "market", "variety_std"]).reset_index(drop=True)
    return d[["date", "market", "variety", "variety_std", "origin_sido", "origin_sigungu", "grade", "unit", "unit_kg", "qty_kg", "price_per_kg", "price_scale_applied", "amount_won"]]


def wavg(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """물량 합(전체 행)과 kg당 가중평균가(가격 파싱된 행만)."""
    df = df.assign(qty_priced=np.where(df.price_per_kg.notna(), df.qty_kg, 0.0))
    g = df.groupby(by)
    out = g.agg(qty_t=("qty_kg", lambda s: s.sum() / 1000), amount_won=("amount_won", "sum"),
                qty_priced_kg=("qty_priced", "sum"), n_rows=("qty_kg", "size")).reset_index()
    out["price_kg"] = np.where(out.qty_priced_kg > 0, out.amount_won / out.qty_priced_kg, np.nan)
    return out.drop(columns="qty_priced_kg")


if __name__ == "__main__":
    EXT.mkdir(parents=True, exist_ok=True); CASE_OUT.mkdir(parents=True, exist_ok=True)
    d = load_all()
    print(f"total {len(d):,} rows  {d.date.min().date()} ~ {d.date.max().date()}", flush=True)
    with gzip.open(EXT / "nongnet_cabbage_panel_2011_2026.csv.gz", "wt", encoding="utf-8", newline="") as f:
        d.to_csv(f, index=False, float_format="%.2f")
    d.to_parquet(EXT / "nongnet_cabbage_panel_2011_2026.parquet", index=False)   # pyarrow, 분석용(빠름)
    g = d[d.market.str.contains("가락")]
    wavg(g, ["date", "variety_std"]).to_csv(EXT / "garak_daily_by_variety.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    top = g.groupby("origin_sigungu").qty_kg.sum().sort_values(ascending=False).head(60).index
    gs = g[g.origin_sigungu.isin(top)]
    gs.groupby(["date", "origin_sido", "origin_sigungu"]).qty_kg.sum().div(1000).rename("qty_t").reset_index().to_csv(EXT / "garak_daily_by_sigungu.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    wavg(d, ["date", "market"]).to_csv(EXT / "market_daily_total.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    st = d[d.variety_std == "저장"].copy(); st["ym"] = st.date.dt.to_period("M").astype(str)
    wavg(st, ["ym", "market"]).to_csv(EXT / "storage_monthly_by_market.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    d["ym"] = d.date.dt.to_period("M").astype(str)
    wavg(d, ["ym", "variety_std"]).to_csv(EXT / "variety_monthly_all.csv", index=False, float_format="%.1f", encoding="utf-8-sig")

    # QC
    lines = [f"rows={len(d):,}  period={d.date.min().date()}~{d.date.max().date()}  markets={d.market.nunique()}  varieties_raw={d.variety.nunique()}"]
    lines.append("rows by year: " + ", ".join(f"{y}:{n:,}" for y, n in d.date.dt.year.value_counts().sort_index().items()))
    lines.append("variety_std share of qty: " + ", ".join(f"{k}:{v:.1%}" for k, v in (d.groupby('variety_std').qty_kg.sum() / d.qty_kg.sum()).sort_values(ascending=False).items()))
    lines.append("raw variety → std: " + "; ".join(f"{k}→{std_variety(k)}" for k in d.variety.value_counts().index[:25]))
    lines.append("top markets by qty: " + ", ".join(f"{k}:{v/1e6:.0f}kt" for k, v in d.groupby('market').qty_kg.sum().sort_values(ascending=False).head(12).items()))
    lines.append("grade values: " + ", ".join(f"{k}:{v:,}" for k, v in d.grade.value_counts().head(8).items()))
    lines.append("unit top: " + ", ".join(f"{k}:{v:,}" for k, v in d.unit.value_counts().head(8).items()))
    # 가락 상 등급 10kg 환산가 vs clean_daily
    cd = pd.read_csv(ROOT / "data" / "raw" / "clean_daily.csv", parse_dates=["date"]).set_index("date")
    lines.append(f"unit_kg 파싱 실패 행 비율: {d.unit_kg.isna().mean():.1%} (물량 기준 {d.loc[d.unit_kg.isna(), 'qty_kg'].sum() / d.qty_kg.sum():.1%}); 미파싱 단위 상위: " +
                 ", ".join(f"{k}:{v:,}" for k, v in d.loc[d.unit_kg.isna(), 'unit'].value_counts().head(6).items()))
    gh = g[(g.grade == "특") & (g.unit_kg == 10) & g.unit.astype(str).str.contains("그물망") & g.price_per_kg.notna()]
    gp = (gh.groupby("date").apply(lambda x: (x.price_per_kg * x.qty_kg).sum() / x.qty_kg.sum() * 10, include_groups=False)).rename("nongnet_high_10kg")
    cmp = pd.concat([gp, cd.price.rename("clean_daily")], axis=1).dropna()
    corr = np.corrcoef(np.log(cmp.nongnet_high_10kg), np.log(cmp.clean_daily))[0, 1]
    ratio = (cmp.nongnet_high_10kg / cmp.clean_daily)
    lines.append(f"가락 '특' 10kg그물망 kg가중가×10 (단위 보정 후) vs clean_daily(KREI 상품): n={len(cmp):,}, log corr={corr:.3f}, ratio median={ratio.median():.3f}, IQR={ratio.quantile(.25):.3f}~{ratio.quantile(.75):.3f}")
    lines.append("  연도별 ratio 중앙값: " + ", ".join(f"{y}:{v:.2f}" for y, v in ratio.groupby(ratio.index.year).median().items()))
    lines.append("  연도별 log corr: " + ", ".join(f"{y}:{np.corrcoef(np.log(v.nongnet_high_10kg), np.log(v.clean_daily))[0,1]:.2f}" for y, v in cmp.groupby(cmp.index.year) if len(v) > 50))
    gq = g.groupby("date").qty_kg.sum().div(1000).rename("nongnet_garak_t")
    cq = pd.concat([gq, cd.quantity.rename("clean_qty_t")], axis=1).dropna()
    lines.append(f"가락 일별 총물량 vs clean_daily quantity: n={len(cq):,}, corr={np.corrcoef(cq.nongnet_garak_t, cq.clean_qty_t)[0,1]:.3f}, ratio median={(cq.nongnet_garak_t/cq.clean_qty_t).median():.3f}")
    # 저장배추 라벨 이력
    sm = st.groupby([st.date.dt.year, st.market]).qty_kg.sum().div(1000).unstack().fillna(0).round(0)
    lines.append("저장배추 연도×시장 물량(t):\n" + sm.to_string())
    s78 = st[st.date.dt.month.isin([7, 8])].groupby([st.date.dt.year, st.market]).qty_kg.sum().div(1000).unstack().fillna(0).round(0)
    lines.append("저장배추 7~8월 연도×시장 물량(t):\n" + s78.to_string())
    txt = "\n".join(lines)
    (EXT / "qc_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
