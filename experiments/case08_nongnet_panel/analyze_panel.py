# -*- coding: utf-8 -*-
"""Case 08 — 농넷 패널 활용성 분석.

A. 저장배추 시계열 복원 가능성: 가락 vs 그 외 시장(광주서부·천안 등) 7~8월 '저장배추' 2011~2026, 연도별 가락 저장 총량.
B. 급등 선행지표 후보: 가락 8월(1~31일, 원점 8/31 가용) 품종·산지별 반입 전년비 — 고랭지 품종, 강원 주요 시군(평창·태백·정선·강릉·삼척·홍천·영월), 2011~2026.
   급등 해(2020·2022·2024)에서 반입 전년비가 비급등 해와 갈리는지.
C. 가락 외 시장 가격 전파: 시장별 kg가격의 가락 대비 비율(연도별 중앙값) — 파운데이션 모델용 다시장 패널 품질 점검.
Outputs: outputs/storage_series.csv, leading_indicators.csv, market_price_ratio.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "data" / "ext" / "nongnet"
OUT = Path(__file__).resolve().parent / "outputs"
HIGHLAND_SGG = ["평창군", "태백시", "정선군", "강릉시", "삼척시", "홍천군", "영월군"]

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 300)
    d = pd.read_parquet(EXT / "nongnet_cabbage_panel_2011_2026.parquet")
    d["year"], d["month"] = d.date.dt.year, d.date.dt.month
    price = pd.read_csv(ROOT / "data" / "raw" / "clean_daily.csv", parse_dates=["date"]); price = price[price.price > 0]
    sep = price[price.date.dt.month == 9].groupby(price.date.dt.year).price.mean()
    aug = price[price.date.dt.month == 8].groupby(price.date.dt.year).price.mean()

    # ---- A. 저장배추
    st = d[d.variety_std == "저장"]
    a = pd.DataFrame({
        "garak_storage_7_8_t": st[st.market.str.contains("가락") & st.month.isin([7, 8])].groupby("year").qty_kg.sum() / 1000,
        "garak_storage_9_t": st[st.market.str.contains("가락") & (st.month == 9)].groupby("year").qty_kg.sum() / 1000,
        "garak_storage_year_t": st[st.market.str.contains("가락")].groupby("year").qty_kg.sum() / 1000,
        "other_storage_7_8_t": st[~st.market.str.contains("가락") & st.month.isin([7, 8])].groupby("year").qty_kg.sum() / 1000,
        "national_storage_7_8_t": st[st.month.isin([7, 8])].groupby("year").qty_kg.sum() / 1000,
    }).fillna(0).round(0)
    a["sep_price"] = sep.reindex(a.index).round(0); a["sep_vs_aug%"] = ((sep / aug - 1) * 100).reindex(a.index).round(1)
    a.to_csv(OUT / "storage_series.csv", encoding="utf-8-sig")
    print("=== A. 저장배추 품종 반입(t) ===\n", a.to_string())

    # ---- B. 선행지표: 가락 8월 반입 전년비
    g = d[d.market.str.contains("가락") & (d.month == 8)]
    tot = g.groupby("year").qty_kg.sum() / 1000
    hl = g[g.variety_std == "고랭지"].groupby("year").qty_kg.sum() / 1000
    sm = g[g.variety_std == "여름"].groupby("year").qty_kg.sum() / 1000
    gw = g[g.origin_sido.astype(str).str.contains("강원")].groupby("year").qty_kg.sum() / 1000
    hs = g[g.origin_sigungu.isin(HIGHLAND_SGG)].groupby("year").qty_kg.sum() / 1000
    late = d[d.market.str.contains("가락") & (d.month == 8) & (d.date.dt.day >= 16)].groupby("year").qty_kg.sum() / 1000
    b = pd.DataFrame({"garak_aug_t": tot, "highland_var_t": hl, "summer_var_t": sm, "gangwon_t": gw, "highland_sgg_t": hs, "late_aug_t": late}).round(0)
    for c in list(b.columns):
        b[c + "_yoy%"] = (b[c] / b[c].shift(1) - 1) * 100
    b["sep_vs_aug%"] = ((sep / aug - 1) * 100).reindex(b.index).round(1)
    b["surge"] = (sep.reindex(b.index) > 20000).astype(int)
    b = b.round(1); b.to_csv(OUT / "leading_indicators.csv", encoding="utf-8-sig")
    print("\n=== B. 가락 8월 반입(t)과 전년비(%) — 급등 해 표시 ===\n", b[["garak_aug_t", "garak_aug_t_yoy%", "gangwon_t_yoy%", "highland_sgg_t_yoy%", "late_aug_t_yoy%", "sep_vs_aug%", "surge"]].to_string())

    # ---- C. 시장별 가격 비율 (상 등급, 10kg 그물망, 가락 대비)
    p = d[(d.grade == "특") & (d.unit_kg == 10) & d.price_per_kg.notna()]     # '상'은 시장 간 잡음이 커(상관 0.1~0.7) 최고 등급 '특'으로
    gk = p[p.market.str.contains("가락")].groupby("date").apply(lambda x: (x.price_per_kg * x.qty_kg).sum() / x.qty_kg.sum(), include_groups=False).rename("garak")
    rows = []
    for mkt, sub in p[~p.market.str.contains("가락")].groupby("market"):
        s = sub.groupby("date").apply(lambda x: (x.price_per_kg * x.qty_kg).sum() / x.qty_kg.sum(), include_groups=False)
        j = pd.concat([s.rename("m"), gk], axis=1).dropna()
        if len(j) < 300:
            continue
        rows.append({"market": mkt, "n_days": len(j), "ratio_median": (j.m / j.garak).median(), "log_corr": np.corrcoef(np.log(j.m), np.log(j.garak))[0, 1],
                     "first": j.index.min().date(), "last": j.index.max().date()})
    c = pd.DataFrame(rows).sort_values("n_days", ascending=False).round(3)
    c.to_csv(OUT / "market_price_ratio.csv", index=False, encoding="utf-8-sig")
    print("\n=== C. 시장별 '특' 10kg 가격, 가락 대비 (일별) ===\n", c.to_string(index=False))
