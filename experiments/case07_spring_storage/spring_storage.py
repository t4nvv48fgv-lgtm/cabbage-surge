"""Case 07 — 봄배추 저장(민간·대량수요처)이 여름 급등을 가르는가.

사용자 가설(2026-10-10): 봄배추 저장량이 여름 급등 여부에 중요한 역할을 한다.
세 가지 측정치를 모아 급등 해·비급등 해 분리 여부와 2024 vs 2025 판별력을 본다.

  S3a  저장량 수준: KREI 월보 본문의 노지봄배추 저장량(대량수요처) 전년비 — 7~9월호 서술(판독 텍스트, 2021~2025) + 김치협회 조사(2025·2026)
  S3b  저장분 출하 흐름: 도매시장 '저장배추' 품종 반입량 7~9월 — 농넷 전국 원장(2020~2024-11), 대아청과 거래(2016~2026-02)
       **라벨 주의**: 농넷 기준 가락시장 7~9월 '저장배추'는 2020~2023 모두 0, 2024부터 등장(1,885t). 대아청과도 같은 패턴(2024 1,178t·2025 4,818t).
       → 2024 이전 0은 "출하 없음"이 아니라 "라벨 미사용"일 가능성이 커 S3b는 2024~2025 두 해만 비교 가능.
  S3c  사용 계획·재고: 김치협회 2026 봄배추 저장량 조사 3차(8/26 기준) — 2026 vs 2025 저장량·재고·월별 사용계획

출력: outputs/storage_level.csv(S3a), storage_shipments.csv(S3b 연×월), kimchi_assoc_2026.csv(S3c), summary.csv
Usage: python experiments/case07_spring_storage/spring_storage.py   (data/local 필요: 농넷·대아청과·김치협회 파일)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
LOCAL = ROOT / "data" / "local"
OUT = CASE_DIR / "outputs"
MAIN_OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "experiments" / "case06_supply_gate"))

# S3a — 월보 본문 서술(CABIS krei_monthly text, 2026-10-10 grep) + 김치협회 조사. 출처 열에 호수 기록.
S3A = pd.DataFrame([
    {"year": 2021, "storage_t": 59000, "storage_yoy%": 17.5, "source": "2021-07호 노지봄배추 저장량 5만9천톤(전년 +17.5%, 평년 +52.8%)"},
    {"year": 2022, "storage_t": 25000, "storage_yoy%": -12.4, "source": "2022-07호 노지봄배추 저장량 2만5천톤(전년 −12.4%, 평년 −14.6%); 작황 부진으로 저장 품위 저하"},
    {"year": 2023, "storage_t": np.nan, "storage_yoy%": 41.8, "source": "2023-07·08호 노지봄배추 저장량 전년 +41.8%, 평년 +33.4%"},
    {"year": 2024, "storage_t": np.nan, "storage_yoy%": 12.5, "source": "2024-09호 노지봄배추 저장량 전년 대비 10~15% 증가 추정(중간값), 9월 저장분 감모 언급. 8월호 텍스트 없음"},
    {"year": 2025, "storage_t": 42672, "storage_yoy%": 4.6, "source": "2025-08호 대량수요처 봄배추 저장량 전년 +4.6%; 김치협회 2026 조사의 2025 저장량 42,672t"},
    {"year": 2026, "storage_t": 46983, "storage_yoy%": 10.1, "source": "김치협회 2026 봄배추 저장량 조사 3차(8/26 기준, 38개 업체) 저장 계획량 46,983t(+10.1%)"},
])


def s3b_shipments() -> pd.DataFrame:
    rows = []
    d = pd.read_excel(LOCAL / "yupgeun" / "농넷_배추.xlsx", sheet_name="raw", usecols=["DATE", "총거래물량", "도매시장", "품종"])
    d["date"] = pd.to_datetime(d["DATE"], errors="coerce")
    g = d[d["도매시장"].astype(str).str.contains("가락")]
    for (y, m), v in (g[g["품종"].astype(str).str.contains("저장")].groupby([g.date.dt.year, g.date.dt.month])["총거래물량"].sum() / 1000).items():
        rows.append({"source": "농넷_가락", "year": y, "month": m, "storage_t": v})
    for (y, m), v in (d[d["품종"].astype(str).str.contains("저장")].groupby([d.date.dt.year, d.date.dt.month])["총거래물량"].sum() / 1000).items():
        rows.append({"source": "농넷_전국", "year": y, "month": m, "storage_t": v})
    da = pd.read_excel(LOCAL / "d_drive" / "이사자료2_zip" / "배추__무__양배추_품종별_거래__현황_(2016.1.1_2026.02.28).xlsx", sheet_name="배추")
    da["date"] = pd.to_datetime(da["거래일자"], errors="coerce")
    for (y, m), v in (da[da["품종"].astype(str).str.contains("저장")].groupby([da.date.dt.year, da.date.dt.month])["물량"].sum() / 1000).items():
        rows.append({"source": "대아청과", "year": y, "month": m, "storage_t": v})
    # 가락 전체 반입 대비 비중(농넷)
    tot = g.groupby([g.date.dt.year, g.date.dt.month])["총거래물량"].sum() / 1000
    for (y, m), v in tot.items():
        rows.append({"source": "농넷_가락_전체반입", "year": y, "month": m, "storage_t": v})
    return pd.DataFrame(rows)


def s3c_kimchi() -> pd.DataFrame:
    x = pd.read_excel(LOCAL / "yupgeun" / "storage" / "2026_봄배추_저장량_정리.xlsx", sheet_name="집계 · 차트", header=None)
    rows = []
    for i in range(len(x)):
        a = str(x.iat[i, 0])
        if a.endswith("월") or a in ("저장량", "재고량", "현재 재고량", "저장 계획량", "소진율"):
            rows.append({"item": a, "y2026": x.iat[i, 1], "y2025": x.iat[i, 2], "diff": x.iat[i, 3], "pct": x.iat[i, 4]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    S3A.to_csv(OUT / "storage_level.csv", index=False, encoding="utf-8-sig")
    sh = s3b_shipments(); sh.to_csv(OUT / "storage_shipments.csv", index=False, encoding="utf-8-sig")
    kc = s3c_kimchi(); kc.to_csv(OUT / "kimchi_assoc_2026.csv", index=False, encoding="utf-8-sig")

    # 연도별 요약: 9월 가격, 급등 여부, S1(KREI 생산 전망 8월호), S3a, S3b(7~8월 저장 반입, 가락·대아)
    price = pd.read_csv(ROOT / "data" / "raw" / "clean_daily.csv", parse_dates=["date"]); price = price[price.price > 0]
    sep = price[price.date.dt.month == 9].groupby(price.date.dt.year).price.mean()
    aug = price[price.date.dt.month == 8].groupby(price.date.dt.year).price.mean()
    sig = pd.read_csv(ROOT / "experiments" / "case06_supply_gate" / "outputs" / "signals.csv", parse_dates=["origin"])
    s1 = sig[sig.origin.dt.month == 8].set_index(sig[sig.origin.dt.month == 8].origin.dt.year)["S1_prod_yoy"]
    piv = sh.pivot_table(index=["source", "year"], columns="month", values="storage_t", aggfunc="sum").fillna(0)
    rows = []
    for y in range(2016, 2027):
        r = {"year": y, "aug_price": round(aug.get(y, np.nan)), "sep_price": round(sep.get(y, np.nan)),
             "sep_vs_aug%": round((sep.get(y, np.nan) / aug.get(y, np.nan) - 1) * 100, 1),
             "surge": int(sep.get(y, 0) > 20000), "S1_krei_aug": s1.get(y, np.nan)}
        a = S3A[S3A.year == y]
        r["S3a_storage_yoy%"] = float(a["storage_yoy%"].iloc[0]) if len(a) else np.nan
        for src in ("농넷_가락", "대아청과"):
            try:
                row = piv.loc[(src, y)]
                r[f"S3b_{src}_7-8월_t"] = round(float(row.get(7, 0) + row.get(8, 0)))
                r[f"S3b_{src}_9월_t"] = round(float(row.get(9, 0)))
            except KeyError:
                r[f"S3b_{src}_7-8월_t"] = np.nan; r[f"S3b_{src}_9월_t"] = np.nan
        rows.append(r)
    summ = pd.DataFrame(rows); summ.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print("=== S3a 저장량 수준 ===\n", S3A[["year", "storage_t", "storage_yoy%"]].to_string(index=False))
    print("\n=== S3b 저장배추 품종 반입(t), 7~9월 ===")
    print(piv.loc[piv.index.get_level_values(0).isin(["농넷_가락", "농넷_전국", "대아청과"])].reindex(columns=[5, 6, 7, 8, 9, 10]).round(0).to_string())
    print("\n=== S3c 김치협회 2026 조사 (vs 2025) ===\n", kc.to_string(index=False))
    print("\n=== 연도별 요약 ===\n", summ.to_string(index=False))
