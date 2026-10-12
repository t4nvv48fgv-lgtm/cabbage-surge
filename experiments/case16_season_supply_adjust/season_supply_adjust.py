"""Case 16 — 여름 외 작형(겨울·봄·가을)의 KREI 월보 생산 전망을 '수준 조정'으로 넣는다 (탄력성 레이어).

배경(2026-10-12): 최종 모형의 20% 초과 오차 17개월 중 7개월이 가을·겨울·봄배추 공급 충격(2021-11, 2022-11, 2023-04, 2024-04, 2025-01, 2025-06, 2021-05).
여름처럼 모형을 바꾸는 스위치가 아니라, 예측값 자체를 생산 전망 전년비로 밀어 주는 방식을 시험한다.

  조정:  pred_adj = pred × (1 + β × S / 100),   S = 대상월 작형의 최신호 생산량 전년비(%, 없으면 면적 전년비), |S| ≤ 20 으로 절단
  β  = 가격 신축성(공급 1% 변화 → 가격 β% 변화). 결과를 보고 맞추지 않도록 **문헌값 −2 를 기본**으로 두고 {0, −0.5, −1, −1.5, −2, −3} 민감도.
      (참고로 2021~2025에 맞춘 사후 β 도 계산해 병기하되 근거로 쓰지 않는다.)

작형 매핑(대상월 기준): 1~4월 겨울(전년 가을 정식분: 발행 8~12월호는 그해 작형, 3~5월호는 전년 작형), 5~6월 봄(전체>노지>시설 순, 그해 3~7월호),
11~12월 가을(그해 7~12월호), 7~10월 여름(메인 게이트 S1). 가용 시점 규약 B(원점 m월 말에 m+1월호까지).
변형: V1 비여름만 조정(여름은 메인 게이트 그대로) / V2 여름 경보 달도 S1로 조정 / V3 모든 달 조정(여름 평시 달 포함).

Usage: python experiments/case16_season_supply_adjust/season_supply_adjust.py
Outputs: outputs/issues_nonsummer_2021_2026.csv(판독표 발췌, 공개 수치), adjust_predictions.csv, adjust_metrics.csv, big_errors_before_after.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
OUT = CASE_DIR / "outputs"
KREI_XLSX = ROOT / "data" / "local" / "cabis" / "krei_series_보완_20260921.xlsx"
BETAS = [0.0, -0.5, -1.0, -1.5, -2.0, -3.0]
BETA0 = -2.0
CAP = 20.0
APRIL_SPRING = "--april-spring" in sys.argv     # 변형 매핑: 4월 대상월을 겨울이 아니라 봄 작형으로
SUFFIX = "_aprilspring" if APRIL_SPRING else ""


def season_of(month: int) -> str:
    if APRIL_SPRING:
        return "겨울" if month <= 3 else "봄" if month <= 6 else "여름" if month <= 10 else "가을"
    return "겨울" if month <= 4 else "봄" if month <= 6 else "여름" if month <= 10 else "가을"


def load_issues() -> pd.DataFrame:
    csv = OUT / "issues_nonsummer_2021_2026.csv"
    if KREI_XLSX.exists():
        x = pd.read_excel(KREI_XLSX)
        x["m"] = x["발행호"].astype(str).str.replace("월", "").astype(int)
        rows = []
        for _, r in x.iterrows():
            name = str(r["작형(연산)"])
            if "겨울" in name:
                crop = int(r["발행연도"]) if r.m >= 8 else int(r["발행연도"]) - 1
                sea, sub = "겨울", "겨울"
            elif "봄" in name:
                crop = int(r["발행연도"]) + (1 if r.m == 12 else 0)
                sea = "봄"
                sub = "전체" if ("전체" in name or name.startswith("봄배추")) else ("노지" if "노지" in name else "시설")
            elif "가을" in name:
                crop, sea, sub = int(r["발행연도"]), "가을", "가을"
            else:
                continue
            rows.append({"season": sea, "sub": sub, "crop_year": crop, "year": int(r["발행연도"]), "issue_month": int(r.m),
                         "prod_yoy": r["생산량 전년대비(%)"], "area_yoy": r["면적 전년대비(%)"], "label": name})
        d = pd.DataFrame(rows).sort_values(["season", "crop_year", "year", "issue_month"]).reset_index(drop=True)
        d.to_csv(csv, index=False, encoding="utf-8-sig")
        return d
    return pd.read_csv(csv, encoding="utf-8-sig")


def signal_at(iss: pd.DataFrame, origin: pd.Timestamp) -> dict:
    tm = origin + pd.offsets.MonthBegin(1)
    sea = season_of(tm.month)
    if sea == "여름":
        return {"season": sea}
    crop = tm.year - 1 if sea == "겨울" else tm.year
    avail_y, avail_m = (origin.year, origin.month + 1) if origin.month < 12 else (origin.year + 1, 1)
    s = iss[(iss.season == sea) & (iss.crop_year == crop) & ((iss.year < avail_y) | ((iss.year == avail_y) & (iss.issue_month <= avail_m)))]
    s = s[s.prod_yoy.notna() | s.area_yoy.notna()]
    if s.empty:
        return {"season": sea}
    if sea == "봄":   # 생산 전년비가 있는 행 중 전체 > 노지 > 시설 우선, 같은 호 안에서
        s = s.sort_values(["year", "issue_month"])
        last = s[(s.year == s.year.iloc[-1]) & (s.issue_month == s.issue_month.iloc[-1])]
        pri = {"전체": 0, "노지": 1, "시설": 2}
        last = last.assign(p=last["sub"].map(pri)).sort_values("p")
        withp = last[last.prod_yoy.notna()]
        r = withp.iloc[0] if len(withp) else last.iloc[0]
    else:
        r = s.iloc[-1]
    S = r.prod_yoy if pd.notna(r.prod_yoy) else r.area_yoy
    return {"season": sea, "issue": f"{int(r.year)}-{int(r.issue_month):02d}", "S": float(S), "basis": "prod" if pd.notna(r.prod_yoy) else "area", "label": r.label}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    iss = load_issues()
    g = pd.read_csv(ROOT / "outputs" / "gate_predictions.csv", parse_dates=["origin"], encoding="utf-8-sig")
    d = g[g.origin >= "2020-12-31"].copy()
    sig = [signal_at(iss, o) for o in d.origin]
    d["season"] = [x["season"] for x in sig]
    d["issue"] = [x.get("issue", np.nan) for x in sig]
    d["S_season"] = [x.get("S", np.nan) for x in sig]
    d["basis"] = [x.get("basis", "") for x in sig]
    # 여름은 메인 S1
    d["S_all"] = np.where(d.season == "여름", d.S1, d.S_season)
    d["S_cap"] = d.S_all.clip(-CAP, CAP)
    nonsummer = (d.season != "여름") & d.S_cap.notna()
    alarm = (d.season == "여름") & (d.gate_calm == 0) & d.S_cap.notna()
    anysig = d.S_cap.notna()
    cols = ["gate_hgb"]
    for b in BETAS:
        f = 1 + b * d.S_cap / 100
        for v, mk in (("V1", nonsummer), ("V2", nonsummer | alarm), ("V3", anysig)):
            c = f"{v}_b{abs(b):.1f}"
            d[c] = np.where(mk, d.gate_hgb * f, d.gate_hgb)
            cols.append(c)
    d.to_csv(OUT / f"adjust_predictions{SUFFIX}.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    y = d.origin.dt.year
    masks = {"all_2021_2025": y.between(2021, 2025), "all_2021_2026": y.between(2021, 2026)}
    for sea in ("겨울", "봄", "여름", "가을"):
        masks[f"{sea}_2021_2025"] = y.between(2021, 2025) & (d.season == sea)
    met = []
    for name, mk in masks.items():
        for c in cols:
            s = d[mk]
            e = s[c] - s.actual
            met.append({"subset": name, "model": c, "n": len(s), "MAE": e.abs().mean(), "MAPE%": (e.abs() / s.actual).mean() * 100, "bias": e.mean()})
    met = pd.DataFrame(met)
    met.to_csv(OUT / f"adjust_metrics{SUFFIX}.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 320)
    print("=== 비여름 대상월 신호 (규약 B) ===")
    s = d[d.season != "여름"]
    print(s.assign(tm=(s.origin + pd.offsets.MonthBegin(1)).dt.strftime("%Y-%m"))[["tm", "season", "issue", "label" if "label" in s else "issue", "S_season", "basis"]].to_string(index=False) if False else
          s.assign(tm=(s.origin + pd.offsets.MonthBegin(1)).dt.strftime("%Y-%m"))[["tm", "season", "issue", "S_season", "basis"]].to_string(index=False))
    print("\n=== MAPE (행: 변형_β) ===")
    piv = met.pivot(index="model", columns="subset", values="MAPE%").loc[cols][list(masks)].round(1)
    print(piv.to_string())
    # 사후 β (참고): 비여름 2021~2025에서 log(actual/pred) ~ β·S/100 최소제곱
    s = d[nonsummer & y.between(2021, 2025)]
    x = s.S_cap / 100; yy = np.log(s.actual / s.gate_hgb)
    print(f"\n[참고·사후] 비여름 2021~2025 최소제곱 β = {float((x * yy).sum() / (x * x).sum()):.2f} (n={len(s)})")
    for sea in ("겨울", "봄", "가을"):
        t = s[s.season == sea]; xx = t.S_cap / 100; yv = np.log(t.actual / t.gate_hgb)
        print(f"   {sea}: β = {float((xx * yv).sum() / (xx * xx).sum()):.2f} (n={len(t)})")
    # 20% 초과 달 전후
    big = d[(d.gate_hgb / d.actual - 1).abs() > 0.2].copy()
    big["tm"] = (big.origin + pd.offsets.MonthBegin(1)).dt.strftime("%Y-%m")
    big["before%"] = ((big.gate_hgb / big.actual - 1) * 100).round(0)
    big["after_V1_b2%"] = ((big["V1_b2.0"] / big.actual - 1) * 100).round(0)
    big["after_V2_b2%"] = ((big["V2_b2.0"] / big.actual - 1) * 100).round(0)
    print("\n=== 20% 초과 달: 조정 전후 (β=−2) ===")
    print(big[["tm", "season", "S_all", "before%", "after_V1_b2%", "after_V2_b2%"]].to_string(index=False))
    big["after_V2_b1%"] = ((big["V2_b1.0"] / big.actual - 1) * 100).round(0)
    print(big[["tm", "season", "S_all", "before%", "after_V2_b1%"]].to_string(index=False))
    big[["tm", "season", "S_all", "before%", "after_V1_b2%", "after_V2_b2%", "after_V2_b1%"]].to_csv(OUT / f"big_errors_before_after{SUFFIX}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
