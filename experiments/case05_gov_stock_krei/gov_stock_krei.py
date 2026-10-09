"""Case 05 — 정부비축 로트별 실적(2017~2026)으로 8월 말 비축 잔량을 추정하고, KREI 작형별 전망치(2021~2026)와 함께
'2024·2025를 가르는 변수' 후보로 검토한다.

입력 (data/ext, 사용자 제공·인용 가능 확인 2026-10-09)
  정부비축_배추무_수매방출폐기_실적_작기별_20260828.xlsx  시트 '배추'  로트별 수매·판매(상장/직배/공매)·폐기·기타, 기간 문자열
  KREI_배추무_전망치_모음_2021_2026.xlsx                시트 Sheet1  작형별 면적·생산량·가격 전망(평년비·전년비), 실제 가격
  data/raw/clean_daily.csv                              9월·8월 실제 가격
  outputs/anomaly_predictions.csv, backtest_predictions.csv  대표 모형·가격만 모형의 8/31 원점 오차

A. 8월 말 비축 잔량 추정 (로트별, 기간 비례 배분)
   잔량(Y-08-31) = Σ_{Y-1 9월~Y 8월 수매 로트} [ 수매량×f(수매기간) − 상장×f − 직배×f − 공매×f − 폐기×f ],  f = 기간 중 8/31 이전 일수 비율
   기타(감모 등)는 종료 시점 반영으로 보고 8/31 잔량에서 빼지 않는다. 기간이 없는 판매·폐기는 수매기간과 같다고 본다.
   검산: at_gov_daily(2021~2025)의 누적(수매−방출)과 비교.
B. KREI 여름작형 면적·생산량 전망 전년비 vs 9월 실제 가격 전년비 및 대표 모형 오차 (2021~2026)

Outputs: outputs/gov_lots_parsed.csv, gov_stock_aug31.csv, krei_summer.csv, summary.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
EXT = ROOT / "data" / "ext"
RAW = ROOT / "data" / "raw"
MAIN_OUT = ROOT / "outputs"
OUT = CASE_DIR / "outputs"
FILE_DATE = pd.Timestamp("2026-08-28")      # 실적 기준일 (열린 기간의 종료일로 사용)

# ---------------------------------------------------------------- 기간 문자열 파싱
TOK = re.compile(r"(?:(\d{2,4})\.)?(\d{1,2})\.(\d{1,2})")


def _r(v):
    return round(v) if pd.notna(v) else np.nan


def parse_period(s, ref_year: int | None = None) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str]:
    """'17.5.29.~'17.8.29 / 24.4.23~8.5 / `23.12.20~`24.3.21 / 23.3.29 / 23.5.3~5.22 / 26.7.23~  → (start, end, note)"""
    if s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() in ("", "0", "-", "nan"):
        return None, None, "none"
    txt = str(s).replace("'", "").replace("`", "").replace("’", "").replace("～", "~").strip()
    parts = re.split(r"[~～]", txt)
    dates, year = [], ref_year
    for i, part in enumerate(parts):
        for seg in part.split("/"):
            m = TOK.search(seg.strip(" ."))
            if not m:
                continue
            y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
            if y:
                year = int(y) + (2000 if int(y) < 100 else 0)
            elif year is None:
                return None, None, f"noyear:{txt}"
            elif dates and mo < dates[-1].month:
                year += 1                                   # 연도 생략 + 월이 작아지면 해 넘김
            try:
                dates.append(pd.Timestamp(year=year, month=mo, day=d))
            except ValueError:
                return None, None, f"bad:{txt}"
    if not dates:
        return None, None, f"unparsed:{txt}"
    start = min(dates)
    end = max(dates) if (len(dates) > 1 or not txt.rstrip().endswith("~")) else FILE_DATE
    if txt.rstrip().endswith("~"):
        end = FILE_DATE
    return start, end, "ok" if len(dates) > 1 else "single"


def frac_before(start, end, cutoff) -> float:
    """기간 [start, end] 중 cutoff 이전 일수 비율 (균등 배분)."""
    if start is None:
        return np.nan
    if end is None or end < start:
        end = start
    total = (end - start).days + 1
    done = (min(end, cutoff) - start).days + 1
    return float(np.clip(done / total, 0.0, 1.0))


# ---------------------------------------------------------------- A. 로트 파싱
def load_lots() -> pd.DataFrame:
    d = pd.read_excel(EXT / "정부비축_배추무_수매방출폐기_실적_작기별_20260828.xlsx", sheet_name="배추", header=None)
    rows = []
    for i in range(6, len(d)):
        r = d.iloc[i]
        if pd.isna(r[1]) and pd.isna(r[2]):
            continue
        year = int(r[1]) if pd.notna(r[1]) else rows[-1]["year"]
        season = str(r[2]).replace("*", "").strip() if pd.notna(r[2]) else ""
        num = lambda v: float(v) if pd.notna(v) and str(v).strip() not in ("", "-", "0", "NaN") and isinstance(v, (int, float, np.integer, np.floating)) else (0.0 if pd.isna(v) or str(v).strip() in ("-", "0", "") else float(str(v).replace(",", "")) if re.fullmatch(r"[\d,.]+", str(v).strip()) else 0.0)
        ps, pe, pn = parse_period(r[3], None)
        if ps is None and rows and rows[-1]["year"] == year and rows[-1]["season"] == season and rows[-1]["p_start"] is not None:
            ps, pe, pn = rows[-1]["p_start"], rows[-1]["p_end"], "inherited"      # 기간 없는 추가 로트(예: 2023 봄 2,000t)는 직전 같은 작기 로트 기간을 따름
        ref = ps.year if ps is not None else year
        ls, le, ln = parse_period(r[6], ref)
        ds, de, dn = parse_period(r[8], ref)
        as_, ae, an = parse_period(r[11], ref)
        ws, we, wn = parse_period(r[13], ref)
        rows.append({"year": year, "season": season, "lot_row": i,
                     "purchase_period": r[3], "purchase_t": num(r[4]), "p_start": ps, "p_end": pe, "p_note": pn,
                     "listed_t": num(r[5]), "listed_period": r[6], "l_start": ls, "l_end": le, "l_note": ln,
                     "direct_t": num(r[7]), "direct_period": r[8], "d_start": ds, "d_end": de, "d_note": dn,
                     "auction_t": num(r[9]), "auction_period": r[11], "a_start": as_, "a_end": ae, "a_note": an,
                     "disposal_t": num(r[12]), "disposal_period": r[13], "w_start": ws, "w_end": we, "w_note": wn,
                     "disposal_method": r[14] if pd.notna(r[14]) else "", "other_t": num(r[15])})
    return pd.DataFrame(rows)


def stock_at(lots: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """cutoff 시점 잔량: 직전 12개월 내 수매 시작 로트만."""
    lo = cutoff - pd.DateOffset(months=12)
    sel = lots[(lots.p_start.notna()) & (lots.p_start <= cutoff) & (lots.p_start > lo)]
    tot = {"purchased": 0.0, "listed": 0.0, "direct": 0.0, "auction": 0.0, "disposed": 0.0, "n_lots": len(sel)}
    for _, r in sel.iterrows():
        fp = frac_before(r.p_start, r.p_end, cutoff)
        tot["purchased"] += r.purchase_t * fp
        for key, t, s, e in (("listed", r.listed_t, r.l_start, r.l_end), ("direct", r.direct_t, r.d_start, r.d_end),
                             ("auction", r.auction_t, r.a_start, r.a_end), ("disposed", r.disposal_t, r.w_start, r.w_end)):
            if t > 0:
                f = frac_before(s, e, cutoff) if s is not None else fp
                tot[key] += t * f
    tot["stock_est"] = tot["purchased"] - tot["listed"] - tot["direct"] - tot["auction"] - tot["disposed"]
    return tot


# ---------------------------------------------------------------- B. KREI 전망
def load_krei() -> pd.DataFrame:
    d = pd.read_excel(EXT / "KREI_배추무_전망치_모음_2021_2026.xlsx", header=None)
    rows, year = [], None
    for i in range(7, 31):
        r = d.iloc[i]
        if pd.notna(r[0]):
            year = int(r[0])
        if pd.isna(r[1]):
            continue
        rows.append({"year": year, "season": str(r[1]).strip("()"),
                     "area_ha": r[2], "area_vs_normal%": r[3], "area_yoy%": r[4],
                     "prod_t": r[5], "prod_vs_normal%": r[6], "prod_yoy%": r[7],
                     "price_fcst": r[8], "price_fcst_vs_normal%": r[9], "price_fcst_yoy%": r[10],
                     "price_actual": r[11], "price_actual_vs_normal%": r[12], "price_actual_yoy%": r[13]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    lots = load_lots()
    lots.to_csv(OUT / "gov_lots_parsed.csv", index=False, encoding="utf-8-sig")
    bad = lots[(lots.p_note != "ok") & (lots.p_note != "single")]
    print(f"로트 {len(lots)}개 파싱, 수매기간 미해석 {len(bad)}개", flush=True)
    if len(bad):
        print(bad[["year", "season", "purchase_period", "p_note"]].to_string(index=False))

    price = pd.read_csv(RAW / "clean_daily.csv", parse_dates=["date"]); price = price[price.price > 0]
    sep = price[price.date.dt.month == 9].groupby(price.date.dt.year).price.mean()
    aug = price[price.date.dt.month == 8].groupby(price.date.dt.year).price.mean()
    an = pd.read_csv(MAIN_OUT / "anomaly_predictions.csv", parse_dates=["origin"]).set_index("origin")
    bt = pd.read_csv(MAIN_OUT / "backtest_predictions.csv", parse_dates=["origin"]).set_index("origin")
    try:
        ag = pd.read_csv(RAW / "at_gov_daily.csv", parse_dates=["date"]).set_index("date")
        ag_cum = (ag.gov_purchase - ag.gov_release).cumsum()
    except FileNotFoundError:
        ag_cum = None

    rows = []
    for y in range(2017, 2027):
        cut = pd.Timestamp(f"{y}-08-31")
        s = stock_at(lots, cut)
        o = cut
        r = {"year": y, **{k: _r(v) for k, v in s.items()}, "aug_price": _r(aug.get(y, np.nan)), "sep_price": _r(sep.get(y, np.nan)),
             "sep_vs_aug%": round((sep.get(y, np.nan) / aug.get(y, np.nan) - 1) * 100, 1),
             "sep_yoy%": round((sep.get(y, np.nan) / sep.get(y - 1, np.nan) - 1) * 100, 1)}
        r["at_gov_cum_aug31"] = _r(ag_cum.loc[:o].iloc[-1]) if ag_cum is not None and o in ag_cum.index else np.nan
        if o in an.index:
            r["err_an_sparse_all%"] = round((an.loc[o, "an_sparse_all"] / an.loc[o, "actual"] - 1) * 100, 1)
        if o in bt.index:
            r["err_price_only%"] = round((bt.loc[o, "price_only_ridge"] / bt.loc[o, "actual"] - 1) * 100, 1)
        rows.append(r)
    stock = pd.DataFrame(rows)
    stock.to_csv(OUT / "gov_stock_aug31.csv", index=False, encoding="utf-8-sig")

    krei = load_krei()
    krei.to_csv(OUT / "krei_forecasts_parsed.csv", index=False, encoding="utf-8-sig")
    ks = krei[krei.season == "여름"].set_index("year")
    summ = stock.set_index("year").join(ks[["area_yoy%", "prod_yoy%", "price_fcst", "price_fcst_yoy%", "price_actual", "price_actual_yoy%"]]
                                        .rename(columns=lambda c: "krei_summer_" + c))
    summ.to_csv(OUT / "summary.csv", encoding="utf-8-sig")

    pd.set_option("display.width", 300)
    print("\n=== A. 8월 말 정부비축 잔량 추정(톤) vs 9월 가격 ===")
    print(stock[["year", "n_lots", "purchased", "listed", "direct", "disposed", "stock_est", "at_gov_cum_aug31",
                 "aug_price", "sep_price", "sep_vs_aug%", "err_an_sparse_all%", "err_price_only%"]].to_string(index=False))
    print("\n=== B. KREI 여름작형 전망(전년비) vs 실제 ===")
    print(ks[["area_ha", "area_yoy%", "prod_t", "prod_yoy%", "price_fcst", "price_fcst_yoy%", "price_actual", "price_actual_yoy%"]].round(1).to_string())
    print("\n=== 2024 vs 2025 비교 ===")
    print(summ.loc[[2020, 2021, 2022, 2023, 2024, 2025, 2026], ["stock_est", "sep_vs_aug%", "krei_summer_area_yoy%", "krei_summer_prod_yoy%", "err_an_sparse_all%"]].to_string())
