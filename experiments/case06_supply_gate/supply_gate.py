"""Case 06 — 공급 상태 게이트: 대표 모형(an_sparse_all, 폭염 편차 선형) 위에 공급 쪽 상태 변수로 경보를 끄고 켠다.

배경: 폭염 피처는 2024(급등)·2025(급등 없음)를 못 가른다(둘 다 폭염). case05에서 두 공급 상태 변수가 두 해를 가름을 확인:
  S1  KREI 월보 여름배추 생산량 전망 전년비 — 원점까지 발행된 최신호 값 (호별 판독표, 2021~2026). 8월호 2024 −7.2% / 2025 +8.8%
  S2  원점 시점 정부비축 잔량 추정 (aT 로트 실적 기간 비례 배분, 2017~2026). 8/31 2024 762t / 2025 7,490t

규칙 (실행 전 고정)
  calm_K = S1 > 0          (생산이 전년보다 늘 것이라는 전망)
  calm_S = S2 >= 2000 t     (풀 수 있는 비축이 남아 있음)
  G_K    : 여름 원점(6~9월)에서 calm_K 이면 평시 모형, 아니면 대표 모형
  G_S    : calm_S 이면 평시 모형, 아니면 대표 모형
  G_both : calm_K and calm_S 이면 평시 모형, 아니면 대표 모형  (= 둘 중 하나라도 위험 신호면 경보 유지)
  G_soft : 예측 = 평시·대표의 로그 가중평균, w_대표 = 0.5·[S1<0] + 0.5·[S2<2000]
  평시 모형 = price_only_ridge (기준선). 참고로 wx_hgb_summer(전체 MAPE 최선) 버전도 함께.
  신호가 없는 원점(비여름, 2021 이전)에서는 대표 모형 그대로.
사후 설계: 임계(0, 2,000t)와 규칙 형태는 2024·2025를 본 뒤 정했다. 발행일은 호 월의 5일로 보수적으로 둔다(8월호 실제 게시 8/1~8/5).

Usage: python experiments/case06_supply_gate/supply_gate.py
Outputs: outputs/signals.csv(원점별 S1·S2), gate_predictions.csv, gate_metrics.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "case05_gov_stock_krei"))
import gov_stock_krei as c5  # noqa: E402  로트 파서·잔량 추정 재사용

OUT = CASE_DIR / "outputs"
MAIN_OUT = ROOT / "outputs"
KREI_SERIES = ROOT / "data" / "local" / "cabis" / "krei_series_보완_20260921.xlsx"   # 공개 저장소 미포함(data/local)
STOCK_T = 2000.0
HEAD, CALM, CALM2 = "an_sparse_all", "price_only_ridge", "wx_hgb_summer"


def krei_summer_signal() -> pd.DataFrame:
    """호별 여름배추(당해년산) 생산량 전년비. 발행일 = 호 월 5일."""
    d = pd.read_excel(KREI_SERIES)
    d2 = d[d["작형(연산)"].astype(str).str.contains("여름|고랭지")].copy()
    d2["issue_month"] = d2["발행호"].astype(str).str.replace("월", "").astype(int)
    d2["pub_date"] = pd.to_datetime(dict(year=d2["발행연도"], month=d2["issue_month"], day=5))
    # 당해년산만 (작형 문자열에 발행연도 포함)
    d2 = d2[[str(y) in str(a) for y, a in zip(d2["발행연도"], d2["작형(연산)"])]]
    out = d2[["발행연도", "issue_month", "pub_date", "재배면적(ha)", "생산량(톤)", "면적 전년대비(%)", "생산량 전년대비(%)"]].rename(
        columns={"발행연도": "year", "면적 전년대비(%)": "area_yoy", "생산량 전년대비(%)": "prod_yoy", "재배면적(ha)": "area_ha", "생산량(톤)": "prod_t"})
    return out.sort_values("pub_date").reset_index(drop=True)


def latest_signal(sig: pd.DataFrame, origin: pd.Timestamp) -> dict:
    """원점 연도의 호 중 발행일 ≤ 원점인 최신호. 생산 전년비가 없으면(4~5월호) 면적 전년비로 대체."""
    s = sig[(sig.year == origin.year) & (sig.pub_date <= origin)]
    if s.empty:
        return {"krei_issue": np.nan, "S1_prod_yoy": np.nan, "S1_area_yoy": np.nan, "S1": np.nan}
    r = s.iloc[-1]
    s1 = r.prod_yoy if pd.notna(r.prod_yoy) else r.area_yoy
    return {"krei_issue": f"{int(r.year)}-{int(r.issue_month):02d}", "S1_prod_yoy": r.prod_yoy, "S1_area_yoy": r.area_yoy, "S1": s1}


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.read_csv(MAIN_OUT / "anomaly_predictions.csv", parse_dates=["origin"])
    b = pd.read_csv(MAIN_OUT / "backtest_predictions.csv", parse_dates=["origin"])
    m = b[["origin", "actual", "summer", CALM, CALM2, "chronos2"]].merge(a[["origin", HEAD, "an_sparse_summer"]], on="origin")
    lots = c5.load_lots()
    sig = krei_summer_signal()
    rows = []
    for _, r in m.iterrows():
        o = r.origin
        st = c5.stock_at(lots, o) if o.year >= 2017 else {"stock_est": np.nan}
        k = latest_signal(sig, o) if r.summer == 1 else {"krei_issue": np.nan, "S1_prod_yoy": np.nan, "S1_area_yoy": np.nan, "S1": np.nan}
        rows.append({"origin": o, **k, "S2_stock": st["stock_est"]})
    s = pd.DataFrame(rows)
    d = m.merge(s, on="origin")
    calm_K = d.S1 > 0
    calm_S = d.S2_stock >= STOCK_T
    hasK, hasS = d.S1.notna(), d.S2_stock.notna()
    lg = lambda c: np.log(d[c])
    for calm_name, calm_col in (("", CALM), ("_hgb", CALM2)):
        d[f"G_K{calm_name}"] = np.where(hasK & calm_K, d[calm_col], d[HEAD])
        d[f"G_S{calm_name}"] = np.where(hasS & calm_S & (d.summer == 1), d[calm_col], d[HEAD])
        d[f"G_both{calm_name}"] = np.where(hasK & hasS & calm_K & calm_S, d[calm_col], d[HEAD])
        w = 0.5 * np.where(hasK, (d.S1 < 0).astype(float), 0.5) + 0.5 * np.where(hasS & (d.summer == 1), (d.S2_stock < STOCK_T).astype(float), 0.5)
        w = np.where(hasK | (hasS & (d.summer == 1)), w, 1.0)
        d[f"G_soft{calm_name}"] = np.exp(w * lg(HEAD) + (1 - w) * lg(calm_col))
    d["gate_both_calm"] = (hasK & hasS & calm_K & calm_S).astype(int)
    return d, s


def metrics(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    y = d.origin.dt.year
    masks = {"all_2021_2025": y.between(2021, 2025), "summer_2021_2025": y.between(2021, 2025) & (d.summer == 1),
             "summer_2021_2026": y.between(2021, 2026) & (d.summer == 1)}
    out = []
    for name, mk in masks.items():
        s = d[mk]
        for c in cols:
            e = s[c] - s.actual
            out.append({"subset": name, "model": c, "n": len(s), "MAE": e.abs().mean(), "MAPE%": (e.abs() / s.actual).mean() * 100, "bias": e.mean()})
    return pd.DataFrame(out)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    d, s = run()
    s.to_csv(OUT / "signals.csv", index=False, encoding="utf-8-sig")
    d.to_csv(OUT / "gate_predictions.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    cols = [CALM, "chronos2", CALM2, HEAD, "G_K", "G_S", "G_both", "G_soft", "G_both_hgb", "G_soft_hgb"]
    met = metrics(d, cols)
    met.to_csv(OUT / "gate_metrics.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 320)
    keys = pd.to_datetime([f"{y}-0{m}-{dd}" for y in range(2021, 2027) for m, dd in ((7, "31"), (8, "31"), (9, "30"))])
    sc = d[d.origin.isin(keys)].copy()
    for c in cols:
        sc[c + "%"] = (sc[c] / sc.actual - 1) * 100
    print("=== 신호 (여름 원점) ===")
    print(sc[["origin", "actual", "krei_issue", "S1_prod_yoy", "S1_area_yoy", "S2_stock", "gate_both_calm"]].round(1).to_string(index=False))
    print("\n=== 오차% (여름 원점 7·8·9월말, 2021~2026) ===")
    print(sc[["origin"] + [c + "%" for c in cols]].round(0).to_string(index=False))
    print("\n=== 지표 ===")
    print(met.round(1).to_string(index=False))
