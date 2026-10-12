"""Case 12 보충(2026-10-12) — 세미나 계열(가락 전 등급 평균, 원/kg)에서 **현재 최종 모형**(게이트 + 겨울 분기 + 잔차 β + 9월 고온 층)을 다시 평가.

run_case.py 가 만든 이 케이스의 backtest_predictions.csv·anomaly_predictions.csv·daily_features.csv 를 그대로 두고, 메인 gate_model 의 현재 규칙만 다시 적용한다
(backtest 재실행 없음). 잔차 β(−0.566·−0.216)는 상품 계열에서 추정한 값을 그대로 쓴다(모형 사양 그대로 이식).
기존 산출물(gate_predictions.csv·case12_metrics.csv)은 덮어쓰지 않고 *_current 로 따로 쓴다.

Usage: python experiments/case12_avg_price_series/rerun_current_gate.py
Outputs: outputs/gate_predictions_current.csv, outputs/case12_metrics_current.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
OUT = CASE_DIR / "outputs"
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import anomaly_model as am  # noqa: E402
import backtest as bt  # noqa: E402
import gate_model as gm  # noqa: E402

SEMINAR = {"MAE": 127.25, "WMAPE%": 12.27}          # 세미나 미세조정 Chronos-2, 1개월 선행, 2024~2025 (원/kg)
SEMINAR_OTHERS = {"세미나 제로샷": 139.4, "세미나 SARIMAX": 135.8, "관측월보(세미나 표기)": 231.76}
SEMINAR_PIXEL = {  # 그림 3-7 픽셀 읽기(인계 노트 ②-6·②-10), 대상월 → 세미나 미세조정 오차 %
    "2024-06": 5, "2024-07": 10, "2024-08": -10, "2024-09": -12, "2024-10": -28, "2024-11": 2, "2024-12": -16,
    "2025-06": 6, "2025-07": 2, "2025-08": 9, "2025-09": 4, "2025-10": 15, "2025-11": -16}


def metrics(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    y = d.origin.dt.year
    masks = {"all_2021_2025": y.between(2021, 2025), "seminar_2024_2025": (d.origin >= "2023-12-31") & (d.origin <= "2025-11-30")}
    out = []
    for name, mk in masks.items():
        s = d[mk]
        for c in cols:
            e = s[c] - s.actual
            out.append({"subset": name, "model": c, "n": len(s), "MAE": e.abs().mean(), "MAPE%": (e.abs() / s.actual).mean() * 100,
                        "WMAPE%": e.abs().sum() / s.actual.sum() * 100, "bias": e.mean()})
    return pd.DataFrame(out)


if __name__ == "__main__":
    bt.PROC, bt.OUT = OUT, OUT
    am.PROC, am.OUT = OUT, OUT
    gm.OUT = OUT
    d = gm.run("M")
    b = pd.read_csv(OUT / "backtest_predictions.csv", parse_dates=["origin"])
    d = d.merge(b[["origin", "mstl_ets", "wx_ridge", "wx_hgb"]], on="origin")
    d.to_csv(OUT / "gate_predictions_current.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    cols = ["price_only_ridge", "chronos2", "mstl_ets", "wx_ridge", "wx_hgb_summer", "an_sparse_all", "gate_hgb", "final_base", "final"]
    met = metrics(d, cols)
    met.to_csv(OUT / "case12_metrics_current.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 300)
    for sub in ("seminar_2024_2025", "all_2021_2025"):
        print(f"\n[{sub}]  (원/kg)")
        print(met[met.subset == sub].set_index("model")[["n", "MAE", "MAPE%", "WMAPE%", "bias"]].loc[cols].round(1).to_string())
    print(f"\n세미나 미세조정 Chronos-2(1개월 선행, 2024~2025): MAE {SEMINAR['MAE']}, WMAPE {SEMINAR['WMAPE%']}% | 참고 {SEMINAR_OTHERS}")
    d["tm"] = (d.origin + pd.Timedelta(days=15)).dt.strftime("%Y-%m")
    k = d[d.tm.isin(SEMINAR_PIXEL)].copy()
    for c in ("gate_hgb", "final", "chronos2"):
        k[c + "%"] = ((k[c] / k.actual - 1) * 100).round(0)
    k["세미나(그림 읽기)%"] = k.tm.map(SEMINAR_PIXEL)
    print("\n대상월별 오차 % (전 등급 평균 계열)")
    print(k[["tm", "actual", "세미나(그림 읽기)%", "gate_hgb%", "final%", "chronos2%"]].round(0).to_string(index=False))
    print(f"\n위 {len(k)}개 달 평균 절대오차: 세미나 {k['세미나(그림 읽기)%'].abs().mean():.1f}% | 게이트만 {k['gate_hgb%'].abs().mean():.1f}% | 현재 최종 {k['final%'].abs().mean():.1f}% | 제로샷 {k['chronos2%'].abs().mean():.1f}%")
