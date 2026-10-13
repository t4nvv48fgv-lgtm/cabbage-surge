"""실시간 전망 — 실제값이 아직 없는 원점에서 최종 모형(gate_model.final)과 같은 규칙으로 앞으로 28일 평균가격을 예측한다.

백테스트(backtest.py·anomaly_model.py·gate_model.py)는 실제값이 있는 원점만 돌린다. 이 스크립트는 같은 부품·같은 학습 규칙
(확장 창, 학습 줄의 대상 기간 종료 ≤ 원점, 사흘에 한 줄)으로 한 원점만 계산한다.

  부품: 대표 an_sparse_all / an_sparse_ship, 평시 wx_hgb_summer, 가격만 ridge, 겨울 작형 가격만 ridge
  규칙: 0) 대표 모형 계절 구분  1) 여름 공급 게이트(생산 전망 + 출하량 전망 방향)  2) 겨울 분기  3) 잔차 β  4) 9월 고온 층(월말 9월 원점만; γ 는 outputs/gate_predictions.csv 의 과거 원점으로 추정)

Usage: python src/forecast_live.py 2026-09-30 [M|W]     (M = 월말 원점 규칙[9월 고온 층 적용 가능], W = 주간 원점 규칙)
Output: 콘솔 + outputs/forecast_live.csv (원점별 한 줄, 같은 원점은 덮어씀)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import anomaly_model as am
import backtest as bt
import gate_model as gm

H = 28


def components(o: pd.Timestamp) -> dict:
    df = pd.read_csv(bt.PROC / "daily_features.csv", parse_dates=["date"]).set_index("date")
    if o not in df.index:
        raise SystemExit(f"{o.date()} 가 daily_features 에 없다(자료 끝 {df.index.max().date()})")
    df[bt.TARGET] = df["y_lp_fwd28"] - df[bt.BASE]
    df = am.add_ship_weather(am.add_normals(df))
    te = df.loc[[o]]
    rec = {"origin": o, "summer": int(te.summer.iloc[0]), "p7": float(np.exp(te.lp7.iloc[0])), "p28": float(np.exp(te.lp28.iloc[0])),
           "normal_fwd": float(np.exp(te.norm_fwd.iloc[0])), "w_summer_ship": float(te.w_summer_ship.iloc[0]), "H_hot30": float(te.L_hot30_sum30.iloc[0])}
    # 대표 모형 둘
    pool = df[(df.is_trading == 1) & df.y_anom.notna()].dropna(subset=am.CORE).iloc[::3]
    tr = pool[pool.index + pd.Timedelta(days=H) <= o]
    for name, feats in (("an_sparse_all", am.SPARSE), ("an_sparse_ship", am.SPARSE_SHIP)):
        m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(tr[feats], tr.y_anom)
        rec[name] = float(np.exp(te.norm_fwd.iloc[0] + m.predict(te[feats])[0]))
    rec["n_train_anomaly"] = len(tr)
    # 가격만 ridge, 평시 HGB
    bpool = df[(df.is_trading == 1) & df[bt.TARGET].notna()].dropna(subset=bt.PRICE_FEATS + bt.SUPPLY_FEATS + bt.WX_FEATS).iloc[::3]
    btr = bpool[bpool.index + pd.Timedelta(days=H) <= o]
    for name in ("price_only_ridge", "wx_hgb_summer"):
        rec[name] = float(np.exp(te[bt.BASE].iloc[0] + bt.fit_predict(name, btr, te)))
    # 겨울 모형
    target = o + pd.Timedelta(days=15)
    rec["tmonth"] = target.month
    rec["winter_ridge"] = np.nan
    if target.month in gm.WINTER_MONTHS:
        wp = df[(df.is_trading == 1) & df[bt.TARGET].notna()].dropna(subset=bt.PRICE_FEATS + bt.SUPPLY_FEATS + bt.WX_FEATS + am.CORE + ["y_anom"]).iloc[::3]
        wp = wp[(wp.index + pd.Timedelta(days=15)).month.isin(gm.WINTER_MONTHS)]
        wtr = wp[wp.index + pd.Timedelta(days=H) <= o]
        if len(wtr) >= gm.MIN_TRAIN_WINTER:
            m = make_pipeline(StandardScaler(), Ridge(alpha=3.0)).fit(wtr[bt.PRICE_FEATS], wtr[bt.TARGET])
            rec["winter_ridge"] = float(np.exp(float(te[bt.BASE].iloc[0]) + m.predict(te[bt.PRICE_FEATS])[0]))
    return rec


def assemble(rec: dict, monthly: bool) -> dict:
    o = rec["origin"]
    issues, fall = gm.load_issues(), gm.load_fall_issues()
    rec["krei_issue"], rec["S1"] = gm.latest_signal(issues, o) if rec["summer"] == 1 else (np.nan, np.nan)
    rec["ship_q"] = gm.load_shipment().get(rec["krei_issue"], np.nan) if rec["summer"] == 1 else np.nan
    calm = pd.notna(rec["S1"]) and (rec["S1"] > 0 or rec["ship_q"] > 0)      # 생산 전망이 줄어도 출하량 전망이 증가면 평시 (case30)
    rec["state"] = "평시" if calm else ("경보" if rec["summer"] == 1 and pd.notna(rec["S1"]) else "해당 없음")
    rec["head"] = rec["an_sparse_all"] if rec["summer"] == 1 else rec["an_sparse_ship"]
    rec["gate_hgb"] = rec["wx_hgb_summer"] if calm else rec["head"]
    ty = (o + pd.Timedelta(days=15)).year
    k, s, beta = np.nan, np.nan, 0.0
    if rec["state"] == "경보":
        k, s = gm.adjust_signal(issues, o, o.year)
        beta = gm.BETA_SUMMER_ALERT if (pd.notna(s) and s <= 0) else 0.0
    elif rec["tmonth"] in gm.FALL_MONTHS:
        k, s = gm.adjust_signal(fall, o, ty, "crop_year")
        beta = gm.BETA_FALL
    f = 1.0 if (pd.isna(s) or beta == 0.0) else 1.0 + beta * float(np.clip(s, -gm.S_CAP, gm.S_CAP)) / 100.0
    rec["adj_issue"], rec["S_adj"], rec["adj_pct"] = k, s, (f - 1.0) * 100
    rec["final_base"] = rec["winter_ridge"] if pd.notna(rec["winter_ridge"]) else rec["gate_hgb"] * f
    # 9월 고온 층: 과거 원점(실제값 있음) + 이 원점을 붙여 메인 함수로 계산
    past = pd.read_csv(gm.OUT / "gate_predictions.csv", parse_dates=["origin"], encoding="utf-8-sig")[["origin", "actual", "tmonth", "final_base"]]
    new = pd.DataFrame([{"origin": o, "actual": np.nan, "tmonth": rec["tmonth"], "final_base": rec["final_base"]}])
    d = gm.heat_hold(pd.concat([past[past.origin < o], new], ignore_index=True), fall, monthly)
    last = d.iloc[-1]
    rec.update({"heat_F": last.heat_F, "heat_gamma_pct": last.heat_gamma_pct, "heat_pct": last.heat_pct, "final": float(last[gm.FINAL])})
    return rec


if __name__ == "__main__":
    o = pd.Timestamp(sys.argv[1])
    monthly = (sys.argv[2] if len(sys.argv) > 2 else "M") == "M"
    rec = assemble(components(o), monthly)
    out = gm.OUT / "forecast_live.csv"
    row = pd.DataFrame([rec])
    row["rule"] = "M" if monthly else "W"
    if out.exists():
        old = pd.read_csv(out, parse_dates=["origin"], encoding="utf-8-sig")
        row = pd.concat([old[~((old.origin == o) & (old.rule == row.rule.iloc[0]))], row], ignore_index=True).sort_values(["origin", "rule"])
    row.to_csv(out, index=False, float_format="%.2f", encoding="utf-8-sig")
    print(f"원점 {o.date()} ({'월말' if monthly else '주간'} 규칙) → 대상 {(o + pd.Timedelta(days=1)).date()} ~ {(o + pd.Timedelta(days=H)).date()}")
    for k in ("p7", "p28", "normal_fwd", "price_only_ridge", "wx_hgb_summer", "an_sparse_all", "an_sparse_ship", "winter_ridge"):
        print(f"  {k:18s} {rec[k]:10.0f}" if pd.notna(rec[k]) else f"  {k:18s}          -")
    print(f"  게이트: 월보 {rec['krei_issue']} 여름 생산 전망 {rec['S1']}, 출하량 전망 방향 {rec['ship_q']} → {rec['state']}, 대표 = {'an_sparse_all' if rec['summer'] == 1 else 'an_sparse_ship'} (고랭지 기상 켜짐 정도 {rec['w_summer_ship']:.2f})")
    print(f"  수준 조정: {rec['adj_issue']} S {rec['S_adj']} → {rec['adj_pct']:+.1f}%  → final_base {rec['final_base']:.0f}")
    print(f"  9월 고온: 30°C 이상 일수 {rec['H_hot30']:.1f}, 가을 전망 {rec['heat_F']}, γ {rec['heat_gamma_pct']}%/일 → {rec['heat_pct']:+.1f}%")
    print(f"  ▶ 최종 예측 {rec['final']:.0f} 원/10kg")
