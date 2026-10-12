"""Case 15 — 4작형 다중모형: 대상월 작형(겨울 1~4월·봄 5~6월·여름 7~10월·가을 11~12월)마다 다른 모형을 쓴다.

사용자 제안(2026-10-12): "4계절을 두고 다중모형으로 쓰면 안 되나?" 작형별 집계에서 겨울·가을은 HGB, 봄은 가격만 ridge, 여름은 스위치가 최선이었음.
두 가지를 시험한다.
  A. 작형별 **선택**: 메인 산출물의 기존 모형 중 작형마다 하나를 고른다.
  B. 작형별 **학습**: 그 작형 대상월의 학습 행만으로 모형(가격만 ridge·기상 ridge·기상 HGB·계절편차 희소 ridge)을 새로 적합해 후보에 더한다.
선택은 2016~2020 월말 원점(설계 창)의 작형별 MAPE 최소로 하고, 2021~2025(검증 창)·2021~2026으로 평가한다. 여름 후보에는 메인 최종 모형(gate_hgb)을 포함한다.
학습 규칙·하이퍼파라미터는 메인(backtest.py·anomaly_model.py)과 동일, 학습 풀만 작형으로 제한(여름특화 HGB가 여름에 하던 것을 네 작형으로 일반화).

Usage: python experiments/case15_season_models/season_models.py
Outputs: outputs/season_predictions.csv, season_metrics.csv, season_selection.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
OUT = CASE_DIR / "outputs"
sys.path.insert(0, str(ROOT / "src"))
import anomaly_model as am  # noqa: E402
import backtest as bt  # noqa: E402

warnings.filterwarnings("ignore")
SEASON = {1: "겨울", 2: "겨울", 3: "겨울", 4: "겨울", 5: "봄", 6: "봄", 7: "여름", 8: "여름", 9: "여름", 10: "여름", 11: "가을", 12: "가을"}
ORDER = ["겨울", "봄", "여름", "가을"]
MAIN_CANDS = ["price_only_ridge", "wx_ridge", "wx_hgb", "wx_hgb_summer", "an_sparse_all", "chronos2", "mstl_ets", "gate_hgb"]


def season_of_target(dates: pd.DatetimeIndex) -> pd.Series:
    return pd.Series([(SEASON[(d + pd.Timedelta(days=15)).month]) for d in dates], index=dates)


def season_trained() -> pd.DataFrame:
    """작형별 학습 행만으로 적합한 4개 모형의 월말 예측(2016~)."""
    df = pd.read_csv(bt.PROC / "daily_features.csv", parse_dates=["date"]).set_index("date")
    df[bt.TARGET] = df["y_lp_fwd28"] - df[bt.BASE]
    df = am.add_normals(df)
    feats_all = bt.PRICE_FEATS + bt.SUPPLY_FEATS + bt.WX_FEATS
    pool = df[(df.is_trading == 1) & df[bt.TARGET].notna()].dropna(subset=feats_all + am.CORE + ["y_anom"]).iloc[::3].copy()
    pool["season"] = [SEASON[(d + pd.Timedelta(days=15)).month] for d in pool.index]
    origins = bt.month_ends(df.index)
    origins = origins[(origins >= bt.FIRST_ORIGIN) & (origins <= df.index.max())]
    rows = []
    for o in origins:
        if o not in df.index:
            continue
        te = df.loc[[o]]
        if te[bt.TARGET].isna().iloc[0] or te[feats_all].isna().any(axis=1).iloc[0] or te.y_anom.isna().iloc[0]:
            continue
        sea = SEASON[(o + pd.Timedelta(days=15)).month]
        tr = pool[(pool.index + pd.Timedelta(days=28) <= o) & (pool.season == sea)]
        if len(tr) < 120:
            continue
        base = float(te[bt.BASE].iloc[0])
        rec = {"origin": o, "season": sea, "n_train": len(tr)}
        # 가격만 ridge (작형 학습)
        m = make_pipeline(StandardScaler(), Ridge(alpha=3.0)).fit(tr[bt.PRICE_FEATS], tr[bt.TARGET])
        rec["s_price_ridge"] = float(np.exp(base + m.predict(te[bt.PRICE_FEATS])[0]))
        # 기상 ridge (작형 학습)
        f = bt.PRICE_FEATS + bt.SUPPLY_FEATS + bt.WX_FEATS
        m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(tr[f], tr[bt.TARGET])
        rec["s_wx_ridge"] = float(np.exp(base + m.predict(te[f])[0]))
        # 기상 HGB (작형 학습)
        f2 = f + ["summer", "month"]
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=1.0, random_state=0).fit(tr[f2], tr[bt.TARGET])
        rec["s_wx_hgb"] = float(np.exp(base + m.predict(te[f2])[0]))
        # 계절편차 희소 ridge (작형 학습, alpha 10)
        m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(tr[am.SPARSE], tr["y_anom"])
        rec["s_an_sparse"] = float(np.exp(float(te.norm_fwd.iloc[0]) + m.predict(te[am.SPARSE])[0]))
        rows.append(rec)
        if o.month == 8:
            print(f"  {o.date()} {sea} n={len(tr)} " + " ".join(f"{k}={rec[k]:7.0f}" for k in ("s_price_ridge", "s_wx_ridge", "s_wx_hgb", "s_an_sparse")), flush=True)
    return pd.DataFrame(rows)


def mape(s: pd.DataFrame, c: str) -> float:
    return float(((s[c] - s.actual).abs() / s.actual).mean() * 100)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    g = pd.read_csv(ROOT / "outputs" / "gate_predictions.csv", parse_dates=["origin"], encoding="utf-8-sig")
    b = pd.read_csv(ROOT / "outputs" / "backtest_predictions.csv", parse_dates=["origin"])
    d = g.merge(b[["origin", "wx_ridge", "wx_hgb", "mstl_ets"]], on="origin")
    st = season_trained()
    d = d.merge(st, on="origin", how="left")
    d["season"] = [SEASON[(o + pd.Timedelta(days=15)).month] for o in d.origin]
    y = d.origin.dt.year
    design, valid = y.between(2016, 2020), y.between(2021, 2025)
    S_CANDS = ["s_price_ridge", "s_wx_ridge", "s_wx_hgb", "s_an_sparse"]
    # 작형별 선택 (설계 창)
    sel_rows, pickA, pickAB = [], {}, {}
    for sea in ORDER:
        mk = design & (d.season == sea)
        scores = {c: mape(d[mk], c) for c in MAIN_CANDS + S_CANDS if d.loc[mk, c].notna().all()}
        bestA = min((c for c in scores if c in MAIN_CANDS), key=scores.get)
        bestAB = min(scores, key=scores.get)
        pickA[sea], pickAB[sea] = bestA, bestAB
        vmk = valid & (d.season == sea)
        for c, v in sorted(scores.items(), key=lambda kv: kv[1]):
            sel_rows.append({"season": sea, "model": c, "design_MAPE_2016_2020": round(v, 2), "valid_MAPE_2021_2025": round(mape(d[vmk], c), 2),
                             "pickA": int(c == bestA), "pickAB": int(c == bestAB)})
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(OUT / "season_selection.csv", index=False, encoding="utf-8-sig")
    d["multiA"] = [r[pickA[r.season]] for _, r in d.iterrows()]
    d["multiAB"] = [r[pickAB[r.season]] for _, r in d.iterrows()]
    # 사후 참고: 검증 창에서 고른 것(상한)
    pickV = {sea: min(MAIN_CANDS + S_CANDS, key=lambda c: mape(d[valid & (d.season == sea)], c)) for sea in ORDER}
    d["multi_posthoc"] = [r[pickV[r.season]] for _, r in d.iterrows()]
    d.to_csv(OUT / "season_predictions.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    cols = ["price_only_ridge", "wx_hgb_summer", "an_sparse_all", "gate_hgb"] + S_CANDS + ["multiA", "multiAB", "multi_posthoc"]
    masks = {"all_2016_2020(design)": design, "all_2021_2025(valid)": valid, "all_2021_2026": y.between(2021, 2026)}
    for sea in ORDER:
        masks[f"{sea}_2021_2025"] = valid & (d.season == sea)
    met = []
    for name, mk in masks.items():
        for c in cols:
            s = d[mk]
            e = s[c] - s.actual
            met.append({"subset": name, "model": c, "n": len(s), "MAE": e.abs().mean(), "MAPE%": (e.abs() / s.actual).mean() * 100, "bias": e.mean()})
    met = pd.DataFrame(met)
    met.to_csv(OUT / "season_metrics.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 320)
    print("\n=== 작형별 선택 (설계 2016~2020 MAPE 기준) ===")
    print(sel.to_string(index=False))
    print("\n선택 A(기존 모형만):", pickA, "\n선택 AB(작형 학습 포함):", pickAB, "\n사후(검증 창에서 고름, 상한):", pickV)
    print("\n=== MAPE ===")
    print(met.pivot(index="model", columns="subset", values="MAPE%").loc[cols][list(masks)].round(1).to_string())
    k = d[(d.origin.dt.month == 8) & (y >= 2021)].copy()
    for c in ["gate_hgb", "s_an_sparse", "s_wx_hgb", "multiA", "multiAB"]:
        k[c + "%"] = ((k[c] / k.actual - 1) * 100).round(0)
    print("\n=== 8월 말 원점 ===")
    print(k[["origin", "actual"] + [c + "%" for c in ["gate_hgb", "s_an_sparse", "s_wx_hgb", "multiA", "multiAB"]]].to_string(index=False))
