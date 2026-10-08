"""Case 03 — 설계(2016~2023) / 검증(2024~2025) 분리.

risk-audit R-1(사후 설계) 해소용. 메인 모형의 사양(피처 집합·규제 강도·여름 전용 여부·HGB 하이퍼)을
**2016~2023 월말 원점의 성적만 보고** 고른 뒤, 그 사양을 **2024~2025 월말 원점**에서 평가한다.
2024-09를 본 뒤 고른 현재 메인 사양이 분리 선택에서도 뽑히는지, 뽑히지 않으면 분리 선택 사양의 2024-09·2025-08 성적은 어떤지 본다.

선택 기준(실행 전 고정, 둘 다 보고)
  A. 2016~2023 전체 원점 MAPE 최소
  B. 2016~2023 여름 원점(6~9월) MAPE 최소   ← 급등 추종이 목적이므로 2020-09·2022-09 급등이 포함된 기준
후보 격자(실행 전 고정)
  ridge  : 피처 {price, price+supply, +wx, +wx×summer} × alpha {1, 3, 10, 30, 100}
  anomaly: 피처 {CORE, SPARSE} × 여름전용 {F, T} × alpha {0.1, 0.3, 1, 3, 10}
  hgb    : 여름전용 {F, T} × (max_depth, lr) {(3,0.04),(4,0.04),(6,0.04),(4,0.1)}
평가 설계(원점·타깃·누수 차단·학습 표본 추출)는 메인 backtest.py / anomaly_model.py 와 동일. 메인은 import만.

Usage: python experiments/case03_split_selection/split_selection.py
Outputs: outputs/grid_predictions.csv (원점×사양 예측), grid_scores.csv (설계/검증 구간 지표), selected.csv
"""
from __future__ import annotations

import itertools
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
sys.path.insert(0, str(ROOT / "src"))
import backtest as bt       # noqa: E402
import anomaly_model as am  # noqa: E402

warnings.filterwarnings("ignore")
PROC = ROOT / "data" / "processed"
OUT = CASE_DIR / "outputs"
DESIGN = (2016, 2023)
VALID = (2024, 2025)

RIDGE_FEATS = {
    "price": bt.PRICE_FEATS,
    "price+supply": bt.PRICE_FEATS + bt.SUPPLY_FEATS,
    "price+supply+wx": bt.PRICE_FEATS + bt.SUPPLY_FEATS + bt.WX_FEATS,
    "price+supply+wx×summer": None,   # interact_summer 적용
}
RIDGE_ALPHAS = [1, 3, 10, 30, 100]
ANOM_FEATS = {"CORE": am.CORE, "SPARSE": am.SPARSE}
ANOM_ALPHAS = [0.1, 0.3, 1, 3, 10]
HGB_GRID = [(3, 0.04), (4, 0.04), (6, 0.04), (4, 0.1)]

# 현재 메인 사양(2024-09를 본 뒤 정한 것) — 분리 선택 결과와 대조
MAIN_SPECS = {
    "wx_ridge": "ridge|price+supply+wx×summer|a10",
    "wx_hgb": "hgb|all|d4_lr0.04",
    "wx_hgb_summer": "hgb|summer|d4_lr0.04",
    "an_ridge_all": "anom|CORE|all|a1",
    "an_sparse_summer": "anom|SPARSE|summer|a0.3",
}


def configs() -> list[dict]:
    cs = []
    for fname, alpha in itertools.product(RIDGE_FEATS, RIDGE_ALPHAS):
        cs.append({"id": f"ridge|{fname}|a{alpha}", "family": "ridge", "feats": fname, "alpha": alpha})
    for fname, summer, alpha in itertools.product(ANOM_FEATS, (False, True), ANOM_ALPHAS):
        cs.append({"id": f"anom|{fname}|{'summer' if summer else 'all'}|a{alpha}", "family": "anom",
                   "feats": fname, "summer": summer, "alpha": alpha})
    for summer, (d, lr) in itertools.product((False, True), HGB_GRID):
        cs.append({"id": f"hgb|{'summer' if summer else 'all'}|d{d}_lr{lr}", "family": "hgb",
                   "summer": summer, "depth": d, "lr": lr})
    return cs


def fit_ridge(c, tr, te):
    if c["feats"] == "price+supply+wx×summer":
        feats = bt.PRICE_FEATS + bt.SUPPLY_FEATS + bt.WX_FEATS + ["summer"]
        Xtr, Xte = bt.interact_summer(tr[feats], bt.WX_FEATS), bt.interact_summer(te[feats], bt.WX_FEATS)
    else:
        feats = RIDGE_FEATS[c["feats"]]
        Xtr, Xte = tr[feats], te[feats]
    m = make_pipeline(StandardScaler(), Ridge(alpha=c["alpha"])).fit(Xtr, tr[bt.TARGET])
    return float(m.predict(Xte)[0])


def fit_anom(c, tr, te):
    feats = ANOM_FEATS[c["feats"]]
    t = tr[tr.summer == 1] if (c["summer"] and te.summer.iloc[0] == 1) else tr
    if len(t) < 120:
        return np.nan
    m = make_pipeline(StandardScaler(), Ridge(alpha=c["alpha"])).fit(t[feats], t["y_anom"])
    return float(m.predict(te[feats])[0])


def fit_hgb(c, tr, te):
    feats = bt.PRICE_FEATS + bt.SUPPLY_FEATS + bt.WX_FEATS + ["summer", "month"]
    t = tr[tr.summer == 1] if (c["summer"] and te.summer.iloc[0] == 1) else tr
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=c["lr"], max_depth=c["depth"],
                                      min_samples_leaf=25, l2_regularization=1.0, random_state=0)
    m.fit(t[feats], t[bt.TARGET])
    return float(m.predict(te[feats])[0])


def run() -> pd.DataFrame:
    df = pd.read_csv(PROC / "daily_features.csv", parse_dates=["date"]).set_index("date")
    df[bt.TARGET] = df["y_lp_fwd28"] - df[bt.BASE]
    df = am.add_normals(df)
    req = bt.PRICE_FEATS + bt.SUPPLY_FEATS + bt.WX_FEATS
    pool = df[(df.is_trading == 1) & df[bt.TARGET].notna() & df.y_anom.notna()].dropna(subset=req + am.CORE).iloc[::3]
    origins = bt.month_ends(df.index)
    origins = origins[(origins >= f"{DESIGN[0]}-01-31") & (origins <= f"{VALID[1]}-12-31")]
    cs = configs()
    rows = []
    for o in origins:
        if o not in df.index:
            continue
        te = df.loc[[o]]
        if te[bt.TARGET].isna().iloc[0] or te.y_anom.isna().iloc[0] or te[req].isna().any(axis=1).iloc[0]:
            continue
        tr = pool[pool.index + pd.Timedelta(days=28) <= o]
        if len(tr) < 300:
            continue
        base, nf = float(te[bt.BASE].iloc[0]), float(te.norm_fwd.iloc[0])
        rec = {"origin": o, "actual": float(te.y_price_fwd28_true.iloc[0]), "summer": int(te.summer.iloc[0])}
        for c in cs:
            if c["family"] == "ridge":
                rec[c["id"]] = float(np.exp(base + fit_ridge(c, tr, te)))
            elif c["family"] == "anom":
                yh = fit_anom(c, tr, te)
                rec[c["id"]] = float(np.exp(nf + yh)) if np.isfinite(yh) else np.nan
            else:
                rec[c["id"]] = float(np.exp(base + fit_hgb(c, tr, te)))
        rows.append(rec)
        print(f"{o.date()} actual={rec['actual']:7.0f}  done {len(cs)} configs", flush=True)
    return pd.DataFrame(rows)


def score(pred: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    y = pred.origin.dt.year
    masks = {
        "design_all": y.between(*DESIGN), "design_summer": y.between(*DESIGN) & (pred.summer == 1),
        "valid_all": y.between(*VALID), "valid_summer": y.between(*VALID) & (pred.summer == 1),
    }
    rows = []
    for cid in ids:
        r = {"id": cid}
        for k, mk in masks.items():
            s = pred[mk].dropna(subset=[cid])
            e = s[cid] - s.actual
            r[f"{k}_MAPE"] = (e.abs() / s.actual).mean() * 100
            r[f"{k}_n"] = len(s)
        for o in ("2020-08-31", "2022-08-31", "2024-08-31", "2025-08-31"):
            s = pred[pred.origin == o]
            r[f"err_{o[:7]}"] = float((s[cid].iloc[0] / s.actual.iloc[0] - 1) * 100) if len(s) and pd.notna(s[cid].iloc[0]) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pred = run()
    pred.to_csv(OUT / "grid_predictions.csv", index=False, float_format="%.1f")
    ids = [c["id"] for c in configs()]
    sc = score(pred, ids)
    sc["family"] = sc.id.str.split("|").str[0]
    sc.to_csv(OUT / "grid_scores.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 300)
    cols = ["id", "design_all_MAPE", "design_summer_MAPE", "valid_all_MAPE", "valid_summer_MAPE",
            "err_2020-08", "err_2022-08", "err_2024-08", "err_2025-08"]
    sel_rows = []
    for fam in ("ridge", "anom", "hgb"):
        f = sc[sc.family == fam]
        for crit in ("design_all_MAPE", "design_summer_MAPE"):
            best = f.loc[f[crit].idxmin()]
            sel_rows.append({"family": fam, "criterion": crit, **best[cols].to_dict()})
        # 사후(검증 구간을 보고 고른) 최선 — 분리 선택과의 격차를 보기 위한 참고값
        hind = f.loc[f["valid_all_MAPE"].idxmin()]
        sel_rows.append({"family": fam, "criterion": "HINDSIGHT valid_all_MAPE", **hind[cols].to_dict()})
    for name, cid in MAIN_SPECS.items():
        r = sc[sc.id == cid].iloc[0]
        sel_rows.append({"family": r.family, "criterion": f"MAIN({name})", **r[cols].to_dict()})
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(OUT / "selected.csv", index=False, float_format="%.2f")
    print("\n=== 분리 선택 결과 (design=2016~2023 로 고름 → valid=2024~2025 평가) ===")
    print(sel.round(1).to_string(index=False))
    print("\n=== 전체 격자 (design_all_MAPE 순) ===")
    print(sc[cols].sort_values("design_all_MAPE").round(1).to_string(index=False))
