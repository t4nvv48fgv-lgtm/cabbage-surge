"""Case 31 — 학습 줄 솎기 시작점에 대한 민감도와 평시 모형(여름특화 HGB) 안정화.

계기(2026-10-13 튜닝 요소 점검): 메인의 모든 부품은 거래일 사흘에 한 줄만 학습한다(`iloc[::3]`, 시작점 0). 시작점은 고른 값이 아니라
기본값인데, 평시 모형(wx_hgb_summer)은 시작점만 1·2로 바꿔도 평시 원점 예측이 크게 움직인다(2021-08-31 +2 → −17%, 2025-08-31 +15 → +30%).
현재 성적이 시작점 0의 덕을 보고 있는지, 시작점에 기대지 않는 값은 얼마인지 본다.

부품별로 시작점 0·1·2 로 다시 학습한다(학습 규칙·재료·누수 차단은 메인과 같다. 시작점 0 이 메인 값과 같은지 확인한다).
  대표 an_sparse_all / an_sparse_ship (ridge)   평시 wx_hgb_summer (HGB, 여름 원점·월보 신호 있는 해)   겨울 가격만 ridge
변형(조립은 메인 함수 gate_model.assemble, 평시·경보 판정은 메인 그대로):
  현행               전 부품 시작점 0
  평시 평균          평시 모형만 세 시작점 예측의 평균(로그 평균). 대표·겨울은 현행      ← 안정화 후보
  평시 전부          평시 모형만 솎지 않고 모든 줄로 학습(참고)
  전 부품 평균       대표·평시·겨울 모두 세 시작점 평균
  전 부품 시작점 1·2 현행과 같은 구조에서 시작점만 바꾼 것(범위 확인용)

Usage: python experiments/case31_thinning_stability/thinning_stability.py [M|W]
Outputs: outputs/thin_predictions{tag}.csv, thin_metrics{tag}.csv, (M) thin_monthly.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
OUT = CASE_DIR / "outputs"
sys.path.insert(0, str(ROOT / "src"))
import anomaly_model as am  # noqa: E402
import backtest as bt  # noqa: E402
import gate_model as gm  # noqa: E402

H = 28
OFFS = (0, 1, 2)
VARIANTS = ["현행", "평시 평균", "평시 전부", "전 부품 평균", "전 부품 시작점 1", "전 부품 시작점 2"]


def components(main: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(bt.PROC / "daily_features.csv", parse_dates=["date"]).set_index("date")
    df[bt.TARGET] = df["y_lp_fwd28"] - df[bt.BASE]
    df = am.add_ship_weather(am.add_normals(df))
    feats_b = bt.PRICE_FEATS + bt.SUPPLY_FEATS + bt.WX_FEATS
    pool_a = df[(df.is_trading == 1) & df.y_anom.notna()].dropna(subset=am.CORE)
    pool_b = df[(df.is_trading == 1) & df[bt.TARGET].notna()].dropna(subset=feats_b)
    pool_w = df[(df.is_trading == 1) & df[bt.TARGET].notna()].dropna(subset=feats_b + am.CORE + ["y_anom"])
    rows = []
    for o, summer, s1, tm, wr in zip(main.origin, main.summer, main.S1, main.tmonth, main.winter_ridge):
        te = df.loc[[o]]
        rec = {"origin": o}
        cut = lambda p: p[p.index + pd.Timedelta(days=H) <= o]      # 누수 차단: 학습 줄의 대상 기간 종료 ≤ 원점
        for k in OFFS:
            tr = cut(pool_a.iloc[k::3])
            for name, feats in (("an_sparse_all", am.SPARSE), ("an_sparse_ship", am.SPARSE_SHIP)):
                mdl = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(tr[feats], tr.y_anom)
                rec[f"{name}_{k}"] = float(np.exp(te.norm_fwd.iloc[0] + mdl.predict(te[feats])[0]))
            if summer == 1 and pd.notna(s1):
                rec[f"hgb_{k}"] = float(np.exp(te[bt.BASE].iloc[0] + bt.fit_predict("wx_hgb_summer", cut(pool_b.iloc[k::3]), te)))
            if pd.notna(wr):
                wp = pool_w.iloc[k::3]
                wtr = cut(wp[(wp.index + pd.Timedelta(days=15)).month.isin(gm.WINTER_MONTHS)])
                mdl = make_pipeline(StandardScaler(), Ridge(alpha=3.0)).fit(wtr[bt.PRICE_FEATS], wtr[bt.TARGET])
                rec[f"winter_{k}"] = float(np.exp(float(te[bt.BASE].iloc[0]) + mdl.predict(te[bt.PRICE_FEATS])[0]))
        if summer == 1 and pd.notna(s1):
            rec["hgb_full"] = float(np.exp(te[bt.BASE].iloc[0] + bt.fit_predict("wx_hgb_summer", cut(pool_b), te)))
        rows.append(rec)
    return pd.DataFrame(rows)


def gmean(c: pd.DataFrame, stem: str) -> pd.Series:
    return np.exp(np.log(c[[f"{stem}_{k}" for k in OFFS]]).mean(axis=1))


def variant(main: pd.DataFrame, c: pd.DataFrame, calm_src, head_src, winter_src) -> pd.DataFrame:
    """부품 열을 바꿔 메인 조립 함수로 다시 조립. *_src: None(현행) | 'mean' | 'full' | 시작점 번호."""
    d = main.copy()
    pick = lambda stem, src: gmean(c, stem) if src == "mean" else c[f"{stem}_{src}"]
    if calm_src is not None:
        new = c["hgb_full"] if calm_src == "full" else pick("hgb", calm_src)
        d[gm.CALM] = new.where(new.notna(), d[gm.CALM]).values
    if head_src is not None:
        d["head"] = np.where(d.summer == 1, pick("an_sparse_all", head_src), pick("an_sparse_ship", head_src))
    if winter_src is not None:
        new = pick("winter", winter_src)
        d["winter_ridge"] = new.where(new.notna(), d.winter_ridge).values
    return d


def run(freq: str) -> None:
    tag = "" if freq == "M" else f"_{freq}"
    OUT.mkdir(parents=True, exist_ok=True)
    main = pd.read_csv(ROOT / "outputs" / f"gate_predictions{tag}.csv", parse_dates=["origin"], encoding="utf-8-sig")
    c = components(main)
    assert (c.origin.values == main.origin.values).all()
    # 시작점 0 = 메인 재현 확인
    for stem, col in (("an_sparse_all", "an_sparse_all"), ("an_sparse_ship", "an_sparse_ship"), ("hgb", gm.CALM), ("winter", "winter_ridge")):
        mk = c[f"{stem}_0"].notna()
        assert np.allclose(c.loc[mk, f"{stem}_0"], main.loc[mk, col], atol=0.11), f"메인 재현 실패: {stem}"
    issues, fall = gm.load_issues(), gm.load_fall_issues()
    calm = main.gate_calm == 1
    monthly = freq == "M"
    spec = {"현행": (None, None, None), "평시 평균": ("mean", None, None), "평시 전부": ("full", None, None),
            "전 부품 평균": ("mean", "mean", "mean"), "전 부품 시작점 1": (1, 1, 1), "전 부품 시작점 2": (2, 2, 2)}
    res = pd.DataFrame({"origin": main.origin, "actual": main.actual, "summer": main.summer, "gate_calm": main.gate_calm, "tmonth": main.tmonth})
    for name, (cs, hs, ws) in spec.items():
        res[name] = gm.assemble(variant(main, c, cs, hs, ws), calm, issues, fall, monthly)[gm.FINAL].values
    assert np.allclose(res["현행"], main[gm.FINAL], atol=0.11), "현행 재현 실패"
    res.merge(c, on="origin").to_csv(OUT / f"thin_predictions{tag}.csv", index=False, float_format="%.1f", encoding="utf-8-sig")

    yr = res.origin.dt.year
    masks = {"전체 2021~2025": yr.between(2021, 2025), "전체 2021~2026": yr.between(2021, 2026), "전체 2016~2020": yr.between(2016, 2020),
             "여름 원점 2021~2025": yr.between(2021, 2025) & (res.summer == 1), "평시 원점 2021~2026": yr.between(2021, 2026) & (res.gate_calm == 1),
             "겨울 대상 2021~2025": yr.between(2021, 2025) & res.tmonth.isin(gm.WINTER_MONTHS)}
    rows = []
    for sub, mk in masks.items():
        for v in VARIANTS:
            e = (res.loc[mk, v] / res.loc[mk, "actual"] - 1).abs() * 100
            rows.append({"subset": sub, "variant": v, "n": int(mk.sum()), "MAPE%": e.mean(), "n_gt20": int((e > 20).sum()), "n_gt30": int((e > 30).sum())})
    met = pd.DataFrame(rows)
    met.to_csv(OUT / f"thin_metrics{tag}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(f"===== {freq} =====")
    print(met.pivot(index="subset", columns="variant", values="MAPE%")[VARIANTS].round(2).to_string())
    print(met.pivot(index="subset", columns="variant", values="n_gt30")[VARIANTS].to_string())
    # 평시 모형의 시작점별 흔들림
    k = res.merge(c, on="origin")
    k = k[(k.gate_calm == 1) & (yr >= 2021)]
    sp = np.log(k[[f"hgb_{i}" for i in OFFS]]).std(axis=1, ddof=0) * 100
    print(f"\n평시 원점 {len(k)}개: 시작점 셋 사이 예측 표준편차 평균 {sp.mean():.1f}%, 최대 {sp.max():.1f}% ({k.loc[sp.idxmax(), 'origin'].date()})")
    for stem, lab in (("an_sparse_all", "대표(여름)"), ("an_sparse_ship", "대표(여름 밖)"), ("winter", "겨울 모형")):
        s = np.log(c[[f"{stem}_{i}" for i in OFFS]]).std(axis=1, ddof=0) * 100
        print(f"{lab}: 시작점 셋 사이 예측 표준편차 평균 {s.mean():.1f}%, 최대 {s.max():.1f}%")
    if monthly:
        keys = pd.to_datetime([f"{y}-{md}" for y in range(2021, 2027) for md in ("06-30", "07-31", "08-31", "09-30")])
        t = res[res.origin.isin(keys)].copy()
        for v in VARIANTS:
            t[v] = ((t[v] / t.actual - 1) * 100).round(1)
        print("\n[여름 월말 원점 오차율 %]")
        print(t[["origin", "gate_calm"] + VARIANTS].to_string(index=False))
        mth = pd.DataFrame({"target_month": (res.origin + pd.Timedelta(days=15)).dt.to_period("M").astype(str), "origin": res.origin.dt.date, "actual": res.actual})
        for v in VARIANTS:
            mth[v] = (res[v] / res.actual - 1) * 100
        mth.to_csv(OUT / "thin_monthly.csv", index=False, float_format="%.1f", encoding="utf-8-sig")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "M")
