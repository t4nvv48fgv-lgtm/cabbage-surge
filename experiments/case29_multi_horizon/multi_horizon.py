"""Case 29 — 여름 자료 없이 6월 말 한 원점에서 7·8·9월을 한꺼번에 예측하기(1·2·3개월 앞).

사용자 지시(2026-10-13): "2026년 여름데이터를 쓰지 않고 7~9월 데이터를 원점에서 예측해봐."
메인 모형은 '앞으로 28일' 한 구간만 예측한다. 7월 말 원점은 7월 자료를, 8월 말 원점은 8월 자료를 쓴다.
여기서는 **6월 30일까지의 자료만으로** 세 구간을 직접 예측한다(구간마다 따로 학습하는 직접 다단계 예측).

  구간 1  원점+1 ~ +28일   (6/30 원점이면 7/1~7/28)
  구간 2  원점+29 ~ +56일  (7/29~8/25)
  구간 3  원점+57 ~ +84일  (8/26~9/22)
부품(메인과 같은 재료·같은 사양, 타깃만 구간별로 바꿈. 학습 줄은 그 줄의 대상 구간 종료일 ≤ 원점, 사흘에 한 줄):
  대표    평년 대비 편차 모형(SPARSE 재료, 릿지 알파 10). 평년치 = 직전 5년 같은 구간 평균
  평시    기상 HGB(여름 원점 줄만 학습), 타깃 = 구간 평균 − 최근 28일 평균
  가격만  가격 재료 15개 릿지(알파 3)
조립(메인의 여름 원점 규칙 그대로): 원점까지 나온 월보 최신호의 여름배추 생산 전망 S1 > 0 → 평시, 아니면 대표 × (1 − 0.566·S/100) (S 는 규약 B).
  월보 신호는 2021년부터. 그 전 해는 대표 모형 그대로. 수준 조정 계수는 구간 1에서 정한 값을 세 구간에 그대로 쓴다(가정).
확인: 6월 말 원점 2016~2026 의 구간별 예측·실제·오차, 구간 1이 메인 값과 같은지, 비교 기준(평년치 그대로, 가격만, 최근 28일 그대로).

Usage: python experiments/case29_multi_horizon/multi_horizon.py [06-30]     (원점의 월-일. 06-30·07-31·08-31 을 차례로 돌리면 원점을 옮겨 가며 본 결과가 된다)
Outputs: outputs/mh_predictions_{월-일}.csv, mh_metrics_{월-일}.csv
"""
from __future__ import annotations

import sys
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
import gate_model as gm  # noqa: E402

HORIZONS = {1: (0, 28), 2: (28, 56), 3: (56, 84)}
HGB_FEATS = bt.PRICE_FEATS + bt.SUPPLY_FEATS + bt.WX_FEATS + ["summer", "month"]


def block_means(price: pd.Series, cal: pd.DatetimeIndex, a: int, b: int) -> tuple[pd.Series, pd.Series]:
    """날짜 d 마다 (d+a, d+b] 구간의 로그가격 평균과 가격 평균(거래일 8일 이상, 구간이 관측 범위 안일 때만)."""
    p = price.reindex(cal)
    ok = p.notna().astype(float)
    cs_l, cs_p, cs_n = np.log(p).fillna(0).cumsum(), p.fillna(0).cumsum(), ok.cumsum()
    n = cs_n.shift(-b) - cs_n.shift(-a)
    ml = ((cs_l.shift(-b) - cs_l.shift(-a)) / n).where(n >= 8)
    mp = ((cs_p.shift(-b) - cs_p.shift(-a)) / n).where(n >= 8)
    last = price.index.max()
    bad = cal + pd.Timedelta(days=b) > last
    ml[bad], mp[bad] = np.nan, np.nan
    return ml, mp


def build() -> pd.DataFrame:
    df = pd.read_csv(bt.PROC / "daily_features.csv", parse_dates=["date"]).set_index("date")
    df[bt.TARGET] = df["y_lp_fwd28"] - df[bt.BASE]
    df = am.add_normals(df)
    price = pd.read_csv(ROOT / "data" / "raw" / "clean_daily.csv", parse_dates=["date"])
    price = price[price.price > 0].set_index("date")["price"]
    for h, (a, b) in HORIZONS.items():
        ml, mp = block_means(price, df.index, a, b)
        if h == 1:                                   # 구간 1은 메인과 같은 타깃 열을 그대로 쓴다(소수 5자리로 저장된 값 → 나무 모형까지 같은 결과)
            ml = df["y_lp_fwd28"]
        df[f"y{h}"], df[f"act{h}"] = ml, mp
        df[f"norm{h}"] = pd.concat([ml.shift(364 * k) for k in range(1, am.NORM_YEARS + 1)], axis=1).mean(axis=1)
        df[f"ya{h}"] = df[f"y{h}"] - df[f"norm{h}"]       # 평년 대비 편차
        df[f"yc{h}"] = df[f"y{h}"] - df["lp28"]            # 최근 28일 평균 대비 변화
    return df


def predict_origin(df: pd.DataFrame, o: pd.Timestamp, issues: pd.DataFrame) -> list[dict]:
    te = df.loc[[o]]
    trading = df[df.is_trading == 1]
    k_issue, s1 = gm.latest_signal(issues, o) if int(te.summer.iloc[0]) == 1 else (np.nan, np.nan)
    _, s_adj = gm.adjust_signal(issues, o, o.year) if pd.notna(s1) else (np.nan, np.nan)
    calm = pd.notna(s1) and s1 > 0
    rows = []
    for h, (a, b) in HORIZONS.items():
        # 학습 줄 정의는 메인과 같다: 대표는 CORE 재료가 있는 줄, 가격만·평시는 backtest 재료가 모두 있는 줄. 사흘에 한 줄.
        pool = trading[trading[f"ya{h}"].notna()].dropna(subset=am.CORE).iloc[::3]
        tr = pool[pool.index + pd.Timedelta(days=b) <= o]             # 누수 차단: 학습 줄의 대상 구간 종료 ≤ 원점
        bpool = trading[trading[f"yc{h}"].notna()].dropna(subset=bt.PRICE_FEATS + bt.SUPPLY_FEATS + bt.WX_FEATS).iloc[::3]
        btr = bpool[bpool.index + pd.Timedelta(days=b) <= o]
        rec = {"origin": o, "h": h, "start": o + pd.Timedelta(days=a + 1), "end": o + pd.Timedelta(days=b), "n_train": len(tr),
               "actual": float(te[f"act{h}"].iloc[0]), "normal": float(np.exp(te[f"norm{h}"].iloc[0])), "last28": float(np.exp(te.lp28.iloc[0])),
               "krei_issue": k_issue, "S1": s1, "state": "평시" if calm else ("경보" if pd.notna(s1) else "신호 없음")}
        if len(tr) < 120:
            rows.append(rec)
            continue
        m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(tr[am.SPARSE], tr[f"ya{h}"])
        rec["head"] = float(np.exp(te[f"norm{h}"].iloc[0] + m.predict(te[am.SPARSE])[0]))
        m = make_pipeline(StandardScaler(), Ridge(alpha=3.0)).fit(btr[bt.PRICE_FEATS], btr[f"yc{h}"])
        rec["price_only"] = float(np.exp(te.lp28.iloc[0] + m.predict(te[bt.PRICE_FEATS])[0]))
        trs = btr[btr.summer == 1] if int(te.summer.iloc[0]) == 1 else btr
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=1.0, random_state=0)
        m.fit(trs[HGB_FEATS], trs[f"yc{h}"])
        rec["calm_hgb"] = float(np.exp(te.lp28.iloc[0] + m.predict(te[HGB_FEATS])[0]))
        f = 1.0
        if rec["state"] == "경보" and pd.notna(s_adj) and s_adj <= 0:
            f = 1.0 + gm.BETA_SUMMER_ALERT * float(np.clip(s_adj, -gm.S_CAP, gm.S_CAP)) / 100.0
        rec["adj_pct"] = (f - 1.0) * 100
        rec["final"] = rec["calm_hgb"] if calm else rec["head"] * f
        rows.append(rec)
    return rows


def main(md: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 300)
    df = build()
    issues = gm.load_issues()
    rows = []
    for y in range(2016, 2027):
        o = pd.Timestamp(f"{y}-{md}")
        if o in df.index:
            rows += predict_origin(df, o, issues)
    d = pd.DataFrame(rows)
    models = ["final", "head", "calm_hgb", "price_only", "normal", "last28"]
    for c in models:
        d[c + "%"] = (d[c] / d.actual - 1) * 100
    d.to_csv(OUT / f"mh_predictions_{md}.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    # 구간 1이 메인과 같은지
    g = pd.read_csv(ROOT / "outputs" / "gate_predictions.csv", parse_dates=["origin"], encoding="utf-8-sig")[["origin", "final", "actual"]]
    c = d[d.h == 1].merge(g, on="origin", suffixes=("", "_main"))
    print(f"[확인] 구간 1 예측과 메인 final 의 최대 차이: {float((c.final - c.final_main).abs().max()):.1f}원, 실제값 차이 {float((c.actual - c.actual_main).abs().max()):.1f}원")
    yr = d.origin.dt.year
    rows = []
    for wname, (a, b) in {"2016~2020": (2016, 2020), "2021~2025": (2021, 2025), "2016~2025": (2016, 2025)}.items():
        for h in HORIZONS:
            s = d[(d.h == h) & yr.between(a, b) & d.actual.notna()]
            row = {"window": wname, "구간": h, "n": len(s)}
            for mname in models:
                row[mname] = s[mname + "%"].abs().mean()
            rows.append(row)
    met = pd.DataFrame(rows)
    met.to_csv(OUT / f"mh_metrics_{md}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    print(f"\n[{md} 원점에서 본 구간별 평균 오차율(%)]")
    print(met.round(1).to_string(index=False))
    show = d.assign(해=yr, 구간=d.h, 기간=d.start.dt.strftime("%m/%d") + "~" + d.end.dt.strftime("%m/%d"))
    cols = ["해", "구간", "기간", "state", "S1", "actual", "final", "final%", "head%", "calm_hgb%", "price_only%", "normal%", "last28%"]
    print(f"\n[{md} 원점 해별 예측]")
    print(show[cols].round({"actual": 0, "final": 0, "final%": 0, "head%": 0, "calm_hgb%": 0, "price_only%": 0, "normal%": 0, "last28%": 0}).to_string(index=False))
    last = d[yr == 2026]
    print("\n[2026년]")
    print(last[["h", "start", "end", "actual", "final", "head", "calm_hgb", "price_only", "normal", "last28", "state", "S1", "adj_pct"]].round(0).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "06-30")
