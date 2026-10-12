"""Case 10 — 공급 게이트(G_K)의 평시 모형 3후보 비교.

배경: case06 G_K 는 여름 원점에서 KREI 최신호 생산 전망 S1 > 0(공급 증가 전망)이면 평시 모형, 아니면 대표 모형(an_sparse_all)을 쓴다.
평시 모형은 S1 > 0 인 해(2021·2023·2025)의 예측을 전담하므로, 2025 거짓 경보 억제(가격만 ridge −9%)와 2021·2023 보통 상승 포착(HGB)이 갈린다.
2026-10-12 해석: 대표 모형의 2021 −14% 는 급등 추종이 아니라 "평년치로의 부분 회귀" 성질이고, 같은 성질이 2025 +21% 과대의 ≈12%p 를 만든다.
폭염 피처 기여(≈+8%p)만 떼면 평년 회귀 성질은 유지하면서 2025 과대를 줄일 수 있다는 가설 → 세 번째 후보.

후보 (실행 전 고정)
  C1 price_only_ridge   가격만 ridge(메인 기준선). 최근 28일 평균이 기준점.
  C2 wx_hgb_summer      기상 HGB 여름특화(메인, 전체 MAPE 최선).
  C3 an_norm            평년 회귀형: 대표 사양(SPARSE·전체 원점 학습·ridge alpha 10·같은 학습 풀)에서 고랭지 기상 피처 H_* 를 모두 뺀
                        3피처 {cur_anom7, dlp7, qyoy28}. 타깃은 대표와 같은 평년 대비 편차(y_anom).
  참고 normal_fwd       평년치 그대로(직전 5년 같은 창 평균).
게이트: G_K[c] = 여름 원점 & S1 > 0 → 후보 c, 아니면 an_sparse_all.  S1 은 case09 outputs/krei_issues.csv(월보 호별 값, 공개)에서 case06 latest_signal 로.
평가: 메인과 같은 2021~2025 월말 60개(주간 261개) + 2026 여름 표본외. 후보 자체의 설계 구간(2016~2020) 성적도 같이 본다(사전 선택 근거).

Usage: python experiments/case10_calm_model/calm_model.py [M|W]
Outputs (experiments/case10_calm_model/outputs/): calm_predictions{tag}.csv, calm_metrics{tag}.csv, calm_key_origins{tag}.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "case06_supply_gate"))
import anomaly_model as am  # noqa: E402  add_normals·CORE·SPARSE·학습 풀 관례 재사용
import supply_gate as sg  # noqa: E402  latest_signal

OUT = CASE_DIR / "outputs"
MAIN_OUT = ROOT / "outputs"
KREI_ISSUES = ROOT / "experiments" / "case09_learned_quantile" / "outputs" / "krei_issues.csv"
HEAD = "an_sparse_all"
NORM_FEATS = [f for f in am.SPARSE if not f.startswith("H_")]      # cur_anom7, dlp7, qyoy28
CANDS = {"price_only_ridge": "C1 가격만 ridge", "wx_hgb_summer": "C2 기상 HGB(여름특화)", "an_norm": "C3 평년 회귀형(폭염 제외)", "normal_fwd": "참고 평년치"}


def an_norm(freq: str) -> pd.DataFrame:
    """대표 모형과 같은 학습 풀·원점·타깃으로 H_* 를 뺀 3피처 ridge(alpha 10)를 적합."""
    df = pd.read_csv(am.PROC / "daily_features.csv", parse_dates=["date"]).set_index("date")
    df = am.add_normals(df)
    pool = df[(df.is_trading == 1) & df.y_anom.notna()].dropna(subset=am.CORE).iloc[::3]
    if freq == "M":
        origins = pd.DatetimeIndex(sorted(set(df.index.to_period("M").to_timestamp("M"))))
    else:
        origins = df.index[df.index.weekday == 6]
    origins = origins[(origins >= am.FIRST_ORIGIN) & (origins <= df.index.max())]
    rows = []
    for o in origins:
        if o not in df.index:
            continue
        te = df.loc[[o]]
        if te.y_anom.isna().iloc[0] or te[am.CORE].isna().any(axis=1).iloc[0]:
            continue
        tr = pool[pool.index + pd.Timedelta(days=28) <= o]
        if len(tr) < 120:
            continue
        m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(tr[NORM_FEATS], tr["y_anom"])
        rows.append({"origin": o, "an_norm": float(np.exp(te.norm_fwd.iloc[0] + m.predict(te[NORM_FEATS])[0]))})
    return pd.DataFrame(rows)


def load_issues() -> pd.DataFrame:
    if sg.KREI_SERIES.exists():
        return sg.krei_summer_signal()
    return pd.read_csv(KREI_ISSUES, parse_dates=["pub_date"])


def run(freq: str) -> pd.DataFrame:
    tag = "" if freq == "M" else f"_{freq}"
    b = pd.read_csv(MAIN_OUT / f"backtest_predictions{tag}.csv", parse_dates=["origin"])
    a = pd.read_csv(MAIN_OUT / f"anomaly_predictions{tag}.csv", parse_dates=["origin"])
    d = b[["origin", "actual", "summer", "price_only_ridge", "wx_hgb_summer", "wx_hgb", "chronos2"]].merge(
        a[["origin", "normal_fwd", HEAD]], on="origin").merge(an_norm(freq), on="origin", how="left")
    sig = load_issues()
    d["S1"] = [sg.latest_signal(sig, o)["S1"] if s == 1 else np.nan for o, s in zip(d.origin, d.summer)]
    calm = d.S1.notna() & (d.S1 > 0)
    for c in CANDS:
        d[f"G_K[{c}]"] = np.where(calm, d[c], d[HEAD])
    d["gate_calm"] = calm.astype(int)
    return d


def metrics(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    y = d.origin.dt.year
    masks = {"all_2021_2025": y.between(2021, 2025), "summer_2021_2025": y.between(2021, 2025) & (d.summer == 1),
             "summer_2021_2026": y.between(2021, 2026) & (d.summer == 1),
             "calm_origins_2021_2026": (d.gate_calm == 1) & y.between(2021, 2026),
             "design_all_2016_2020": y.between(2016, 2020), "design_summer_2016_2020": y.between(2016, 2020) & (d.summer == 1)}
    out = []
    for name, mk in masks.items():
        s = d[mk]
        for c in cols:
            e = s[c] - s.actual
            out.append({"subset": name, "model": c, "n": len(s), "MAE": e.abs().mean(), "MAPE%": (e.abs() / s.actual).mean() * 100, "bias": e.mean()})
    return pd.DataFrame(out)


if __name__ == "__main__":
    freq = sys.argv[1] if len(sys.argv) > 1 else "M"
    tag = "" if freq == "M" else f"_{freq}"
    OUT.mkdir(parents=True, exist_ok=True)
    d = run(freq)
    cols = list(CANDS) + [HEAD] + [f"G_K[{c}]" for c in CANDS]
    d.to_csv(OUT / f"calm_predictions{tag}.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    met = metrics(d, cols)
    met.to_csv(OUT / f"calm_metrics{tag}.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 320)
    if freq == "M":
        keys = pd.to_datetime([f"{y}-0{m}-{dd}" for y in range(2021, 2027) for m, dd in ((6, "30"), (7, "31"), (8, "31"), (9, "30"))])
    else:
        keys = d.origin[(d.origin.dt.month.isin([6, 7, 8, 9])) & (d.origin.dt.year >= 2021)]
    ko = d[d.origin.isin(keys)].copy()
    for c in cols:
        ko[c + "%"] = (ko[c] / ko.actual - 1) * 100
    ko[["origin", "actual", "S1", "gate_calm"] + [c + "%" for c in cols]].to_csv(OUT / f"calm_key_origins{tag}.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    print(f"[{freq}] 원점 {len(d)}개, 평시(calm) 원점 {int(d.gate_calm.sum())}개")
    print("\n=== 여름 원점 오차% (2021~2026) ===")
    show = ko if freq == "M" else ko[ko.origin.dt.month.isin([8, 9]) & (ko.origin.dt.day <= 7) | (ko.origin.dt.month == 8) & (ko.origin.dt.day >= 25)]
    print(show[["origin", "actual", "S1", "gate_calm"] + [c + "%" for c in cols]].round(0).to_string(index=False))
    print("\n=== MAPE ===")
    print(met.pivot(index="model", columns="subset", values="MAPE%").loc[cols].round(1).to_string())
    print("\n=== bias (2021~2025 전체 / calm 원점) ===")
    print(met[met.subset.isin(["all_2021_2025", "calm_origins_2021_2026"])].pivot(index="model", columns="subset", values="bias").loc[cols].round(0).to_string())
