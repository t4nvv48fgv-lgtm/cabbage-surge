"""Case 09 — 학습으로 조절되는 분위수 수준: 공급 신호 → Chronos-2 분위수 수준(τ)을 자료에서 배우고 해마다 갱신한다.

배경
  case01은 트리거 심각도 → 분위수 수준을 손으로 정한 상수(300t/0.5/0.10)로 매겼고(사후 설계), case06 G_K는 임계 하나(S1 ≤ 0)로
  대표 모형 ↔ 평시 모형을 전환한다. 둘 다 "급등 국면에서 어느 분위수를 점예측으로 쓸지"를 사람이 정한다.
  이 케이스는 그 수준을 자료에서 배운다: 과거 원점마다 실제값이 Chronos-2 예측분포의 몇 번째 분위수에 떨어졌는지(τ*, PIT)를 구하고,
  공급 신호 S → τ* 의 사상을 **원점 이전에 타깃 창이 닫힌 원점만으로**(확장 창) 적합해 그 원점의 분위수 수준 τ̂ 로 쓴다.
  새 해가 들어오면 사상이 자동으로 갱신된다.

신호 (원점 시점 가용)
  S1  KREI 월보 여름배추 생산량 전망 전년비, 원점까지 발행된 최신호(case06 krei_summer_signal 재사용, 2021~2026, 여름 원점 6~9월만)
  S2  정부비축 잔량 추정 (case05 stock_at, 2017~2026, 여름 원점만 사용)

사상(mapping) 형태 — 손 상수는 수축 강도 k 하나
  lin  τ̂ = 0.5 + β·z,  z = 표준화한 S(학습 표본 기준), β 는 ridge(alpha=k)·부호 제약(β ≤ 0: 신호가 낮을수록 τ 높음).
       k 는 가상 관측 수에 해당(z 표준화 후 Σz² ≈ n 이므로 β 가 n/(n+k) 배로 수축).
  iso  단조 감소 isotonic 회귀 iso(S) → τ̂ = (n·iso(S) + k·0.5)/(n+k).
  두 신호 동시: lin 은 2변수 ridge, iso 는 단일 신호 τ̂ 의 평균.
  학습 표본 < n_min(3) 이면 τ̂ = 0.5 (학습 불가, 중위수). τ̂ 는 [0.01, 0.99] 로 자른다.
  여름이 아닌 원점·신호 없는 원점은 τ̂ = 0.5.

점예측  LQ_<form>_<sig> = Chronos-2 분위수를 τ̂ 에서 선형보간.  LQh_* = 여름 원점은 LQ, 비여름은 대표 모형(an_sparse_all).
비교    chronos2(중위수)·an_sparse_all·price_only_ridge·G_K(case06 규칙 재계산)·case01 trig_cont(월말 2021~, 있으면).

누수 방지: 학습 표본은 target_end ≤ 원점. 신호는 원점까지 발행·추정분. 평가 창은 메인과 같은 2021~2025 + 2026 여름 표본외 + 2017~2020 여름(S2).

Usage: python experiments/case09_learned_quantile/learned_quantile.py [M|W]   (메인 outputs/backtest_predictions{tag}.csv, anomaly_predictions{tag}.csv 필요)
Outputs (experiments/case09_learned_quantile/outputs/):
  chronos_quantiles{tag}.csv   원점별 Chronos-2 분위수 0.01~0.99 (캐시, 재실행 시 재사용)
  krei_issues.csv              월보 호별 여름배추 전망치(S1 원천의 공개 가능한 값)
  lq_predictions{tag}.csv      원점별 신호·τ*·τ̂·학습 표본 수·β·예측
  lq_metrics{tag}.csv          지표(2021~2025 전체·여름, 2021~2026 여름, 2017~2020 여름)
  lq_trajectory{tag}.csv       해마다 8월 말 원점에서 학습된 사상(β·n·τ̂)의 궤적
  lq_sensitivity{tag}.csv      수축 강도 k·n_min 민감도
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "case05_gov_stock_krei"))
sys.path.insert(0, str(ROOT / "experiments" / "case06_supply_gate"))
import ts_baselines  # noqa: E402
import gov_stock_krei as c5  # noqa: E402
import supply_gate as sg  # noqa: E402

warnings.filterwarnings("ignore")
MAIN_OUT = ROOT / "outputs"
OUT = CASE_DIR / "outputs"
QL = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
QCOLS = [f"q{int(round(l * 100))}" for l in QL]
FIRST_ORIGIN = pd.Timestamp("2017-01-31")      # S2 시작
HEAD, CALM = "an_sparse_all", "price_only_ridge"
K_DEFAULT, NMIN_DEFAULT = 2.0, 3
SIGS = {"S1": "S1", "S2": "S2_kt", "S12": None}


# ---------------------------------------------------------------- 입력
def chronos_quantiles(origins: pd.Series, freq: str, tag: str) -> pd.DataFrame:
    """원점별 Chronos-2 분위수(향후 h스텝 평균의 분위수 근사, case01과 같은 방식). 캐시 파일이 원점을 다 덮으면 재사용."""
    cache = OUT / f"chronos_quantiles{tag}.csv"
    if cache.exists():
        c = pd.read_csv(cache, parse_dates=["origin"])
        if set(origins) <= set(c.origin) and all(col in c.columns for col in QCOLS):
            return c[c.origin.isin(origins)].reset_index(drop=True)
    import torch
    rows = []
    for i, o in enumerate(origins):
        s, h = ts_baselines.history(o, freq)
        ctx = torch.tensor(s.values, dtype=torch.float32).reshape(1, 1, -1)
        q, _ = ts_baselines._chronos().predict_quantiles(ctx, prediction_length=h, quantile_levels=QL)
        q = np.asarray(q[0] if isinstance(q, list) else q)[0].mean(axis=0)
        rows.append({"origin": o, **dict(zip(QCOLS, q))})
        if i % 20 == 0:
            print(f"  chronos {o.date()} q50={q[QL.index(0.5)]:.0f}", flush=True)
    c = pd.DataFrame(rows)
    c.to_csv(cache, index=False, float_format="%.2f")
    return c


def signals(d: pd.DataFrame) -> pd.DataFrame:
    """S1(여름 원점, 최신호)·S2(잔량). 월보 호별 값은 krei_issues.csv 로 저장(판독표 없이 재현 가능하게)."""
    issues_csv = OUT / "krei_issues.csv"
    if sg.KREI_SERIES.exists():
        sig = sg.krei_summer_signal()
        sig.to_csv(issues_csv, index=False, encoding="utf-8-sig")
    else:
        sig = pd.read_csv(issues_csv, parse_dates=["pub_date"])
    lots = c5.load_lots()
    s1 = [sg.latest_signal(sig, o)["S1"] if sm == 1 else np.nan for o, sm in zip(d.origin, d.summer)]
    s2 = [c5.stock_at(lots, o)["stock_est"] if o >= FIRST_ORIGIN else np.nan for o in d.origin]
    d = d.copy()
    d["S1"] = s1
    d["S2_t"] = s2
    d["S2_kt"] = np.where(d.summer == 1, d.S2_t / 1000.0, np.nan)   # 여름 원점만 사용, 천 톤 단위
    return d


def pit(actual: float, qv: np.ndarray) -> float:
    """실제값이 예측분포의 몇 번째 분위수에 떨어졌는가(분위수 값에 대한 선형보간, 범위 밖은 0.01/0.99로 절단)."""
    qv = np.maximum.accumulate(qv)   # 단조 보장
    return float(np.interp(actual, qv, QL, left=QL[0], right=QL[-1]))


# ---------------------------------------------------------------- 사상 학습
def fit_lin(S: np.ndarray, y: np.ndarray, s_new: np.ndarray, k: float, intercept: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """τ = a + β·z. a = 학습 τ* 평균을 0.5 쪽으로 k 수축(intercept=False 면 0.5 고정), β 는 ridge(alpha=k)·β ≤ 0 (각 열).
    y 는 τ*−0.5. 반환: 새 점의 τ̂−0.5 와 β."""
    from sklearn.linear_model import Ridge
    n = len(y)
    a = n * y.mean() / (n + k) if intercept else 0.0
    mu, sd = S.mean(axis=0), S.std(axis=0, ddof=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    z, z_new = (S - mu) / sd, (s_new - mu) / sd
    m = Ridge(alpha=k, fit_intercept=False, positive=True).fit(-z, y - a)   # positive on −z  ⇔  β ≤ 0 on z
    beta = -m.coef_
    return a + z_new @ beta, beta


def fit_iso(S: np.ndarray, y: np.ndarray, s_new: np.ndarray, k: float) -> np.ndarray:
    from sklearn.isotonic import IsotonicRegression
    n = len(y)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(S, y)
    return (n * iso.predict(s_new) + k * 0.5) / (n + k) - 0.5


def learn_tau(d: pd.DataFrame, k: float = K_DEFAULT, n_min: int = NMIN_DEFAULT, scope: str = "summer", intercept: bool = True) -> pd.DataFrame:
    """원점마다 확장 창(target_end ≤ 원점, 여름, 신호 있음)으로 사상을 적합해 τ̂ 를 만든다.
    scope='summer': 과거 여름 원점(6~9월) 전부로 학습. scope='month': 같은 달 원점만(8월 말 원점은 과거 8월 말 원점만 → "8월 발표치 → 9월 가격")."""
    d = d.sort_values("origin").reset_index(drop=True)
    sfx = "" if scope == "summer" else "_m"
    out = {}
    for name in SIGS:
        cols = ["S1", "S2_kt"] if name == "S12" else [SIGS[name]]
        tau_l, tau_i, ntr, betas = [], [], [], []
        for i, r in d.iterrows():
            ok_new = (r.summer == 1) and all(np.isfinite(r[c]) for c in cols)
            mk = (d.target_end <= r.origin) & (d.summer == 1) & d[cols].notna().all(axis=1) & d.tau_star.notna()
            if scope == "month":
                mk &= d.origin.dt.month == r.origin.month
            T = d[mk]
            n = len(T)
            if not ok_new or n < n_min:
                tau_l.append(0.5); tau_i.append(0.5); ntr.append(n if ok_new else 0); betas.append([np.nan] * len(cols))
                continue
            S, y, s_new = T[cols].values.astype(float), (T.tau_star - 0.5).values, r[cols].values.astype(float)[None, :]
            dl, b = fit_lin(S, y, s_new, k, intercept)
            if name == "S12":
                di = np.mean([fit_iso(S[:, [j]], y, s_new[:, [j]], k)[0] for j in range(len(cols))])
            else:
                di = fit_iso(S, y, s_new, k)[0]
            tau_l.append(float(np.clip(0.5 + dl[0], QL[0], QL[-1])))
            tau_i.append(float(np.clip(0.5 + di, QL[0], QL[-1])))
            ntr.append(n); betas.append(list(b))
        out[f"tau_lin_{name}{sfx}"], out[f"tau_iso_{name}{sfx}"], out[f"ntr_{name}{sfx}"] = tau_l, tau_i, ntr
        for j, c in enumerate(cols):
            out[f"beta_{name}{sfx}_{c}"] = [b[j] for b in betas]
    return pd.concat([d, pd.DataFrame(out)], axis=1)


def learn_all(d: pd.DataFrame, k: float = K_DEFAULT, n_min: int = NMIN_DEFAULT, intercept: bool = True) -> pd.DataFrame:
    d = learn_tau(d, k, n_min, "summer", intercept)
    return learn_tau(d, k, n_min, "month", intercept)


def point_preds(d: pd.DataFrame) -> pd.DataFrame:
    Q = d[QCOLS].values.astype(float)
    for tc in [c for c in d.columns if c.startswith("tau_") and c != "tau_star"]:
        tau = d[tc].values
        p = np.array([np.interp(t, QL, np.maximum.accumulate(q)) for t, q in zip(tau, Q)])
        key = tc[len("tau_"):]
        d[f"LQ_{key}"] = p
        d[f"LQh_{key}"] = np.where(d.summer == 1, p, d[HEAD])
    hasK = d.S1.notna()
    d["G_K"] = np.where(hasK & (d.S1 > 0), d[CALM], d[HEAD])
    return d


# ---------------------------------------------------------------- 평가
def metrics(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    y = d.origin.dt.year
    masks = {"all_2021_2025": y.between(2021, 2025), "summer_2021_2025": y.between(2021, 2025) & (d.summer == 1),
             "summer_2021_2026": y.between(2021, 2026) & (d.summer == 1), "summer_2017_2020": y.between(2017, 2020) & (d.summer == 1)}
    out = []
    for name, mk in masks.items():
        s = d[mk]
        for c in cols:
            e = s[c] - s.actual
            out.append({"subset": name, "model": c, "n": len(s), "MAE": e.abs().mean(), "MAPE%": (e.abs() / s.actual).mean() * 100, "bias": e.mean()})
    return pd.DataFrame(out)


def key_origins(d: pd.DataFrame, freq: str) -> pd.DataFrame:
    """해마다 8/31(주간은 그 직후 일요일) 원점 한 줄."""
    keys = []
    for yr in range(2017, 2027):
        o = d.origin[d.origin >= f"{yr}-08-31"]
        if freq == "M":
            o = o[o <= f"{yr}-08-31"]
        if len(o):
            keys.append(o.min())
    return d[d.origin.isin(keys)].copy()


def sensitivity(d0: pd.DataFrame, freq: str) -> pd.DataFrame:
    rows = []
    for k in (0.0, 1.0, 2.0, 5.0, 10.0):
        for n_min, intercept in ((3, True), (3, False), (6, True)):
            d = point_preds(learn_all(d0, k, n_min, intercept))
            ko = key_origins(d, freq)
            ko = ko.set_index(ko.origin.dt.year)
            y = d.origin.dt.year
            for key in [c[len("tau_"):] for c in d.columns if c.startswith("tau_") and c != "tau_star"]:
                    form, name = key.split("_", 1)
                    c = f"LQh_{key}"
                    rec = {"k": k, "n_min": n_min, "intercept": int(intercept), "form": form, "sig": name}
                    for sub, mk in (("MAPE_all_21_25", y.between(2021, 2025)), ("MAPE_summer_21_26", y.between(2021, 2026) & (d.summer == 1))):
                        s = d[mk]; rec[sub] = ((s[c] - s.actual).abs() / s.actual).mean() * 100
                    for yr in (2020, 2022, 2024, 2025, 2026):
                        rec[f"err_{yr}%"] = (ko.loc[yr, c] / ko.loc[yr, "actual"] - 1) * 100 if yr in ko.index else np.nan
                    rows.append(rec)
    return pd.DataFrame(rows)


def run(freq: str) -> None:
    tag = "" if freq == "M" else f"_{freq}"
    OUT.mkdir(parents=True, exist_ok=True)
    b = pd.read_csv(MAIN_OUT / f"backtest_predictions{tag}.csv", parse_dates=["origin", "target_end"])
    a = pd.read_csv(MAIN_OUT / f"anomaly_predictions{tag}.csv", parse_dates=["origin"])
    d = b[["origin", "target_end", "actual", "summer", CALM, "chronos2"]].merge(a[["origin", HEAD]], on="origin")
    d = d[d.origin >= FIRST_ORIGIN].reset_index(drop=True)
    print(f"[{freq}] 원점 {len(d)}개 {d.origin.min().date()}~{d.origin.max().date()}", flush=True)
    q = chronos_quantiles(d.origin, freq, tag)
    d = d.merge(q, on="origin")
    d["q50_gap%"] = (d.q50 / d.chronos2 - 1) * 100            # 메인 chronos2 중위수와의 일치 확인
    d = signals(d)
    d["tau_star"] = [pit(a_, r) for a_, r in zip(d.actual, d[QCOLS].values.astype(float))]
    d["beyond"] = np.where(d.actual > d.q99, "above_q99", np.where(d.actual < d.q1, "below_q1", ""))
    d = point_preds(learn_all(d))
    c01 = ROOT / "experiments" / "case01_release_trigger" / "outputs" / f"trigger_quantile_predictions{tag}.csv"
    if c01.exists():
        t = pd.read_csv(c01, parse_dates=["origin"])[["origin", "trig_cont"]].rename(columns={"trig_cont": "case01_trig_cont"})
        d = d.merge(t, on="origin", how="left")
        d["case01_trig_cont"] = d.case01_trig_cont.fillna(d.chronos2)
    lq_cols = [c for c in d.columns if c.startswith(("LQ_", "LQh_"))]
    cols = [CALM, "chronos2", HEAD, "G_K"] + (["case01_trig_cont"] if "case01_trig_cont" in d else []) + lq_cols
    d.to_csv(OUT / f"lq_predictions{tag}.csv", index=False, float_format="%.4f", encoding="utf-8-sig")
    met = metrics(d, cols)
    met.to_csv(OUT / f"lq_metrics{tag}.csv", index=False, float_format="%.2f")

    pd.set_option("display.width", 320)
    print(f"\nq50 vs 메인 chronos2 중위수 차이: 최대 {d['q50_gap%'].abs().max():.2f}%")
    ko = key_origins(d, freq)
    print("\n=== 8월 말 원점: 신호·τ*·학습된 τ̂ (k=2, n_min=3, 절편 있음) — 여름 풀(summer) ===")
    base = ["origin", "actual", "S1", "S2_t", "tau_star", "beyond"]
    show = ["ntr_S1", "tau_lin_S1", "tau_iso_S1", "beta_S1_S1", "ntr_S2", "tau_lin_S2", "tau_iso_S2", "beta_S2_S2_kt", "tau_lin_S12"]
    print(ko[base + show].round(3).to_string(index=False))
    print("\n=== 같은 달만(month): 8월 말 원점은 과거 8월 말 원점으로만 학습 ===")
    show_m = ["ntr_S1_m", "tau_lin_S1_m", "tau_iso_S1_m", "beta_S1_m_S1", "ntr_S2_m", "tau_lin_S2_m", "tau_iso_S2_m", "beta_S2_m_S2_kt", "tau_lin_S12_m"]
    print(ko[base + show_m].round(3).to_string(index=False))
    print("\n=== 8월 말 원점 오차% ===")
    e = ko[["origin", "actual"]].copy()
    for c in cols:
        e[c] = ((ko[c] / ko.actual - 1) * 100).round(0)
    print(e.to_string(index=False))
    print("\n=== 지표 ===")
    print(met.round(1).pivot(index="model", columns="subset", values="MAPE%").loc[cols].to_string())
    traj = ko[["origin", "S1", "S2_t", "tau_star"] + [c for c in d.columns if c.startswith(("ntr_", "tau_lin", "tau_iso", "beta_"))]]
    traj.to_csv(OUT / f"lq_trajectory{tag}.csv", index=False, float_format="%.3f")
    sens = sensitivity(d.drop(columns=[c for c in d.columns if c.startswith(("tau_lin", "tau_iso", "ntr_", "beta_", "LQ"))]), freq)
    sens.to_csv(OUT / f"lq_sensitivity{tag}.csv", index=False, float_format="%.2f")
    print("\n=== 민감도 (LQh, k × n_min) ===")
    print(sens.round(1).to_string(index=False))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "M")
