"""6차 편입(2026-10-14, 평시 모형을 학습 줄 솎기 시작점 0·1·2 평균으로)의 수치 검산.

docs/연구노트_2026-10-14_평시평균편입.md 에 적은 수치를 산출물에서 다시 계산해 기대값과 견준다(모형을 다시 학습하지 않는다. 몇 초).
  현재 값    outputs/gate_predictions{,_W}.csv, gate_metrics{,_W}.csv, forecast_live.csv, monthly_error_by_version_avg.csv
  종전 값    5차 편입 커밋 7873051 의 outputs/gate_predictions{,_W}.csv (git 에서 읽음)
  독립 계산  experiments/case31_thinning_stability/outputs/thin_predictions.csv (편입 전에 케이스 스크립트가 따로 계산한 시작점별 값)

모형 재학습까지 확인하려면: python src/gate_model.py M  (약 1분; 시작점 0 이 backtest 산출물과 다르면 assert 로 멈춘다) 뒤 git diff outputs/

Usage: python tools/verify_stage6.py      (전부 통과하면 종료 코드 0)
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))
import gate_model as gm  # noqa: E402

PREV_COMMIT = "7873051"
RESULTS: list[tuple[str, str, str, bool]] = []


def check(name: str, got, want, tol: float = 0.0) -> None:
    ok = (got == want) if tol == 0.0 else abs(float(got) - float(want)) <= tol
    RESULTS.append((name, f"{got:.2f}" if isinstance(got, float) else str(got), str(want), bool(ok)))


def load(tag: str = "", commit: str | None = None) -> pd.DataFrame:
    if commit is None:
        return pd.read_csv(OUT / f"gate_predictions{tag}.csv", parse_dates=["origin"], encoding="utf-8-sig")
    raw = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:outputs/gate_predictions{tag}.csv"], capture_output=True, check=True).stdout
    return pd.read_csv(io.BytesIO(raw), parse_dates=["origin"], encoding="utf-8-sig")


def mape(d: pd.DataFrame, col: str, mask) -> float:
    return float(((d[col] / d.actual - 1).abs() * 100)[mask].mean())


def err(d: pd.DataFrame, origin: str, col: str = "final") -> float:
    r = d[d.origin == origin].iloc[0]
    return float((r[col] / r.actual - 1) * 100)


m, w = load(), load("_W")
m0, w0 = load("", PREV_COMMIT), load("_W", PREV_COMMIT)
y, yw = m.origin.dt.year, w.origin.dt.year

# 1. 성적 (현재)
check("월말 2021~2025 MAPE final", mape(m, "final", y.between(2021, 2025)), 11.39, 0.005)
check("월말 2021~2026 MAPE final", mape(m, "final", y.between(2021, 2026)), 10.80, 0.005)
check("월말 2016~2020 MAPE final", mape(m, "final", y.between(2016, 2020)), 17.37, 0.005)
check("월말 평시 원점 2021~2026 MAPE final", mape(m, "final", y.between(2021, 2026) & (m.gate_calm == 1)), 9.26, 0.005)
check("월말 2021~2025 MAPE 가격만", mape(m, "price_only_ridge", y.between(2021, 2025)), 15.10, 0.005)
check("월말 2021~2025 MAE final", float((m.final - m.actual).abs()[y.between(2021, 2025)].mean()), 1133.3, 0.05)
check("주간 2021~2025 MAPE final", mape(w, "final", yw.between(2021, 2025)), 15.02, 0.005)
check("주간 2021~2026 MAPE final", mape(w, "final", yw.between(2021, 2026)), 14.64, 0.005)
# 2. 성적 (종전, 5차 편입 커밋)
check("종전 월말 2021~2025", mape(m0, "final", y.between(2021, 2025)), 11.38, 0.005)
check("종전 월말 2021~2026", mape(m0, "final", y.between(2021, 2026)), 10.72, 0.005)
check("종전 월말 평시 원점", mape(m0, "final", y.between(2021, 2026) & (m0.gate_calm == 1)), 9.09, 0.005)
check("종전 주간 2021~2025", mape(w0, "final", yw.between(2021, 2025)), 15.15, 0.005)
check("종전 주간 2021~2026", mape(w0, "final", yw.between(2021, 2026)), 14.75, 0.005)
# 3. 범위
check("9월 고온 층 제외(final_base) 2021~2025", mape(m, "final_base", y.between(2021, 2025)), 11.92, 0.005)
nogate = gm.assemble(m, m.gate_calm == -1, gm.load_issues(), gm.load_fall_issues(), monthly=True)
check("게이트 제외(항상 대표 모형) 2021~2025", mape(nogate, "final", y.between(2021, 2025)), 13.76, 0.005)
# 4. 바뀐 원점
chg = (m.final - m0.final).abs() > 0.11
check("바뀐 월말 원점 수", int(chg.sum()), 15)
check("바뀐 월말 원점 중 평시", int((chg & (m.gate_calm == 1)).sum()), 14)
check("바뀐 경보 원점은 2024-09-30 뿐", [str(o.date()) for o in m.origin[chg & (m.gate_calm == 0)]], ["2024-09-30"])
check("평시 판정은 그대로", bool((m.gate_calm == m0.gate_calm).all()), True)
for col in ("head", "winter_ridge", "price_only_ridge", "wx_hgb_summer", "an_sparse_all", "an_sparse_ship"):
    check(f"열 {col} 불변", bool(((m[col] - m0[col]).abs().fillna(0) <= 0.11).all()), True)
e1, e0 = (m.final / m.actual - 1).abs()[chg], (m0.final / m0.actual - 1).abs()[chg]
check("바뀐 원점 중 좋아진 수", int((e1 < e0).sum()), 5)
check("바뀐 원점 중 나빠진 수", int((e1 > e0).sum()), 10)
check("바뀐 주간 원점 수", int(((w.final - w0.final).abs() > 0.11).sum()), 65)
# 5. 평시 모형 열의 정의
check("gate_hgb = 평시면 calm_avg, 아니면 head", bool(np.allclose(m.gate_hgb, np.where(m.gate_calm == 1, m.calm_avg, m["head"]), atol=0.11)), True)
c = pd.read_csv(ROOT / "experiments" / "case31_thinning_stability" / "outputs" / "thin_predictions.csv", parse_dates=["origin"], encoding="utf-8-sig")
k = m.merge(c[["origin", "hgb_0", "hgb_1", "hgb_2"]], on="origin")
k = k[k.hgb_0.notna()]
gm3 = np.exp(np.log(k[["hgb_0", "hgb_1", "hgb_2"]]).mean(axis=1))
check("case31 시작점별 값이 있는 원점 수", len(k), 23)
check("calm_avg = 세 시작점 로그 평균(case31 독립 계산과의 최대 차, 원)", float((gm3 - k.calm_avg).abs().max()), 0.0, 0.11)
check("wx_hgb_summer = 시작점 0(최대 차, 원)", float((k.hgb_0 - k.wx_hgb_summer).abs().max()), 0.0, 0.11)
sd = np.log(k.loc[k.gate_calm == 1, ["hgb_0", "hgb_1", "hgb_2"]]).std(axis=1, ddof=0) * 100
check("평시 원점의 시작점 사이 표준편차 평균(%)", float(sd.mean()), 4.4, 0.05)
check("평시 원점의 시작점 사이 표준편차 최대(%)", float(sd.max()), 10.3, 0.05)
# 6. 달별 값
for o, want, old in (("2021-08-31", -4.2, 2.2), ("2022-08-31", -10.8, -10.8), ("2023-08-31", 0.0, -3.0), ("2024-08-31", -6.1, -6.1),
                     ("2025-08-31", 19.6, 15.2), ("2026-08-31", -3.8, -3.8), ("2023-06-30", 21.6, 33.5), ("2024-09-30", 2.6, 0.0)):
    check(f"{o} 오차율(%) 현재", err(m, o), want, 0.05)
    check(f"{o} 오차율(%) 종전", err(m0, o), old, 0.05)
e = (m.final / m.actual - 1).abs() * 100
tgt = (m.origin + pd.Timedelta(days=15)).dt.to_period("M").astype(str)
check("30% 초과(원점 연도 2021~2025)", int((e[y.between(2021, 2025)] > 30).sum()), 3)
check("30% 초과(대상월 2021-01 이후)", sorted(tgt[(tgt >= "2021-01") & (e > 30)]), ["2021-01", "2022-08", "2022-11", "2023-12"])
# 7. 9월 고온 층·실시간 전망
r = m[m.origin == "2024-09-30"].iloc[0]
check("2024-09-30 γ(%/일) 현재", float(r.heat_gamma_pct), 2.9, 0.05)
check("2024-09-30 γ(%/일) 종전", float(m0[m0.origin == "2024-09-30"].heat_gamma_pct.iloc[0]), 2.8, 0.05)
fl = pd.read_csv(OUT / "forecast_live.csv", parse_dates=["origin"], encoding="utf-8-sig")
r = fl[(fl.origin == "2026-09-30") & (fl.rule == "M")].iloc[0]
check("2026-09-30 final_base(원)", float(r.final_base), 7871.58, 0.01)
check("2026-09-30 final(원)", float(r.final), 8778.35, 0.01)
check("2026-09-30 γ(%/일)", float(r.heat_gamma_pct), 2.80, 0.005)
check("2026-09-30 상태", r.state, "경보")
# 8. 대상월별 표
a = pd.read_csv(OUT / "monthly_error_by_version_avg.csv", encoding="utf-8-sig").set_index("대상월")
check("대상월 69개 평균 5차", float(a.loc["전체", "5차"]), 11.0, 0.05)
check("대상월 69개 평균 6차", float(a.loc["전체", "6차"]), 11.1, 0.05)

width = max(len(n) for n, *_ in RESULTS)
for name, got, want, ok in RESULTS:
    print(f"{'통과' if ok else '실패'}  {name:<{width}}  계산 {got}  기대 {want}")
bad = [n for n, _, _, ok in RESULTS if not ok]
print(f"\n{len(RESULTS) - len(bad)} / {len(RESULTS)} 통과" + (f"  — 실패: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
