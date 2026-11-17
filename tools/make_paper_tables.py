"""원고(manuscript/main.md)의 표 2~5 값을 outputs/*.csv 에서 계산해 마크다운으로 찍는다. 원고의 표는 이 출력과 같아야 한다.

결합 모형 = gate_predictions 의 final (계절편차 모형 + 겨울 전용 모형 + 모형 전환 + 수준 보정 + 9월 고온 보정). 2026-11-17 사용자 결정 "원고에 9월 고온 층 넣어줘".
단계별 값은 gate_model.assemble 을 관측 전망정보 없이/일부만으로 다시 불러 계산한다.

Usage: python tools/make_paper_tables.py   (원고에 붙일 표와, 본문에 쓰는 낱수치 목록을 출력)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import gate_model as gm  # noqa: E402

OUT = ROOT / "outputs"
MODEL = "final"
SHIP = {1.0: "증가", 0.0: "비슷", -1.0: "감소"}


def load(tag=""):
    g = pd.read_csv(OUT / f"gate_predictions{tag}.csv", parse_dates=["origin"], encoding="utf-8-sig")
    b = pd.read_csv(OUT / f"backtest_predictions{tag}.csv", parse_dates=["origin"])
    return g.merge(b[["origin", "mstl_arima", "mstl_ets", "wx_ridge", "wx_hgb"]], on="origin")


def stages(d: pd.DataFrame, monthly: bool) -> pd.DataFrame:
    """관측 전망정보를 단계별로 얹은 예측값."""
    issues, fall = gm.load_issues(), gm.load_fall_issues()
    calm, none = d.gate_calm == 1, d.gate_calm == -1
    bs, bf = gm.BETA_SUMMER_ALERT, gm.BETA_FALL
    gm.BETA_SUMMER_ALERT = gm.BETA_FALL = 0.0
    try:
        winter = gm.assemble(d, none, issues, fall, monthly).final_base
        gate = gm.assemble(d, calm, issues, fall, monthly).final_base
    finally:
        gm.BETA_SUMMER_ALERT, gm.BETA_FALL = bs, bf
    return pd.DataFrame({"가격만 ridge": d.price_only_ridge, "Chronos-2 제로샷": d.chronos2, "MSTL+AutoETS": d.mstl_ets,
                         "계절편차 모형": d["head"], "+ 겨울 전용 모형": winter, "+ 모형 전환": gate, "+ 수준 보정": d["final_base"],
                         "+ 9월 고온 보정(결합 모형)": d[MODEL]})


def pct(v, signed=True):
    r = int(np.floor(abs(v) + 0.5)) * (1 if v >= 0 else -1)
    return "0" if r == 0 else (f"{r:+d}".replace("-", "−") if signed else f"{r}")


def main():
    m, w = load(), load("_W")
    y, yw = m.origin.dt.year, w.origin.dt.year

    print("### 표 2. 기초 통계량")
    rows = [("28일 평균가격, 월말 원점 전체", m.actual), ("28일 평균가격, 6~9월 원점", m.actual[m.summer == 1]),
            ("28일 평균가격, 8월 말 원점", m.actual[m.origin.dt.month == 8]), ("여름배추 생산량 전망 전년비", m.S1.dropna())]
    print("| 변수 | n | 평균 | 중앙값 | 최소 | 최대 | 표준편차 | 왜도 | 첨도 |\n|:---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, x in rows:
        f = (lambda v: f"{v:,.0f}") if x.mean() > 100 else (lambda v: f"{v:.1f}".replace("-", "−"))
        print(f"| {name} | {len(x)} | {f(x.mean())} | {f(x.median())} | {f(x.min())} | {f(x.max())} | {f(x.std())} | "
              f"{st.skew(x, bias=False):.2f} | {st.kurtosis(x, bias=False):.2f} |".replace("-", "−"))

    print("\n### 표 3. 8월 말 원점의 9월 가격 예측 (과소폭·과대폭 %, 2021~2025 평균 오차율)")
    models = [("가격만 ridge", "price_only_ridge", "가격"), ("Chronos-2 제로샷", "chronos2", "가격(월평균 계열)"), ("MSTL+AutoARIMA", "mstl_arima", "가격(월평균 계열)"),
              ("MSTL+AutoETS", "mstl_ets", "가격(월평균 계열)"), ("기상 ridge", "wx_ridge", "가격·반입량·기상"), ("기상 HGB", "wx_hgb", "가격·반입량·기상"),
              ("여름특화 HGB", "calm_avg", "가격·반입량·기상"), ("계절편차 모형", "head", "평년 대비 가격 편차·반입량·고랭지 기상"), ("결합 모형", MODEL, "계절편차 모형·여름특화 HGB·농업관측 전망")]
    yrs = (2020, 2022, 2024, 2025, 2026)
    print("| 모형 | 쓰는 정보 | " + " | ".join(f"{yy}년" for yy in yrs) + " | 평균 오차율 |\n|:---|:---|" + "---:|" * (len(yrs) + 1))
    for name, col, info in models:
        e = [pct((m.loc[m.origin == f"{yy}-08-31", col].iloc[0] / m.loc[m.origin == f"{yy}-08-31", "actual"].iloc[0] - 1) * 100) for yy in yrs]
        mp = ((m[col] / m.actual - 1).abs() * 100)[y.between(2021, 2025)].mean()
        print(f"| {name} | {info} | " + " | ".join(e) + f" | {mp:.1f} |")
    print("실제(원/10kg):", {yy: round(float(m.loc[m.origin == f'{yy}-08-31', 'actual'].iloc[0])) for yy in yrs})

    print("\n### 표 4. 8월 말 원점의 연도별 예측")
    print("| 연도 | 생산량 전망 전년비(%) | 출하량 전망 | 적용 모형 | 실제 가격(원/10kg) | 가격만 ridge | 계절편차 모형 | 여름특화 HGB | 결합 모형 |\n|---|---:|---|---|---:|---:|---:|---:|---:|")
    for yy in range(2021, 2027):
        r = m[m.origin == f"{yy}-08-31"].iloc[0]
        e = lambda c: pct((r[c] / r.actual - 1) * 100)
        print(f"| {yy} | {r.S1:+.1f} | {SHIP[r.ship_q]} | {'여름특화 HGB' if r.gate_calm == 1 else '계절편차 모형'} | {r.actual:,.0f} | {e('price_only_ridge')} | {e('an_sparse_all')} | {e('calm_avg')} | {e(MODEL)} |".replace("-", "−"))

    print("\n### 표 5. 단계별 예측 성과")
    S, SW = stages(m, True), stages(w, False)
    print("| 모형 | 평균 오차율(%) | 평균 절대오차(원/10kg) | 6~9월 원점 평균 오차율(%) | 오차 30% 초과 원점 수 | 주간 원점 평균 오차율(%) | 2016~2020년 평균 오차율(%) |\n|:---|---:|---:|---:|---:|---:|---:|")
    for c in S.columns:
        e = (S[c] / m.actual - 1).abs() * 100
        ew = (SW[c] / w.actual - 1).abs() * 100
        mk = y.between(2021, 2025)
        print(f"| {c} | {e[mk].mean():.1f} | {(S[c] - m.actual).abs()[mk].mean():,.0f} | {e[mk & (m.summer == 1)].mean():.1f} | {int((e[mk] > 30).sum())} | "
              f"{ew[yw.between(2021, 2025)].mean():.1f} | {e[y.between(2016, 2020)].mean():.1f} |")

    print("\n### 본문 낱수치")
    e = (m[MODEL] / m.actual - 1) * 100
    tm = (m.origin + pd.Timedelta(days=15))
    s = tm >= "2021-01-01"
    bym = e[s].abs().groupby(tm[s].dt.month).mean()
    print("대상월별 평균 오차율(대상월 2021-01~2026-09, n=%d): " % int(s.sum()), {int(k): round(v, 1) for k, v in bym.items()}, "전체", round(e[s].abs().mean(), 1))
    print("30% 초과 대상월:", [(str(t.date())[:7], pct(v)) for t, v in zip(tm[s & (e.abs() > 30)], e[s & (e.abs() > 30)])])
    print("9월 말 원점 → 10월:", {int(o.year): pct(v) for o, v in zip(m.origin, e) if o.month == 9 and o.year >= 2020})
    print("월말 2021~2026 평균 오차율:", round(e.abs()[y.between(2021, 2026)].mean(), 1), "/ 주간 2021~2026:", round(((w[MODEL] / w.actual - 1).abs() * 100)[yw.between(2021, 2026)].mean(), 1))
    print("원점 수: 월말", len(m), str(m.origin.min().date()), "~", str(m.origin.max().date()), "/ 주간", len(w), "/ 6~9월 월말 원점 중 증가 전망", int((m.gate_calm == 1).sum()),
          "감소 전망", int(((m.summer == 1) & m.S1.notna() & (m.gate_calm == 0)).sum()))
    wk = w[w.origin.isin(pd.to_datetime(["2024-08-18", "2024-09-01", "2024-09-22", "2024-10-13", "2024-10-20"]))]
    print("주간 2024:", [(str(o.date()), {c: pct((wk.loc[wk.origin == o, c].iloc[0] / a - 1) * 100) for c in ("price_only_ridge", "chronos2", "mstl_ets", "an_sparse_all", MODEL)}) for o, a in zip(wk.origin, wk.actual)])
    r = m[m.origin == "2024-08-31"].iloc[0]
    print("2024-08-31 가격만 ridge 예측가격:", round(float(r.price_only_ridge)), "/ 계절편차 모형:", round(float(r["head"])), "/ 결합 모형:", round(float(r[MODEL])))
    # 모형 선택에 쓰는 호를 발간 시점(호 월 5일) 뒤로 한 호 늦춘 경우 — 수준 보정의 전망은 그대로
    iss, ship = gm.load_issues(), gm.load_shipment()
    late = []
    for o, sm in zip(m.origin, m.summer):
        x = iss[(iss.year == o.year) & (iss.pub_date <= o)] if sm == 1 else iss.iloc[0:0]
        if x.empty:
            late.append(False)
            continue
        rr = x.iloc[-1]
        s1 = rr.prod_yoy if pd.notna(rr.prod_yoy) else rr.area_yoy
        late.append(bool(s1 > 0 or ship.get(f"{int(rr.year)}-{int(rr.issue_month):02d}", np.nan) > 0))
    late = pd.Series(late, index=m.index)
    diff = m[(late != (m.gate_calm == 1)) & (m.summer == 1) & y.between(2021, 2025)]
    lp = gm.assemble(m, late, iss, gm.load_fall_issues(), True)[MODEL]
    print("한 호 늦춘 모형 선택: 2021~2025 6~9월 월말 원점 중 바뀌는 수", len(diff), [str(o.date()) for o in diff.origin],
          "평균 오차율", round(((lp / m.actual - 1).abs() * 100)[y.between(2021, 2025)].mean(), 2))
    h = m[(m.origin.dt.month == 9) & (m.tmonth == 10)]
    print("9월 고온 보정(9월 말 원점):", [(int(o.year), f"H {hh:.1f}", f"F {ff}", f"γ {gg:.1f}%/일", f"보정 {pp:+.1f}%", f"보정 전 {pct((fb / a - 1) * 100)}", f"후 {pct((fn / a - 1) * 100)}")
                                for o, hh, ff, gg, pp, fb, fn, a in zip(h.origin, h.heat_H, h.heat_F, h.heat_gamma_pct, h.heat_pct, h.final_base, h.final, h.actual)])
    prod = gm.assemble(m, (m.S1 > 0) & m.S1.notna(), gm.load_issues(), gm.load_fall_issues(), True)[MODEL]
    print("출하량 전망 규칙 제외(생산 전망만): 2021~2025", round(((prod / m.actual - 1).abs() * 100)[y.between(2021, 2025)].mean(), 2), "2021~2026", round(((prod / m.actual - 1).abs() * 100)[y.between(2021, 2026)].mean(), 2))


if __name__ == "__main__":
    main()
