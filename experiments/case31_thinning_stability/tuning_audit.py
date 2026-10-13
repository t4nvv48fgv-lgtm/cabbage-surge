"""Case 31 부속 — 메인 모형의 튜닝 요소 점검(2026-10-13): 요소별 기여·영향 원점·집중도·민감도. 읽기 전용, 화면 출력만.

Usage: python experiments/case31_thinning_stability/tuning_audit.py > experiments/case31_thinning_stability/outputs/tuning_audit.txt
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import gate_model as gm  # noqa: E402
from monthly_error_table import past_column  # noqa: E402

pd.set_option("display.width", 250)


def ape(d, c):
    return (d[c] / d.actual - 1).abs() * 100


def mape(d, c, a=2021, b=2025):
    k = d[d.origin.dt.year.between(a, b)]
    return ape(k, c).mean()


m = pd.read_csv(ROOT / "outputs" / "gate_predictions.csv", parse_dates=["origin"], encoding="utf-8-sig")
w = pd.read_csv(ROOT / "outputs" / "gate_predictions_W.csv", parse_dates=["origin"], encoding="utf-8-sig")
m["v3"] = past_column("b5a34ee", "final").reindex(m.origin).values
w["v3"] = past_column("b5a34ee", "final", "_W").reindex(w.origin).values

print("=== A. 단계별 MAPE ===")
cols = ["price_only_ridge", "an_sparse_all", "head", "gate_hgb", "gate_wr", "gate_beta", "final_base", "v3", "final_prod_only", "final"]
rows = []
for c in cols:
    rows.append({"열": c, "M 21-25": mape(m, c), "M 21-26": mape(m, c, 2021, 2026), "M 16-20": mape(m, c, 2016, 2020),
                 "W 21-25": mape(w, c), "W 16-20": mape(w, c, 2016, 2020)})
print(pd.DataFrame(rows).round(2).to_string(index=False))


def effect(d, before, after, label, a=2021, b=2025):
    k = d[d.origin.dt.year.between(a, b)].copy()
    ch = k[(k[before] - k[after]).abs() > 0.5]
    eb, ea = ape(ch, before), ape(ch, after)
    gain_each = (eb - ea) / len(k)                      # 전체 MAPE 에 대한 원점별 기여(%p)
    tot = gain_each.sum()
    by_year = gain_each.groupby(ch.origin.dt.year).sum().round(2).to_dict()
    top = gain_each.sort_values(ascending=False)
    top1 = f"{ch.loc[top.index[0], 'origin'].date()} {top.iloc[0]:+.2f}" if len(ch) else "-"
    print(f"[{label}] 바뀐 원점 {len(ch)}/{len(k)}, 좋아짐 {(ea < eb).sum()} 나빠짐 {(ea > eb).sum()}, 전체 MAPE 기여 {tot:+.2f}%p, 최대 기여 원점 {top1}, 해별 {by_year}")
    return ch


print("\n=== B. 요소별 영향(월말 2021~2025) ===")
effect(m, "head", "gate_hgb", "게이트(대표→평시 HGB)")
effect(m, "gate_hgb", "gate_wr", "겨울 분기")
effect(m, "gate_hgb", "gate_beta", "잔차 β(여름 경보+가을)")
s = m[m.summer == 1]; f = m[m.summer != 1]
effect(s, "gate_hgb", "gate_beta", "  └ 여름 경보 β(여름 원점 20개 기준)")
effect(f, "gate_hgb", "gate_beta", "  └ 가을 β(비여름 원점 40개 기준)")
effect(m, "final_base", "final", "9월 고온 층")
effect(m, "v3", "final_prod_only", "고랭지 기상 출하기 한정(4차)")
effect(m, "final_prod_only", "final", "출하량 전망(5차)")
effect(m, "final_prod_only", "final", "출하량 전망(5차) 2021~2026", 2021, 2026)
print("\n=== B'. 주간 2021~2025 ===")
effect(w, "head", "gate_hgb", "게이트")
effect(w, "gate_hgb", "gate_wr", "겨울 분기")
effect(w, "gate_hgb", "gate_beta", "잔차 β")
effect(w, "v3", "final_prod_only", "고랭지 기상 출하기 한정(+게이트 호 수정)")
effect(w, "final_prod_only", "final", "출하량 전망(5차)")
print("\n=== B''. 월말 2016~2020 (신호 없는 구간에서 확인 가능한 요소) ===")
effect(m, "gate_hgb", "gate_wr", "겨울 분기", 2016, 2020)
effect(m, "v3", "final_prod_only", "고랭지 기상 출하기 한정", 2016, 2020)
effect(m, "final_base", "final", "9월 고온 층", 2016, 2020)

print("\n=== C. 9월 고온 층 상세 ===")
h = m[m.origin.dt.month == 9][["origin", "actual", "final_base", "heat_H", "heat_F", "heat_gamma_pct", "heat_pct", "final"]].copy()
h["base%"] = (h.final_base / h.actual - 1) * 100; h["final%"] = (h.final / h.actual - 1) * 100
h["nocap%"] = (h.final_base * np.exp(h.heat_gamma_pct.fillna(0) / 100 * h.heat_H.fillna(0)) / h.actual - 1) * 100
print(h.round(1).to_string(index=False))

print("\n=== D. 민감도: 상수 바꿔 다시 조립(월말) ===")
base_run = gm.run("M")


def rerun(label, **kw):
    old = {k: getattr(gm, k) for k in kw}
    for k, v in kw.items():
        setattr(gm, k, v)
    try:
        d = gm.run("M")
    finally:
        for k, v in old.items():
            setattr(gm, k, v)
    e24 = (d[d.origin == "2024-08-31"].final.iloc[0] / d[d.origin == "2024-08-31"].actual.iloc[0] - 1) * 100
    e25 = (d[d.origin == "2025-08-31"].final.iloc[0] / d[d.origin == "2025-08-31"].actual.iloc[0] - 1) * 100
    n30 = int((ape(d[d.origin.dt.year.between(2021, 2025)], "final") > 30).sum())
    print(f"{label:38s} 21-25 {mape(d, 'final'):6.2f}  21-26 {mape(d, 'final', 2021, 2026):6.2f}  16-20 {mape(d, 'final', 2016, 2020):6.2f}  2024-09 {e24:+5.1f}  2025-09 {e25:+5.1f}  >30%: {n30}")
    return d


rerun("현행")
for b in (0.0, -0.283, -0.849, -1.132):
    rerun(f"여름 경보 β {b}", BETA_SUMMER_ALERT=b)
for b in (0.0, -0.108, -0.432, -1.0):
    rerun(f"가을 β {b}", BETA_FALL=b)
rerun("겨울 대상월 1~3", WINTER_MONTHS=(1, 2, 3))
rerun("겨울 대상월 12~3", WINTER_MONTHS=(12, 1, 2, 3))
rerun("겨울 대상월 12~4", WINTER_MONTHS=(12, 1, 2, 3, 4))
rerun("평시 = wx_hgb(여름특화 아님)", CALM="wx_hgb")
rerun("평시 = 가격만 ridge", CALM="price_only_ridge", CALM_REF="naive_last28")
rerun("평시 = wx_ridge", CALM="wx_ridge")

print("\n=== D'. 조립 변형(월말) ===")

def show(label, d):
    e = lambda o: (d[d.origin == o].final.iloc[0] / d[d.origin == o].actual.iloc[0] - 1) * 100
    n30 = int((ape(d[d.origin.dt.year.between(2021, 2025)], "final") > 30).sum())
    print(f"{label:38s} 21-25 {mape(d, 'final'):6.2f}  21-26 {mape(d, 'final', 2021, 2026):6.2f}  16-20 {mape(d, 'final', 2016, 2020):6.2f}  2022-09 {e('2022-08-31'):+5.1f}  2024-09 {e('2024-08-31'):+5.1f}  2025-09 {e('2025-08-31'):+5.1f}  2026-09 {e('2026-08-31'):+5.1f}  >30%: {n30}")


issues, fall = gm.load_issues(), gm.load_fall_issues()
d = gm.run("M")
show("현행", d)
d2 = d.copy(); d2["head"] = d2["an_sparse_ship"]
show("대표를 여름 원점도 an_sparse_ship", gm.assemble(d2, d2.gate_calm == 1, issues, fall, True))
d3 = d.copy(); d3["head"] = d3["an_sparse_all"]
show("대표를 1년 내내 an_sparse_all", gm.assemble(d3, d3.gate_calm == 1, issues, fall, True))
show("게이트 없음(항상 대표)", gm.assemble(d, pd.Series(False, index=d.index), issues, fall, True))
calm_all = (d.summer == 1) & d.S1.notna()
show("게이트 없음(여름 신호 있으면 항상 평시)", gm.assemble(d, calm_all, issues, fall, True))
for thr in (-5.0, -2.0, 2.0, 5.0):
    c = d.S1.notna() & ((d.S1 > thr) | (d.ship_q > 0))
    show(f"게이트 기준값 {thr:+.0f}", gm.assemble(d, c, issues, fall, True))
old = gm.S_CAP; gm.S_CAP = 10.0; show("S 절단 ±10", gm.assemble(d, d.gate_calm == 1, issues, fall, True)); gm.S_CAP = old
x = d.copy(); x["final"] = x.final_base; show("9월 고온 층 없음", x)

print("\n=== E. 게이트 기준값: 월말 여름 원점의 S1 분포 ===")
sm = d[(d.summer == 1) & d.S1.notna()]
k = sm[sm.origin.dt.year <= 2025]
print("2021~2025 양수 최소", k[k.S1 > 0].S1.min(), "음수 최대", k[k.S1 <= 0].S1.max())
print(sm[sm.origin.dt.year == 2026][["origin", "S1", "ship_q", "gate_calm"]].to_string(index=False))
w = pd.read_csv(ROOT / "outputs" / "gate_predictions_W.csv", parse_dates=["origin"], encoding="utf-8-sig")
sw = w[(w.summer == 1) & w.S1.notna() & (w.origin.dt.year <= 2025)]
print("주간 2021~2025 양수 최소", sw[sw.S1 > 0].S1.min(), "음수 최대", sw[sw.S1 <= 0].S1.max(), "|S1|<4 원점 수", int((sw.S1.abs() < 4).sum()), sorted(sw[sw.S1.abs() < 4].S1.unique()))
