"""Case 23 — 자문회의·월보에 담긴 전문가 정보가 모형 오차를 줄이는가: 월보 가격 전망치와의 결합 시험.

사용자 지시(2026-10-13): 30% 초과 달의 원인을 월보·자문회의 자료로 확인한 뒤 "이 정보들로 개선이 되는지 확인".
원인 조사 결과(CASE.md): 출하 시기 이동(조기 출하·지연 뒤 몰림), 작황 판단, 저장·비축 물량, 수요가 되풀이되는 원인이었고,
그 정보는 숫자 열로 쌓여 있지 않다. 호마다 빠짐없이 숫자로 남아 있는 것은 그 정보를 종합한 **월보 가격 전망치 K** 하나다.
그래서 K 를 전문가 정보의 요약으로 보고, 최종 모형(final)과 결합했을 때 오차가 줄어드는지 본다.

주의: case14 에서 사용자는 월보 가격 전망을 모형 입력으로 쓰지 않기로 했다(순환). 이 케이스는 "정보가 값이 있는가"를 재는 평가이며,
채택은 그 결정을 바꿀지에 대한 사용자 판단이다.

K: data/local/yupgeun/(KREI)(21_26)_배추_기초자료_관측월보기준.xlsx 1)절(3~12월호, 원/10kg; 관측월보 우선, 일부는 자문원고 값).
   2016년 5~8월은 월보 본문에서 채움(8,500·5,000·6,500·9,000). 로컬 전용 파일이라 달별 값은 저장소에 쓰지 않는다(data/local/case23/).
가용 시점: 규약 B(m월호는 (m−1)월 말 원점에서 가용). 대상월 = 원점 + 15일의 달. 1·2월 대상은 K 가 없어 final 그대로.
주의: K 의 대상은 달력 월평균, 모형의 실제는 원점 뒤 28일 평균이라 기준이 조금 다르다.

규칙(모두 log 가격에서 결합: pred = final × (K/final)^b):
  K       b = 1 (월보 전망 그대로)
  avg     b = 0.5 (기하평균)
  w_fix   b = 2016~2020 에서 log(실제/final) ~ b·log(K/final) 로 추정(0~1 로 절단) → 2021~ 에 고정 적용
  w_exp   b 를 그 해 이전 해들로 매해 다시 추정(확장 창, 최소 2개 해)
  dis     |log(final/K)| > τ 일 때만 b 로 당김. (τ, b) 는 2016~2020 MAPE 최소로 선택
  sea_sel 대상 작형(겨울 3~4·봄 5~6·여름 7~10·가을 11~12월)별로 2016~2020 에서 avg 가 final 보다 나은 작형에만 avg
  spring  봄 대상월(5~6월)에만 avg (sea_sel 에서 두 창 모두 이득이 난 조각만 떼어 본 것 — 결과를 본 뒤 정의)

Usage: python experiments/case23_expert_info/expert_blend.py
Outputs: outputs/blend_metrics.csv, blend_season.csv, blend_picks.csv
         data/local/case23/blend_predictions.csv, blend_big_months.csv (달별 값은 미추적)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
OUT = CASE_DIR / "outputs"
LOCAL = ROOT / "data" / "local" / "case23"
KREI_XLSX = ROOT / "data" / "local" / "yupgeun" / "(KREI)(21_26)_배추_기초자료_관측월보기준.xlsx"
FILL_2016 = {5: 8500.0, 6: 5000.0, 7: 6500.0, 8: 9000.0}     # 월보 2016년 5~8월호 본문
WINDOWS = {"2016~2020": (2016, 2020), "2021~2025": (2021, 2025), "2021~2026": (2021, 2026)}
TAUS, BS = [0.1, 0.15, 0.2, 0.3, 0.4], [0.25, 0.5, 0.75, 1.0]


def load() -> pd.DataFrame:
    g = pd.read_csv(ROOT / "outputs" / "gate_predictions.csv", parse_dates=["origin"], encoding="utf-8-sig")
    t = g.origin + pd.Timedelta(days=15)
    g["ty"], g["tmo"], g["oy"] = t.dt.year, t.dt.month, g.origin.dt.year
    x = pd.read_excel(KREI_XLSX, header=None)
    k = [(int(x.iloc[i, 0]), m, float(x.iloc[i, j])) for i in range(4, 15) for j, m in enumerate(range(3, 13), start=1) if pd.notna(x.iloc[i, j])]
    k += [(2016, m, v) for m, v in FILL_2016.items()]
    k = pd.DataFrame(k, columns=["ty", "tmo", "K"]).drop_duplicates(["ty", "tmo"])
    return g.merge(k, on=["ty", "tmo"], how="left")


def blend(d: pd.DataFrame, b) -> np.ndarray:
    r = np.log(d.K / d.final)
    return np.where(d.K.notna(), d.final * np.exp(np.asarray(b) * r.fillna(0.0)), d.final)


def est_b(s: pd.DataFrame) -> float:
    s = s[s.K.notna()]
    x, y = np.log(s.K / s.final), np.log(s.actual / s.final)
    return float(np.clip((x * y).sum() / (x ** 2).sum(), 0.0, 1.0)) if len(s) >= 10 else 0.0


def ape(s: pd.DataFrame, c: str) -> pd.Series:
    return (s[c] / s.actual - 1) * 100


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 300)
    d = load()
    design = d.oy.between(2016, 2020)
    d["K_only"] = blend(d, 1.0)
    d["avg"] = blend(d, 0.5)
    b_fix = est_b(d[design])
    d["w_fix"] = blend(d, b_fix)
    b_exp = {y: (est_b(d[d.oy < y]) if d[(d.oy < y) & d.K.notna()].oy.nunique() >= 2 else 0.0) for y in sorted(d.oy.unique())}
    d["w_exp"] = blend(d, d.oy.map(b_exp).values)
    gap = np.log(d.final / d.K).abs()
    best = None
    for tau in TAUS:
        for b in BS:
            p = blend(d, np.where(gap > tau, b, 0.0))
            m = float(((p - d.actual).abs() / d.actual)[design].mean() * 100)
            if best is None or m < best[0] - 1e-9:
                best = (m, tau, b)
    _, tau, bd = best
    d["dis"] = blend(d, np.where(gap > tau, bd, 0.0))
    sea = d.tmo.map({3: "겨울", 4: "겨울", 5: "봄", 6: "봄", 7: "여름", 8: "여름", 9: "여름", 10: "여름", 11: "가을", 12: "가을"})
    use = {}
    for k in ("겨울", "봄", "여름", "가을"):
        s = d[design & d.K.notna() & (sea == k)]
        use[k] = bool(ape(s, "avg").abs().mean() < ape(s, "final").abs().mean())
    d["sea_sel"] = np.where(sea.map(use).eq(True) & d.K.notna(), d.avg, d.final)
    d["spring"] = np.where((sea == "봄") & d.K.notna(), d.avg, d.final)
    print("sea_sel 이 avg 를 쓰는 작형(2016~2020 선택):", use)
    rules = ["final", "K_only", "avg", "w_fix", "w_exp", "dis", "sea_sel", "spring"]
    picks = pd.DataFrame([{"b_fix(2016~2020)": b_fix, "dis_tau": tau, "dis_b": bd, **{f"b_exp_{y}": v for y, v in b_exp.items()}}])
    picks.to_csv(OUT / "blend_picks.csv", index=False, float_format="%.3f")
    print("고른 값:", picks.round(3).to_dict("records")[0])

    rows = []
    for wname, (a, b) in WINDOWS.items():
        for sub, mk in (("K 있는 달(3~12월 대상)", d.K.notna()), ("전체 달", pd.Series(True, index=d.index))):
            s = d[d.oy.between(a, b) & mk]
            for r in rules:
                if r == "K_only" and sub == "전체 달":
                    continue
                e = ape(s, r)
                rows.append({"window": wname, "subset": sub, "rule": r, "n": len(s), "MAPE%": e.abs().mean(), "bias%": e.mean(),
                             "n_gt20": int((e.abs() > 20).sum()), "n_gt30": int((e.abs() > 30).sum()), "n_gt40": int((e.abs() >= 40).sum())})
    met = pd.DataFrame(rows)
    met.to_csv(OUT / "blend_metrics.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    for col in ("MAPE%", "n_gt30"):
        print(f"\n=== {col} ===")
        print(met.pivot_table(index=["subset", "window"], columns="rule", values=col)[[r for r in rules]].round(2).to_string())

    # 두 예측의 관계
    print("\n=== final 과 K 의 오차 관계 (K 있는 달) ===")
    for wname, (a, b) in WINDOWS.items():
        s = d[d.oy.between(a, b) & d.K.notna()]
        ef, ek = np.log(s.final / s.actual), np.log(s.K / s.actual)
        print(f"{wname}: n={len(s)}, 오차 상관 {np.corrcoef(ef, ek)[0, 1]:+.2f}, K 가 더 가까운 달 {int((ek.abs() < ef.abs()).sum())}, "
              f"부호가 반대인 달 {int((np.sign(ef) != np.sign(ek)).sum())}, |log(final/K)| 중앙값 {np.log(s.final / s.K).abs().median():.2f}")

    # 큰 오차 달 before/after
    big = d[(ape(d, "final").abs() > 30) | (d.K.notna() & (ape(d, "K_only").abs() > 30))].copy()
    tab = pd.DataFrame({"대상월": big.ty.astype(str) + "-" + big.tmo.astype(str).str.zfill(2), **{r: ape(big, r).round(0) for r in rules}})
    tab.loc[big.K.isna().values, "K_only"] = np.nan
    tab.to_csv(LOCAL / "blend_big_months.csv", index=False, encoding="utf-8-sig")
    print("\n=== final 또는 K 가 30% 넘게 틀린 달의 오차율(%) ===")
    print(tab.to_string(index=False))
    key = d[(d.origin.dt.month == 8) & (d.oy >= 2020)]
    print("\n=== 8월 말 원점(9월 대상) 오차율(%) ===")
    print(pd.DataFrame({"origin": key.origin.dt.date, **{r: ape(key, r).round(0) for r in rules}}).to_string(index=False))
    # 작형별
    print("\n=== 대상 작형별 MAPE% (K 있는 달) ===")
    srows = []
    for wname, (a, b) in WINDOWS.items():
        s = d[d.oy.between(a, b) & d.K.notna()]
        t = pd.DataFrame({r: ape(s, r).abs().groupby(sea[s.index]).mean() for r in rules})
        t.insert(0, "n", s.groupby(sea[s.index]).size())
        print(wname)
        print(t.round(1).to_string())
        srows.append(t.reset_index().rename(columns={"tmo": "season"}).assign(window=wname))
    pd.concat(srows).to_csv(OUT / "blend_season.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    sp = d[(sea == "봄") & d.K.notna()]
    yr = pd.DataFrame({r: ape(sp, r).abs().groupby(sp.oy).mean() for r in ("final", "K_only", "avg")}).round(0)
    print("\n=== 봄 대상월(5~6월) 해별 평균 오차율(%) ===")
    print(yr.T.to_string())
    d[["origin", "actual", "ty", "tmo", "K"] + rules].to_csv(LOCAL / "blend_predictions.csv", index=False, float_format="%.1f", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
