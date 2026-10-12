"""Case 13 — KREI 월보 공개 API로 2008~2020년 호를 받아 여름배추 생산 전망 신호(S1)를 소급하고, 게이트를 2016~2020 원점에서 검증.

배경: 메인 게이트(src/gate_model.py)의 신호 S1은 2021-03호~(CABIS 판독표). "신호가 6개 해뿐"이 가장 큰 약점.
사용자 지시(2026-10-12): CABIS `data_pipeline/krei_monthly` 수집기를 이용 → 그 코드가 쓰는 KREI 공개 API(aglook.krei.re.kr montlyReport,
엽근채소 OVR0000000008)로 2008-03호~ 188개 호의 목록·게시일을 받고(outputs/krei_monthly_index_2008_2026.csv), 2008~2020년 6·7·8·9월호 PDF를
`data/local/krei_monthly/`(미추적)에 내려받아 고랭지(여름)배추 생산량 전년비를 판독했다(2008~2017 텍스트, 2018~2020 이미지는 눈으로).
판독 결과는 `krei_summer_issues_2010_2020.csv`(호·게시일·값·근거). 2012·2013 8월호는 생산량 전망이 없어 출하량 전년비를 대리값으로 썼다(basis 열).

규칙은 메인과 동일: 여름 원점(6~9월)에서 원점까지 게시된 최신호 S1 > 0 → 평시(wx_hgb_summer), 아니면 대표(an_sparse_all). 게시일은 실제 reg_date.
평가: 2016~2020 월말 여름 원점(메인 예측이 있는 구간). 2010~2015는 모형 예측이 없어 신호 vs 급등 여부 표만.

Usage: python experiments/case13_krei_backfill/backfill_gate.py
Outputs: outputs/backfill_predictions.csv, backfill_metrics.csv, signal_vs_surge_2008_2026.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
OUT = CASE_DIR / "outputs"
sys.path.insert(0, str(ROOT / "src"))
import gate_model as gm  # noqa: E402

HEAD, CALM, BASE = "an_sparse_all", "wx_hgb_summer", "price_only_ridge"


def load_issues_ext() -> pd.DataFrame:
    old = pd.read_csv(CASE_DIR / "krei_summer_issues_2010_2020.csv", parse_dates=["pub_date"])
    new = pd.read_csv(ROOT / "data" / "ext" / "krei_summer_issues.csv", encoding="utf-8-sig")
    new["pub_date"] = pd.to_datetime(new.pub_date)
    new["basis"] = "production(판독표)"
    cols = ["year", "issue_month", "pub_date", "prod_yoy", "area_yoy", "basis"]
    return pd.concat([old[cols], new[cols]]).sort_values("pub_date").reset_index(drop=True)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(ROOT / "outputs" / "gate_predictions.csv", parse_dates=["origin"], encoding="utf-8-sig")
    iss = load_issues_ext()
    sig = [gm.latest_signal(iss, o) if s == 1 else (np.nan, np.nan) for o, s in zip(d.origin, d.summer)]
    d["krei_issue_ext"] = [k for k, _ in sig]
    d["S1_ext"] = [v for _, v in sig]
    d["calm_ext"] = (d.S1_ext.notna() & (d.S1_ext > 0)).astype(int)
    d["gate_hgb_ext"] = np.where(d.calm_ext == 1, d[CALM], d[HEAD])
    d["gate_ridge_ext"] = np.where(d.calm_ext == 1, d[BASE], d[HEAD])
    # 신호 vs 급등 여부 (8월 말 원점, 9월 평균 / 8월 평균 ≥ 1.3 을 급등으로)
    price = pd.read_csv(ROOT / "data" / "raw" / "clean_daily.csv", parse_dates=["date"])
    price = price[price.price > 0]
    mth = price.groupby(price.date.dt.to_period("M")).price.mean()
    rows = []
    for y in range(2008, 2027):
        s = iss[(iss.year == y) & (iss.issue_month == 8)]
        s1 = float(s.prod_yoy.iloc[0]) if len(s) and pd.notna(s.prod_yoy.iloc[0]) else np.nan
        aug, sep = mth.get(pd.Period(f"{y}-08")), mth.get(pd.Period(f"{y}-09"))
        ratio = sep / aug if (aug and sep) else np.nan
        rows.append({"year": y, "S1_aug_issue": s1, "basis": s.basis.iloc[0] if len(s) else "", "aug_mean": aug, "sep_mean": sep,
                     "sep_over_aug": ratio, "surge(sep/aug>=1.3)": int(ratio >= 1.3) if pd.notna(ratio) else np.nan,
                     "alarm(S1<=0)": int(s1 <= 0) if pd.notna(s1) else np.nan})
    return d, pd.DataFrame(rows)


def metrics(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    y = d.origin.dt.year
    masks = {"summer_2016_2020": y.between(2016, 2020) & (d.summer == 1), "all_2016_2020": y.between(2016, 2020),
             "summer_2016_2026": y.between(2016, 2026) & (d.summer == 1), "all_2016_2025": y.between(2016, 2025),
             "summer_2021_2025": y.between(2021, 2025) & (d.summer == 1), "all_2021_2025": y.between(2021, 2025)}
    out = []
    for name, mk in masks.items():
        s = d[mk]
        for c in cols:
            e = s[c] - s.actual
            out.append({"subset": name, "model": c, "n": len(s), "MAE": e.abs().mean(), "MAPE%": (e.abs() / s.actual).mean() * 100, "bias": e.mean()})
    return pd.DataFrame(out)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    d, sv = run()
    cols = [BASE, CALM, HEAD, "gate_hgb", "gate_ridge_ext", "gate_hgb_ext"]
    d.to_csv(OUT / "backfill_predictions.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    met = metrics(d, cols)
    met.to_csv(OUT / "backfill_metrics.csv", index=False, float_format="%.2f")
    sv.to_csv(OUT / "signal_vs_surge_2008_2026.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    print("=== 8월호 신호 vs 9월 급등 (2008~2026) ===")
    print(sv.round(2).to_string(index=False))
    s = d[(d.origin.dt.year.between(2016, 2020)) & (d.summer == 1)].copy()
    for c in cols:
        s[c + "%"] = ((s[c] / s.actual - 1) * 100).round(0)
    print("\n=== 2016~2020 여름 원점 ===")
    print(s[["origin", "actual", "krei_issue_ext", "S1_ext", "calm_ext"] + [c + "%" for c in cols]].to_string(index=False))
    print("\n=== MAPE ===")
    print(met.pivot(index="model", columns="subset", values="MAPE%").loc[cols].round(1).to_string())
