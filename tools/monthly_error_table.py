"""월말 원점의 대상월별 오차율을 메인 모형 판(편입 단계)별로 나란히 놓는다.

지난 판의 예측은 그 판을 편입한 커밋의 outputs/gate_predictions.csv 를 git 에서 읽는다(재계산하지 않음).
  가격만      현재 파일의 price_only_ridge
  1차 게이트  2741e43 gate_hgb   (2026-10-12, 대표 + 생산 전망 게이트)
  2차         3569174 final      (겨울 분기 + 잔차 β)
  3차         b5a34ee final      (9월 고온 층)
  4차         7873051 final_prod_only  (고랭지 기상은 출하기에만 + 게이트 호 수정. 게이트는 생산 전망만)
  5차         7873051 final      (게이트에 출하량 전망 방향을 엮음, case30)
  6차         현재 파일의 final  (평시 모형을 학습 줄 솎기 시작점 0·1·2 평균으로, case31)

Usage: python tools/monthly_error_table.py
Outputs: outputs/monthly_error_by_version.csv        (대상월 한 줄: 실제가격, 판별 오차율 %)
         outputs/monthly_error_by_version_avg.csv    (달력 월별·전체 평균 오차율, 대상월 2021-01 이후)
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PAST = {"1차 게이트": ("2741e43", "gate_hgb"), "2차": ("3569174", "final"), "3차": ("b5a34ee", "final"),
        "4차": ("7873051", "final_prod_only"), "5차": ("7873051", "final")}
VERSIONS = ["가격만", "1차 게이트", "2차", "3차", "4차", "5차", "6차"]


def past_column(commit: str, col: str, tag: str = "") -> pd.Series:
    """그 커밋의 outputs/gate_predictions{tag}.csv 에서 열 하나(원점 색인). tag '' = 월말, '_W' = 주간."""
    raw = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:outputs/gate_predictions{tag}.csv"], capture_output=True, check=True).stdout
    d = pd.read_csv(io.BytesIO(raw), parse_dates=["origin"], encoding="utf-8-sig")
    return d.set_index("origin")[col]


def build() -> pd.DataFrame:
    d = pd.read_csv(OUT / "gate_predictions.csv", parse_dates=["origin"], encoding="utf-8-sig").set_index("origin")
    pred = pd.DataFrame({"가격만": d.price_only_ridge, "6차": d.final})
    for name, (commit, col) in PAST.items():
        pred[name] = past_column(commit, col).reindex(d.index)
    err = pred[VERSIONS].div(d.actual, axis=0).sub(1).mul(100)
    target = (d.index + pd.Timedelta(days=15)).to_period("M").astype(str)
    out = pd.DataFrame({"target_month": target, "origin": d.index.date, "actual": d.actual.values}).join(err.reset_index(drop=True))
    return out


if __name__ == "__main__":
    out = build()
    out.to_csv(OUT / "monthly_error_by_version.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    s = out[out.target_month >= "2021-01"]
    a = s[VERSIONS].abs().groupby(s.target_month.str[5:].astype(int)).mean()
    a.index = [f"{m}월" for m in a.index]
    a.loc["전체"] = s[VERSIONS].abs().mean()
    a.insert(0, "n", list(s.groupby(s.target_month.str[5:].astype(int)).size()) + [len(s)])
    a.to_csv(OUT / "monthly_error_by_version_avg.csv", index_label="대상월", float_format="%.1f", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(f"대상월 {s.target_month.min()} ~ {s.target_month.max()} ({len(s)}개)")
    print(a.round(1).to_string())
    g = s.assign(y=s.target_month.str[:4], m=s.target_month.str[5:].astype(int))
    for v in ("5차", "6차"):
        print(f"\n[{v}] 대상월별 오차율(%)")
        print(g.pivot(index="y", columns="m", values=v).round(0).to_string())
    print("\n30% 초과:", {v: int((s[v].abs() > 30).sum()) for v in VERSIONS})
    print("20% 초과:", {v: int((s[v].abs() > 20).sum()) for v in VERSIONS})
