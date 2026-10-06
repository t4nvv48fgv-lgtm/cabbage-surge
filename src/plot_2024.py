"""Plot weekly-origin forecasts of the next-28-day mean price for summer 2024.

Reads outputs/backtest_predictions_W.csv and outputs/anomaly_predictions_W.csv.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# 한글 폰트
for cand in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

# fixed categorical order; validated with scripts/validate_palette.js (light surface)
SERIES = [
    ("actual", "실제(향후 28일 평균)", "#1f1f1f", 2.2, "-"),
    ("price_only_ridge", "가격만(ridge)", "#9467bd", 1.8, "--"),
    ("chronos2", "Chronos-2 제로샷", "#2ca02c", 1.5, ":"),
    ("mstl_ets", "MSTL+ETS", "#8c564b", 1.5, ":"),
    ("wx_ridge", "기상 ridge", "#1f77b4", 1.8, "-"),
    ("an_ridge_all", "계절편차 ridge", "#d6604d", 1.8, "-"),
]


def main() -> None:
    b = pd.read_csv(OUT / "backtest_predictions_W.csv", parse_dates=["origin"])
    a = pd.read_csv(OUT / "anomaly_predictions_W.csv", parse_dates=["origin"])
    df = b.merge(a[["origin", "an_ridge_all"]], on="origin")
    df = df[(df.origin >= "2024-06-01") & (df.origin <= "2024-11-10")]

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    for col, label, color, lw, ls in SERIES:
        ax.plot(df.origin, df[col] / 1000, color=color, lw=lw, ls=ls, label=label)
    ax.axvspan(pd.Timestamp("2024-08-31"), pd.Timestamp("2024-09-30"), color="#000", alpha=0.05, lw=0)
    ax.text(pd.Timestamp("2024-09-15"), ax.get_ylim()[1] * 0.97, "9월 대상 구간", ha="center", va="top", fontsize=8.5, color="#666")
    ax.set_title("2024년 여름 — 주간 원점별 향후 28일 평균가격 예측 (가락 상품, 천원/10kg)", fontsize=11, loc="left")
    ax.set_xlabel("예측 원점(매주 일요일)")
    ax.set_ylabel("천원/10kg")
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#bbb"); ax.spines["bottom"].set_color("#bbb")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    out = OUT / "surge_2024_weekly_tracking.png"
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
