"""원고(manuscript/main.md) 그림 생성. 수치는 outputs/*.csv 에서 직접 읽는다.

  그림 1  월평균 실제가격과 전월 말 예측(대상월 2021-01~2026-09): 실제, 결합 모형, 가격만 ridge
  그림 2  2024년 여름 주간 원점의 28일 평균가격 예측: 실제, 가격만 ridge, Chronos-2, MSTL+AutoETS, 계절편차 모형, 결합 모형
결합 모형 = gate_predictions 의 final (모형 전환 + 겨울 전용 모형 + 수준 보정 + 9월 고온 보정).
그림 속 이름은 원고 표의 모형 이름과 같게 쓴다(docs/EDIT_CHECKLIST.md §6-1). 제목은 그림에 넣지 않는다(캡션은 원고에).

Usage: python tools/make_paper_figures.py
Output: figures/out/fig1_monthly_track.png, fig2_weekly_2024.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures" / "out"
MODEL = "final"
INK, INK2, GRID = "#1f1f1f", "#55534f", "#e4e2dd"
ORANGE, GREY, BLUE, GREEN, VIOLET = "#e8622c", "#9a9893", "#2a78d6", "#1b9e77", "#7b5cd6"

fm.fontManager.addfont(r"C:\Windows\Fonts\malgun.ttf")
plt.rcParams["font.family"] = fm.FontProperties(fname=r"C:\Windows\Fonts\malgun.ttf").get_name()
plt.rcParams["axes.unicode_minus"] = False


def _axes(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#bdbab4")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig1():
    g = pd.read_csv(OUT / "gate_predictions.csv", parse_dates=["origin"], encoding="utf-8-sig")
    g = g[(g.origin + pd.Timedelta(days=15)) >= "2021-01-01"].copy()
    g["tm"] = (g.origin + pd.Timedelta(days=15)).dt.to_period("M").dt.to_timestamp()
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.plot(g.tm, g.actual / 1000, color=INK, lw=2.0, label="실제")
    ax.plot(g.tm, g[MODEL] / 1000, color=ORANGE, lw=2.0, label="결합 모형")
    ax.plot(g.tm, g.price_only_ridge / 1000, color=GREY, lw=1.5, ls=(0, (4, 2)), label="가격만 ridge")
    ax.set_ylabel("천원/10kg", fontsize=9, color=INK2)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(0, g.actual.max() / 1000 * 1.15)
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=3)
    _axes(ax)
    fig.savefig(FIG / "fig1_monthly_track.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig2():
    w = pd.read_csv(OUT / "gate_predictions_W.csv", parse_dates=["origin"], encoding="utf-8-sig")
    b = pd.read_csv(OUT / "backtest_predictions_W.csv", parse_dates=["origin"])[["origin", "mstl_ets"]]
    d = w.merge(b, on="origin")
    d = d[(d.origin >= "2024-06-01") & (d.origin <= "2024-11-10")]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.axvspan(pd.Timestamp("2024-08-31"), pd.Timestamp("2024-09-30"), color="#f0efec", lw=0)
    ax.plot(d.origin, d.actual / 1000, color=INK, lw=2.2, label="실제")
    ax.plot(d.origin, d.price_only_ridge / 1000, color=GREY, lw=1.5, ls=(0, (4, 2)), label="가격만 ridge")
    ax.plot(d.origin, d.chronos2 / 1000, color=GREEN, lw=1.5, ls=(0, (1, 1.5)), label="Chronos-2 제로샷")
    ax.plot(d.origin, d.mstl_ets / 1000, color=VIOLET, lw=1.5, ls=(0, (3, 1, 1, 1)), label="MSTL+AutoETS")
    ax.plot(d.origin, d.an_sparse_all / 1000, color=BLUE, lw=1.6, label="계절편차 모형")
    ax.plot(d.origin, d[MODEL] / 1000, color=ORANGE, lw=2.0, label="결합 모형")
    ax.set_ylabel("천원/10kg", fontsize=9, color=INK2)
    ax.set_xlabel("원점(매주 일요일)", fontsize=9, color=INK2)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=2)
    _axes(ax)
    fig.savefig(FIG / "fig2_weekly_2024.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig1()
    fig2()
    print("wrote", FIG / "fig1_monthly_track.png", FIG / "fig2_weekly_2024.png")
