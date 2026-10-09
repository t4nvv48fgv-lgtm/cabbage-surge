# -*- coding: utf-8 -*-
"""쉬운 말 결과 보고서 PDF 생성기.

모든 모형 수치는 저장소의 산출물에서 직접 읽는다(README 결과표와 같은 원본):
  outputs/backtest_predictions.csv, backtest_predictions_W.csv      (가격만·Chronos-2·MSTL·기상 ridge/HGB)
  outputs/anomaly_predictions.csv,  anomaly_predictions_W.csv       (계절편차 모형)
  experiments/case03_split_selection/outputs/selected.csv          (분리 선택 결과)
  data/raw/clean_daily.csv                                         (그림 1 월평균 가격)
수치가 아닌 서술(해석·한계·다음 단계)은 이 파일의 TEXT 상수에 있고, README·CASE.md의 문장 상한을 넘지 않게 쓴다.

실행:  python tools/make_report.py [--out docs/배추급등예측_결과보고서_YYYY-MM-DD.pdf]
요구:  reportlab, matplotlib, pandas. 한글 글꼴은 Windows(맑은 고딕) / macOS(AppleGothic·NanumGothic) 순으로 찾는다.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RAW = ROOT / "data" / "raw"
CASE03 = ROOT / "experiments" / "case03_split_selection" / "outputs" / "selected.csv"
TODAY = date.today().isoformat()

# ---------------------------------------------------------------- 글꼴 (Windows / macOS / Linux 순)
FONT_CANDIDATES = [
    ("MalgunGothic", r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunbd.ttf"),
    ("AppleGothic", "/System/Library/Fonts/Supplemental/AppleGothic.ttf", None),
    ("AppleSDGothicNeo", "/System/Library/Fonts/AppleSDGothicNeo.ttc", None),
    ("NanumGothic", "/Library/Fonts/NanumGothic.ttf", "/Library/Fonts/NanumGothicBold.ttf"),
    ("NanumGothic", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
]


def register_font() -> tuple[str, str]:
    for name, regular, bold in FONT_CANDIDATES:
        if os.path.exists(regular):
            pdfmetrics.registerFont(TTFont(name, regular))
            bname = name
            if bold and os.path.exists(bold):
                bname = name + "-Bold"
                pdfmetrics.registerFont(TTFont(bname, bold))
            fm.fontManager.addfont(regular)
            plt.rcParams["font.family"] = fm.FontProperties(fname=regular).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return name, bname
    raise SystemExit("한글 글꼴을 찾지 못했습니다. FONT_CANDIDATES에 경로를 추가하세요.")


FONT, FONT_B = register_font()

# ---------------------------------------------------------------- 팔레트 (dataviz 스킬 검증 기본 팔레트, 고정 순서)
PAL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
       "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

# ---------------------------------------------------------------- 모형 표기 (보고서 이름 ↔ CSV 열)
MODELS = [  # (열, 보고서 이름, 성격)
    ("price_only_ridge", "가격만 보는 모형", "기준선. 최근 가격 흐름만. KREI 연구와 같은 한계를 재현"),
    ("chronos2", "Chronos-2 (AI 시계열)", "KREI 연구 1 계열. 2010년부터의 월평균 가격만, 학습 없이 사용"),
    ("mstl_ets", "MSTL+ETS (통계 분해)", "KREI 연구 2 계열. 월평균 가격만"),
    ("wx_ridge", "날씨 추가 (선형)", "가격 + 고랭지 폭염·강수를 직선 관계로"),
    ("wx_hgb", "날씨 추가 (트리)", "같은 변수, 규칙을 자동으로 찾는 방식"),
    ("wx_hgb_summer", "여름 특화 (트리)", "위와 같되 여름에는 여름 자료만 학습"),
    ("an_sparse_all", "계절편차 (사전선택) ★대표", "'평년 이맘때보다 얼마나 더운가' 등 8변수. 설정을 2016~2023 자료만으로 고름"),
    ("an_sparse_summer", "계절편차 (여름만, 사후)", "같은 8변수를 여름 자료만으로 학습. 2024 결과를 본 뒤 정한 설정, 참고용"),
]
NAME = {c: n for c, n, _ in MODELS}
HEAD = "an_sparse_all"
BASE_MODEL = "price_only_ridge"
TARGET_SURGE, TARGET_MAPE = -10.0, None   # MAPE 기준은 가격만 모형 값에서 계산


# ---------------------------------------------------------------- 데이터
def load() -> dict:
    b = pd.read_csv(OUT / "backtest_predictions.csv", parse_dates=["origin"])
    a = pd.read_csv(OUT / "anomaly_predictions.csv", parse_dates=["origin"])
    m = b.merge(a[["origin", "an_sparse_all", "an_sparse_summer", "an_ridge_all"]], on="origin")
    bw = pd.read_csv(OUT / "backtest_predictions_W.csv", parse_dates=["origin"])
    aw = pd.read_csv(OUT / "anomaly_predictions_W.csv", parse_dates=["origin"])
    w = bw.merge(aw[["origin", "an_sparse_all", "an_sparse_summer"]], on="origin")
    price = pd.read_csv(RAW / "clean_daily.csv", parse_dates=["date"]).set_index("date")["price"]
    price = price[price > 0]
    sel = pd.read_csv(CASE03) if CASE03.exists() else None
    return {"m": m, "w": w, "price": price, "sel": sel}


def err_at(df: pd.DataFrame, origin: str, col: str) -> float:
    s = df[df.origin == origin]
    return float((s[col].iloc[0] / s.actual.iloc[0] - 1) * 100)


def mape(df: pd.DataFrame, col: str, years=(2021, 2025), summer_only=False) -> float:
    mk = df.origin.dt.year.between(*years)
    if summer_only:
        mk &= df.origin.dt.month.isin([6, 7, 8, 9])
    s = df[mk]
    return float(((s[col] - s.actual).abs() / s.actual).mean() * 100)


def fmt_pct(x: float, signed=True) -> str:
    return f"{x:+.0f}%" if signed else f"{x:.1f}%"


# ---------------------------------------------------------------- 그림
def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def to_img(fig, width_mm=165) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    w_px, h_px = fig.get_size_inches() * 200
    img = Image(buf, width=width_mm * mm, height=width_mm * mm * h_px / w_px)
    return img


def fig_monthly(price: pd.Series):
    mth = price["2022":"2025"].resample("MS").mean() / 1000
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(mth.index, mth.values, color=PAL["blue"], lw=2)
    peak = mth.idxmax()
    ax.scatter([peak], [mth.max()], s=50, color=PAL["orange"], zorder=3)
    ax.annotate(f"2024년 9월 {mth.max():.1f}천원", (peak, mth.max()), xytext=(10, -2), textcoords="offset points",
                fontsize=9, color=INK)
    ax.set_ylabel("월평균 가격 (천원/10kg, 상품)", color=INK2, fontsize=9)
    ax.set_title("그림 1. 2022~2025년 가락시장 배추 월평균 가격", loc="left", fontsize=10, color=INK)
    _style(ax)
    return to_img(fig)


def fig_surge(m: pd.DataFrame):
    cols = [c for c, _, _ in MODELS]
    e24 = [err_at(m, "2024-08-31", c) for c in cols]
    e25 = [err_at(m, "2025-08-31", c) for c in cols]
    y = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axvspan(-10, 10, color="#f0efec", zorder=0)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.barh(y + 0.19, e24, height=0.36, color=PAL["blue"], label="2024년 (급등 있음)")
    ax.barh(y - 0.19, e25, height=0.36, color=PAL["orange"], label="2025년 (급등 없음)")
    for yi, v in zip(y + 0.19, e24):
        ax.text(v + (1.2 if v >= 0 else -1.2), yi, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK)
    for yi, v in zip(y - 0.19, e25):
        ax.text(v + (1.2 if v >= 0 else -1.2), yi, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([NAME[c] for c in cols], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("8월 말 시점의 9월 평균 가격 예측 오차 (예측 ÷ 실제 - 1)", color=INK2, fontsize=9)
    ax.set_xlim(-45, 40)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title("그림 2. 급등을 얼마나 따라갔나 (회색 띠 = ±10%)", loc="left", fontsize=10, color=INK)
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    return to_img(fig)


def fig_mape(m: pd.DataFrame):
    cols = [c for c, _, _ in MODELS]
    v = [mape(m, c) for c in cols]
    base = mape(m, BASE_MODEL)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(cols))
    colors_ = [PAL["blue"] if c != HEAD else PAL["violet"] for c in cols]
    ax.barh(y, v, height=0.55, color=colors_)
    ax.axvline(base, color=PAL["orange"], lw=1.4, ls="--")
    ax.text(base + 0.2, -0.6, f"기준선 {base:.1f}% (가격만 보는 모형)", color=PAL["orange"], fontsize=8)
    for yi, val in zip(y, v):
        ax.text(val + 0.2, yi, f"{val:.1f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([NAME[c] for c in cols], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(v) + 4)
    ax.set_xlabel("2021~2025년 월말 예측 60회 평균 오차율 MAPE (낮을수록 좋음)", color=INK2, fontsize=9)
    ax.set_title("그림 3. 평소 정확도", loc="left", fontsize=10, color=INK)
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    return to_img(fig)


def fig_weekly(w: pd.DataFrame):
    d = w[(w.origin >= "2024-06-01") & (w.origin <= "2024-11-10")]
    series = [("actual", "실제 (이후 4주 평균)", INK, 2.2, "-"),
              ("price_only_ridge", NAME["price_only_ridge"], PAL["violet"], 1.6, "--"),
              ("chronos2", NAME["chronos2"], PAL["aqua"], 1.4, ":"),
              ("wx_ridge", NAME["wx_ridge"], PAL["blue"], 1.6, "-"),
              ("an_sparse_all", NAME["an_sparse_all"], PAL["orange"], 2.0, "-")]
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, label, color, lw, ls in series:
        ax.plot(d.origin, d[col] / 1000, color=color, lw=lw, ls=ls, label=label)
    ax.axvspan(pd.Timestamp("2024-08-15"), pd.Timestamp("2024-09-30"), color="#f0efec", zorder=0)
    ax.text(pd.Timestamp("2024-08-17"), ax.get_ylim()[0] + 0.4, "급등 국면", fontsize=8, color=INK2, va="bottom")
    ax.set_ylabel("이후 4주 평균 가격 (천원/10kg)", color=INK2, fontsize=9)
    ax.set_xlabel("예측 시점 (매주 일요일)", color=INK2, fontsize=9)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_title("그림 4. 2024년 여름 주간 예측 추적", loc="left", fontsize=10, color=INK)
    _style(ax)
    return to_img(fig)


# ---------------------------------------------------------------- 문서
def styles():
    base = dict(fontName=FONT, leading=15, textColor=INK, alignment=TA_LEFT)
    return {
        "title": ParagraphStyle("title", fontName=FONT_B, fontSize=17, leading=22, textColor=INK, spaceAfter=4),
        "sub": ParagraphStyle("sub", fontName=FONT, fontSize=9.5, leading=13, textColor=INK2, spaceAfter=10),
        "h1": ParagraphStyle("h1", fontName=FONT_B, fontSize=12.5, leading=17, textColor=INK, spaceBefore=12, spaceAfter=5),
        "h2": ParagraphStyle("h2", fontName=FONT_B, fontSize=10.5, leading=14, textColor=INK, spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("body", fontSize=9.8, **base),
        "bullet": ParagraphStyle("bullet", fontSize=9.8, leftIndent=10, bulletIndent=0, **base),
        "small": ParagraphStyle("small", fontName=FONT, fontSize=8.3, leading=11.5, textColor=INK2),
        "cell": ParagraphStyle("cell", fontName=FONT, fontSize=8.6, leading=11.5, textColor=INK),
        "cellb": ParagraphStyle("cellb", fontName=FONT_B, fontSize=8.6, leading=11.5, textColor=INK),
        "box": ParagraphStyle("box", fontName=FONT, fontSize=9.8, leading=15, textColor=INK, backColor="#f3f4f7",
                              borderPadding=(7, 9, 7, 9), spaceBefore=4, spaceAfter=8),
    }


def table(rows, widths, st, header=True, bold_rows=()):
    data = []
    for i, r in enumerate(rows):
        sty = st["cellb"] if (header and i == 0) or i in bold_rows else st["cell"]
        data.append([Paragraph(str(c), sty) for c in r])
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if header else 0)
    ts = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(GRID)),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        ts.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0efec")))
    for i in bold_rows:
        ts.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#eef2fb")))
    t.setStyle(TableStyle(ts))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT, 8)
    canvas.setFillColor(colors.HexColor(INK2))
    canvas.drawString(20 * mm, 12 * mm, f"배추 급등 예측모형 결과 보고서 · {TODAY} · cabbage-surge")
    canvas.drawRightString(190 * mm, 12 * mm, str(doc.page))
    canvas.restoreState()


def build(out_path: Path) -> None:
    D = load()
    m, w, sel = D["m"], D["w"], D["sel"]
    st = styles()
    P = lambda t, s="body": Paragraph(t, st[s])
    B = lambda t: Paragraph(t, st["bullet"], bulletText="•")

    # ---- 계산값
    e24 = {c: err_at(m, "2024-08-31", c) for c, _, _ in MODELS}
    e25 = {c: err_at(m, "2025-08-31", c) for c, _, _ in MODELS}
    e22 = {c: err_at(m, "2022-08-31", c) for c, _, _ in MODELS}
    mp = {c: mape(m, c) for c, _, _ in MODELS}
    mps = {c: mape(m, c, summer_only=True) for c, _, _ in MODELS}
    mpw = {c: mape(w, c) for c in ("price_only_ridge", "chronos2", "wx_hgb", "an_sparse_all", "an_sparse_summer")}
    base_mape = mp[BASE_MODEL]
    n60 = int(m.origin.dt.year.between(2021, 2025).sum())
    nW = int(w.origin.dt.year.between(2021, 2025).sum())
    best_mape = min(mp, key=mp.get)
    wk = w.set_index("origin")
    wk_err = lambda d, c: float((wk.loc[d, c] / wk.loc[d, "actual"] - 1) * 100)
    sel_row = None
    if sel is not None:
        r = sel[(sel.family == "anom") & (sel.criterion == "design_all_MAPE")]
        sel_row = r.iloc[0] if len(r) else None

    story = []
    # ---- 표지·요약
    story += [P("배추 가격 급등 예측모형, 지금까지의 결과", "title"),
              P(f"{date.fromisoformat(TODAY).strftime('%Y년 %m월 %d일')} 기준 · 프로젝트 cabbage-surge · 쉬운 말로 정리한 보고서 · "
                "수치는 저장소 outputs/*.csv에서 직접 계산 (tools/make_report.py)", "sub"),
              P("<b>한 줄 요약</b>", "h2"),
              P(f"2024년 9월 배추값이 한 달 만에 45% 뛰었는데, 기존 연구의 모형은 이를 거의 예측하지 못했습니다. "
                f"강원도 고랭지의 폭염 데이터를 더한 단순 선형 모형은 이 급등을 <b>{e24[HEAD]:+.0f}%</b> 차이까지 따라갔고, "
                f"이 모형의 설정은 <b>2024년 결과를 보지 않고 2016~2023년 자료만으로 고른 것</b>입니다. "
                f"다만 평소 정확도는 가격만 보는 모형보다 조금 나쁘고(오차율 {mp[HEAD]:.1f}% 대 {base_mape:.1f}%), "
                f"급등이 없던 2025년에는 {e25[HEAD]:+.0f}%로 오를 거라고 잘못 경보를 냈습니다. "
                "그래서 '목표 달성'이 아니라 <b>'급등은 따라가되 그 대가가 있다'</b>로 평가합니다.", "box"),
              P("<b>이 보고서에서 다루는 것</b>", "h2"),
              B("문제: 왜 2024년 9월 급등이 중요한가, 기존 연구는 왜 못 맞췄나"),
              B("방법: 어떤 데이터로 어떻게 예측을 시험했나, 성공 기준은 무엇인가"),
              B("결과: 모형별로 급등을 얼마나 따라갔나, 평소 정확도는 어떤가, 기준 판정"),
              B("알게 된 것과 한계, 실험 케이스, 다음 단계")]

    # ---- 1. 문제
    story += [P("1. 문제: 2024년 9월에 무슨 일이 있었나", "h1"),
              P("가락시장 배추(상품) 도매가격은 보통 10kg에 1만원 안팎입니다. 그런데 2024년 9월에는 월평균 24,714원으로 8월보다 45% 올랐습니다. "
                "여름 내내 이어진 강원도 고랭지 폭염으로 배추가 제대로 자라지 못한 탓입니다."),
              fig_monthly(D["price"]),
              P("한국농촌경제연구원(KREI)은 최근 배추 가격 예측 연구를 두 건 냈습니다. 하나는 아마존의 AI 시계열 모형 Chronos-2를 쓴 것이고, "
                "다른 하나는 통계적 분해 기법(STL)을 쓴 것입니다. 두 연구 모두 평소에는 잘 맞추지만 2024년 9월은 실제보다 10~24% 낮게 예측했고, "
                "보고서 스스로 '급등은 별도 과제'라고 적었습니다. 이 프로젝트는 그 빈틈을 메우는 것이 목표입니다."),
              PageBreak()]

    # ---- 2. 방법
    story += [P("2. 방법: 어떻게 시험했나", "h1"), P("<b>예측 과제</b>", "h2"),
              P("매달 말일에 서서, 그 시점까지 알 수 있는 정보만으로 다음 달 평균 가격을 맞히는 과제입니다. 2016년 1월부터 2025년 12월까지 "
                f"반복해 평균 성적을 내고(2021~2025년 {n60}회를 지표로 사용), 2024년 8월 말(9월 예측)을 특별히 봅니다. "
                "매주 일요일에 서서 4주 뒤를 맞히는 주간 버전도 같이 돌렸습니다. 학습에는 예측 시점 이전에 결과가 다 나온 자료만 씁니다(정보 누수 금지)."),
              P("<b>쓴 데이터</b> (농식품 가격정보 시스템 CABIS에서 복사, 2026-10-06 기준)", "h2"),
              table([["데이터", "내용", "기간"],
                     ["가락시장 일별 가격·반입량", "맞혀야 할 값. 상품 등급, 원/10kg", "2010~2026.06"],
                     ["고랭지 날씨", "평창·태백·정선·강릉 4곳의 기온·강수. '평년 대비 얼마나 더웠나'로 바꿔 사용", "2010~2026.08"],
                     ["정부 비축 방출량", "2022년부터만 있어 본 모형에서는 제외 (4절 참고). 별도 실험(case01)에서만 사용", "2022~"],
                     ["KREI 월보", "작형별 면적·생산량 전망. 아직 모형에는 안 씀", "2022~2025"]],
                    [42, 98, 28], st),
              P("<b>비교한 모형 8가지</b>", "h2"),
              table([["보고서 표기", "무엇을 보는가 / 성격"]] + [[n, d] for _, n, d in MODELS], [52, 116], st, bold_rows=(7,)),
              P("'선형'은 변수와 가격 사이를 직선 관계로 보는 단순한 모형(ridge), '트리'는 조건을 나눠가며 규칙을 찾는 모형(HGB)입니다. "
                "단순한 모형은 겪어보지 못한 폭염에도 예측을 늘려 잡을 수 있고, 트리 모형은 경험 범위를 벗어나면 평평해집니다. "
                "★대표 모형은 설정(변수 8개, 규제 강도)을 2016~2023년 성적만으로 골랐기 때문에 '2024년 정답을 보고 맞췄다'는 비판을 피합니다.", "small"),
              P("<b>성공 기준</b> (프로젝트 시작 때 정함)", "h2"),
              B("2024년 9월 예측이 실제보다 10% 이상 낮지 않을 것 (기존 연구는 10~40% 낮았음)"),
              B(f"그러면서 2021~2025년 전체 평균 오차율이 가격만 보는 모형({base_mape:.1f}%)보다 나쁘지 않을 것"),
              PageBreak()]

    # ---- 3. 결과
    story += [P("3. 결과", "h1"), P("3-1. 급등을 얼마나 따라갔나", "h2"),
              P("8월 말에 서서 9월 평균을 예측했을 때의 오차입니다. 파란 막대는 급등이 있었던 2024년, 주황 막대는 급등이 없었던 2025년입니다. "
                "좋은 모형이라면 둘 다 회색 띠(±10%) 안에 있어야 합니다."),
              fig_surge(m),
              B(f"가격만 보는 모형은 {e24['price_only_ridge']:+.0f}%. 기존 연구의 한계가 그대로 나타납니다."),
              B(f"가격만 보더라도 Chronos-2({e24['chronos2']:+.0f}%)와 MSTL+ETS({e24['mstl_ets']:+.0f}%)는 절반 이상 따라갑니다. "
                "8월 하순의 상승세를 그대로 연장하기 때문입니다."),
              B(f"<b>대표 모형 계절편차(사전선택)은 {e24[HEAD]:+.0f}%</b>로 기준선(-10%)에 걸립니다. 그러나 급등이 없던 2025년에는 "
                f"{e25[HEAD]:+.0f}%로, 오르지 않을 값을 오른다고 한 셈입니다. 여름만 학습한 사후 설정은 {e24['an_sparse_summer']:+.0f}%까지 따라가지만 "
                f"2025년 거짓 경보가 {e25['an_sparse_summer']:+.0f}%로 더 큽니다."),
              B("급등을 잘 따라가는 모형일수록 급등 없는 해에 과대 예측하는 맞교환 관계가 뚜렷합니다."),
              P("3-2. 평소 정확도는 어떤가", "h2"),
              P(f"2021~2025년 {n60}번의 월말 예측에서 평균 오차율(MAPE)입니다. 낮을수록 좋고, 주황 점선이 가격만 보는 모형의 기준선입니다."),
              fig_mape(m),
              B(f"{NAME[best_mape]} 모형이 {mp[best_mape]:.1f}%로 평소 정확도는 가장 좋지만, 2024년 9월은 {e24[best_mape]:+.0f}%로 급등을 못 따라갑니다."),
              B(f"<b>대표 모형은 {mp[HEAD]:.1f}%로 기준선({base_mape:.1f}%)을 넘지 못합니다.</b> 급등을 따라가는 대신 평소 오차가 {mp[HEAD] - base_mape:.1f}%p 늘어난 것입니다. "
                f"사후 설정(여름만)의 {mp['an_sparse_summer']:.1f}%는 기준선을 넘지만, 학습 자료 초기 100행(6%)을 빼면 15.2%로 뒤집히는 불안정한 값이라 근거로 쓰지 않습니다."),
              B(f"Chronos-2와 MSTL은 급등은 절반쯤 잡지만 평소 오차율은 {mp['chronos2']:.1f}~{mp['mstl_ets']:.1f}%로 기준선보다 나쁩니다. 추세를 연장하는 방식의 비용입니다."),
              P("3-3. 주간으로 보면: 상승은 미리 잡고, 하락 전환은 놓친다", "h2"),
              P("매주 일요일에 서서 4주 뒤 평균을 예측한 결과입니다. 검은 선이 실제, 색 선이 모형입니다."),
              fig_weekly(w),
              B(f"대표 모형은 8월 하순~9월 중순에 {wk_err('2024-09-01', HEAD):+.0f}~{wk_err('2024-09-08', HEAD):+.0f}% 안에서 실제에 붙어 "
                f"상승 국면을 따라갑니다(가격만 보는 모형은 같은 시점 {wk_err('2024-09-01', 'price_only_ridge'):+.0f}%)."),
              B(f"그러나 10월 들어 가격이 떨어질 때는 모든 모형이 높게 예측합니다(10월 20일 시점 대표 모형 {wk_err('2024-10-20', HEAD):+.0f}%). "
                "가을배추가 나오기 시작하면 값이 내린다는 신호가 데이터에 없기 때문입니다."),
              B(f"주간 전체(2021~2025, {nW}번)로는 Chronos-2가 오차율 {mpw['chronos2']:.1f}%, 날씨 추가(트리) {mpw['wx_hgb']:.1f}%로 좋고, "
                f"대표 모형은 {mpw['an_sparse_all']:.1f}%로 가격만 보는 모형({mpw['price_only_ridge']:.1f}%)보다 나쁩니다."),
              P("3-4. 성공 기준 판정 (대표 모형 기준)", "h2"),
              table([["기준", "대표 모형 (계절편차, 사전선택)", "판정"],
                     ["2024년 9월 오차 -10% 이내", f"{e24[HEAD]:+.0f}%", "경계 (충족)"],
                     [f"전체 오차율 {base_mape:.1f}% 이하", f"{mp[HEAD]:.1f}%", "미충족"],
                     ["급등 없는 해에 거짓 경보 없을 것", f"2025년 9월 {e25[HEAD]:+.0f}%", "미충족"],
                     ["설계를 결과 보기 전에 정했을 것", "변수·규제 강도를 2016~2023년 자료만으로 선택 (case03)", "충족"],
                     ["학습 자료 일부를 빼도 결과가 유지될 것", "학습 풀·추출 시작점·규제 18조합에서 2024년 -9.5~-12.4%, 2025년 +20~+22%", "충족"]],
                    [50, 88, 30], st),
              P(f"<b>종합:</b> 지난 보고서(사후 설정 기준)에서는 두 숫자 기준은 넘었지만 설계 절차와 안정성이 미충족이었습니다. 대표 모형을 사전 선택 설정으로 바꾼 지금은 "
                f"반대로 <b>설계 절차와 안정성은 충족하지만 평소 오차율 기준을 넘지 못합니다.</b> 어느 쪽이든 '달성'이라고 쓰지 않습니다. 정직한 서술은 이렇습니다. "
                f"\"2024년을 보지 않고 고른 고랭지 기상 편차 선형 모형이 2024년 9월 급등을 {e24[HEAD]:+.0f}% 차이로 따라가며, "
                f"그 대가는 급등 없던 2025년 9월의 {e25[HEAD]:+.0f}% 과대 예측과 평소 오차율 {mp[HEAD] - base_mape:.1f}%p 악화입니다.\"", "box"),
              PageBreak()]

    # ---- 4. 알게 된 것
    story += [P("4. 알게 된 것", "h1"),
              B("<b>문제는 '가격 정보 부족'이 아니라 '모형 구조'였습니다.</b> 같은 가격 자료로도 추세를 연장하는 모형(Chronos-2, MSTL)은 급등의 절반 이상을 잡습니다. "
                f"가격만 보는 선형 모형이 {e24['price_only_ridge']:+.0f}%였던 것은 '평균으로 돌아간다'는 가정이 강하게 들어가 있었기 때문입니다."),
              B("<b>고랭지 폭염이 가장 큰 개선 요인이고, 이 결론은 2024년을 보지 않고 고른 설정에서도 성립합니다.</b> 2024년 8월은 30일 중 24일이 30도를 넘어 자료 역사상 없던 더위였습니다. "
                "이런 '경험 밖' 상황은 단순한 선형 모형이 트리 모형보다 잘 늘려 잡습니다. 분리 선택 실험(case03)에서 변수 8개짜리 계절편차 모형은 규제 강도를 어떻게 잡아도 2024년 9월을 -11% 안팎으로 따라갔습니다."),
              B("<b>그러나 날씨 변수가 평소 정확도까지 높여 주지는 않습니다.</b> 2016~2023년만 보면 가격만 보는 모형(18.3%)이 날씨 추가 선형 모형(18.7~21.1%)보다 오차율이 낮았습니다. "
                "날씨 모형의 우위는 2024~2025년에 집중되어 있습니다. '날씨가 평소 예측도 개선한다'고는 쓸 수 없습니다."),
              B(f"<b>정부 비축 방출량 변수는 함정이었습니다.</b> 자료가 2022년부터만 있어서 그 전 기간이 전부 0으로 채워졌고, 2022년 8월에 처음 숫자가 나타나자 모형이 엉뚱하게 반응해 "
                f"'2022년 9월 -59%' 같은 가짜 실패를 만들었습니다. 빼고 나니 날씨 추가(선형)의 2022년 9월 오차는 {e22['wx_ridge']:+.0f}%로 돌아왔습니다. "
                "5년밖에 없는 자료는 학습 변수로 쓰지 않는다는 규칙을 얻었습니다."),
              B("<b>변수를 더 넣는다고 나아지지 않았습니다.</b> 정부 매입, 전년 생산량, 전체 관측소, 90일 누적 폭염 등 쓸 수 있는 자료를 전부 넣어 봤지만(case02) 두 기준을 동시에 넘는 조합은 없었습니다. "
                "90일 누적 폭염은 오히려 2025년이 2024년보다 컸습니다. 병목은 자료의 양이 아니라 2024년과 2025년을 가르는 변수(산지 작황, 정식 지연, 재배면적)가 없다는 것입니다."),
              B("<b>비교 기준선을 바꿔야 합니다.</b> '날씨가 도움이 되는가'를 따질 때는 가격만 보는 모형이 아니라 Chronos-2·MSTL과 비교해야 합니다.")]

    # ---- 5. 케이스
    case_rows = [["케이스", "아이디어", "결과", "상태"],
                 ["case01 방출 트리거", "정부 방출이 급감하면서 가격이 오르면 Chronos-2의 예측을 상단 분위수로 올림",
                  "월말 2024-09 -18% → +0.4%, 거짓 경보 없음. 주간에서는 거짓 경보로 실패. 발동이 1건뿐이고 사후 설계", "보류"],
                 ["case02 전체 데이터", "쓸 수 있는 자료 전부를 변수로", "두 기준 동시 충족 없음. 변수 추가 방향은 중단", "종결"],
                 ["case03 분리 선택", "모형 설정 48개를 2016~2023 자료만으로 고르고 2024~2025에서 검증",
                  (f"계절편차 8변수 모형은 미리 골라도 2024-09 {sel_row['err_2024-08']:+.0f}%(2025-08 {sel_row['err_2025-08']:+.0f}%). " if sel_row is not None else "계절편차 8변수 모형은 미리 골라도 2024-09 -11%. ")
                  + "날씨 변수는 2016~2023 평소 정확도를 높이지 않음. 미리 고른 설정이 2026-10-09 본 모형의 대표 설정으로 채택됨",
                  "완료 → 대표 설정으로 채택"]]
    story += [P("5. 실험 케이스 세 가지", "h1"),
              P("본 모형을 바꾸지 않고 별도 폴더(experiments/)에서 해본 시도입니다. 케이스가 본 모형에 들어가려면 각 케이스 노트의 승격 조건을 채우고 사용자가 승인해야 합니다. "
                "case03이 첫 사례입니다."),
              table(case_rows, [30, 50, 68, 20], st, bold_rows=(3,))]

    # ---- 6. 한계
    story += [P("6. 한계와 조심할 점", "h1"),
              B("급등 사례가 한 번뿐입니다. 2024년 9월 한 점에 맞춰 설계를 바꾸지 않도록 2020년·2022년 9월, 2025년 9월을 항상 같은 표에 놓고 봅니다."),
              B("'미리 골랐다'에도 한도가 있습니다. 후보 설정의 범위와 정부 방출 변수 제외는 2024년 결과를 본 뒤의 지식입니다. 격자 안에서의 선택만 2024년을 가렸습니다."),
              B("10월 하락 전환은 아직 어떤 모형도 못 잡습니다. 가을배추 출하 신호를 넣는 것이 다음 과제입니다."),
              B("KREI 두 연구와 CABIS 자료는 인용 가능하다고 확인됐지만(2026-10-07), 공개 저장소에 원문 PDF를 올릴지는 별도 판단입니다."),
              B("트리 모형 결과는 실행 환경이 다르면 ±2~3%p 흔들릴 수 있습니다. 선형 모형은 같은 자료면 같은 결과가 나옵니다(Windows와 맥에서 소수점까지 일치 확인).")]

    # ---- 7. 다음 단계
    story += [P("7. 다음 단계", "h1"), P("<b>결정이 필요한 것 (사용자)</b>", "h2"),
              B("이 결과를 논문으로 쓸지, 쓴다면 저널인지 연구보고인지 (타깃 저널은 보류, 자유형식으로 작성 중)"),
              B("연구 질문과 기여를 어떻게 한 문장으로 쓸지. 범위를 '상승 국면 추종'으로 한정할지"),
              B("정부 방출 자료를 2010~2020년까지 확보할 수 있는지 (case01 트리거를 학습 변수로 올리는 조건)"),
              P("<b>바로 할 수 있는 분석</b>", "h2"),
              B("날씨 변수가 Chronos-2·MSTL 기준선 대비 실제로 얼마나 보태는지 순효과 계산"),
              B("급등 시기와 평시를 나눠서 보는 지표, 또는 예측 구간(분위수)이 실제를 얼마나 덮는지"),
              B("가을배추 출하 전환 신호 탐색 (10월 하락을 잡기 위해)"),
              B("2024년과 2025년을 가르는 산지 변수(작황·정식 지연·재배면적) 출처 찾기")]

    # ---- 용어
    story += [P("용어 풀이", "h1"),
              table([["용어", "뜻"],
                     ["MAPE (평균 오차율)", "예측이 실제와 평균적으로 몇 % 어긋나는지. 낮을수록 좋음"],
                     ["오차 (이 보고서)", "예측 ÷ 실제 - 1. -10%면 실제보다 10% 낮게 본 것"],
                     ["원점", "예측을 하는 시점. '8월 말 원점'은 8월 31일에 서서 예측한다는 뜻"],
                     ["ridge / 선형", "변수와 가격을 직선 관계로 보는 단순 모형. 지나치게 큰 반응은 억제함(규제)"],
                     ["HGB / 트리", "조건을 나눠가며 규칙을 찾는 모형. 경험 범위 밖에서는 평평해짐"],
                     ["Chronos-2", "아마존이 공개한 AI 시계열 예측 모형. 별도 학습 없이 써도 됨"],
                     ["MSTL+ETS", "계절성을 분리한 뒤 추세를 연장하는 전통적 통계 기법"],
                     ["계절편차", "'평년 이맘때'와 비교해 얼마나 다른가. 가격도 날씨도 이 편차로 바꿔 사용"],
                     ["사전선택 / 사후", "설정을 결과 보기 전 자료(2016~2023)만으로 골랐으면 사전선택, 2024 결과를 본 뒤 골랐으면 사후"],
                     ["거짓 경보", "급등이 없는데 급등한다고 예측하는 것"]],
                    [40, 128], st)]

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=20 * mm,
                            title="배추 급등 예측모형 결과 보고서", author="cabbage-surge")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / f"배추급등예측_결과보고서_{TODAY}.pdf"))
    a = ap.parse_args()
    build(Path(a.out))
