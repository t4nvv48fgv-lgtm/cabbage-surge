# -*- coding: utf-8 -*-
"""쉬운 말 결과 보고서 PDF 생성기.

모든 모형 수치는 저장소의 산출물에서 직접 읽는다(README 결과표와 같은 원본):
  outputs/backtest_predictions.csv, backtest_predictions_W.csv      (가격만·Chronos-2·MSTL·기상 ridge/HGB)
  outputs/anomaly_predictions.csv,  anomaly_predictions_W.csv       (계절편차 모형)
  outputs/gate_predictions.csv,     gate_predictions_W.csv          (최종 모형과 단계별 열. final_prod_only = 출하량 전망을 엮기 전 값)
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
    ("an_sparse_all", "계절편차 (사전선택) — 경보 모형", "평년치 기준 + '8월 말 가격이 평년보다 얼마나 높은가'의 지속 + 고랭지 기상 편차, 7변수 선형. 설정을 2016~2023 자료만으로 고름. 이 줄은 고랭지 기상을 1년 내내 쓰는 종전 사양"),
    ("an_sparse_summer", "계절편차 (여름만, 사후)", "같은 7변수를 여름 자료만으로 학습. 2024 결과를 본 뒤 정한 설정, 참고용"),
    ("gate_hgb", "스위치만 적용", "KREI 월보 여름배추 생산 전망이 '줄어듦'이고 그 달 출하량 전망도 '증가'가 아니면 경보 모형, 아니면 여름 특화(트리). 여름 밖의 경보 모형은 고랭지 기상을 여름배추 출하기에만 씀(2026-10-13)"),
    ("final", "최종 모형 (스위치+겨울 분기+수준 조정+9월 고온) ★", "위 스위치에 더해 1~4월 예측은 겨울 전용 가격 모형, 여름 경보 달과 11~12월은 생산 전망만큼 수준 조정, 9월 말의 10월 예측은 가을 산지 9월 고온 일수만큼 하락을 늦춤. 2026-10-12~13 다섯 차례 편입"),
]
NAME = {c: n for c, n, _ in MODELS}
HEAD = "an_sparse_all"
FINAL = "final"
PREV = "final_prod_only"   # 출하량 전망을 엮기 전(4차 편입까지)의 최종값
GATE = "gate_hgb"
BASE_MODEL = "price_only_ridge"
TARGET_SURGE, TARGET_MAPE = -10.0, None   # MAPE 기준은 가격만 모형 값에서 계산


# ---------------------------------------------------------------- 데이터
def load() -> dict:
    b = pd.read_csv(OUT / "backtest_predictions.csv", parse_dates=["origin"])
    a = pd.read_csv(OUT / "anomaly_predictions.csv", parse_dates=["origin"])
    g = pd.read_csv(OUT / "gate_predictions.csv", parse_dates=["origin"], encoding="utf-8-sig")
    m = b.merge(a[["origin", "an_sparse_all", "an_sparse_summer", "an_ridge_all"]], on="origin").merge(g[["origin", "head", "gate_hgb", "final_base", "final", PREV, "gate_calm", "krei_issue", "S1", "ship_q", "adj_pct", "heat_H", "heat_pct"]], on="origin")
    bw = pd.read_csv(OUT / "backtest_predictions_W.csv", parse_dates=["origin"])
    aw = pd.read_csv(OUT / "anomaly_predictions_W.csv", parse_dates=["origin"])
    gw = pd.read_csv(OUT / "gate_predictions_W.csv", parse_dates=["origin"], encoding="utf-8-sig")
    w = bw.merge(aw[["origin", "an_sparse_all", "an_sparse_summer"]], on="origin").merge(gw[["origin", "gate_hgb", "final", PREV, "gate_calm"]], on="origin")
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


SHIP_WORD = {1.0: "증가", 0.0: "비슷", -1.0: "감소"}


def sig_at(df: pd.DataFrame, origin: str) -> dict:
    """그 원점에서 스위치가 본 월보 호: 몇 월호, 생산 전망 전년비, 그 달 출하량 전망 방향, 상태."""
    r = df[df.origin == origin].iloc[0]
    return {"issue": int(str(r.krei_issue)[5:]), "S": float(r.S1), "Q": SHIP_WORD.get(r.ship_q, "표현 없음"),
            "state": "평시" if r.gate_calm == 1 else "경보", "adj": float(r.adj_pct)}


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
    colors_ = [PAL["violet"] if c in (HEAD, FINAL) else PAL["blue"] for c in cols]
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
              ("an_sparse_all", NAME["an_sparse_all"], PAL["orange"], 2.0, "-"),
              (FINAL, NAME[FINAL], PAL["yellow"], 2.4, "--")]
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
    canvas.drawString(20 * mm, 12 * mm, f"배추 도매가격 예측모형 결과 보고서 · {TODAY} · cabbage-surge")
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
    mpw = {c: mape(w, c) for c in ("price_only_ridge", "chronos2", "wx_hgb", "an_sparse_all", "an_sparse_summer", "gate_hgb", "final", PREV)}
    mp[PREV] = mape(m, PREV)
    big30w = lambda d, c: int(((d.loc[d.origin.dt.year.between(2021, 2025), c] / d.loc[d.origin.dt.year.between(2021, 2025), "actual"] - 1).abs() > 0.30).sum())
    tmon = (m.origin + pd.Timedelta(days=15))
    yr = m.origin.dt.year.between(2021, 2025)
    seas_mape = lambda months, c: float(((m.loc[yr & tmon.dt.month.isin(months), c] / m.loc[yr & tmon.dt.month.isin(months), "actual"] - 1).abs()).mean() * 100)
    wtm = (w.origin + pd.Timedelta(days=15)).dt.month
    wyr = w.origin.dt.year.between(2021, 2025)
    wwin = lambda c: float(((w.loc[wyr & wtm.isin((1, 2, 3, 4)), c] / w.loc[wyr & wtm.isin((1, 2, 3, 4)), "actual"] - 1).abs()).mean() * 100)
    rng = m.origin >= "2020-12-31"
    big = {c: [(t.strftime("%Y-%m"), (pv / av - 1) * 100) for t, pv, av in zip(tmon[rng], m.loc[rng, c], m.loc[rng, "actual"]) if abs(pv / av - 1) > 0.30] for c in (GATE, FINAL)}
    big_txt = ", ".join(f"{t} {v:+.0f}%" for t, v in big[FINAL])
    adj24 = float(m[m.origin == "2024-08-31"].adj_pct.iloc[0])
    e21 = {c: err_at(m, "2021-08-31", c) for c, _, _ in MODELS}
    e23 = {c: err_at(m, "2023-08-31", c) for c, _, _ in MODELS}
    base_mape = mp[BASE_MODEL]
    n60 = int(m.origin.dt.year.between(2021, 2025).sum())
    nW = int(w.origin.dt.year.between(2021, 2025).sum())
    best_mape = min((c for c in mp if c not in (FINAL, PREV, GATE)), key=mp.get)   # 조립하지 않은 단일 모형 가운데 최선
    wk = w.set_index("origin")
    wk_err = lambda d, c: float((wk.loc[d, c] / wk.loc[d, "actual"] - 1) * 100)
    oos = None   # 2026-08-31 표본외 원점(자료 갱신 후 존재할 때만)
    if (m.origin == "2026-08-31").any():
        oos = {"actual": float(m[m.origin == "2026-08-31"].actual.iloc[0])}
        oos.update({c: err_at(m, "2026-08-31", c) for c, _, _ in MODELS})
        oos["sig"] = sig_at(m, "2026-08-31")
    s24, s25 = sig_at(m, "2024-08-31"), sig_at(m, "2025-08-31")
    mp26 = {c: mape(m, c, years=(2021, 2026)) for c in (PREV, FINAL)}
    mpw26 = {c: mape(w, c, years=(2021, 2026)) for c in (PREV, FINAL)}
    n68 = int(m.origin.dt.year.between(2021, 2026).sum())
    chg = m[(m[FINAL] - m[PREV]).abs() > 0.5]                       # 출하량 전망을 엮어서 바뀐 월말 원점
    chg_w = w[(w[FINAL] - w[PREV]).abs() > 0.5]
    chg_rows = []
    for o in ("2026-06-30", "2026-07-31", "2026-08-31"):
        if (m.origin == o).any():
            s = sig_at(m, o)
            prev_state = "평시" if s["S"] > 0 else "경보"
            chg_rows.append([f"{o} → {int(o[5:7]) + 1}월", f"{s['issue']}월호 {s['S']:+.1f}% / {s['Q']}",
                             f"{prev_state}, {err_at(m, o, PREV):+.0f}%", f"{s['state']}, {err_at(m, o, FINAL):+.0f}%"])
    mp_v3, n30w_v3 = None, None                                      # 3차 편입까지(고랭지 기상을 1년 내내 쓰던 때)의 값: 그 커밋의 산출물에서
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from monthly_error_table import past_column
        v3 = m[["origin", "actual"]].assign(v3=past_column("b5a34ee", "final").reindex(m.origin).values)
        mp_v3 = mape(v3, "v3")
        v3w = w[["origin", "actual"]].assign(v3=past_column("b5a34ee", "final", "_W").reindex(w.origin).values)
        n30w_v3 = big30w(v3w, "v3")
    except Exception:
        pass
    sel_row = None
    if sel is not None:
        r = sel[(sel.family == "anom") & (sel.criterion == "design_all_MAPE")]
        sel_row = r.iloc[0] if len(r) else None

    story = []
    # ---- 표지·요약
    story += [P("배추 도매가격 예측모형, 지금까지의 결과", "title"),
              P(f"{date.fromisoformat(TODAY).strftime('%Y년 %m월 %d일')} 기준 · 프로젝트 cabbage-surge · 쉬운 말로 정리한 보고서 · "
                "수치는 저장소 outputs/*.csv에서 직접 계산 (tools/make_report.py)", "sub"),
              P("<b>한 줄 요약</b>", "h2"),
              P(f"2024년 9월 배추값이 한 달 만에 45% 뛰었는데, 기존 연구의 모형은 이를 거의 예측하지 못했습니다. "
                f"'평년 이맘때'를 기준점으로 삼는 단순 선형 모형(경보 모형)은 이 급등을 <b>{e24[HEAD]:+.0f}%</b> 차이까지 따라갔고, "
                f"이 모형의 설정은 <b>2024년 결과를 보지 않고 2016~2023년 자료만으로 고른 것</b>입니다. "
                f"다만 그 모형 혼자서는 평소 정확도가 가격만 보는 모형보다 나쁘고(오차율 {mp[HEAD]:.1f}% 대 {base_mape:.1f}%), 급등이 없던 2025년에는 {e25[HEAD]:+.0f}%로 잘못 경보를 냈습니다. "
                f"그래서 <b>KREI 월보의 여름배추 생산 전망</b>을 스위치로 두어, 공급이 줄 전망이면 경보 모형을, 늘 전망이면 평소에 잘 맞는 트리 모형을 쓰는 <b>최종 모형</b>을 만들었습니다(2026-10-12). "
                f"같은 날 세 가지를 더 얹었습니다. 1~4월 예측은 겨울 전용 가격 모형으로, 여름 경보 달과 11~12월은 생산 전망만큼 예측을 올리거나 내리고, 9월 말의 10월 예측은 가을 산지의 9월 고온 일수만큼 하락을 늦춥니다. "
                f"10월 13일에는 두 가지를 고쳤습니다. 경보 모형의 고랭지 날씨는 <b>여름배추가 출하되는 때에만</b> 쓰고, 스위치는 생산이 줄 전망이어도 월보가 <b>그 달 출하량을 '증가'로 본 호</b>에서는 경보를 켜지 않습니다. "
                f"최종 모형은 2024년 9월 <b>{e24[FINAL]:+.0f}%</b>(스위치만 {e24[GATE]:+.0f}%), 2021~2025년 오차율 <b>{mp[FINAL]:.1f}%</b>(스위치만 {mp[GATE]:.1f}%, 기준 {base_mape:.1f}%)로 성공 기준 두 가지를 함께 넘습니다. "
                f"남는 비용은 2025년 9월 {e25[FINAL]:+.0f}% 과대 예측, 오차가 30%를 넘는 달 {len(big[FINAL])}개({len(big[GATE])}개에서 늘어남), 주간 예측에서의 겨울 분기 손해이고, 스위치 기록이 여섯 해뿐이라는 한계는 그대로입니다. "
                f"출하량 전망을 엮은 것은 2021~2025년 월말 예측을 하나도 바꾸지 않았고, 바뀐 것은 2026년 여름 {len(chg)}개 달뿐입니다(2021~2026년 오차율 {mp26[PREV]:.1f}%에서 {mp26[FINAL]:.1f}%로).", "box"),
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
              P("매달 말일에 서서, 그 시점까지 알 수 있는 정보만으로 다음 달 평균 가격을 맞히는 과제입니다. 2016년 1월부터 2026년 8월까지 "
                f"반복해 평균 성적을 내고(2021~2025년 {n60}회를 지표로 사용), 2024년 8월 말(9월 예측)을 특별히 봅니다. "
                "매주 일요일에 서서 4주 뒤를 맞히는 주간 버전도 같이 돌렸습니다. 학습에는 예측 시점 이전에 결과가 다 나온 자료만 씁니다(정보 누수 금지)."),
              P("<b>쓴 데이터</b> (농식품 가격정보 시스템 CABIS에서 복사, 2026-10-09 갱신)", "h2"),
              table([["데이터", "내용", "기간"],
                     ["가락시장 일별 가격·반입량", "맞혀야 할 값. 상품 등급, 원/10kg", "2010~2026.10"],
                     ["고랭지 날씨", "평창·태백·정선·강릉 4곳의 기온·강수. '평년 대비 얼마나 더웠나'로 바꿔 사용. 여름 밖의 예측에서는 여름배추가 출하되는 기간만큼만 반영(2026-10-13)", "2010~2026.10"],
                     ["가을배추 주산지 날씨", "해남·고창·아산 등 9곳의 9월 30도 이상 일수. 9월 말의 10월 예측에만 사용", "2010~2026.10"],
                     ["정부 비축 방출량", "2022년부터만 있어 본 모형에서는 제외 (4절 참고). 별도 실험(case01)에서만 사용", "2022~"],
                     ["KREI 월보 호별 여름배추 전망", "생산량·면적 전년비. 최종 모형의 스위치 신호와 경보 달 수준 조정(data/ext/krei_summer_issues.csv)", "2021~2026"],
                     ["KREI 월보 호별 그 달 출하량 전망", "6~10월호가 전망한 그 달 배추 출하량의 전년 대비 방향(증가·비슷·감소)만. 스위치의 경보를 끄는 데만 사용, 2026-10-13(data/ext/krei_shipment_outlook.csv)", "2021~2026"],
                     ["KREI 월보 호별 가을배추 전망", "생산량·면적 전년비. 11~12월 예측의 수준 조정(data/ext/krei_nonsummer_issues.csv)", "2021~2026"]],
                    [42, 98, 28], st),
              P("월보는 담당자가 전달 말에 다음 달 호를 쓰기 때문에, 월말에 설 때는 다음 달 호의 값을 봅니다(8월 말에는 9월호). 월보의 가격 전망치와 포전거래 가격은 쓰지 않습니다.", "small"),
              P("<b>비교한 모형</b>", "h2"),
              table([["보고서 표기", "무엇을 보는가 / 성격"]] + [[n, d] for _, n, d in MODELS], [52, 116], st, bold_rows=(len(MODELS),)),
              P("'선형'은 변수와 가격 사이를 직선 관계로 보는 단순한 모형(ridge), '트리'는 조건을 나눠가며 규칙을 찾는 모형(HGB)입니다. "
                "단순한 모형은 겪어보지 못한 폭염에도 예측을 늘려 잡을 수 있고, 트리 모형은 경험 범위를 벗어나면 평평해집니다. "
                "경보 모형은 설정(변수 7개, 규제 강도)을 2016~2023년 성적만으로 골랐기 때문에 '2024년 정답을 보고 맞췄다'는 비판을 피합니다.", "small"),
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
              B(f"<b>경보 모형 계절편차(사전선택)은 {e24[HEAD]:+.0f}%</b>로 기준선(-10%)에 걸립니다. 그러나 급등이 없던 2025년에는 "
                f"{e25[HEAD]:+.0f}%로, 오르지 않을 값을 오른다고 한 셈입니다. 여름만 학습한 사후 설정은 {e24['an_sparse_summer']:+.0f}%까지 따라가지만 "
                f"2025년 거짓 경보가 {e25['an_sparse_summer']:+.0f}%로 더 큽니다."),
              B("급등을 잘 따라가는 모형일수록 급등 없는 해에 과대 예측하는 맞교환 관계가 뚜렷합니다."),
              B(f"<b>최종 모형은 2024년 {e24[FINAL]:+.0f}%, 2025년 {e25[FINAL]:+.0f}%</b>입니다. 2024년 8월 말에 본 {s24['issue']}월호는 생산 전망 {s24['S']:+.1f}%에 {s24['issue']}월 출하량도 {s24['Q']} 전망이라 경보 모형이 쓰였고({e24[GATE]:+.0f}%), "
                f"같은 값만큼 수준 조정({adj24:+.1f}%)이 더해졌습니다. "
                f"2025년 {s25['issue']}월호는 {s25['S']:+.1f}%라 여름 특화 트리로 바뀌어 과대 예측이 {e25[HEAD]:+.0f}%에서 {e25[FINAL]:+.0f}%로 줄었습니다. 같은 스위치로 2021년은 {e21[HEAD]:+.0f}%→{e21[FINAL]:+.0f}%, "
                f"2023년은 {e23[HEAD]:+.0f}%→{e23[FINAL]:+.0f}%가 됩니다."),
              ] + ([B(f"<b>2026년 9월(표본외, 자료 갱신으로 새로 열린 원점)</b>: 2026년 여름은 고랭지 폭염이 없어 서늘했고 9월 가격은 {oos['actual']:,.0f}원으로 급등이 없었습니다. "
                      f"8월 말에 본 {oos['sig']['issue']}월호는 생산 전망 {oos['sig']['S']:+.1f}%에 출하량도 {oos['sig']['Q']} 전망이라 최종 모형은 경보 모형에 작은 수준 조정({oos['sig']['adj']:+.1f}%)을 더해 썼는데도 "
                      f"{oos[FINAL]:+.0f}%로 거짓 경보를 내지 않았습니다(가격만 보는 모형 {oos['price_only_ridge']:+.0f}%, Chronos-2 {oos['chronos2']:+.0f}%, 평시 모형이었다면 {oos['wx_hgb_summer']:+.0f}%). "
                      "8월 말 가격이 평년보다 낮았기 때문에 경보 모형도 낮게 본 것입니다. 이 값은 이 해를 보기 전에 정한 규칙에서 나왔고 나중에 넣은 출하량 전망 규칙으로도 바뀌지 않아 뜻이 있지만, 한 건이므로 사례로만 봅니다.")]
                   if oos is not None else []) + [
              P("3-2. 평소 정확도는 어떤가", "h2"),
              P(f"2021~2025년 {n60}번의 월말 예측에서 평균 오차율(MAPE)입니다. 낮을수록 좋고, 주황 점선이 가격만 보는 모형의 기준선입니다."),
              fig_mape(m),
              B(f"모형 하나만 쓴다면 {NAME[best_mape]}이 {mp[best_mape]:.1f}%로 평소 정확도가 가장 좋지만, 2024년 9월은 {e24[best_mape]:+.0f}%로 급등을 못 따라갑니다."),
              B(f"<b>최종 모형은 {mp[FINAL]:.1f}%로 기준선({base_mape:.1f}%)을 넘습니다.</b> 스위치만으로 {mp[GATE]:.1f}%(급등 없는 해를 트리 모형이 맡음), "
                f"겨울 분기가 1~4월을 {seas_mape((1, 2, 3, 4), GATE):.1f}%에서 {seas_mape((1, 2, 3, 4), FINAL):.1f}%로(고랭지 날씨를 1년 내내 쓰던 종전 경보 모형은 {seas_mape((1, 2, 3, 4), HEAD):.1f}%), 수준 조정이 여름 원점을 {mps[GATE]:.1f}%에서 {mape(m, 'final_base', summer_only=True):.1f}%로, 11~12월을 {seas_mape((11, 12), GATE):.1f}%에서 {seas_mape((11, 12), FINAL):.1f}%로 줄였습니다. "
                f"9월 고온 층은 2024년 10월 한 달을 {err_at(m, '2024-09-30', 'final_base'):+.0f}%에서 {err_at(m, '2024-09-30', FINAL):+.0f}%로 바꿨고, 다른 해는 사실상 그대로입니다(전체 {mape(m, 'final_base'):.1f}%에서 {mp[FINAL]:.1f}%로)."),
              B(f"<b>오차가 30%를 넘는 달은 오히려 늘었습니다.</b> 2021년 1월~2026년 9월 69개 달 중 스위치만 {len(big[GATE])}개, 최종 모형 {len(big[FINAL])}개({big_txt}). "
                "겨울 전용 모형이 2021년 1월의 바닥 반등을 더 못 봤고, 수준 조정이 이미 높게 본 2022년 8월을 더 올렸기 때문입니다."),
              B(f"경보 모형 혼자는 {mp[HEAD]:.1f}%로 기준선을 넘지 못합니다. 급등을 따라가는 대신 평소 오차가 {mp[HEAD] - base_mape:.1f}%p 늘어난 것입니다. "
                f"사후 설정(여름만)의 {mp['an_sparse_summer']:.1f}%는 기준선을 넘지만, 학습 자료 초기 100행(6%)을 빼면 15.2%로 뒤집히는 불안정한 값이라 근거로 쓰지 않습니다."),
              B(f"Chronos-2와 MSTL은 급등은 절반쯤 잡지만 평소 오차율은 {mp['chronos2']:.1f}~{mp['mstl_ets']:.1f}%로 기준선보다 나쁩니다. 추세를 연장하는 방식의 비용입니다."),
              P("3-3. 주간으로 보면: 상승은 미리 잡고, 하락 전환은 놓친다", "h2"),
              P("매주 일요일에 서서 4주 뒤 평균을 예측한 결과입니다. 검은 선이 실제, 색 선이 모형입니다."),
              fig_weekly(w),
              B(f"경보 모형(2024년 여름은 스위치가 내내 경보라 최종 모형은 여기에 +4% 안팎의 수준 조정을 더한 값)은 8월 하순~9월 중순에 {wk_err('2024-09-01', HEAD):+.0f}~{wk_err('2024-09-08', HEAD):+.0f}% 안에서 실제에 붙어 "
                f"상승 국면을 따라갑니다(가격만 보는 모형은 같은 시점 {wk_err('2024-09-01', 'price_only_ridge'):+.0f}%)."),
              B(f"그러나 10월 들어 가격이 떨어질 때는 모든 모형이 높게 예측합니다(10월 20일 시점 경보 모형 {wk_err('2024-10-20', HEAD):+.0f}%). "
                "가을배추가 나오기 시작하면 값이 내린다는 신호가 데이터에 없기 때문입니다."),
              B(f"주간 전체(2021~2025, {nW}번)로는 최종 모형이 오차율 {mpw[FINAL]:.1f}%(스위치만 {mpw[GATE]:.1f}%)로 "
                + ("가장 좋고" if mpw[FINAL] <= min(mpw[c] for c in ("chronos2", "wx_hgb", "price_only_ridge")) else "비슷한 수준이고")
                + f", 날씨 추가(트리) {mpw['wx_hgb']:.1f}%, Chronos-2 {mpw['chronos2']:.1f}%, "
                f"경보 모형 혼자는 {mpw['an_sparse_all']:.1f}%로 가격만 보는 모형({mpw['price_only_ridge']:.1f}%)보다 나쁩니다. "
                f"단, 주간에서는 겨울 분기가 손해입니다(1~4월 대상: 종전 경보 모형 {wwin(HEAD):.1f}%, 겨울 분기 {wwin(FINAL):.1f}%). 주간의 소폭 개선은 수준 조정에서만 옵니다."),
              P("3-4. 10월 13일에 고친 두 가지", "h2"),
              B("<b>고랭지 날씨는 여름배추가 출하되는 때에만.</b> 종전에는 강원 고랭지 4곳의 날씨를 1년 내내 변수로 썼습니다. 봄·가을 가격을 예측할 때도 여름배추 산지의 날씨를 본 셈입니다. "
                "이제는 '앞으로 28일 가운데 여름 작형 수확 기간(6월 중순~10월)에 드는 날의 비율'만큼만 반영하고, 6~9월 원점은 종전 그대로 둡니다. "
                + (f"2021~2025년 월말 오차율이 {mp_v3:.1f}%에서 {mp[FINAL]:.1f}%로 조금 줄었습니다. " if mp_v3 is not None else "")
                + (f"이득이 작고 주간 예측에서는 30%를 넘게 틀리는 원점이 {n30w_v3}개에서 {big30w(w, PREV)}개로 늘어난다는 점을 함께 적어 둡니다. " if n30w_v3 is not None else "이득은 작습니다. ")
                + "다른 작형(봄·가을·겨울) 주산지의 날씨는 시기를 맞춰 넣어도 나빠져서 넣지 않았습니다(case27)."),
              B("<b>스위치에 월보의 '그 달 출하량 전망'을 엮음.</b> 급등 해(2022·2024)의 월보는 여름 내내 생산 전망과 출하량 전망이 둘 다 '감소'였습니다. 2026년 7·8월호는 '생산은 소폭 줄지만 출하량은 는다'였는데, "
                "생산 전망만 보던 스위치는 이 두 달을 경보로 분류해 높게 예측했습니다. 그래서 <b>생산 전망이 0 이하이고 출하량 전망도 '증가'가 아닐 때만 경보</b>로 바꿨습니다. "
                "출하량 전망은 경보를 끄는 데만 씁니다. 생산이 늘 전망인 해에 출하량 감소 문장 하나로 경보를 켜면 주간 예측에서 크게 틀렸기 때문입니다(case30)."),
              ] + ([table([["원점 → 예측 대상", "본 호: 생산 전망 / 출하량 전망", "생산 전망만 볼 때", "출하량 전망을 엮은 뒤"]] + chg_rows, [36, 52, 40, 40], st)] if chg_rows else []) + [
              B(f"2021~2025년 월말 예측은 하나도 바뀌지 않습니다({mp[PREV]:.1f}% 그대로). 2026년까지 넣으면 월말 {n68}번의 오차율이 {mp26[PREV]:.1f}%에서 {mp26[FINAL]:.1f}%로, 주간은 {mpw26[PREV]:.1f}%에서 {mpw26[FINAL]:.1f}%로 줄어듭니다. "
                f"주간에서 바뀌는 원점은 {len(chg_w)}개이고 2021~2025년 주간 오차율은 {mpw[PREV]:.2f}%에서 {mpw[FINAL]:.2f}%로 사실상 같습니다."),
              B("<b>이 표는 검증이 아닙니다.</b> 출하량 전망을 엮는 규칙은 2026년 8월의 과대 예측을 본 뒤에 만들었습니다. 2021~2025년 월말에는 '생산 감소 + 출하 증가'인 달이 없어서 확인할 표본이 없고, "
                "주간의 2024년 6월 네 원점이 유일한 과거 사례인데 좋아진 것과 나빠진 것이 반반입니다. 2026년은 이 규칙으로 한 해 안에서 상태가 바뀐 첫 해가 됐습니다(6·7월 말 평시, 8월 말 경보)."),
              P("3-5. 성공 기준 판정 (최종 모형 기준)", "h2"),
              table([["기준", "최종 모형 (스위치 + 겨울 분기 + 수준 조정)", "판정"],
                     ["2024년 9월 오차 -10% 이내", f"{e24[FINAL]:+.0f}% (경보 모형 {e24[GATE]:+.0f}%에 수준 조정 {adj24:+.1f}%)", "충족"],
                     [f"전체 오차율 {base_mape:.1f}% 이하", f"{mp[FINAL]:.1f}%", "충족"],
                     ["급등 없는 해에 거짓 경보 없을 것", f"2025년 9월 {e25[FINAL]:+.0f}% (경보 모형 혼자면 {e25[HEAD]:+.0f}%)", "부분 충족"],
                     ["설계를 결과 보기 전에 정했을 것", "경보 모형의 변수·규제 강도는 2016~2023년 자료만으로 선택(case03). 스위치 규칙과 평시 모형 선택은 2024·2025년을 본 뒤 정함(case06·case10). 겨울 분기와 수준 조정 배율은 2016~2020년에서 정했으나 둘을 함께 얹기로 한 것은 결과를 본 뒤(case15·case16). 9월 고온 층은 2023·2024년 두 해를 본 뒤 만든 규칙(case17). 스위치에 출하량 전망을 엮은 것은 2026년 8월 오차를 본 뒤(case30)", "부분 충족"],
                     ["학습 자료 일부를 빼도 결과가 유지될 것", "경보 모형: 학습 풀·추출 시작점·규제 18조합에서 2024년 -9.5~-12.4%, 2025년 +20~+22%. 스위치: 여섯 해 기록이라 검사 불가", "경보 모형만 충족"]],
                    [50, 88, 30], st),
              P(f"<b>종합:</b> 최종 모형은 두 숫자 기준을 함께 넘습니다. 그러나 세 가지를 분명히 해야 합니다. 첫째, 2024년 9월은 경보 모형 혼자 {e24[GATE]:+.0f}%로 기준선에 딱 걸리고, {e24[FINAL]:+.0f}%가 된 것은 경보 달을 일괄로 조금 올리는 수준 조정 덕분입니다. "
                f"둘째, 스위치 규칙(전망 0 이하)과 평시 모형 선택은 2024·2025년을 본 뒤, 출하량 전망을 엮는 것은 2026년을 본 뒤 정한 것이며 기록은 2021~2026년 여섯 해(급등 두 번)뿐입니다. "
                f"셋째, 평균 오차율은 줄었지만 오차가 30%를 넘는 달은 {len(big[GATE])}개에서 {len(big[FINAL])}개로 늘었고, 겨울 분기는 주간 예측에서는 손해입니다. 정직한 서술은 이렇습니다. "
                f"\"2024년을 보지 않고 고른 계절편차 선형 모형이 2024년 9월 급등을 {e24[HEAD]:+.0f}% 차이로 따라가고, KREI 월보 생산 전망으로 경보를 끄고 켜면 2021~2025년 오차율이 "
                f"{mp[GATE]:.1f}%로 가격만 보는 모형({base_mape:.1f}%)보다 낫다. 여기에 2016~2020년에 정한 겨울 전용 모형과 생산 전망 수준 조정을 더하면 {mape(m, 'final_base'):.1f}%, 2024년 9월 {e24[FINAL]:+.0f}%가 되고, 두 해 사례로 만든 9월 고온 층까지 넣으면 {mp[FINAL]:.1f}%다. "
                f"비용은 2025년 9월 {e25[FINAL]:+.0f}% 과대 예측과 30% 초과 달의 증가, 한계는 사후 설계와 여섯 해의 기록이다.\"", "box"),
              PageBreak()]

    # ---- 4. 알게 된 것
    story += [P("4. 알게 된 것", "h1"),
              B("<b>문제는 '가격 정보 부족'이 아니라 '모형 구조'였습니다.</b> 같은 가격 자료로도 추세를 연장하는 모형(Chronos-2, MSTL)은 급등의 절반 이상을 잡습니다. "
                f"가격만 보는 선형 모형이 {e24['price_only_ridge']:+.0f}%였던 것은 '평균으로 돌아간다'는 가정이 강하게 들어가 있었기 때문입니다."),
              B("<b>급등을 따라간 힘은 폭염 변수가 아니라 '기준점'이었습니다(2026-10-12 정정).</b> 경보 모형의 변수별 기여를 나눠 보니 2024년 8월 말의 기온 변수 기여(+7%p)는 강수 변수(-6%p)가 거의 상쇄했고, "
                "2024년 9월 -10%는 '8월 말 가격이 이미 평년보다 25% 높고 그 상태가 이어진다'는 항에서 나왔습니다. 폭염 변수를 뺀 3변수 모형도 2024년 -10%로 같습니다. "
                "가격만 보는 모형(-37%)과의 차이는 기준점을 '최근 한 달 평균'으로 두느냐 '평년 이맘때'로 두느냐입니다. 이 결론은 2024년을 보지 않고 고른 설정(case03)에서도 성립합니다."),
              B("<b>2024년과 2025년을 가른 것은 날씨가 아니라 공급 전망이었습니다.</b> 두 해 모두 기록적 폭염이었지만 KREI 8월호 여름배추 생산 전망은 2024년 -7.2%, 2025년 +8.8%였습니다. "
                "이 전망을 스위치로 쓴 최종 모형이 거짓 경보를 줄였습니다. 봄배추 저장 소진율(case07)도 같은 방향을 가리키지만 확인 신호로만 둡니다."),
              B("<b>생산이 조금 줄어도 출하가 늘 전망이면 급등 조건이 아니었습니다(2026-10-13, 한 해 사례).</b> 2026년 7·8월호는 생산 전망이 -3.5%·-0.9%였지만 그 달 출하량은 '증가'로 봤고, 실제로 급등이 없었습니다. "
                "담당자 설명으로는 <b>봄배추 저장량이 늘어난 영향</b>이 큽니다. 2026년 대량수요처 저장 계획량은 4.7만 톤으로 전년보다 10% 많았고, 8월 가락시장 반입의 31%가 저장배추였습니다(급등한 2024년은 12%). "
                "저장분 출하는 해마다 9월 둘째 주에 끝나는데, 9월호가 출하량 '감소'로 돌아선 것과 맞습니다. 여름배추 생산 전망에는 저장분이 들어 있지 않으므로 출하량 전망이 그 빈칸을 메우는 셈입니다. "
                "급등 해의 월보는 생산과 출하량이 함께 '감소'였습니다. 다만 급등 해를 가르는 것은 여전히 생산 전망이고, 출하량 전망은 경보를 끄는 데만 씁니다. 저장량 수준만으로는 급등을 가르지 못했습니다(2024년은 저장량이 늘었다고 추정됐는데도 급등, case07)."),
              B("<b>여름 밖에서는 고랭지 날씨가 잡음이었습니다(2026-10-13).</b> 그 지역에서 배추가 나오지 않는 철에는 그 지역 날씨를 보지 않는 편이 조금 나았습니다. 봄·가을·겨울 산지의 날씨를 시기에 맞춰 더 넣는 것은 도움이 되지 않았습니다."),
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
                  (f"계절편차 7변수 모형은 미리 골라도 2024-09 {sel_row['err_2024-08']:+.0f}%(2025-08 {sel_row['err_2025-08']:+.0f}%). " if sel_row is not None else "계절편차 7변수 모형은 미리 골라도 2024-09 -11%. ")
                  + "날씨 변수는 2016~2023 평소 정확도를 높이지 않음. 미리 고른 설정이 2026-10-09 본 모형의 대표 설정으로 채택됨",
                  "완료 → 경보 모형 설정으로 채택"],
                 ["case05·06 공급 스위치", "aT 비축 잔량과 KREI 월보 생산 전망으로 2024(급등)·2025(급등 없음)를 가르고, '공급 늘 전망'이면 경보를 끔",
                  f"KREI 8월호 전망 2024 -7.2% / 2025 +8.8%. 스위치를 얹은 모형이 2025-09 {e25[HEAD]:+.0f}%→{e25[GATE]:+.0f}%, 오차율 {mp[HEAD]:.1f}%→{mp[GATE]:.1f}%. 2017~2020 잔량·주간 점검 통과", "승격 → 최종 모형"],
                 ["case10 평시 모형 선택", "스위치가 꺼졌을 때 쓸 모형 3후보(가격만 / 여름 특화 트리 / 폭염 뺀 계절편차) 비교",
                  "폭염 뺀 모형은 2025 +22%로 효과 없음(폭염 변수 기여가 거의 0임을 확인). 여름 특화 트리를 얹은 모형이 오차율 13.2%(당시 값)로 최선, 가격만은 2025 -9%가 유일한 우위. 사용자가 트리 선택", "승격 → 최종 모형"],
                 ["case15 겨울 분기", "예측 대상 달의 작형(겨울·봄·여름·가을)마다 다른 모형을 쓰는 다중모형",
                  f"미리 고른 4작형 다중모형은 스위치 모형보다 못함(16.0~16.6%). 두 구간에서 모두 안정적인 조각은 겨울 전용 가격 모형뿐 → 1~4월 {seas_mape((1, 2, 3, 4), HEAD):.1f}%→{seas_mape((1, 2, 3, 4), FINAL):.1f}%. 주간은 {wwin(HEAD):.1f}%→{wwin(FINAL):.1f}%로 손해", "겨울만 승격"],
                 ["case16 생산 전망 수준 조정", "겨울·봄·가을배추 생산 전망을 예측값에 곱하는 배율로",
                  f"문헌·구조 신축성을 그대로 곱하면 붕괴(28%). 2016~2020년 잔차 반응으로 정한 작은 배율(여름 경보 -0.57, 가을 -0.22)만 유효 → 여름 원점 {mps[GATE]:.1f}%→{mape(m, 'final_base', summer_only=True):.1f}%, 2024-09 {e24[GATE]:+.0f}%→{e24[FINAL]:+.0f}%", "여름 경보·가을만 승격"],
                 ["case17 9월 고온 층", "가을배추 주산지의 9월 30도 이상 일수가 많으면 10월 하락이 늦어진다는 가설",
                  f"더웠던 2023·2024년은 10월 가격이 9월의 0.8배 수준(다른 해 0.36~0.47배). 월말 9월 원점에만 넣으면 2024-10 {err_at(m, '2024-09-30', 'final_base'):+.0f}%→{err_at(m, '2024-09-30', FINAL):+.0f}%, 다른 해는 그대로. 근거 두 해, 주간에는 역효과라 미적용", "사용자 결정으로 승격"],
                 ["case27 작형별 산지 날씨", "산지마다 그 지역에서 배추가 자라고 나오는 시기에만 그 지역 날씨를 보기",
                  (f"고랭지 날씨를 여름배추 출하기에만 켜면 월말 2021~2025 {mp_v3:.1f}%→{mp[PREV]:.1f}%. " if mp_v3 is not None else "고랭지 날씨를 여름배추 출하기에만 켜면 소폭 개선. ")
                  + "봄·가을·겨울 주산지 날씨는 시기를 맞춰 넣어도 악화", "고랭지 출하기 한정만 승격"],
                 ["case30 출하량 전망 엮기", "월보가 전망한 그 달 배추 출하량의 방향을 생산 전망과 엮어 스위치를 판정",
                  f"'생산 감소 + 출하량도 증가 아님'일 때만 경보. 2021~2025 월말 불변, 2026년 {len(chg)}개 달만 바뀜(2021~2026 {mp26[PREV]:.1f}%→{mp26[FINAL]:.1f}%). 출하량 감소로 경보를 켜는 규칙은 주간에서 악화. 근거가 사실상 2026년 한 해", "사용자 결정으로 승격"],
                 ["case09 학습 분위수", "공급 신호로 Chronos-2 분위수 수준을 과거 기록에서 학습해 해마다 갱신",
                  "같은 달 기록 3~5개로는 Chronos-2의 계절 편향을 먼저 배워 2026 +22% 거짓 경보. 기록이 8개 해 이상 쌓이기 전에는 재시도 안 함", "보류"]]
    story += [P("5. 주요 실험 케이스", "h1"),
              P("본 모형을 바꾸지 않고 별도 폴더(experiments/)에서 해본 시도입니다. 케이스가 본 모형에 들어가려면 각 케이스 노트의 승격 조건을 채우고 사용자가 승인해야 합니다. "
                "case03(경보 모형 설정), case06·case10(스위치와 평시 모형), case15·case16(겨울 분기와 수준 조정), case17(9월 고온 층), case27(고랭지 날씨는 출하기에만), case30(출하량 전망)이 승격된 사례입니다. "
                "나머지 가운데 반입량으로 공급 부족을 재는 시도(case08·20·25·26), 예측을 범위로 자르는 가드(case21), 월보 가격 전망과의 결합(case23), 봄 급락 경보(case24)는 기각됐습니다. 케이스 01~30 전체 목록은 README."),
              table(case_rows, [30, 50, 68, 20], st, bold_rows=(3, 4, 5, 6, 7, 8, 9, 10))]

    # ---- 6. 한계
    story += [P("6. 한계와 조심할 점", "h1"),
              B("급등 사례가 적습니다(2021년 이후 2022·2024년 두 번). 2024년 9월 한 점에 맞춰 설계를 바꾸지 않도록 2020년·2022년 9월, 2025년 9월을 항상 같은 표에 놓고 봅니다."),
              B("'미리 골랐다'에도 한도가 있습니다. 후보 설정의 범위와 정부 방출 변수 제외는 2024년 결과를 본 뒤의 지식입니다. 격자 안에서의 선택만 2024년을 가렸습니다."),
              B("10월 하락 전환은 여전히 약점입니다. 9월 고온 층은 2023·2024년 두 해 사례로 만든 조정이고, 과거 검증에서 실제로 바꾼 달은 2024년 10월 하나입니다. 올리는 폭은 2023년 한 해에서 배운 값을 늘려 쓴 것이라 2024년 오차가 0%가 된 것은 우연으로 봐야 합니다. "
                "주간 예측에는 반대로 작용해 적용하지 않습니다. 2026년 10월이 첫 표본외 점검입니다."),
              B(f"오차가 30%를 넘는 달이 {len(big[FINAL])}개 남아 있습니다({big_txt}). 대부분 값이 오르거나 내리는 때가 평년보다 몇 주 늦거나 빨랐던 달입니다."),
              B("스위치와 수준 조정은 담당자가 전달 말일에 다음 달 월보 내용을 안다는 전제로 다음 달 호 값을 씁니다(8월 말에는 9월호). 그 전제가 없는 사용자에게는 한 호 늦은 값이 들어갑니다."),
              B("출하량 전망을 엮은 규칙은 근거가 사실상 2026년 한 해입니다. 월보 문장의 방향만 읽은 것이고(판독이 약한 호가 있음), 출하량 전망은 월보의 가격 전망과 같은 판단에서 나오므로 모형이 월보 판단에 그만큼 더 기댑니다. "
                "생산 전망이 0 근처일 때를 어떻게 다룰지는 아직 정하지 않았습니다. 출하량 전망이 '감소'인 호는 생산 전망이 -0.1%여도 경보입니다."),
              B("평시·경보 구분은 6~9월 말 원점에만 있습니다. 그 밖의 달에는 급등을 따로 다루는 장치가 없고, 급락을 미리 알리는 장치도 없습니다."),
              B("KREI 두 연구와 CABIS 자료는 인용 가능하다고 확인됐지만(2026-10-07), 공개 저장소에 원문 PDF를 올릴지는 별도 판단입니다."),
              B("트리 모형 결과는 실행 환경이 다르면 ±2~3%p 흔들릴 수 있습니다. 선형 모형은 같은 자료면 같은 결과가 나옵니다(Windows와 맥에서 소수점까지 일치 확인).")]

    # ---- 7. 다음 단계
    story += [P("7. 다음 단계", "h1"), P("<b>결정이 필요한 것 (사용자)</b>", "h2"),
              B("생산 전망이 0 근처(±3% 안)일 때의 처리. 2026년이 첫 사례입니다"),
              B("이 결과를 논문으로 쓸지, 쓴다면 저널인지 연구보고인지. 방향이 '진단'에서 '작동하는 방안'으로 바뀌었으므로 연구 질문을 다시 쓸 것"),
              B("정부 방출 자료를 2010~2020년까지 확보할 수 있는지 (case01 트리거를 학습 변수로 올리는 조건)"),
              P("<b>바로 할 수 있는 분석</b>", "h2"),
              B("급등 국면 전용 모형: 경보 상태에서 경보 모형 대신 급등 해(2010·2012·2016·2019·2020·2022·2024)만으로 학습한 모형을 써서 2022·2024의 남은 과소를 줄일 수 있는지"),
              B("작형이 바뀌는 달(5·7·10·12월 대상)은 점 예측을 고치는 대신 범위를 함께 제시하는 방법. 월말에 볼 수 있는 가격·날씨·월보·산지 반입으로는 이 달들의 오차를 줄이지 못했습니다(case18~20)"),
              B("준고랭지 2기작 정식면적(2021~2025 읍면별 자료)을 스위치의 보조 신호로"),
              B("2026년 10월호가 나오면 판독표 셋(여름 생산 전망, 가을 전망, 10월 출하량 방향)에 넣고 10월 전망을 다시 계산. 10월 실적이 나오면 9월 고온 층의 첫 표본외 점검"),
              B("봄배추 공급 정보: 봄에서만 신호가 세 번 나왔으므로(월보 전망, 봄 주산지 날씨, 실측 단수) 2026년산 실측이 들어오면 표본외로 확인"),
              B("매해 새 호가 나올 때마다 KREI 월보 값을 추가하고 스위치 규칙·평시 모형 선택을 다시 점검 (기록이 쌓일수록 사후 설계 비판이 약해짐)")]

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
                            title="배추 도매가격 예측모형 결과 보고서", author="cabbage-surge")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / f"배추급등예측_결과보고서_{TODAY}.pdf"))
    a = ap.parse_args()
    build(Path(a.out))
