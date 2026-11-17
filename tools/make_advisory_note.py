"""자문회의 원고용 참고자료(쉬운 모형 설명 + 11월 가격 전망)를 PDF와 Word로 만든다.

양식은 사용자의 자문회의 원고 '참고' 꼭지(제목 상자, ■ 소제목, • 본문, - 세부, * 주, 〈그림 제목〉, 쪽 번호)를 따른다.
수치는 outputs/forecast_live.csv, outputs/gate_predictions{,_W}.csv, data/raw/clean_daily.csv, data/ext/krei_nonsummer_issues.csv 에서 읽는다.
손으로 적은 값: 관측 11월호 11월 출하량 전망(전년 대비 +24.1%, 사용자 제공 문구). 범위 병기 값 11.9%·13.8%(README 6차 편입 절).

Usage: python tools/make_advisory_note.py [원점 YYYY-MM-DD] [참고 번호]     (기본 2026-10-25, 3)
Output: docs/참고자료_배추가격전망_<원점>.pdf, .docx, _그림1_*.png, _그림2_*.png (그림은 제목 없이, 220dpi)
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT, RAW, EXT, DOCS = ROOT / "outputs", ROOT / "data" / "raw", ROOT / "data" / "ext", ROOT / "docs"
sys.path.insert(0, str(ROOT / "src"))
import gate_model as gm  # noqa: E402

ORIGIN = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2026-10-25")
NOTE_NO = sys.argv[2] if len(sys.argv) > 2 else "3"
SHIP_NOV_YOY = 24.1          # 관측 11월호: 11월 배추 출하량 전년 대비(사용자 제공 문구)
RANGE_NO_THIN, RANGE_NO_GATE = 11.9, 13.8   # 범위 병기(README 6차 편입 절, tools/verify_stage6.py 로 검산)

FONT_FILE, FONT_FILE_B = r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunbd.ttf"
INK, INK2, GRID = "#1f1f1f", "#55534f", "#e4e2dd"
ORANGE, GREY, BLUE = "#e8622c", "#a3a19c", "#2a78d6"

fm.fontManager.addfont(FONT_FILE)
fm.fontManager.addfont(FONT_FILE_B)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_FILE).get_name()
plt.rcParams["axes.unicode_minus"] = False


# ------------------------------------------------------------------ 수치
def numbers() -> dict:
    live = pd.read_csv(OUT / "forecast_live.csv", parse_dates=["origin"], encoding="utf-8-sig")
    r = live[live.origin == ORIGIN].iloc[0]
    fall = gm.load_fall_issues()
    ty = (ORIGIN + pd.Timedelta(days=15)).year
    nov = fall[(fall.crop_year == ty) & (fall.issue_month == 11)]
    s_nov = float(nov.prod_yoy.iloc[0])
    pulled = float(r["head"]) * (1.0 + gm.BETA_FALL * float(np.clip(s_nov, -gm.S_CAP, gm.S_CAP)) / 100.0)
    p = pd.read_csv(RAW / "clean_daily.csv", parse_dates=["date"])
    p = p[p.price > 0]
    t0, t1 = ORIGIN + pd.Timedelta(days=1), ORIGIN + pd.Timedelta(days=28)
    ly = p[(p.date >= t0 - pd.DateOffset(years=1)) & (p.date <= t1 - pd.DateOffset(years=1))].price.mean()
    g = pd.read_csv(OUT / "gate_predictions.csv", parse_dates=["origin"], encoding="utf-8-sig")
    w = pd.read_csv(OUT / "gate_predictions_W.csv", parse_dates=["origin"], encoding="utf-8-sig")
    yr = g.origin.dt.year
    mape = lambda d, mk, c="final": float(((d[c] / d.actual - 1).abs() * 100)[mk].mean())
    oct_end = g[(g.origin.dt.month == 10) & yr.between(2021, 2025)]
    md = w.origin.dt.strftime("%m-%d")
    late = w[(md >= "10-18") & (md <= "11-01") & w.origin.dt.year.between(2021, 2025)]
    late_err = (late.final / late.actual - 1) * 100
    tm = (g.origin + pd.Timedelta(days=15)).dt.month
    nov_tgt = g[(tm == 11) & yr.between(2021, 2025)]
    e = lambda o, c="final": float((g.loc[g.origin == o, c] / g.loc[g.origin == o, "actual"] - 1).iloc[0] * 100)
    return {
        "rule": float(r.final), "pulled": pulled, "head": float(r["head"]), "s_rule": float(r.S_adj), "issue_rule": str(r.adj_issue),
        "adj_rule": float(r.adj_pct), "s_nov": s_nov, "adj_nov": gm.BETA_FALL * s_nov, "normal": float(r.normal_fwd), "p7": float(r.p7),
        "p28": float(r.p28), "price_only": float(r.price_only_ridge), "last_year": float(ly), "t0": t0, "t1": t1,
        "last_price_day": p.date.max(), "mape": mape(g, yr.between(2021, 2025)), "mape_price": mape(g, yr.between(2021, 2025), "price_only_ridge"),
        "e2024": e("2024-08-31"), "e2022": e("2022-08-31"), "e2024_price": e("2024-08-31", "price_only_ridge"), "e2025": e("2025-08-31"),
        "oct_end": [(int(o.year), float((f / a - 1) * 100)) for o, f, a in zip(oct_end.origin, oct_end.final, oct_end.actual)],
        "nov_mape": mape(nov_tgt, nov_tgt.index == nov_tgt.index), "late_n": len(late), "late_mape": float(late_err.abs().mean()),
        "late_gt20": int((late_err.abs() > 20).sum()), "late_min": float(late_err.min()), "late_max": float(late_err.max()),
        "g": g, "p": p, **no_outlook(g),
    }


def no_outlook(g: pd.DataFrame) -> dict:
    """농업관측 전망정보(여름 전환·수준 보정·9월 고온 층의 가을 전망 끄기)를 모두 뺀 변형의 월말 2021~2025 성적."""
    bs, bf = gm.BETA_SUMMER_ALERT, gm.BETA_FALL
    gm.BETA_SUMMER_ALERT = gm.BETA_FALL = 0.0
    try:
        fall = gm.load_fall_issues()
        x = gm.assemble(g, g.gate_calm == -1, gm.load_issues(), fall, monthly=True)
        x = gm.heat_hold(x, fall.iloc[0:0], monthly=True)
    finally:
        gm.BETA_SUMMER_ALERT, gm.BETA_FALL = bs, bf
    err = (x[gm.FINAL] / g.actual - 1) * 100
    at = lambda o: float(err[g.origin == o].iloc[0])
    return {"mape_noobs": float(err.abs()[g.origin.dt.year.between(2021, 2025)].mean()),
            "noobs2024": at("2024-08-31"), "noobs2022": at("2022-08-31"), "noobs2025": at("2025-08-31")}


def won(x: float) -> str:
    return f"{x:,.0f}"


# ------------------------------------------------------------------ 그림
def _axes(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#bdbab4")
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_track(N: dict) -> bytes:
    """월평균 실제가격과 한 달 전(전월 말) 예측, 대상월 2021-01 이후."""
    g = N["g"]
    g = g[(g.origin + pd.Timedelta(days=15)) >= "2021-01-01"].copy()
    g["tm"] = (g.origin + pd.Timedelta(days=15)).dt.to_period("M").dt.to_timestamp()
    fig, ax = plt.subplots(figsize=(7.4, 2.2))
    ax.plot(g.tm, g.actual / 1000, color=INK, lw=2.0, label="실제 (월평균)")
    ax.plot(g.tm, g.final / 1000, color=ORANGE, lw=2.0, label="관측결합ML (전월 말 예측)")
    ax.plot(g.tm, g.price_only_ridge / 1000, color=GREY, lw=1.4, ls=(0, (4, 2)), label="가격 흐름만 쓰는 모형")
    pk = g.loc[g.actual.idxmax()]
    ax.annotate(f"2024년 9월\n실제 {pk.actual/1000:.1f}천원\n예측 {pk.final/1000:.1f}천원", (pk.tm, pk.actual / 1000), xytext=(18, -8),
                textcoords="offset points", fontsize=8, color=INK2, va="top")
    ax.set_ylabel("천원/10kg (상품)", fontsize=8.5, color=INK2)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(0, max(g.actual.max(), g.final.max()) / 1000 * 1.12)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    _axes(ax)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def fig_outlook(N: dict) -> bytes:
    """최근 일별 가격과 전망 구간."""
    p = N["p"]
    p = p[p.date >= ORIGIN - pd.Timedelta(days=80)]
    t0, t1 = N["t0"], N["t1"]
    fig, ax = plt.subplots(figsize=(7.4, 2.4))
    ax.plot(p.date, p.price / 1000, color=INK, lw=1.8, marker="o", ms=2.6, label="실제 (일별, 상품)")
    lo, hi = sorted([N["pulled"], N["rule"]])
    ax.fill_between([t0, t1], lo / 1000, hi / 1000, color=ORANGE, alpha=0.35, lw=0)
    ax.plot([t0, t1], [lo / 1000] * 2, color=ORANGE, lw=2.0, label="관측결합ML 전망 (28일 평균)")
    ax.plot([t0, t1], [hi / 1000] * 2, color=ORANGE, lw=2.0)
    ax.plot([t0, t1], [N["normal"] / 1000] * 2, color=GREY, lw=1.6, ls=(0, (4, 2)), label="평년 (같은 기간)")
    ax.plot([t0, t1], [N["last_year"] / 1000] * 2, color=BLUE, lw=1.6, ls=(0, (1, 1.5)), label="전년 (같은 기간)")
    ax.annotate(f"{won(lo)}~{won(hi)}원", (t0 + (t1 - t0) / 2, hi / 1000), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.axvline(ORIGIN, color="#bdbab4", lw=0.8)
    ax.set_ylabel("천원/10kg", fontsize=8.5, color=INK2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    top = max(p.price.max(), N["last_year"]) / 1000 * 1.15
    ax.set_ylim(0, top)
    ax.annotate(f"전망 기준일 {ORIGIN.month}/{ORIGIN.day}", (ORIGIN, top), xytext=(4, -2), textcoords="offset points",
                ha="left", va="top", fontsize=8, color=INK2)
    ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=2)
    _axes(ax)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# ------------------------------------------------------------------ 본문 (양식 무관한 내용 목록)
def content(N: dict) -> list[tuple]:
    lo, hi = sorted([N["pulled"], N["rule"]])
    oct_txt = ", ".join(f"{y}년 {v:+.0f}%" for y, v in N["oct_end"])
    under = [y for y, v in N["oct_end"] if v < 0]
    over = [y for y, v in N["oct_end"] if v > 0]
    return [
        ("h", "관측결합ML 모형 개요"),
        ("b", "가락시장 배추(상품, 10kg) <b>앞으로 4주 평균가격</b>을 매달 말·매주 일요일 기준으로 예측하는 "
              "<b>머신러닝 예측모형에 「농업관측」 전망정보를 결합해 고도화한 모형</b>(관측정보 결합 배추가격 머신러닝 모형, 약칭 관측결합ML, 시험 운영)"),
        ("d", "2024년 9월 급등(상품 월평균 24,714원)을 기존 예측모형(인공지능 시계열 모형 등)이 공통으로 크게 낮게 본 문제에서 출발"),
        ("b", "<b>(1단계) 머신러닝 예측</b>: 가락시장 가격·반입량, 고랭지·가을배추 산지 기상을 과거 자료로 학습(예측 때마다 그날까지의 자료로 다시 학습)"),
        ("d", "<b>(기본 모형)</b> 같은 시기 평년 가격을 기준으로, 지금 가격이 평년보다 높거나 낮은 정도가 일부 이어진다고 봄 → 급등기에 가격을 따라감"),
        ("d", "<b>(평시 모형)</b> 가격·반입량·기상의 평소 관계를 나무 모형으로 학습 → 평소 정확도가 높음"),
        ("d", "1~4월은 겨울배추 시기 자료만으로 학습한 전용 모형"),
        ("b", "<b>(2단계) 농업관측 전망정보로 고도화</b>: 가격·기상 자료에는 없는 산지 공급 정보를 월보 전망으로 반영"),
        ("d", "<b>(모형 전환)</b> 6~9월 말에는 월보의 여름배추 생산 전망이 '증가'이거나 그 달 출하량 전망이 '증가'면 평시 모형, "
              "둘 다 아니면 기본 모형 → 공급이 넉넉한 해의 과대 예측을 막고, 공급이 줄 해는 급등 가능성을 열어 둠"),
        ("d", "<b>(수준 보정)</b> 여름 공급 감소 전망 때는 생산 전망 1% 감소당 예측을 약 0.6% 올리고, "
              "11~12월은 가을배추 생산 전망 1% 증가당 약 0.2% 낮춤"),
        ("b", f"2021~2025년 매달 말 기준 다음 달 평균가격 예측의 평균 오차율 <b>{N['mape']:.1f}%</b>로, 가격 흐름만 쓰는 모형({N['mape_price']:.1f}%)보다 작음*"),
        ("d", f"2024년 9월 급등은 8월 말 기준 {abs(N['e2024']):.0f}% 낮게 예측(가격 흐름만 쓰는 모형 {abs(N['e2024_price']):.0f}% 낮게)"),
        ("d", f"<b>농업관측 전망정보 결합 효과</b>: 전망정보를 빼면 오차율 {N['mape_noobs']:.1f}% → 결합 시 {N['mape']:.1f}%. "
              f"2022년 9월 {N['noobs2022']:+.0f}% → {N['e2022']:+.0f}%, 2024년 9월 {N['noobs2024']:+.0f}% → {N['e2024']:+.0f}%"),
        ("d", f"반면 급등이 없던 2025년 9월은 {N['e2025']:.0f}% 높게 예측 → 공급 증가 전망 해에도 높게 볼 수 있음"),
        ("n", f"* 근거 사례가 1~2개뿐인 보정을 빼면 {RANGE_NO_THIN}%, 여름 모형 전환까지 빼면 {RANGE_NO_GATE}%. "
              "모형 전환 기준은 2024·2025년 결과를 본 뒤 정했고, 월보 신호가 있는 해는 2021년 이후 6개 해(급등 2번)"),
        ("fig", "track", "〈월평균 가격과 한 달 전 예측 (2021.1~2026.9)〉"),
        ("h", f"11월 가격 전망 ({ORIGIN.month}월 {ORIGIN.day}일 기준)"),
        ("b", f"{N['t0'].month}월 {N['t0'].day}일~{N['t1'].month}월 {N['t1'].day}일 가락시장 배추 상품 평균가격은 10kg당 <b>{won(lo)}~{won(hi)}원</b> 수준으로 전망"),
        ("d", f"평년(같은 기간, {won(N['normal'])}원)의 {lo / N['normal'] * 100:.0f}~{hi / N['normal'] * 100:.0f}%, "
              f"전년 같은 기간({won(N['last_year'])}원)의 {lo / N['last_year'] * 100:.0f}~{hi / N['last_year'] * 100:.0f}% 수준"),
        ("d", f"10월 하순 가격이 이미 5천원대(최근 1주 평균 {won(N['p7'])}원)로 내려왔고, 모형은 이 수준이 대체로 이어진다고 봄"),
        ("d", f"가을배추 생산 전망(관측 11월호, 전년 대비 {N['s_nov']:+.1f}%)을 반영하면 {won(N['pulled'])}원, 반영 전 {won(N['rule'])}원**"),
        ("d", f"관측 11월호의 11월 출하량 증가 전망(전년 대비 +{SHIP_NOV_YOY}%)과 같은 방향(출하량 전망은 6~9월 모형 전환에만 쓰여 이번 전망의 입력은 아님)"),
        ("n", f"** 모형은 m월호를 (m-1)월 말부터 쓰므로 {ORIGIN.month}월 {ORIGIN.day}일 기준 규칙상으로는 9월호(가을배추 재배면적 {N['s_rule']:+.1f}%)를 씀. "
              f"10월 31일 기준으로 다시 계산하면 11월호가 반영된 값이 됨"),
        ("n", "※ 평년: 직전 5년 같은 기간 평균(관측 평년과 산출 방식이 다름). 가격 자료는 "
              f"{N['last_price_day'].month}월 {N['last_price_day'].day}일까지"),
        ("fig", "outlook", "〈최근 가격 흐름과 전망〉"),
        ("h", "유의사항"),
        ("b", f"<b>이 시기는 모형 오차가 가장 큰 때임.</b> 10월 말 기준 11월 예측의 과거 오차는 {oct_txt}"),
        ("d", f"10월 하순 매주 기준으로 보면 2021~2025년 {N['late_n']}번 가운데 {N['late_gt20']}번이 20% 넘게 빗나감"
              f"(오차 {N['late_min']:+.0f}~{N['late_max']:+.0f}%)"),
        ("d", f"낮게 본 해({'·'.join(map(str, under))})와 높게 본 해({'·'.join(map(str, over))})가 섞여 있어, 방향도 해마다 다름 "
              "→ 가을배추 출하 시기와 김장 수요에 따라 가격이 크게 움직이는 달"),
        ("b", "전망치는 '현재 가격 수준이 이어질 경우'의 참고값으로만 활용"),
        ("d", "시험 운영 중인 모형 결과로, 농업관측 공식 전망과는 별개"),
    ]


# ------------------------------------------------------------------ PDF
def build_pdf(N: dict, figs: dict, path: Path) -> None:
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    pdfmetrics.registerFont(TTFont("Malgun", FONT_FILE))
    pdfmetrics.registerFont(TTFont("Malgun-Bold", FONT_FILE_B))
    from reportlab.lib.fonts import addMapping
    addMapping("Malgun", 0, 0, "Malgun")
    addMapping("Malgun", 1, 0, "Malgun-Bold")
    addMapping("Malgun", 0, 1, "Malgun")
    addMapping("Malgun", 1, 1, "Malgun-Bold")
    base = dict(fontName="Malgun", textColor=INK)
    st = {
        "title": ParagraphStyle("t", fontName="Malgun-Bold", fontSize=16, leading=21, textColor=INK),
        "h": ParagraphStyle("h", fontName="Malgun-Bold", fontSize=12, leading=17, spaceBefore=9, spaceAfter=3, textColor=INK),
        "b": ParagraphStyle("b", **base, fontSize=11, leading=17.5, leftIndent=11, firstLineIndent=-11, spaceBefore=3),
        "d": ParagraphStyle("d", **base, fontSize=10.5, leading=16.5, leftIndent=26, firstLineIndent=-10, spaceBefore=1),
        "n": ParagraphStyle("n", fontName="Malgun", fontSize=9, leading=13, leftIndent=26, textColor=INK2, spaceBefore=1),
        "cap": ParagraphStyle("c", fontName="Malgun-Bold", fontSize=10.5, leading=14, alignment=TA_CENTER, spaceBefore=8, spaceAfter=2),
    }
    W = A4[0] - 40 * mm
    story = []
    tbox = Table([[Paragraph(f"참고 {NOTE_NO}. 관측결합ML 모형의 11월 배추가격 전망", st["title"])]], colWidths=[W])
    tbox.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors_hex("#e9e8e6")), ("LINEABOVE", (0, 0), (-1, 0), 2.2, colors_hex("#3a3a3a")),
                              ("LINEBELOW", (0, 0), (-1, -1), 0.6, colors_hex("#9a9893")), ("TOPPADDING", (0, 0), (-1, -1), 7),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    story += [tbox, Spacer(1, 6)]
    for item in content(N):
        kind = item[0]
        if kind == "h":
            story.append(Paragraph(f"■ {item[1]}", st["h"]))
        elif kind == "b":
            story.append(Paragraph(f"•&nbsp;{item[1]}", st["b"]))
        elif kind == "d":
            story.append(Paragraph(f"-&nbsp;&nbsp;{item[1]}", st["d"]))
        elif kind == "n":
            story.append(Paragraph(item[1], st["n"]))
        elif kind == "fig":
            png = figs[item[1]]
            from PIL import Image as PILImage
            wpx, hpx = PILImage.open(io.BytesIO(png)).size
            img = Image(io.BytesIO(png), width=W * 0.98, height=W * 0.98 * hpx / wpx)
            story.append(KeepTogether([Paragraph(item[2], st["cap"]), img]))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors_hex("#d6d5d2"))
        canvas.rect(20 * mm, 10 * mm, W, 6 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors_hex(INK))
        canvas.setFont("Malgun", 9)
        canvas.drawCentredString(A4[0] / 2, 11.8 * mm, f"- {doc.page} -")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=22 * mm,
                            title=f"참고 {NOTE_NO}. 관측결합ML 모형의 11월 배추가격 전망", author="cabbage-surge")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def colors_hex(h: str):
    from reportlab.lib import colors
    return colors.HexColor(h)


# ------------------------------------------------------------------ Word
def build_docx(N: dict, figs: dict, path: Path) -> None:
    import re

    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Mm, Pt, RGBColor

    d = Document()
    sec = d.sections[0]
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    sec.left_margin = sec.right_margin = Mm(20)
    sec.top_margin, sec.bottom_margin = Mm(18), Mm(20)

    def font(run, size, bold=False, color=INK):
        run.font.name = "맑은 고딕"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "맑은 고딕")
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = RGBColor.from_string(color.lstrip("#"))

    def para(text, size, indent_mm=0.0, hanging_mm=0.0, bold_all=False, color=INK, before=2, align=None):
        p = d.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent, pf.first_line_indent = Mm(indent_mm), Mm(-hanging_mm)
        pf.space_before, pf.space_after, pf.line_spacing = Pt(before), Pt(0), 1.35
        if align:
            p.alignment = align
        for i, part in enumerate(re.split(r"</?b>", text.replace("&nbsp;", " "))):
            if part:
                font(p.add_run(part), size, bold_all or i % 2 == 1, color)
        return p

    t = d.add_table(rows=1, cols=1)
    cell = t.cell(0, 0)
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "E9E8E6")
    tcPr.append(shd)
    borders = OxmlElement("w:tcBorders")
    for edge, sz, col in (("top", "18", "3A3A3A"), ("bottom", "4", "9A9893")):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), sz)
        e.set(qn("w:color"), col)
        borders.append(e)
    tcPr.append(borders)
    cp = cell.paragraphs[0]
    cp.paragraph_format.space_before = cp.paragraph_format.space_after = Pt(4)
    font(cp.add_run(f"참고 {NOTE_NO}. 관측결합ML 모형의 11월 배추가격 전망"), 16, True)

    for item in content(N):
        kind = item[0]
        if kind == "h":
            para(f"■ {item[1]}", 12, bold_all=True, before=10)
        elif kind == "b":
            para(f"• {item[1]}", 11, indent_mm=4, hanging_mm=4, before=3)
        elif kind == "d":
            para(f"- {item[1]}", 10.5, indent_mm=9, hanging_mm=3.5, before=1)
        elif kind == "n":
            para(item[1], 9, indent_mm=9, color=INK2, before=1)
        elif kind == "fig":
            para(item[2], 10.5, bold_all=True, before=8, align=WD_ALIGN_PARAGRAPH.CENTER)
            p = d.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(io.BytesIO(figs[item[1]]), width=Mm(168))

    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    font(fp.add_run("- "), 9)
    r = fp.add_run()
    for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = txt
        r._element.append(el)
    font(r, 9)
    font(fp.add_run(" -"), 9)
    d.save(str(path))


if __name__ == "__main__":
    N = numbers()
    figs = {"track": fig_track(N), "outlook": fig_outlook(N)}
    stem = DOCS / f"참고자료_배추가격전망_{ORIGIN.date()}"
    build_pdf(N, figs, stem.with_suffix(".pdf"))
    build_docx(N, figs, stem.with_suffix(".docx"))
    for i, (k, cap) in enumerate((("track", "월평균가격과_한달전예측"), ("outlook", "최근가격흐름과_전망")), 1):
        (DOCS / f"{stem.name}_그림{i}_{cap}.png").write_bytes(figs[k])          # 원고에 붙일 그림(제목 없음)
    print(f"wrote {stem.with_suffix('.pdf')}\nwrote {stem.with_suffix('.docx')}")
    print(f"전망 {won(min(N['rule'], N['pulled']))}~{won(max(N['rule'], N['pulled']))}원, 평년 {won(N['normal'])}, 전년 {won(N['last_year'])}, "
          f"10월 하순 원점 {N['late_n']}개 MAPE {N['late_mape']:.1f}%")
