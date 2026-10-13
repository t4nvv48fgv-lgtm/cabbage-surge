# -*- coding: utf-8 -*-
"""최종 모형(공급 게이트[생산 전망 + 출하량 전망 방향] + 계절편차 + 여름특화 HGB + 겨울 분기 + 생산 전망 수준 조정 + 9월 고온 층) 쉬운 말 설명서 PDF 생성기.

읽는 사람: 모형을 만들지 않은 동료. "이 모형이 무엇을 보고, 어떻게 고르고, 매달 어떻게 돌리며, 무엇을 못 하는가"를 설명한다.
수치는 전부 저장소 산출물에서 직접 읽는다:
  outputs/gate_predictions.csv, gate_predictions_W.csv, gate_metrics.csv, gate_metrics_W.csv   (src/gate_model.py)
  data/ext/krei_summer_issues.csv, krei_nonsummer_issues.csv                                   (게이트 신호·수준 조정 원천)
  data/ext/krei_shipment_outlook.csv                                                           (월보의 그 달 출하량 전망 방향. 게이트의 경보를 끄는 신호, 2026-10-13)
글꼴·스타일·표·그림 저장은 tools/make_report.py 의 것을 재사용한다.

실행:  python tools/make_model_guide.py [--out docs/배추급등예측_모형설명서_YYYY-MM-DD.pdf]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_report import FONT, GRID, INK, INK2, PAL, ROOT, _style, styles, table, to_img  # noqa: E402

OUT = ROOT / "outputs"
EXT = ROOT / "data" / "ext"
TODAY = date.today().isoformat()
FINAL, GATE, HEAD, CALM, BASE = "final", "gate_hgb", "an_sparse_all", "wx_hgb_summer", "price_only_ridge"
PREV = "final_prod_only"   # 출하량 전망을 엮기 전(생산 전망만 보던 스위치)의 최종값
SHIP_WORD = {1.0: "증가", 0.0: "비슷", -1.0: "감소"}
NAME = {BASE: "가격만 보는 모형", HEAD: "경보 모형(계절편차)", CALM: "평시 모형(여름특화 트리)", GATE: "스위치만 적용",
        FINAL: "최종 모형(스위치+겨울 분기+수준 조정+9월 고온)"}
YEAR_NOTE = {2021: "공급 늘 전망 → 평시 모형. 9월의 보통 상승을 거의 맞힘",
             2022: "공급 줄 전망 → 경보 모형 + 수준 조정. 급등을 대부분 따라감",
             2023: "공급 늘 전망 → 평시 모형. 보합을 맞힘",
             2024: "공급 줄 전망 → 경보 모형 + 수준 조정. 급등을 한 자릿수 차이로 따라감",
             2025: "공급 늘 전망 → 평시 모형. 급등 없음. 평시 모형이 15% 높게 봄(이 모형의 알려진 약점)",
             2026: "생산·출하량 모두 줄 전망 → 경보 모형 + 수준 조정. 급등 없었고 8월 말 가격이 평년보다 낮아 예측도 낮았음"}


def load() -> dict:
    m = pd.read_csv(OUT / "gate_predictions.csv", parse_dates=["origin"], encoding="utf-8-sig")
    w = pd.read_csv(OUT / "gate_predictions_W.csv", parse_dates=["origin"], encoding="utf-8-sig")
    met = pd.read_csv(OUT / "gate_metrics.csv")
    metw = pd.read_csv(OUT / "gate_metrics_W.csv")
    iss = pd.read_csv(EXT / "krei_summer_issues.csv", encoding="utf-8-sig")
    ship = pd.read_csv(EXT / "krei_shipment_outlook.csv", encoding="utf-8-sig")
    return {"m": m, "w": w, "met": met, "metw": metw, "iss": iss, "ship": ship}


def err(df: pd.DataFrame, origin: str, col: str) -> float:
    s = df[df.origin == origin].iloc[0]
    return float((s[col] / s.actual - 1) * 100)


def mape_of(met: pd.DataFrame, subset: str, col: str) -> float:
    return float(met[(met.subset == subset) & (met.model == col)]["MAPE%"].iloc[0])


# ---------------------------------------------------------------- 그림
def fig_flow():
    """그림 1. 모형 흐름도 (상자 4개 + 화살표)."""
    fig, ax = plt.subplots(figsize=(8, 4.1))
    ax.set_xlim(0, 100); ax.set_ylim(-16, 110); ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2", fc=fc, ec=GRID, lw=1))
        ax.text(x + w / 2, y + h - 7, title, ha="center", va="top", fontsize=9.5, color=INK, fontweight="bold")
        ax.text(x + w / 2, y + h - 20, body, ha="center", va="top", fontsize=7.8, color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1, label=None, color=INK2, side="above"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.2, color=color))
        if label and side == "above":
            ax.text((x0 + x1) / 2 + 2, (y0 + y1) / 2 + 4, label, ha="left", va="bottom", fontsize=8, color=color)
        elif label:
            ax.text((x0 + x1) / 2 + 1.5, (y0 + y1) / 2, label, ha="left", va="center", fontsize=8, color=color)

    box(2, 36, 26, 46, "① 입력 (원점 당일까지)", "가락 일별 가격·반입량\n고랭지 4곳 기온·강수\n직전 5년 평년치\nKREI 월보(다음 달 호)", "#f3f4f7")
    box(36, 64, 30, 38, "② 스위치", "KREI 월보 여름배추 생산 전망\n+ 그 달 출하량 전망\n(6~9월 원점에만 적용)", "#fff4e8")
    box(36, -4, 30, 48, "③-A 경보 모형", "계절편차 희소 선형\n평년치 기준 + 현재 가격\n편차 지속 + 기상 편차\n(기상은 여름배추 출하기에만)", "#eef2fb")
    box(72, 2, 26, 42, "③-B 평시 모형", "여름특화 트리(HGB)\n여름에는 여름 자료만\n학습. 평소 정확도 최선", "#eaf7f1")
    box(2, -14, 26, 40, "④~⑥ 덧붙임", "1~4월: 겨울 전용 모형\n경보 달·11~12월: 수준 조정\n9월 말: 고온이면 하락 늦춤", "#f7f1fb")
    arrow(28, 64, 36, 82)
    arrow(28, 56, 36, 28)
    arrow(28, 60, 72, 30)
    arrow(46, 64, 46, 44, "생산 줄 전망이고\n출하량도 증가 아님", PAL["red"], side="right")
    arrow(66, 80, 85, 44, "생산 늘 전망\n또는 출하량 증가 전망", PAL["green"])
    ax.text(50, 109, "출력: 다음 28일(다음 달) 평균 가격 한 값 + 상태(경보/평시)", ha="center", va="top", fontsize=9, color=INK)
    ax.set_title("그림 1. 최종 모형의 구조", loc="left", fontsize=10, color=INK)
    return to_img(fig)


def fig_years(m: pd.DataFrame):
    """그림 2. 해마다 8월 말 → 9월 예측 오차: 가격만 / 경보 모형 / 최종 모형."""
    years = [y for y in range(2021, 2027) if (m.origin == f"{y}-08-31").any()]
    series = [(BASE, PAL["violet"]), (HEAD, PAL["blue"]), (FINAL, PAL["orange"])]
    x = np.arange(len(years)); wd = 0.26
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.axhspan(-10, 10, color="#f0efec", zorder=0)
    ax.axhline(0, color=INK2, lw=0.8)
    for i, (c, col) in enumerate(series):
        v = [err(m, f"{y}-08-31", c) for y in years]
        bars = ax.bar(x + (i - 1) * wd, v, width=wd - 0.03, color=col, label=NAME[c])
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + (1.5 if val >= 0 else -1.5), f"{val:+.0f}", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=7.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{y}년\n{'경보' if m[m.origin == f'{y}-08-31'].gate_calm.iloc[0] == 0 else '평시'}" for y in years], fontsize=8.5)
    ax.set_ylabel("8월 말 시점 9월 예측 오차 (%)", color=INK2, fontsize=9)
    ax.set_ylim(-45, 30)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.set_title("그림 2. 해마다 8월 말에 섰을 때 9월 가격을 얼마나 맞혔나 (회색 띠 = ±10%)", loc="left", fontsize=10, color=INK)
    _style(ax)
    return to_img(fig)


def fig_weekly(w: pd.DataFrame, year: int):
    d = w[(w.origin >= f"{year}-06-01") & (w.origin <= f"{year}-10-31")]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(d.origin, d.actual / 1000, color=INK, lw=2.2, label="실제 (이후 4주 평균)")
    ax.plot(d.origin, d[BASE] / 1000, color=PAL["violet"], lw=1.4, ls="--", label=NAME[BASE])
    ax.plot(d.origin, d[FINAL] / 1000, color=PAL["orange"], lw=2.0, label=NAME[FINAL])
    calm = d[d.gate_calm == 1]
    if len(calm):
        ax.scatter(calm.origin, calm[FINAL] / 1000, s=14, color=PAL["green"], zorder=4, label="평시 모형이 쓰인 주")
    ax.set_ylabel("천원/10kg", color=INK2, fontsize=9)
    ax.legend(fontsize=7.8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_title(f"그림 {3 if year == 2024 else 4}. {year}년 여름, 매주 일요일에 선 예측", loc="left", fontsize=10, color=INK)
    _style(ax)
    return to_img(fig)


# ---------------------------------------------------------------- 문서
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT, 8)
    canvas.setFillColor(colors.HexColor(INK2))
    canvas.drawString(20 * mm, 12 * mm, f"배추 도매가격 예측모형 설명서 · {TODAY} · cabbage-surge")
    canvas.drawRightString(190 * mm, 12 * mm, str(doc.page))
    canvas.restoreState()


def build(out_path: Path) -> None:
    D = load()
    m, w, met, metw, iss, ship = D["m"], D["w"], D["met"], D["metw"], D["iss"], D["ship"]
    st = styles()
    P = lambda t, s="body": Paragraph(t, st[s])
    B = lambda t: Paragraph(t, st["bullet"], bulletText="•")

    years = [y for y in range(2021, 2027) if (m.origin == f"{y}-08-31").any()]
    e = {y: {c: err(m, f"{y}-08-31", c) for c in (BASE, HEAD, CALM, GATE, FINAL)} for y in years}
    row = {y: m[m.origin == f"{y}-08-31"].iloc[0] for y in years}
    ALLC = (BASE, HEAD, CALM, GATE, FINAL)
    mp = {c: mape_of(met, "all_2021_2025", c) for c in ALLC}
    mps = {c: mape_of(met, "summer_origin_2021_2025", c) for c in ALLC}
    mpw = {c: mape_of(metw, "all_2021_2025", c) for c in ALLC}
    win = {c: mape_of(met, "winter_target_2021_2025", c) for c in (GATE, FINAL)}
    winw = {c: mape_of(metw, "winter_target_2021_2025", c) for c in (GATE, FINAL)}
    fal = {c: mape_of(met, "fall_target_2021_2025", c) for c in (GATE, FINAL)}
    tm = (m.origin + pd.Timedelta(days=15)).dt.strftime("%Y-%m")
    rng = m.origin >= "2020-12-31"
    big = {c: [(t, (p / a - 1) * 100) for t, p, a in zip(tm[rng], m.loc[rng, c], m.loc[rng, "actual"]) if abs(p / a - 1) > 0.30] for c in (GATE, FINAL)}
    big_txt = ", ".join(f"{t} {v:+.0f}%" for t, v in big[FINAL])
    adj24 = float(m[m.origin == "2024-08-31"].adj_pct.iloc[0])
    h24 = m[m.origin == "2024-09-30"].iloc[0]
    eh = lambda col: float((h24[col] / h24.actual - 1) * 100)
    n60 = int(met[(met.subset == "all_2021_2025") & (met.model == FINAL)].n.iloc[0])
    nW = int(metw[(metw.subset == "all_2021_2025") & (metw.model == FINAL)].n.iloc[0])
    latest = iss.sort_values(["year", "issue_month"]).iloc[-1]
    qword = lambda q: SHIP_WORD.get(q, "표현 없음")
    # 종전 경보 모형(고랭지 기상을 1년 내내)의 겨울 성적: 겨울 분기의 득실은 이것과 견준다
    win_old, winw_old = mape_of(met, "winter_target_2021_2025", HEAD), mape_of(metw, "winter_target_2021_2025", HEAD)
    spr = {c: mape_of(met, "spring_target_2021_2025", c) for c in (HEAD, "head")}
    mp26 = {c: mape_of(met, "all_2021_2026", c) for c in (PREV, FINAL)}
    mpw26 = {c: mape_of(metw, "all_2021_2026", c) for c in (PREV, FINAL)}
    mpw_prev = mape_of(metw, "all_2021_2025", PREV)
    n68 = int(met[(met.subset == "all_2021_2026") & (met.model == FINAL)].n.iloc[0])
    chg = m[(m[FINAL] - m[PREV]).abs() > 0.5]
    chg_w = w[(w[FINAL] - w[PREV]).abs() > 0.5]
    summer = m[(m.summer == 1) & m.S1.notna()]
    n_calm, n_alert = int((summer.gate_calm == 1).sum()), int((summer.gate_calm == 0).sum())
    # 호별 신호표: 7~10월호의 생산 전망 / 그 달 출하량 전망
    sq = iss.merge(ship[["year", "issue_month", "q_sign"]], on=["year", "issue_month"], how="left")
    sig_rows = [["해", "7월호", "8월호", "9월호", "10월호", "그 해 9월 가격"]]
    surge = {2022: "급등", 2024: "급등"}
    for y in sorted(sq.year.unique()):
        cells = []
        for mth in (7, 8, 9, 10):
            r = sq[(sq.year == y) & (sq.issue_month == mth)]
            cells.append(f"{r.prod_yoy.iloc[0]:+.1f}% / {qword(r.q_sign.iloc[0])}" if len(r) and pd.notna(r.prod_yoy.iloc[0]) else "아직 없음")
        sig_rows.append([f"{y}년"] + cells + [surge.get(y, "급등 없음")])
    # 2026년 여름 원점: 생산 전망만 볼 때와 출하량 전망을 엮은 뒤
    chg_rows = [["원점 → 예측 대상", "본 호: 생산 전망 / 출하량 전망", "생산 전망만 볼 때", "출하량 전망을 엮은 뒤"]]
    for o in ("2026-06-30", "2026-07-31", "2026-08-31"):
        if (m.origin == o).any():
            r = m[m.origin == o].iloc[0]
            chg_rows.append([f"{o} → {int(o[5:7]) + 1}월", f"{int(str(r.krei_issue)[5:])}월호 {r.S1:+.1f}% / {qword(r.ship_q)}",
                             f"{'평시' if r.S1 > 0 else '경보'}, {err(m, o, PREV):+.0f}%", f"{'평시' if r.gate_calm == 1 else '경보'}, {err(m, o, FINAL):+.0f}%"])

    story = [P("배추 도매가격 예측모형 설명서", "title"),
             P(f"{date.fromisoformat(TODAY).strftime('%Y년 %m월 %d일')} 기준 · 프로젝트 cabbage-surge · 최종 모형(2026-10-12 편입, 같은 날 겨울 분기·수준 조정·9월 고온 층 추가, 10-13 고랭지 기상 출하기 한정·출하량 전망 추가)을 쉬운 말로 · "
               "수치는 outputs/gate_predictions*.csv에서 직접 계산 (tools/make_model_guide.py)", "sub"),
             P("<b>한 줄로</b>", "h2"),
             P("매달 말일(또는 매주 일요일)에 서서 <b>다음 한 달 가락시장 배추 평균 가격</b>을 숫자 하나로 예측합니다. "
               "모형은 두 개를 갖고 있고, 여름(6~9월 말)에는 <b>한국농촌경제연구원(KREI) 월보의 여름배추 생산 전망</b>이 '줄어든다'이고 <b>그 달 출하량 전망</b>도 '증가'가 아니면 급등에 대비한 <b>경보 모형</b>을, "
               "생산이 '늘어난다'거나 출하량이 '증가'한다는 전망이면 평소에 잘 맞는 <b>평시 모형</b>을 씁니다. 그 밖의 달에는 경보 모형을 쓰되 고랭지 날씨는 여름배추가 출하되는 때에만 반영합니다. "
               "여기에 세 가지를 덧붙였습니다. 1~4월을 예측할 때는 <b>겨울 전용 모형</b>을 쓰고, 경보 상태인 여름과 11~12월에는 생산 전망만큼 예측을 조금 <b>올리거나 내리며</b>, 9월 말에는 가을 산지의 9월 고온 일수만큼 10월 하락을 <b>늦춰</b> 봅니다. "
               f"2021~2025년 월말 예측 {n60}번의 평균 오차율은 <b>{mp[FINAL]:.1f}%</b>(가격만 보는 모형 {mp[BASE]:.1f}%)이고, "
               f"2024년 9월 급등은 <b>{e[2024][FINAL]:+.0f}%</b> 차이로 따라갔습니다.", "box"),
             P("<b>이 문서의 차례</b>", "h2"),
             B("1. 무엇을 예측하나 · 2. 여섯 가지 부품 · 3. 해마다 어떻게 판정했나 · 4. 성적표"),
             B("5. 매달 돌리는 법 · 6. 결과 읽는 법 · 7. 못 하는 것과 조심할 점 · 용어")]

    # 1
    story += [P("1. 무엇을 예측하나", "h1"),
              P("<b>대상</b>: 가락시장 배추(상품 등급) 도매가격, 10kg 한 망 기준 원. 예측 시점(원점) 다음 날부터 28일 동안의 평균입니다. "
                "월말에 서면 '다음 달 평균'과 거의 같습니다."),
              P("<b>입력</b>: 원점 당일까지 알 수 있는 것만 씁니다. 가락 일별 가격과 반입량, 강원 고랭지 4곳(평창·태백·정선·강릉)의 기온·강수, "
                "직전 5년 같은 시기의 평균 가격(평년치), 그리고 KREI 월보의 여름배추 생산 전망과 그 달 배추 출하량 전망의 방향. "
                "월보는 담당자가 전달 말에 다음 달 호를 쓰기 때문에 월말에 설 때는 다음 달 호를 봅니다(7월 말에는 8월호, 8월 말에는 9월호). 미래의 가격·날씨 정보는 들어가지 않습니다."),
              P("<b>출력</b>: 가격 한 값과 상태 한 가지(경보 또는 평시). 상태는 어느 모형이 쓰였는지를 뜻하며, '경보'는 공급이 줄 것으로 전망돼 "
                "급등 가능성을 열어 둔 상태입니다.")]

    # 2
    story += [KeepTogether([P("2. 여섯 가지 부품", "h1"), fig_flow()]),
              P("<b>② 스위치 — KREI 월보의 생산 전망과 출하량 전망</b>", "h2"),
              P("KREI 농업관측센터는 매달 초 월보를 냅니다. 스위치는 그 안의 두 가지를 봅니다. 하나는 '여름배추(그해 산) 생산량, 전년 대비 %'이고, 다른 하나는 그 호가 전망한 '그 달 배추 출하량이 작년보다 느는가 주는가'입니다. "
                "월말에 설 때는 담당자가 이미 써 둔 다음 달 호를 봅니다(7월 말에는 8월호, 8월 말에는 9월호)."),
              B("생산 전망이 0보다 크면(올해 생산이 작년보다 늘 전망) <b>평시</b>."),
              B("생산 전망이 0 이하여도 그 달 출하량이 '증가' 전망이면 <b>평시</b>(2026-10-13 추가)."),
              B("생산 전망이 0 이하이고 출하량 전망도 '증가'가 아니면(감소·비슷·표현 없음) <b>경보</b>."),
              P("6~9월 원점에만 적용하고, 그 밖의 달은 경보 모형(계절편차)을 그대로 씁니다. 출하량 전망은 경보를 끄는 데만 씁니다. 생산이 늘 전망인 호에 출하량 감소 문장이 있어도 경보를 켜지 않습니다. "
                f"지금 파일의 마지막 호는 {int(latest.year)}년 {int(latest.issue_month)}월호(생산 전망 {latest.prod_yoy:+.1f}%)입니다."),
              table(sig_rows, [16, 32, 32, 32, 32, 24], st),
              P("칸마다 '여름배추 생산 전망 전년비 / 그 달 출하량 전망 방향'입니다. 급등이 있었던 2022·2024년은 여름 내내 둘 다 감소였습니다.", "small"),
              P("왜 이 값인가: 2024년(급등)과 2025년(급등 없음)은 둘 다 기록적 폭염이었는데, 날씨로는 두 해를 가를 수 없었습니다. "
                "8월호 생산 전망은 2024년 -7.2%, 2025년 +8.8%로 두 해를 갈랐고, 2021~2025년 다섯 해는 급등 여부와 부호가 맞았습니다. "
                "2026년은 생산 전망이 소폭 감소(-3.5%, -0.9%)였는데 급등이 없었고, 월보는 그 두 달의 출하량을 '증가'로 봤습니다. 출하량 전망을 함께 보게 된 계기입니다. "
                "담당자 설명으로는 봄배추 저장량이 늘어난 영향이 큽니다(대량수요처 저장 계획량 4.7만 톤, 전년 대비 +10%. 8월 가락시장 반입의 31%가 저장배추, 급등한 2024년은 12%). "
                "여름배추 생산 전망에는 저장분이 들어 있지 않아 출하량 전망이 그 빈칸을 메웁니다. 저장분 출하는 9월 둘째 주에 끝나므로 9월호가 출하량 '감소'로 돌아선 것과도 맞습니다. "
                "다만 여섯 해, 급등 두 번이라는 짧은 기록이고, 생산 전망 규칙은 2024·2025년을 본 뒤, 출하량 전망 규칙은 2026년을 본 뒤 정한 것입니다(7절).", "small"),
              P("<b>③-A 경보 모형 — 계절편차 희소 선형</b>", "h2"),
              P("'평년 이맘때보다 얼마나 다를까'를 예측합니다. 기준점은 직전 5년 같은 시기 평균(평년치)이고, 거기에 지금 가격이 평년보다 얼마나 높은지, "
                "최근 일주일 변화, 반입량 전년비, 고랭지 기온·강수가 평년보다 얼마나 다른지를 더해 직선 관계로 계산합니다. 변수 7개, 2010년부터의 전 기간 학습."),
              P("<b>고랭지 날씨는 여름배추가 출하되는 때에만 봅니다(2026-10-13).</b> 날씨 변수 4개는 강원 고랭지 값이라 여름배추가 시장에 나오는 때에만 뜻이 있습니다. "
                "그래서 '앞으로 28일 가운데 여름 작형 수확 기간(준고랭지 1기작 6/11~7/20, 고랭지 7/11~9/30, 준고랭지 2기작 9/21~10/31)에 드는 날의 비율'만큼만 반영합니다. "
                "6~9월 원점은 종전 계산을 그대로 쓰고, 그 밖의 원점에서만 바뀝니다. "
                f"월말 봄 대상(5~6월) 오차율이 {spr[HEAD]:.1f}%에서 {spr['head']:.1f}%로 줄었고 전체로는 이득이 작습니다(7절)."),
              P("이 모형의 힘은 <b>'8월 말에 이미 평년보다 비싸면 9월도 평년보다 비싸다'</b>는 항에서 나옵니다. 2024년 8월 말 가격은 평년보다 25% 높았고, 모형은 그 상태가 이어진다고 봐 "
                f"9월을 {e[2024][HEAD]:+.0f}% 차이로 따라갔습니다. 가격만 보는 모형은 '최근 한 달 평균으로 돌아간다'고 보기 때문에 {e[2024][BASE]:+.0f}%였습니다. "
                "고랭지 기온 변수는 생각보다 작게 작동합니다. 2024·2025년 8월 말에는 기온이 올리는 만큼 강수가 깎아 거의 상쇄됐습니다(2022년에는 강수가 +13%p 보탬).", "small"),
              P("<b>③-B 평시 모형 — 여름특화 트리(HGB)</b>", "h2"),
              P("같은 종류의 변수(가격 흐름, 반입량, 고랭지 날씨)를 '조건을 나눠가며 규칙을 찾는' 트리 모형으로 학습합니다. 여름 원점에서는 여름 자료만 학습해 계절 특성을 살립니다. "
                f"평소 정확도가 가장 좋지만(2021~2025 오차율 {mp[CALM]:.1f}%), 경험해 본 범위를 벗어난 급등은 늘려 잡지 못합니다(2024년 9월 {e[2024][CALM]:+.0f}%). "
                "그래서 공급이 넉넉하다고 전망된 해에만 씁니다."),
              P("<b>④ 겨울 분기 — 1~4월은 겨울 전용 모형</b>", "h2"),
              P("예측 대상이 1~4월이면(원점으로는 12월 말~3월 말) 스위치 대신 <b>겨울 자료만으로 학습한 가격만 보는 모형</b>을 씁니다. 날씨·평년치·생산 전망은 쓰지 않고 최근 가격 흐름만 봅니다. "
                "겨울은 저장배추가 꾸준히 풀리는 철이라 최근 흐름만으로 잘 맞고, 2016~2020년과 2021~2025년 두 구간 모두에서 겨울에 가장 안정적이었던 조각입니다. "
                f"월말 예측의 겨울 오차율은 경보 모형을 그대로 쓸 때의 {win_old:.1f}%에서 {win[FINAL]:.1f}%로 줄었습니다. 다만 주간 예측에서는 {winw_old:.1f}%에서 {winw[FINAL]:.1f}%로 오히려 조금 나빠집니다(7절). "
                "여기서 견준 경보 모형은 고랭지 날씨를 1년 내내 쓰던 종전 사양입니다."),
              P("<b>⑤ 수준 조정 — 생산 전망만큼 예측을 밀어 주기</b>", "h2"),
              P("두 경우에만 예측값에 작은 배율을 곱합니다. (가) 여름 경보 상태: 여름배추 생산이 1% 줄 전망이면 예측을 약 0.57% 올립니다. "
                "(나) 11~12월 예측: 가을배추 생산이 1% 줄 전망이면 약 0.22% 올리고, 늘 전망이면 그만큼 내립니다. "
                f"2024년 8월 말에는 9월호 전망 -6.6%로 예측이 {adj24:+.1f}% 올라갔습니다. "
                "이 배율은 2016~2020년에 '모형이 틀린 만큼이 생산 전망과 얼마나 같이 움직였나'를 재서 정한 값이고, 2021년 이후 자료로 맞춘 것이 아닙니다. "
                "조정에는 담당자가 전달 말에 이미 아는 다음 달 호 값을 씁니다(예: 8월 말에는 9월호)."),
              P("이 조정은 경보 달을 일괄로 조금 올리는 장치입니다. 모형이 낮게 본 달(급등 진입·봉우리)에는 도움이 되고, 이미 높게 본 달(봉우리 직후)에는 손해입니다.", "small"),
              P("<b>⑥ 9월 고온 층 — 9월이 더웠으면 10월 하락을 늦춰 보기</b>", "h2"),
              P("9월 말에 10월을 예측할 때만 작동합니다. 가을배추 주산지 9곳(해남·진도·고창·아산·괴산·제천·영양·문경·춘천)에서 9월에 낮 기온이 30도를 넘은 날이 많으면, "
                "가을배추가 늦게 자리 잡아 10월 출하가 밀리고 값이 평년만큼 내리지 않는다고 보고 예측을 올립니다. 가을배추 생산이 늘 전망이면 올리지 않고, 올린 값이 9월 말 최근 7일 가격을 넘지 않게 묶습니다. "
                "얼마나 올릴지는 정해진 숫자가 아니라 그 전 해들의 9월 말 예측이 얼마나 틀렸는지에서 매해 다시 계산합니다. "
                f"2024년 9월은 30도 넘은 날이 {h24.heat_H:.1f}일이라 예측이 {h24.heat_pct:+.0f}% 올라가 10월 오차가 {eh('final_base'):+.0f}%에서 {eh(FINAL):+.0f}%가 됐습니다."),
              P("이 부품은 근거가 가장 약합니다. 9월이 이만큼 더웠던 해가 2023·2024·2025년 셋뿐이고, 과거 기록에서 실제로 예측이 바뀐 달은 2024년 10월 하나입니다. "
                "2024년에 쓴 폭은 2023년 한 해에서 배운 것이라 0%에 맞은 것은 우연으로 봐야 합니다. 주간 예측에는 쓰지 않습니다(7절).", "small")]

    # 3
    rows = [["8월 말 원점", "9월호 생산 / 출하량 전망", "상태", "실제 9월", "예측", "오차", "무슨 일이 있었나"]]
    for y in years:
        r = row[y]
        rows.append([f"{y}년", f"{r.S1:+.1f}% / {qword(r.ship_q)}", "경보" if r.gate_calm == 0 else "평시", f"{r.actual:,.0f}", f"{r[FINAL]:,.0f}",
                     f"{e[y][FINAL]:+.0f}%", YEAR_NOTE.get(y, "")])
    r26 = row.get(2026)
    story += [P("3. 해마다 어떻게 판정했나 (8월 말에 서서 9월을 예측)", "h1"),
              table(rows, [20, 28, 12, 18, 18, 14, 58], st),
              ] + ([P(f"2026년 8월 말 원점의 값은 이 해를 보기 전에 정한 규칙에서 나왔고, 나중에 넣은 출하량 전망 규칙으로도 바뀌지 않았습니다. 8월 말에 본 9월호는 생산 전망 {r26.S1:+.1f}%에 9월 출하량도 {qword(r26.ship_q)} 전망이라 경보 모형이 쓰였지만, 그 해 8월 말 가격이 평년보다 낮아 경보 모형도 낮게 예측했고 결과는 {e[2026][FINAL]:+.0f}%였습니다. "
                      f"평시 모형이 쓰였다면 {e[2026][CALM]:+.0f}%였습니다.", "small")] if r26 is not None else []) + [
              fig_years(m),
              B(f"급등 두 해(2022·2024)는 경보 모형에 수준 조정이 더해져 {e[2022][FINAL]:+.0f}%·{e[2024][FINAL]:+.0f}%(조정 전 {e[2022][GATE]:+.0f}%·{e[2024][GATE]:+.0f}%). 급등의 대부분을 따라가지만 끝까지는 못 올라갑니다."),
              B(f"급등 없는 세 해(2021·2023·2025)는 평시 모형이 쓰여 {e[2021][FINAL]:+.0f}%·{e[2023][FINAL]:+.0f}%·{e[2025][FINAL]:+.0f}%. "
                f"경보 모형을 그대로 썼다면 {e[2021][HEAD]:+.0f}%·{e[2023][HEAD]:+.0f}%·{e[2025][HEAD]:+.0f}%였습니다. 스위치의 역할이 여기에 있습니다."),
              fig_weekly(w, 2024), fig_weekly(w, 2025),
              P("주간 그림에서 2024년은 6월(6월호가 생산 -2.9%에 출하량 증가 전망이라 평시)을 빼고 여름 내내 경보 상태라 최종 모형이 경보 모형에 수준 조정(+4% 안팎)을 더한 값이고, 2025년은 평시 상태(초록 점)라 평시 모형이 쓰였습니다. "
                "두 해 모두 10월에 가격이 내릴 때는 높게 예측합니다(7절).", "small"),
              P("<b>2026년 여름: 출하량 전망을 엮은 뒤 달라진 달</b>", "h2"),
              table(chg_rows, [36, 52, 40, 40], st),
              P(f"월말 예측에서 바뀐 것은 이 표의 {len(chg)}개 달뿐이고 2021~2025년은 하나도 바뀌지 않습니다. 2021~2026년 월말 {n68}번의 오차율은 {mp26[PREV]:.1f}%에서 {mp26[FINAL]:.1f}%로, 주간은 {mpw26[PREV]:.1f}%에서 {mpw26[FINAL]:.1f}%로 줄었습니다"
                f"(주간에서 바뀐 원점 {len(chg_w)}개, 2021~2025년 주간은 {mpw_prev:.2f}%에서 {mpw[FINAL]:.2f}%로 사실상 같음). "
                "이 규칙은 2026년 8월의 과대 예측을 본 뒤에 만든 것이라 이 표는 검증이 아닙니다. 2026년은 한 해 안에서 상태가 바뀐 첫 해입니다(6·7월 말 평시, 8월 말 경보).", "small")]

    # 4
    story += [P("4. 성적표", "h1"),
              table([["모형", f"2021~2025 월말 {n60}회 오차율", "여름 원점만(20회)", f"주간 {nW}회 오차율", "2024년 9월", "2025년 9월"],
                     [NAME[BASE], f"{mp[BASE]:.1f}%", f"{mps[BASE]:.1f}%", f"{mpw[BASE]:.1f}%", f"{e[2024][BASE]:+.0f}%", f"{e[2025][BASE]:+.0f}%"],
                     [NAME[HEAD], f"{mp[HEAD]:.1f}%", f"{mps[HEAD]:.1f}%", f"{mpw[HEAD]:.1f}%", f"{e[2024][HEAD]:+.0f}%", f"{e[2025][HEAD]:+.0f}%"],
                     [NAME[CALM], f"{mp[CALM]:.1f}%", f"{mps[CALM]:.1f}%", f"{mpw[CALM]:.1f}%", f"{e[2024][CALM]:+.0f}%", f"{e[2025][CALM]:+.0f}%"],
                     [NAME[GATE], f"{mp[GATE]:.1f}%", f"{mps[GATE]:.1f}%", f"{mpw[GATE]:.1f}%", f"{e[2024][GATE]:+.0f}%", f"{e[2025][GATE]:+.0f}%"],
                     [NAME[FINAL], f"{mp[FINAL]:.1f}%", f"{mps[FINAL]:.1f}%", f"{mpw[FINAL]:.1f}%", f"{e[2024][FINAL]:+.0f}%", f"{e[2025][FINAL]:+.0f}%"]],
                    [50, 30, 24, 26, 19, 19], st, bold_rows=(5,)),
              P("오차율(MAPE)은 예측이 실제와 평균 몇 % 어긋났는지입니다. 최종 모형은 평소에는 평시 모형만큼 정확하고, 급등 해에는 경보 모형만큼 따라갑니다. "
                f"프로젝트의 성공 기준 두 가지(2024년 9월 -10% 이내, 전체 오차율이 가격만 보는 모형 {mp[BASE]:.1f}% 이하)를 함께 넘습니다. "
                f"스위치만 적용했을 때 2024년은 {e[2024][GATE]:+.0f}%로 기준선에 딱 걸렸고, 수준 조정을 더해 {e[2024][FINAL]:+.0f}%가 됐습니다. "
                f"겨울 분기는 월말 겨울 오차율을 {win_old:.1f}%에서 {win[FINAL]:.1f}%로, 수준 조정은 11~12월을 {fal[GATE]:.1f}%에서 {fal[FINAL]:.1f}%로 줄였습니다. "
                f"반대로 오차가 30%를 넘는 달은 {len(big[GATE])}개에서 {len(big[FINAL])}개로 늘었습니다({big_txt})."),
              P(f"표의 '경보 모형(계절편차)' 줄은 고랭지 날씨를 1년 내내 쓰던 종전 사양 혼자의 성적입니다. '스위치만 적용' 줄부터는 여름 밖 원점에서 날씨를 출하기에만 반영합니다. "
                f"출하량 전망을 엮은 것은 2021~2025년 월말 성적을 바꾸지 않았고, 2026년까지 넣은 월말 {n68}번에서는 {mp26[PREV]:.1f}%가 {mp26[FINAL]:.1f}%가 됐습니다.", "small")]

    # 5
    story += [P("5. 매달 돌리는 법", "h1"),
              P("저장소 루트에서 아래 순서로 실행합니다(Python 3.13, requirements.txt). 월말 한 번이면 1~3번은 수 분, 4번은 Chronos-2 기준모형 때문에 5분쯤 걸립니다."),
              table([["순서", "명령", "하는 일"],
                     ["1", "python src/sync_from_cabis.py", "CABIS에서 가격·날씨 최신 자료를 data/raw로 복사 (CABIS가 있는 PC에서만)"],
                     ["2", "KREI 월보 값 입력", "data/ext/krei_summer_issues.csv(여름배추)·krei_nonsummer_issues.csv(가을배추)·krei_shipment_outlook.csv(그 달 출하량 방향, 6~10월호)에 새 호 한 줄씩 추가 (아래 참고)"],
                     ["3", "python src/build_features.py", "일별 변수 만들기"],
                     ["4", "python src/backtest.py M", "평시 모형(HGB)·기준모형 예측"],
                     ["5", "python src/anomaly_model.py M", "경보 모형(계절편차) 예측"],
                     ["6", "python src/gate_model.py M", "스위치·겨울 분기·수준 조정·9월 고온 층 적용 → outputs/gate_predictions.csv (실제값이 나온 원점까지의 과거 검증)"],
                     ["7", "python src/forecast_live.py 2026-09-30 M", "<b>이번 달 예측</b>. 실제값이 아직 없는 원점 하나를 같은 규칙으로 계산 → 화면과 outputs/forecast_live.csv (날짜는 서는 날, 주간 규칙은 W)"],
                     ["(주간)", "3~6번 명령에 W", "매주 일요일 원점 버전. 4번이 15분쯤"]],
                    [14, 58, 96], st),
              P("6번까지는 실제 가격이 나온 원점만 계산합니다. 이번 달처럼 아직 실제값이 없는 원점의 예측은 7번이 냅니다. 과거 원점을 7번으로 계산하면 6번과 같은 값이 나옵니다.", "small"),
              P("<b>KREI 월보 값 입력</b>", "h2"),
              P("파일의 열은 year, issue_month, pub_date, area_ha, prod_t, area_yoy, prod_yoy 입니다. 월보 '여름배추(그해 산)' 표에서 재배면적(ha)·생산량(톤)과 각각의 전년 대비 %를 옮겨 적습니다. "
                "pub_date는 발간 월 5일로 적는 것이 약속입니다(실제 게시는 1~5일. 모형은 이 날짜가 아니라 '그 호는 전달 말일부터 안다'는 규칙으로 호를 고릅니다). 생산량 전년비가 아직 없는 봄 호(4~5월호)는 면적 전년비를 대신 씁니다. "
                "스위치가 보는 값은 prod_yoy 의 부호(0 이하 / 초과)이고, 경보 상태의 수준 조정은 같은 값의 크기를 씁니다."),
              P("출하량 전망은 krei_shipment_outlook.csv 에 year, issue_month, q_sign 을 적습니다(6~10월호). q_sign 은 그 호가 전망한 그 달 배추 출하량의 전년 대비 방향으로 증가 1, 비슷 0, 감소 -1 이고, 표현이 없으면 비워 둡니다. "
                "wording 열에 근거 문장을 짧게 적어 두면 나중에 확인하기 쉽습니다. 평년 대비 표현만 있는 호는 전년 대비가 아니므로 basis 열에 그렇게 적습니다."),
              P("가을배추는 krei_nonsummer_issues.csv 에 season=가을, crop_year(그해), year, issue_month, prod_yoy, area_yoy 를 적습니다(7~12월호). 11~12월 예측에만 쓰이고, "
                "생산 전년비가 아직 없으면 면적 전년비가 대신 쓰입니다. 같은 파일의 겨울·봄 줄은 지금 모형에서 쓰이지 않습니다."),
              P("<b>6. 결과 읽는 법</b>", "h1"),
              B("gate_predictions.csv 한 줄 = 원점 하나. 열 final 이 최종 예측, gate_calm 이 상태(0 경보 / 1 평시), S1 이 그때 쓰인 KREI 생산 전망, ship_q 가 같은 호의 출하량 전망 방향(1 증가 / 0 비슷 / -1 감소), krei_issue 가 몇 월호인지. "
                "head 는 그 원점에 쓰인 경보 모형 값(6~9월 원점은 an_sparse_all, 그 밖은 고랭지 날씨를 출하기에만 반영한 an_sparse_ship), final_prod_only 는 출하량 전망을 보지 않았다면 나왔을 최종값입니다. "
                "gate_hgb 는 스위치만 적용한 값, winter_ridge 는 겨울 전용 모형 값(1~4월 대상일 때만), adj_pct 는 수준 조정 폭(%), adj_issue·S_adj 는 조정에 쓴 호와 전망. "
                "final_base 는 9월 고온 층을 넣기 전 값이고, heat_H(9월 30도 넘은 날 수)·heat_pct(그에 따른 조정 폭)는 9월 말 원점 줄에만 채워집니다."),
              B(f"오차의 크기 감각: 평소에는 ±{mp[FINAL]:.0f}% 안팎, 여름에는 ±{mps[FINAL]:.0f}% 안팎으로 봅니다. 경보 상태에서 급등이 실제로 오면 예측이 6~11% <b>낮을</b> 가능성이 큽니다(2022·2024). "
                "평시 상태에서 급등이 없으면 예측이 15%까지 <b>높을</b> 수 있습니다(2025)."),
              B("'경보'는 '급등한다'가 아니라 '공급이 줄 전망이라 급등 가능성을 열어 둔 모형을 썼다'는 뜻입니다. 2026년 8월 말처럼 경보 상태에서도 가격이 낮으면 낮게 예측합니다. "
                "상태는 달마다 다시 판정하므로 한 해 안에서 바뀔 수 있습니다(2026년: 6·7월 말 평시, 8월 말 경보)."),
              B("같은 원점의 an_sparse_all(경보 모형)·wx_hgb_summer(평시 모형)·price_only_ridge(가격만)·gate_hgb(조정 전) 값도 같은 파일에 있어 '스위치가 반대였다면', '조정이 없었다면' 을 바로 볼 수 있습니다."),
              PageBreak()]

    # 7
    story += [P("7. 못 하는 것과 조심할 점", "h1"),
              B("<b>기록이 짧습니다.</b> KREI 월보 스위치는 2021~2026년 여섯 해, 그중 급등은 두 번뿐입니다. 스위치 규칙(0 기준)과 경보·평시의 조합은 2024·2025년을 본 뒤 정했습니다. "
                "새 해가 쌓일 때마다 규칙을 다시 점검해야 하고, 2021년 이전으로 거슬러 적용하면 오히려 나빠져서 그 구간에는 쓰지 않습니다."),
              B(f"<b>출하량 전망을 엮은 규칙은 근거가 사실상 2026년 한 해입니다.</b> 2026년 8월의 과대 예측을 본 뒤에 만들었고, 월말에서 바뀌는 것이 2026년 {len(chg)}개 달뿐이라 그 해에서 좋아진 것은 검증이 아닙니다. "
                "2021~2025년 월말에는 '생산 감소 + 출하 증가'인 달이 없었습니다. 월보 문장의 방향만 읽은 값이고 판독이 약한 호도 있습니다(2022·2024년 10월호, 2025·2026년 6월호는 표현 없음)."),
              B("<b>생산 전망이 0 근처일 때의 처리는 정해지지 않았습니다.</b> 출하량 전망이 '감소'인 호는 생산 전망이 -0.1%여도 -10%와 똑같이 경보입니다."),
              B("<b>평시·경보 구분은 6~9월 말 원점에만 있습니다.</b> 봄·가을·겨울을 같은 방식으로 나누는 시험은 효과가 없었고, 급락을 미리 알리는 장치도 없습니다."),
              B(f"<b>고랭지 날씨를 출하기에만 쓰는 것은 이득이 작습니다.</b> 봄 대상 오차율은 {spr[HEAD]:.1f}%에서 {spr['head']:.1f}%로 줄었지만 주간 예측에서는 30%를 넘게 틀리는 원점이 오히려 조금 늘었습니다. "
                "다른 작형(봄·가을·겨울) 주산지의 날씨는 시기를 맞춰 넣어도 나빠져서 넣지 않았습니다."),
              B(f"<b>급등 해에도 끝까지는 못 올라갑니다.</b> 2022년 {e[2022][FINAL]:+.0f}%, 2024년 {e[2024][FINAL]:+.0f}%. 급등 해에 더 높이 잡는 별도 모형은 다음 과제입니다."),
              B(f"<b>급등 없는 해에 높게 볼 수 있습니다.</b> 2025년 9월 {e[2025][FINAL]:+.0f}%. 평시 모형(트리)이 폭염 해에 과대 예측한 탓입니다. "
                "평시 모형을 가격만 보는 모형으로 바꾸면 2025년은 -9%가 되지만 2021·2023년의 보통 상승을 -26%·-17%로 놓칩니다. 정확도 전체로는 트리가 나아 트리를 골랐습니다."),
              B(f"<b>오차가 30%를 넘는 달이 남아 있습니다.</b> 2021년 1월~2026년 9월 69개 달 가운데 {len(big[FINAL])}개({big_txt})입니다. 대부분 값이 오르거나 내리는 때가 평년보다 몇 주 늦거나 빨랐던 달이고, "
                "겨울 분기와 수준 조정이 그중 둘(2021년 1월, 2022년 8월)을 30% 밖으로 밀어냈습니다."),
              B(f"<b>겨울 분기는 주간 예측에서는 이득이 없습니다.</b> 월말 예측은 겨울 오차율이 {win_old:.1f}%에서 {win[FINAL]:.1f}%로 줄지만 주간 예측은 {winw_old:.1f}%에서 {winw[FINAL]:.1f}%로 늘었습니다(견준 대상은 고랭지 날씨를 1년 내내 쓰던 종전 경보 모형). 주간으로 쓸 때는 an_sparse_all 열과 함께 보십시오."),
              B("<b>수준 조정은 봉우리 뒤에 손해입니다.</b> 경보 달을 일괄로 올리므로 값이 이미 꺾인 달(2022년 8월처럼)에는 과대 예측이 더 커집니다."),
              B("<b>10월 하락 전환은 여전히 약합니다.</b> 가을배추가 나오기 시작하면 값이 내리는데, 언제 나오는지에 대한 신호가 입력에 없습니다. 9월 고온 층은 '더운 9월 뒤에는 늦게 내린다'는 두 해(2023·2024)의 경험을 넣은 것이고, "
                "10월에 들어선 뒤의 주간 예측에는 오히려 해로워 월말 9월 원점에만 씁니다. 2026년 10월 실적이 이 부품의 첫 검증입니다."),
              B("<b>KREI의 판단을 물려받습니다.</b> 스위치는 통계가 아니라 관측센터의 전망치입니다. 전망이 틀린 해(줄어든다고 했는데 급등이 없거나 그 반대)에는 모형도 같이 틀립니다. "
                "출하량 전망의 방향까지 쓰면서 이 의존이 더 커졌습니다. 월보의 가격 전망치와 포전거래 가격은 쓰지 않습니다."),
              B("<b>트리 모형은 실행 환경에 조금 민감합니다.</b> 같은 패키지 버전이면 결과가 똑같지만(시드 고정), 버전이 바뀌면 ±2~3%p 움직일 수 있습니다. 경보 모형(선형)은 어디서 돌려도 같습니다."),
              B("<b>월보를 아는 시점의 전제.</b> 스위치와 수준 조정 모두 담당자가 전달 말일에 다음 달 호 내용을 안다는 전제로 다음 달 호 값을 씁니다(7월 말에는 8월호). 담당자가 아닌 사람이 쓰면 이 전제가 성립하지 않고, 한 호 늦은 값을 넣어야 합니다.")]

    # 용어
    story += [P("용어 풀이", "h1"),
              table([["용어", "뜻"],
                     ["원점", "예측을 하는 시점. '8월 말 원점'은 8월 31일에 서서 그 뒤 28일을 예측한다는 뜻"],
                     ["평년치", "직전 5년 같은 시기의 평균 가격. 경보 모형의 기준점"],
                     ["계절편차", "'평년 이맘때'와 비교해 얼마나 다른가. 가격도 날씨도 이 편차로 바꿔 씀"],
                     ["ridge / 선형", "변수와 결과를 직선 관계로 보는 단순 모형. 지나친 반응은 억제(규제)"],
                     ["HGB / 트리", "조건을 나눠가며 규칙을 찾는 모형. 경험 밖에서는 평평해짐"],
                     ["스위치 / 게이트", "어느 모형을 쓸지 고르는 규칙. 여기서는 KREI 여름배추 생산 전망의 부호와 그 달 출하량 전망의 방향"],
                     ["경보 / 평시", "경보 모형이 쓰인 상태(생산 전망이 같거나 줄고 출하량 전망도 늘지 않음) / 평시 모형이 쓰인 상태(생산 전망이 늘거나 출하량 전망이 늚)"],
                     ["출하기 한정", "고랭지 날씨를 앞으로 28일 가운데 여름배추 수확 기간에 드는 날의 비율만큼만 반영하는 것"],
                     ["MAPE (오차율)", "예측이 실제와 평균 몇 % 어긋났는지. 낮을수록 좋음"],
                     ["오차 (이 문서)", "예측 ÷ 실제 - 1. -10%면 실제보다 10% 낮게 본 것"],
                     ["표본외", "모형을 만들 때 전혀 보지 않은 자료. 여기서는 2026년 8월 말 원점(9월 예측). 2026년 6·7월 말 원점은 그 오차를 본 뒤 출하량 전망 규칙을 만들었으므로 해당하지 않음"]],
                    [36, 132], st)]

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=20 * mm, title="배추 도매가격 예측모형 설명서", author="cabbage-surge")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / f"배추급등예측_모형설명서_{TODAY}.pdf"))
    a = ap.parse_args()
    build(Path(a.out))
