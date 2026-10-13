# -*- coding: utf-8 -*-
"""마크다운 문서를 PDF로 옮기는 간단한 변환기(이 저장소의 설명 문서용).

지원: 제목(#·##·###), 문단, 글머리표(- , 두 칸 들여쓴 하위 항목), 번호 목록, 파이프 표, ``` 코드 블록, **굵게**, `코드`.
글꼴·스타일·표는 tools/make_report.py 의 것을 재사용한다(한글 글꼴 등록 포함).

실행:  python tools/md_to_pdf.py docs/모형설명_2026-10-13.md [--out docs/모형설명_2026-10-13.pdf]
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_report import FONT, FONT_B, INK, INK2, styles, table  # noqa: E402

WIDTH_MM = 170.0


def inline(t: str) -> str:
    t = t.replace("−", "–")     # 등록 글꼴에 수학 빼기 부호(U+2212)가 없어 빈칸으로 찍힌다 → 같은 폭의 줄표로
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", rf'<font name="{FONT_B}">\1</font>', t)
    t = re.sub(r"`(.+?)`", rf'<font color="{INK2}">\1</font>', t)
    return t


def disp_width(t: str) -> int:
    t = re.sub(r"\*\*|`", "", t)
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in t)


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def col_widths(rows: list[list[str]]) -> list[float]:
    n = len(rows[0])
    w = [min(max(max(disp_width(r[i]) for r in rows if i < len(r)), 6), 70) for i in range(n)]
    tot = sum(w)
    w = [max(x / tot * WIDTH_MM, 18.0) for x in w]
    s = sum(w)
    return [x / s * WIDTH_MM for x in w]


def build(src: Path, out: Path) -> None:
    st = styles()
    st["bullet2"] = ParagraphStyle("bullet2", parent=st["bullet"], leftIndent=22, bulletIndent=12)
    st["num"] = ParagraphStyle("num", parent=st["bullet"], leftIndent=14, bulletIndent=0)
    st["code"] = ParagraphStyle("code", fontName="Courier", fontSize=8.6, leading=12, textColor=colors.HexColor(INK),
                                backColor="#f3f4f7", leftIndent=4, borderPadding=(2, 4, 2, 4))
    lines = src.read_text(encoding="utf-8").splitlines()
    story, title = [], src.stem
    i, para = 0, []

    def flush():
        if para:
            story.append(Paragraph(inline(" ".join(para)), st["body"]))
            story.append(Spacer(1, 3))
            para.clear()

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("```"):
            flush()
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code = lines[i].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;")
                font = "Courier" if lines[i].isascii() else FONT
                story.append(Paragraph(f'<font name="{font}">{code or "&nbsp;"}</font>', st["code"]))
                i += 1
            story.append(Spacer(1, 5))
        elif s.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            flush()
            rows = [split_row(ln)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            n = len(rows[0])
            rows = [(r + [""] * n)[:n] for r in rows]
            story.append(Spacer(1, 2))
            story.append(table([[inline(c) for c in r] for r in rows], col_widths(rows), st))
            story.append(Spacer(1, 6))
            continue
        elif s.startswith("# "):
            flush()
            title = s[2:].strip()
            story.append(Paragraph(inline(title), st["title"]))
        elif s.startswith("## "):
            flush()
            story.append(Paragraph(inline(s[3:]), st["h1"]))
        elif s.startswith("### "):
            flush()
            story.append(Paragraph(inline(s[4:]), st["h2"]))
        elif re.match(r"^\s*- ", ln):
            flush()
            deep = len(ln) - len(ln.lstrip()) >= 2
            story.append(Paragraph(inline(s[2:]), st["bullet2" if deep else "bullet"], bulletText="–" if deep else "•"))
        elif re.match(r"^\s*\d+\. ", ln):
            flush()
            num, rest = s.split(". ", 1)
            story.append(Paragraph(inline(rest), st["num"], bulletText=f"{num}."))
        elif not s:
            flush()
        else:
            para.append(s)
        i += 1
    flush()

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT, 8)
        canvas.setFillColor(colors.HexColor(INK2))
        canvas.drawString(20 * mm, 12 * mm, f"{title} · cabbage-surge")
        canvas.drawRightString(190 * mm, 12 * mm, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=20 * mm,
                            title=title, author="cabbage-surge")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    src = Path(a.src)
    build(src, Path(a.out) if a.out else src.with_suffix(".pdf"))
