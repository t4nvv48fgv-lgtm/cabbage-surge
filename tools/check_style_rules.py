# -*- coding: utf-8 -*-
"""원고 문체·용어·무결성 검산 — `Desktop\\논문`의 scratchpad/check_style_rules.py를 이식해 범용화한 판 (2026-10-07).

사용:
    py check_style_rules.py [원고.md | 원고.docx] [--bib references.bib] [--no-cite] [--pii]

기본 대상: <프로젝트 루트>/manuscript/main.md (이 스크립트의 상위 폴더를 루트로 본다).
모든 원고 수정 후 실행해 "전부 0건"을 확인한다. 위반이 있으면 종료 코드 1.

검사 항목:
  1. 금지 패턴 — BAD_GENERAL(일반 규칙: 직역투·구어·종결·'즉'·'증거' 등)
     + 같은 폴더의 style_rules_local.py 에 정의한 BAD_EXTRA(프로젝트 고유 용어). ALLOW에 적은 문자열은 빼고 센다.
  2. 자리표시자 — [CITATION NEEDED], [VALUE NEEDED], TODO, ???, <미정> 류
  3. 인용 무결성 (md만) — 본문 [@key]/@key ↔ references.bib 키 대조. 누락 키는 위반, 미사용 bib 항목은 참고 출력.
  4. 괄호 — 산문의 통계 보고값 괄호((β=…)·(R²=…)·(p=…)·(n=…)), 괄호 겹침
  5. 여는따옴표 방향 오류(공백 뒤 ’)
  6. --pii: 전화번호 패턴(개인정보가 든 원자료를 다루는 프로젝트용)

허용 예외: 표·그림 라벨/주석 문단(표/그림/주:/Table/Fig./Note:/Source: 로 시작), 표 행(| 로 시작),
          코드 블록·HTML 주석·YAML 머리말은 검사하지 않는다.
규칙 정본: docs/STYLE_GUIDE.md, 실전 요약: docs/EDIT_CHECKLIST.md, 배경: docs/PAPER_WRITING_PLAYBOOK.md §4.
"""
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ── 일반 금지 패턴 (주제와 무관하게 논문 프로젝트에서 사용자가 확정한 것) ────────────
BAD_GENERAL = [
    # 번역투 (STYLE_GUIDE 번역투 금지 목록)
    "의 그것", "공백으로 남아", "차별성을 가진다", "보고한다", "함을 확인한다",
    "식별의 원천", "해석의 경계", "해석의 범위", "집계적", "시사적", "에서의", "으로의",
    "구조를 갖는다", "방향을 가리", "크기를 주장", "유보가 필요", "중요한 유보",
    "독립 관측", "보조 관찰", "무게중심", "변이", "벤치마크", "간명한", "사양",
    # '증거' 전면 폐기(07-27) — '볼 근거는 없다'·'보기 어렵다'·'결론지을 수 없다'로
    "증거",
    # 동격·요약의 '즉' 금지(07-29)
    ", 즉 ", ". 즉 ",
    # 구어·과교정
    "걷어내", "통째로", "커 보이게", "밝혀 둔다", "그대로다", "떠안", "다가서",
    "잡아도", "재는 것", "재야 할", "모아 쓴", "출렁", "못박", "쥔 주체",
    "변동인데", "만든 충격", "다시 뽑",
    # 종결(모음 끝 명사 + '다.' → '이다.')·과거형
    "효과다.", "결과다.", "단위다.", "가치다.", "단계다.", "형태다.", "해다.", "정보다.", "원리다.", "렌트다.", "과제다.",
    "측정했다", "적용했다", "확인했다", "검정했으나", "추정했다",
    # 외래어 조어 (robustness → 강건성)
    "로버스트니스", "견고성",
]

LABEL_PREFIX = ("표 ", "그림 ", "주:", "주 1)", "주 2)", "주 3)", "주 4)", "주1)", "주2)", "주3)", "주4)",
                "Table ", "Fig.", "Figure ", "Note:", "Source:", "자료:")
PLACEHOLDERS = ["[CITATION NEEDED", "[VALUE NEEDED", "TODO", "???", "<미정>", "[미정", "[미작성", "확인 필요]"]
STAT_PAREN = re.compile(r"\((?:β|beta|R²|R2|p|n|t|SE|CI|φ|σ|λ|δ)\s*[=<>≈]")

# 'AI가 쓴 느낌' 의심 문형 — 경고만 낸다 (PLAYBOOK §14.2 유형표). 핵심 결론의 'A가 아니라 B'처럼 남길 것이 섞여 있다.
AI_TONE = [
    ("예고 문장 '~절에서 확인/논의/제시한다'", r"\d(?:\.\d)?절에서\s*(?:확인|논의|제시|다룬|살펴본|검토)"),
    ("예고 문장 '~N가지이다/가 있다. 첫째'", r"(?:두|세|네|다섯)\s*가지(?:이다|가 있다|로)\s*.{0,40}첫째"),
    ("논문 구성 문단", r"논문의 구성은|논문은 다음과 같이 구성"),
    ("되풀이 요약 '따라서/요컨대/종합하면 … 이다'", r"(?:^|\.\s)(?:따라서|요컨대|종합하면)\s[^.]{0,60}이다\."),
    ("'A가 아니라 B' 대구", r"가 아니라"),
    ("점검 보고 '성립하였다/위반 0건/표준오차'", r"위반(?:은|이)?\s*0건|성립하였다|표준오차의 중앙값"),
    ("내부 근거 언급 '원문/내부자료와 일치'", r"(?:원문|내부자료|내부 자료)(?:과|와)\s*일치"),
    ("연결 수사 '하나의 결론으로 모인다/수렴한다'", r"하나의 결론으로|하나로 (?:모인다|수렴)"),
]


def load_local():
    extra, allow, label_extra, placeholders_ignore = [], [], (), []
    path = os.path.join(HERE, "style_rules_local.py")
    if os.path.exists(path):
        ns = {}
        with open(path, encoding="utf-8") as f:
            exec(compile(f.read(), path, "exec"), ns)
        extra = list(ns.get("BAD_EXTRA", []))
        allow = list(ns.get("ALLOW", []))
        label_extra = tuple(ns.get("LABEL_PREFIX_EXTRA", ()))
        placeholders_ignore = list(ns.get("PLACEHOLDERS_IGNORE", []))
    return extra, allow, label_extra, placeholders_ignore


def read_md(path):
    text = open(path, encoding="utf-8").read()
    # YAML 머리말·HTML 주석·코드 블록 제거
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4:]
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras, text


def read_docx(path):
    import docx  # python-docx
    doc = docx.Document(path)
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    tables = [cell.text for tbl in doc.tables for row in tbl.rows for cell in row.cells]
    return paras + ["| " + t for t in tables], "\n".join(paras + tables)


def split_prose(paras, label_prefix):
    prose, labels = [], []
    for p in paras:
        first = p.lstrip()
        if first.startswith("|") or first.startswith(label_prefix) or first.startswith("#"):
            labels.append(p)
        else:
            prose.append(p)
    return "\n".join(prose), "\n".join(labels)


def cite_check(text, bib_path):
    keys_in_text = set(re.findall(r"@\{?([A-Za-z][\w:-]*?)\}?(?=[\s\];,.)ㄱ-힝]|$)", text))
    keys_in_text = {k.rstrip("}") for k in keys_in_text if not k.startswith("fig")}
    bib = open(bib_path, encoding="utf-8").read() if os.path.exists(bib_path) else ""
    bib_keys = set(re.findall(r"^@\w+\s*\{\s*([^,\s]+)\s*,", bib, flags=re.M))
    missing = sorted(keys_in_text - bib_keys)
    unused = sorted(bib_keys - keys_in_text)
    return keys_in_text, bib_keys, missing, unused


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    path = args[0] if args else os.path.join(ROOT, "manuscript", "main.md")
    bib_path = sys.argv[sys.argv.index("--bib") + 1] if "--bib" in sys.argv else os.path.join(ROOT, "references.bib")
    extra, allow, label_extra, ph_ignore = load_local()
    label_prefix = LABEL_PREFIX + label_extra

    if path.lower().endswith(".docx"):
        paras, full = read_docx(path)
        is_md = False
    else:
        paras, full = read_md(path)
        is_md = True
    prose, labels = split_prose(paras, label_prefix)

    print(f"대상: {os.path.normpath(path)}")
    problems = []

    # 1. 금지 패턴 (산문 + 표/라벨 모두)
    for bad in BAD_GENERAL + extra:
        n = full.count(bad)
        for ok in allow:
            if bad in ok:
                n -= full.count(ok)
        if n > 0:
            problems.append((f"금지 패턴 '{bad}'", n))

    # 2. 자리표시자
    for ph in PLACEHOLDERS:
        if ph in ph_ignore:
            continue
        n = full.count(ph)
        if n:
            problems.append((f"자리표시자 '{ph}'", n))

    # 3. 인용 무결성
    if is_md and "--no-cite" not in flags:
        keys, bib_keys, missing, unused = cite_check(full, bib_path)
        print(f"인용 키 {len(keys)}건 / bib 항목 {len(bib_keys)}건")
        for k in missing:
            problems.append((f"bib에 없는 인용 키 '{k}'", 1))
        if unused:
            print(f"  (참고) 본문 미인용 bib 항목 {len(unused)}건: " + ", ".join(unused[:12]) + (" …" if len(unused) > 12 else ""))

    # 4. 괄호
    n_stat = len(STAT_PAREN.findall(prose))
    if n_stat:
        problems.append(("산문 통계 괄호 (β=·R²=·p=·n= …) — 서술로 풀 것", n_stat))
    n_nest = len(re.findall(r"\([^()\n]*\([^()\n]*\)", prose))
    if n_nest:
        problems.append(("괄호 겹침", n_nest))

    # 5. 여는따옴표 방향
    wrong_quotes = len(re.findall(r"\s’\S", prose))
    if wrong_quotes:
        problems.append(("여는따옴표 방향(’)", wrong_quotes))

    # 6. 개인정보
    if "--pii" in flags:
        n_phone = len(re.findall(r"01[016789]-?\d{3,4}-?\d{4}", full))
        if n_phone:
            problems.append(("전화번호 패턴(개인정보)", n_phone))

    # 7. 'AI가 쓴 느낌' 의심 문형 (경고만 — 2026-10-03~06 사용자 지적 유형, 판단은 사람이)
    warns = []
    for label, pat in AI_TONE:
        n = len(re.findall(pat, prose))
        if n:
            warns.append((label, n))
    if warns:
        print("AI투 의심(경고, 위반 아님 — PLAYBOOK §14.2로 판정):")
        for b, n in sorted(warns, key=lambda x: -x[1]):
            print(f"  {n:3d}건  {b}")

    if problems:
        print("잔존 위반:")
        for b, n in sorted(problems, key=lambda x: -x[1]):
            print(f"  {n:3d}건  {b}")
        sys.exit(1)
    print("전부 0건 ✓ (문체·용어·무결성 검산 통과)")


if __name__ == "__main__":
    main()
