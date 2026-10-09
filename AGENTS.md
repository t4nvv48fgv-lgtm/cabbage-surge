# cabbage-surge 에이전트 가드레일 (Authoritative)

이 문서는 cabbage-surge(2024년 9월 배추 급등 추종 예측모형) 프로젝트의 **권위 가드레일 파일**이다.
`CLAUDE.md`와 이 파일이 충돌하면 **이 파일을 따른다**.

> 뼈대 출처: `C:\Users\user\Desktop\논문\AGENTS.md`(겨울배추 저장 옵션 논문) → `Desktop\생육\AGENTS.md`를 거쳐 2026-10-07 이식.
> 규칙의 배경·절차·프롬프트는 `docs/PAPER_WRITING_PLAYBOOK.md`. 세 프로젝트는 **별개**다 — 다른 논문의 원고·위키·수치를 승인 없이 끌어오지 않는다.

## 프로젝트 루트

```text
C:\Users\user\Desktop\cabbage-surge
```

CABIS 저장소(`C:\Users\user\Desktop\CABIS`)의 지침·가드레일은 여기에 적용되지 않으며, CABIS 안의 파일을 수정·실행하지 않는다(읽기만).

## 작업 전 확인

- 현재 작업 디렉터리와 편집할 파일의 절대 경로를 확인한다.
- 구조·논지·인용·데이터에 영향을 주는 작업 전에 먼저 읽는다:
  - `docs\PAPER_SPEC.md` — 주제·기여·타깃·분량·마감
  - `docs\OUTLINE.md` — 목차와 각 절의 핵심 주장
  - `docs\STYLE_GUIDE.md` — 인용·용어·표기 규칙
  - `wiki\handoff-next-session.md` / `docs\HANDOFF_20261006.md` — 현황·결과·실패 기록
  - `wiki\risk-audit.md` — 논지 리스크(집필 전·투고 전 재독)

## 보호 파일 및 영역

사용자가 명시적으로 승인하지 않으면 편집·삭제·덮어쓰기 금지:

- `data\raw\*` — CABIS 스냅숏(읽기 전용). 갱신은 `src\sync_from_cabis.py`로만(단방향 복사).
- `data\ext\*` — 사용자가 제공한 외부 자료 원본(읽기 전용, 2026-10-09~). 등록 시 `data\ext\README.md`에 출처·기준일·인용 가능 확인을 적는다. 공개 저장소이므로 사용자가 인용·커밋 가능을 확인한 자료만.
- `manuscript\main.md` — 마스터 원고(**진실의 원천**)
- `references.bib` — 참고문헌 원본
- `docs\PAPER_SPEC.md`, `docs\OUTLINE.md` — 논지·구조의 정본(**현재 미승인 초안** — 사용자 승인 후 정본으로 승격)
- `src\*` 모형 코드의 **평가 설계**(원점 집합·타깃 정의·누수 차단 규칙) — 바꾸면 결과표 전체가 비교 불능이 되므로 제안 후 승인
- `outputs\*.png` — 추적되는 그림(재생성은 가능하나 결과표와 함께 갱신)
- `submitted\*`, `deliverable\*` — (생기면) 제출본·최종본, 읽기 전용

`data\processed\`·`outputs\*.csv`·`build\`는 스크립트로 재생성하는 파생물이라 보호 대상이 아니다(단, `data\processed\`·`outputs\*.csv`는 2026-10-09부터 git 추적하므로 코드·원자료 변경 시 재실행해 같은 커밋에 넣는다). 마스터 원고·bib를 대량 치환·재배열하기 전에는 백업한다.

## 절대 규칙 (학술 무결성)

이 규칙은 사용자가 요청하더라도 완화되지 않는다.

### 인용
- **존재하지 않는 문헌을 인용하지 않는다.** 서지를 기억으로 재구성하지 않는다.
- 인용은 (a) `references.bib` 등록 키 (b) 사용자가 제공·확인한 출처만. 불확실하면 `[CITATION NEEDED: 무엇에 대한 근거]`.
- KREI 두 연구(Chronos-2 결과세미나 자료 2026-09-30, 배추가격예측력개선 연구보고 2026-06)와 CABIS 내부 산출물은 **미발간·내부 자료일 수 있다.** 인용 가능 여부와 범위(공개 표시·저자 동의)를 사용자에게 확인한 뒤 bib에 등록하고 쓴다. 원본 PDF는 `docs\references\`에 보존(참조 자료 보존 규칙).

### 데이터·결과
- **수치를 지어내지 않는다.** 원고의 모든 수치는 `src\` 스크립트 재실행으로 재현되는 `outputs\*.csv`에서 온다. 수치 흐름: `outputs\*.csv` → `README.md` 결과표 → 원고. 값이 바뀌면 이 순서로 갱신한다.
- 값이 없으면 `[VALUE NEEDED]`. 실행 때마다 ±2~3%p 흔들리는 값(트리 모형)은 고정 시드·실행 일자를 함께 적는다.
- **누수 금지**: 학습 표본의 타깃 창 종료일 ≤ 원점, 피처는 원점 당일까지의 정보만. 엄밀하지 않은 부분(기후 평년값 2010~2019 고정창 등)은 고치거나 §4 한계에 명시한다.
- **같은 원점 집합·같은 타깃**으로만 모형을 비교한다. 비교 기준선은 가격만 ridge가 아니라 Chronos-2 제로샷·MSTL(추세 외삽 단변량).
- **2024-09 한 점 과적합 경계**: 2020-09·2022-09·2025-08(급등 없던 해, 거짓 경보 비용)을 반드시 함께 보고한다. 하이퍼파라미터는 2016~2023에서 고르고 2024~2025로 검증하는 분리를 지킨다(미완이면 한계로).
- 단위: 가락 상품 원/10kg. KREI 세미나자료는 원/kg이므로 비교 시 10배. 집계 방식이 다른 외부 수치와의 비교는 "엄밀 비교 아님"을 명시.

### 논지·주장
- 핵심 주장·기여·결론을 조용히 바꾸지 않는다. 변경은 제안 후 승인.
- 강도를 임의로 높이지 않는다. 확정된 결론의 상한: **"한계는 가격 정보가 아니라 평균회귀 구조에 있다"**, **"고랭지 폭염 피처가 학습범위 밖 외삽에서 선형 구조로 급등의 절반 이상을 잡는다"**. "급등을 예측한다"·"10월 전환을 잡는다"는 결과가 뒷받침하지 않는다.
- 사후에 고른 피처·원점(2024-09를 보고 설계한 것)은 사전 예측처럼 서술하지 않는다(post hoc 명시).

### 표절·저작권
- 출처 문장을 그대로 옮기지 않는다. 직접 인용은 따옴표+페이지. KREI 보고서의 그림·표를 허가·출처 없이 복제하지 않는다.

### 참조 자료 보존
- 세션 중 읽은 외부 자료(투고규정·작성요령·심사평·KREI 보고서)는 **저장소 안에 복사**한다(`docs\journal-guidelines\`, `docs\references\`). 바이너리는 텍스트 추출본 동봉. 적용한 조항 번호를 `docs\STYLE_GUIDE.md`에 적는다.

### 외부 행위
- 원고·데이터를 투고 시스템·이메일·외부 서비스로 전송·제출하지 않는다. 다운로드·업로드는 승인 후.
- 원격 저장소는 공개(public)다 — CABIS 원본 경로·내부 자료 실명·API 키를 커밋하지 않는다.

## 검증 (안전, 비파괴)

```powershell
cd C:\Users\user\Desktop\cabbage-surge

# 0. 모형 결과 재현 (수치 변경 시)
python src\build_features.py; python src\backtest.py M; python src\anomaly_model.py M

# 1. 인용 무결성 + 자리표시자 + 문체 검산 (원고 수정 후 0건이 완료 조건)
py tools\check_style_rules.py manuscript\main.md

# 2. 빌드 (--lua-filter는 반드시 --citeproc 뒤)
pandoc manuscript\main.md -o build\main.docx --citeproc --lua-filter=styles\refs-korean-first.lua --bibliography=references.bib --csl=styles\citation.csl
```

이 PC: Python은 `python`(이 프로젝트에서는 동작 확인됨)과 `py` 모두 가능, pandoc은 `%LOCALAPPDATA%\Pandoc`. 콘솔 한글은 `PYTHONIOENCODING=utf-8`.

## 보고 형식

- 수정한 파일 / 생성한 파일·폴더
- 추가·변경한 인용 (키, 출처 확인 여부)
- 삽입한 자리표시자 위치
- 논지·주장 변경 여부
- 데이터·수치 변경 여부 (재실행 명령·실행 일자 포함)
- 실행한 검증과 결과
- 남은 위험 / 다음 단계
