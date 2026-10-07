---
title: "[제목 미정]"
author:
  - "[저자 미정]"
lang: ko
bibliography: ../references.bib
csl: ../styles/citation.csl
---

<!--
마스터 원고 (진실의 원천). 2026-10-07 뼈대만 생성 — 본문 미작성.
구조: 범용 4절 (docs/OUTLINE.md). 타깃 저널 미정 — 특정 학회지 양식에 맞추지 않는다.
주제·연구 질문이 확정되기 전에는 본문을 쓰지 않는다 (docs/PAPER_SPEC.md "확정 대기 목록").
인용: [@key] — references.bib 등록분만. 불확실하면 [CITATION NEEDED: ...] / 수치가 없으면 [VALUE NEEDED].
수치: outputs/*.csv(스크립트 재실행으로 재현) → README 결과표 → 여기. 실행 일자를 표 주에 적는다.
수정 전 필독: docs/EDIT_CHECKLIST.md. 수정 후: py tools/check_style_rules.py manuscript/main.md 0건.
빌드: pandoc manuscript\main.md -o build\main.docx --citeproc --lua-filter=styles\refs-korean-first.lua --bibliography=references.bib --csl=styles\citation.csl
-->

**초록**

[초록 미작성 — 배경·방법·결과·결론]

**주요어**: [미정]

# 1. 서론

[미작성 — 문제의식(2024-09 급등과 두 KREI 연구의 공통 한계), 선행연구, 연구 질문, 기여, 논문 구성]

# 2. 자료 및 방법

## 2.1. 자료

[미작성 — 가락 일별 가격·반입, 고랭지 기상, 정부 방출. 근거: docs/HANDOFF_20261006.md §3]

## 2.2. 타깃과 원점

[미작성 — 향후 28일 평균가격, 월말·주간 원점, 누수 차단 규칙]

## 2.3. 피처

[미작성 — 가격군·수급군·고랭지 기상군, 평년 편차 정의]

## 2.4. 모형

[미작성 — 선형·트리·계절 정상치 편차·단변량 기준모형]

## 2.5. 평가

[미작성 — 확장창 롤링, 지표, 사례 원점 집합]

# 3. 결과

## 3.1. 2024년 여름의 특이성

[미작성]

## 3.2. 월말 원점 비교

[미작성]

## 3.3. 주간 원점 추적

[미작성]

## 3.4. 거짓 경보와 강건성

[미작성]

# 4. 요약 및 결론

[미작성 — 기여 재확인, 한계(단일 사건·사후 설계·전환 신호 부재·평년값 창), 후속 연구]

# 참고문헌

::: {#refs}
:::
