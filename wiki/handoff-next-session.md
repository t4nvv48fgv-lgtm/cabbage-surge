---
type: reference
tags: [handoff]
date: 2026-10-07
---

# 인계 노트 — 다음 세션 시작점

> 세션을 마칠 때마다 맨 위에 날짜 블록을 추가한다(양식: 사용자 결정 → 한 일 → 현재 상태 → 다음 할 일 → 함정 → 판단 대기 → 기각안). 모형 쪽 상세 이력은 `docs/HANDOFF_20261006.md`(2026-10-06 정본 인수인계서)가 담당하고, 이 파일은 그 뒤의 세션 블록을 쌓는다.

## 2026-10-07 — 논문 토대 이식

**① 사용자 지시.** `Desktop\논문`에서 쌓은 논문 작성·수정 규칙·절차·프롬프트를 정리해 이 프로젝트와 `Desktop\생육`에서 쓸 수 있게 할 것. 정리본만이 아니라 **요소 자체를 심을 것**.

**② 한 일.**
- 정리본 `docs/PAPER_WRITING_PLAYBOOK.md`(§0~§14: 하네스 체계·절대 규칙·작업 리듬·집필 순서·문체 규칙·리스크 감사·디펜스·심사 대응·인계·발표·축약판·분량 축소와 AI투 제거·프롬프트 모음). 생육에도 동일 사본.
- 검산 `tools/check_style_rules.py` + 프로젝트 사전 `tools/style_rules_local.py`(금지 패턴·자리표시자·인용↔bib·괄호·AI투 경고).
- 논문 구조 이식: `AGENTS.md`(권위 가드레일, 이 프로젝트 고유 규칙 포함) · `docs/PAPER_SPEC.md`·`OUTLINE.md`·`STYLE_GUIDE.md`·`EDIT_CHECKLIST.md`(전부 **미승인 초안**) · `manuscript/main.md`(뼈대) · `references.bib`(0건) · `styles/`(APA CSL·국문 우선 lua, 논문 폴더에서 복사) · `templates/`(Obsidian 노트 4종) · `wiki/risk-audit.md`(R-1~R-6 첫 목록) · `wiki/presentation-defense.md`(양식+질문 후보) · 이 파일.
- `CLAUDE.md` 읽기 순서 갱신.
- 논문 폴더 로컬 `main`을 `origin/main`(c266108)에 맞춤(사용자 승인) — 10-03~06 맥북 세션 커밋 13건 반영.

**③ 현재 상태.** 모형·결과는 10-06과 동일(원고에 들어간 수치 0). 뼈대 원고 pandoc 빌드·검산 통과 여부는 이 블록 아래 "검증"에. Git 미커밋 상태 — 커밋은 사용자 지시 후.

**④ 다음 할 일.**
1. 사용자와 **논문화 여부·RQ·기여 확정** → PAPER_SPEC 승격.
2. [[risk-audit]] 실행 목록 1~3(하이퍼파라미터 분리, 기준모형 대비 순효과, 급등/비급등 분리 지표) — 집필 전 필요.
3. KREI 두 연구·CABIS 산출물 인용 가능 범위 확인 → `references.bib` 등록·`docs/references/` 보존.
4. 선행연구 조사 → `wiki/literature/@key.md`(status: unread) → 원문 읽은 뒤 수치 인용.
5. `.obsidian/` 볼트 설정은 넣지 않았다(논문 폴더 것을 복사할지 사용자 결정).

**⑤ 함정.**
- 원격 저장소는 **공개**다. CABIS 내부 경로·실명·API 키를 올리지 않는다.
- 트리 모형 수치는 실행마다 ±2~3%p — 원고에 넣을 때 시드·실행 일자 고정.
- `Desktop\논문/side-notes/`에 10-03 추가된 "Chronos-2 배추 가격 예측 연구 따라가기 가이드"는 이 프로젝트에 참고가 되지만, 논문 폴더 세션에서는 격리 구역이다(여기서 읽는 것은 무방).
