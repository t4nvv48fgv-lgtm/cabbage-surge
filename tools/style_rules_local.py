# -*- coding: utf-8 -*-
"""cabbage-surge 프로젝트 고유 검산 사전 (check_style_rules.py가 읽음).

BAD_EXTRA : 이 논문에서 폐기·통일한 용어. 사용자가 확정할 때마다 날짜와 함께 추가한다.
ALLOW     : BAD에 걸리지만 허용하는 더 긴 문자열(정식 용어 등).
LABEL_PREFIX_EXTRA : 검사에서 제외할 라벨 문단 접두어 추가분.
PLACEHOLDERS_IGNORE: 자리표시자 중 이 프로젝트에서 무시할 것.
"""
BAD_EXTRA = [
    # 예) "급등락 예측"  # 2026-xx-xx 사용자 확정: "급등 추종"으로 통일
]
ALLOW = [
    "경쟁적 저장",   # 이론 용어 (competitive storage) — 금지어 '경쟁작형'과 구분
]
LABEL_PREFIX_EXTRA = ()
PLACEHOLDERS_IGNORE = ["[미작성", "[미정", "<미정>"]   # 뼈대 단계에서는 자리표시자가 정상 — 집필 시작 후 이 줄을 비울 것
