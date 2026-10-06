from __future__ import annotations

import argparse
from pathlib import Path

"""
주차별 수집 지점 규칙 (엑셀 반영본)

작성 기준
- 입력 파일: 주차별 날씨 수집 지점.xlsx
- WEEKLY_REGION_RULES_TEMPLATE의 주차 키는 엑셀의 행 순서(1~52행)를 기준으로 작성
- 튜플 형식: (source, 지점ID, 지역명)

주의
- 엑셀 A열 값은 반복되는 월/구간 번호로 보이며, 여기서는 템플릿 규칙 형식에 맞춰
  1~52행을 그대로 1~52주 규칙으로 변환했습니다.
"""
WEEKLY_REGION_RULES_TEMPLATE = {
    1: [("asos", "261", "해남"), ("asos", "268", "진도군")],
    2: [("asos", "261", "해남"), ("asos", "268", "진도군")],
    3: [("asos", "261", "해남"), ("asos", "268", "진도군")],
    4: [("asos", "261", "해남"), ("asos", "268", "진도군"), ("aws", "628", "예산")],
    5: [("asos", "261", "해남"), ("asos", "268", "진도군"), ("aws", "628", "예산")],
    6: [("asos", "261", "해남"), ("asos", "268", "진도군"), ("aws", "628", "예산")],
    7: [("asos", "261", "해남"), ("asos", "268", "진도군"), ("aws", "628", "예산")],
    8: [("asos", "261", "해남"), ("asos", "268", "진도군"), ("aws", "628", "예산")],
    9: [("asos", "261", "해남"), ("aws", "628", "예산"), ("asos", "263", "의령군")],
    10: [("asos", "261", "해남"), ("aws", "628", "예산"), ("asos", "263", "의령군"), ("asos", "273", "문경"), ("aws", "634", "아산")],
    11: [("asos", "261", "해남"), ("aws", "628", "예산"), ("asos", "263", "의령군"), ("asos", "273", "문경"), ("aws", "634", "아산")],
    12: [("asos", "261", "해남"), ("aws", "628", "예산"), ("asos", "263", "의령군"), ("asos", "273", "문경"), ("aws", "634", "아산")],
    13: [("asos", "261", "해남"), ("aws", "628", "예산"), ("asos", "263", "의령군"), ("asos", "273", "문경"), ("aws", "801", "영양"), ("asos", "121", "영월")],
    14: [("asos", "261", "해남"), ("aws", "628", "예산"), ("asos", "263", "의령군"), ("asos", "273", "문경"), ("aws", "801", "영양"), ("asos", "121", "영월")],
    15: [("asos", "261", "해남"), ("aws", "628", "예산"), ("asos", "263", "의령군"), ("asos", "273", "문경"), ("aws", "801", "영양"), ("asos", "121", "영월")],
    16: [("asos", "261", "해남"), ("aws", "628", "예산"), ("asos", "263", "의령군"), ("asos", "273", "문경"), ("aws", "801", "영양"), ("asos", "121", "영월")],
    17: [("asos", "261", "해남"), ("aws", "628", "예산"), ("asos", "263", "의령군"), ("asos", "273", "문경"), ("aws", "801", "영양"), ("asos", "121", "영월"), ("aws", "526", "평창")],
    18: [("asos", "261", "해남"), ("aws", "628", "예산"), ("asos", "263", "의령군"), ("asos", "273", "문경"), ("aws", "801", "영양"), ("asos", "121", "영월"), ("aws", "526", "평창")],
    19: [("asos", "261", "해남"), ("aws", "628", "예산"), ("asos", "263", "의령군"), ("asos", "273", "문경"), ("aws", "801", "영양"), ("asos", "121", "영월"), ("aws", "526", "평창")],
    20: [("asos", "263", "의령군"), ("asos", "273", "문경"), ("aws", "801", "영양"), ("asos", "121", "영월"), ("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군")],
    21: [("asos", "273", "문경"), ("aws", "801", "영양"), ("asos", "121", "영월"), ("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군")],
    22: [("asos", "273", "문경"), ("aws", "801", "영양"), ("asos", "121", "영월"), ("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군")],
    23: [("asos", "273", "문경"), ("aws", "801", "영양"), ("asos", "121", "영월"), ("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군")],
    24: [("asos", "273", "문경"), ("aws", "801", "영양"), ("asos", "121", "영월"), ("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군")],
    25: [("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군"), ("asos", "105", "강릉")],
    26: [("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군"), ("asos", "105", "강릉")],
    27: [("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군"), ("asos", "105", "강릉")],
    28: [("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군"), ("asos", "105", "강릉")],
    29: [("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군"), ("asos", "105", "강릉")],
    30: [("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군"), ("asos", "105", "강릉")],
    31: [("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군"), ("asos", "105", "강릉")],
    32: [("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군"), ("asos", "105", "강릉")],
    33: [("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군"), ("asos", "105", "강릉")],
    34: [("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군"), ("asos", "105", "강릉"), ("asos", "101", "춘천"), ("asos", "261", "해남"), ("asos", "268", "진도군"), ("asos", "172", "고창"), ("aws", "634", "아산"), ("aws", "603", "괴산"), ("asos", "221", "제천"), ("aws", "801", "영양"), ("asos", "273", "문경")],
    35: [("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군"), ("asos", "105", "강릉"), ("asos", "101", "춘천"), ("asos", "261", "해남"), ("asos", "268", "진도군"), ("asos", "172", "고창"), ("aws", "634", "아산"), ("aws", "603", "괴산"), ("asos", "221", "제천"), ("aws", "801", "영양"), ("asos", "273", "문경")],
    36: [("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군"), ("asos", "105", "강릉"), ("asos", "101", "춘천"), ("asos", "261", "해남"), ("asos", "268", "진도군"), ("asos", "172", "고창"), ("aws", "634", "아산"), ("aws", "603", "괴산"), ("asos", "221", "제천"), ("aws", "801", "영양"), ("asos", "273", "문경")],
    37: [("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군"), ("asos", "105", "강릉"), ("asos", "101", "춘천"), ("asos", "261", "해남"), ("asos", "268", "진도군"), ("asos", "172", "고창"), ("aws", "634", "아산"), ("aws", "603", "괴산"), ("asos", "221", "제천"), ("aws", "801", "영양"), ("asos", "273", "문경")],
    38: [("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군"), ("asos", "105", "강릉"), ("asos", "101", "춘천"), ("asos", "261", "해남"), ("asos", "268", "진도군"), ("asos", "172", "고창"), ("aws", "634", "아산"), ("aws", "603", "괴산"), ("asos", "221", "제천"), ("aws", "801", "영양"), ("asos", "273", "문경")],
    39: [("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군"), ("asos", "105", "강릉"), ("asos", "101", "춘천"), ("asos", "261", "해남"), ("asos", "268", "진도군"), ("asos", "172", "고창"), ("aws", "634", "아산"), ("aws", "603", "괴산"), ("asos", "221", "제천"), ("aws", "801", "영양"), ("asos", "273", "문경")],
    40: [("aws", "526", "평창"), ("asos", "216", "태백"), ("asos", "217", "정선군"), ("asos", "105", "강릉"), ("asos", "101", "춘천"), ("asos", "261", "해남"), ("asos", "268", "진도군"), ("asos", "172", "고창"), ("aws", "634", "아산"), ("aws", "603", "괴산"), ("asos", "221", "제천"), ("aws", "801", "영양"), ("asos", "273", "문경")],
    41: [("asos", "105", "강릉"), ("asos", "101", "춘천"), ("asos", "261", "해남"), ("asos", "268", "진도군"), ("asos", "172", "고창"), ("aws", "634", "아산"), ("aws", "603", "괴산"), ("asos", "221", "제천"), ("aws", "801", "영양"), ("asos", "273", "문경")],
    42: [("asos", "105", "강릉"), ("asos", "101", "춘천"), ("asos", "261", "해남"), ("asos", "268", "진도군"), ("asos", "172", "고창"), ("aws", "634", "아산"), ("aws", "603", "괴산"), ("asos", "221", "제천"), ("aws", "801", "영양"), ("asos", "273", "문경")],
    43: [("asos", "105", "강릉"), ("asos", "101", "춘천"), ("asos", "261", "해남"), ("asos", "268", "진도군"), ("asos", "172", "고창"), ("aws", "634", "아산"), ("aws", "603", "괴산"), ("asos", "221", "제천"), ("aws", "801", "영양"), ("asos", "273", "문경")],
    44: [("asos", "105", "강릉"), ("asos", "101", "춘천"), ("asos", "261", "해남"), ("asos", "268", "진도군"), ("asos", "172", "고창"), ("aws", "634", "아산"), ("aws", "603", "괴산"), ("asos", "221", "제천"), ("aws", "801", "영양"), ("asos", "273", "문경")],
    45: [("asos", "105", "강릉"), ("asos", "101", "춘천"), ("asos", "261", "해남"), ("asos", "268", "진도군"), ("asos", "172", "고창"), ("aws", "634", "아산"), ("aws", "603", "괴산"), ("asos", "221", "제천"), ("aws", "801", "영양"), ("asos", "273", "문경")],
    46: [("asos", "105", "강릉"), ("asos", "101", "춘천"), ("asos", "261", "해남"), ("asos", "268", "진도군"), ("asos", "172", "고창"), ("aws", "634", "아산"), ("aws", "603", "괴산"), ("asos", "221", "제천"), ("aws", "801", "영양"), ("asos", "273", "문경")],
    47: [("asos", "105", "강릉"), ("asos", "101", "춘천"), ("asos", "261", "해남"), ("asos", "268", "진도군"), ("asos", "172", "고창"), ("aws", "634", "아산"), ("aws", "603", "괴산"), ("asos", "221", "제천"), ("aws", "801", "영양"), ("asos", "273", "문경")],
    48: [("asos", "261", "해남"), ("asos", "268", "진도군")],
    49: [("asos", "261", "해남"), ("asos", "268", "진도군")],
    50: [("asos", "261", "해남"), ("asos", "268", "진도군")],
    51: [("asos", "261", "해남"), ("asos", "268", "진도군")],
    52: [("asos", "261", "해남"), ("asos", "268", "진도군")]
}

WEEKLY_REGION_RULES = WEEKLY_REGION_RULES_TEMPLATE

_EXPECTED_WEEK_COUNT = 52
_REQUIRED_WEEKLY_RULES = {
    1: {("asos", "261", "해남"), ("asos", "268", "진도군")},
    4: {("aws", "628", "예산")},
    10: {("asos", "273", "문경"), ("aws", "634", "아산")},
    17: {("aws", "526", "평창")},
    25: {("asos", "105", "강릉")},
    34: {("asos", "101", "춘천"), ("aws", "603", "괴산"), ("asos", "221", "제천")},
    48: {("asos", "261", "해남"), ("asos", "268", "진도군")},
}


def get_weekly_region_rules() -> dict[int, list[tuple[str, str, str]]]:
    return WEEKLY_REGION_RULES


def count_weekly_region_rules() -> tuple[int, int, int]:
    week_count = len(WEEKLY_REGION_RULES)
    entry_count = sum(len(rules) for rules in WEEKLY_REGION_RULES.values())
    unique_station_count = len(
        {
            (str(source), str(station_id), str(region))
            for rules in WEEKLY_REGION_RULES.values()
            for source, station_id, region in rules
        }
    )
    return week_count, entry_count, unique_station_count


def validate_weekly_region_rules() -> list[str]:
    errors: list[str] = []
    rules = WEEKLY_REGION_RULES

    if not isinstance(rules, dict) or not rules:
        return ["WEEKLY_REGION_RULES is empty or not a dict"]

    expected_weeks = set(range(1, _EXPECTED_WEEK_COUNT + 1))
    actual_weeks = set(rules)
    missing_weeks = sorted(expected_weeks - actual_weeks)
    extra_weeks = sorted(actual_weeks - expected_weeks)
    if missing_weeks:
        errors.append(f"missing week keys: {missing_weeks}")
    if extra_weeks:
        errors.append(f"unexpected week keys: {extra_weeks}")

    for week, week_rules in sorted(rules.items()):
        if not isinstance(week, int):
            errors.append(f"week key is not int: {week!r}")
        if not isinstance(week_rules, list) or not week_rules:
            errors.append(f"week {week} has no rule list")
            continue
        for rule in week_rules:
            if not (isinstance(rule, tuple) and len(rule) == 3):
                errors.append(f"week {week} has invalid rule shape: {rule!r}")
                continue
            source, station_id, region = rule
            if source not in {"asos", "aws"}:
                errors.append(f"week {week} has invalid source: {rule!r}")
            if not str(station_id).strip():
                errors.append(f"week {week} has blank station id: {rule!r}")
            if not str(region).strip():
                errors.append(f"week {week} has blank region: {rule!r}")

    for week, required_rules in sorted(_REQUIRED_WEEKLY_RULES.items()):
        actual_rules = set(rules.get(week, []))
        missing_rules = sorted(required_rules - actual_rules)
        if missing_rules:
            errors.append(f"week {week} missing required rule(s): {missing_rules!r}")

    return errors


def check_paths() -> int:
    script_path = Path(__file__).resolve()
    week_count, entry_count, unique_station_count = count_weekly_region_rules()
    errors = validate_weekly_region_rules()

    print(f"SCRIPT_PATH: {script_path}")
    print(f"RULE_NAME: WEEKLY_REGION_RULES")
    print(f"TEMPLATE_NAME: WEEKLY_REGION_RULES_TEMPLATE")
    print(f"WEEK_COUNT: {week_count}")
    print(f"RULE_ENTRY_COUNT: {entry_count}")
    print(f"UNIQUE_STATION_COUNT: {unique_station_count}")
    print("NO_API_CALLS: true")
    print("NO_FILE_WRITES: true")

    if errors:
        print("RULE_CHECK: FAIL")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("RULE_CHECK: PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate static weekly weather region rules without API calls or writes."
    )
    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="print import/rule diagnostics and validate rule integrity",
    )
    parser.add_argument(
        "--check-rules",
        action="store_true",
        help="alias for --check-paths",
    )
    args = parser.parse_args(argv)

    if args.check_paths or args.check_rules:
        return check_paths()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
