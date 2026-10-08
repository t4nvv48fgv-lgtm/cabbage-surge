# cabbage-surge — Claude Code 가이드

독립 프로젝트. **CABIS 저장소(`C:\Users\user\Desktop\CABIS`)의 CLAUDE.md·AGENTS.md·가드레일은 여기에 적용되지 않는다.**
CABIS 쪽 파일은 읽기 전용 원천일 뿐이며, 이 저장소에서 CABIS 폴더 안의 파일을 수정·실행하지 않는다.

## 목적

KREI 두 연구(Chronos-2 파운데이션 모형 세미나자료 2026-09-30, STL 분해-예측조합 연구보고 2026-06)가
공통으로 놓친 **2024년 9월 배추 급등**(가락 상품 월평균 24,714원/10kg)을 따라가는 예측모형을 만든다.
첫 번째 성공 기준: 2024-08-31 원점에서 9월 평균가격 과소폭을 −10% 이내로 줄이면서 2021~2025 전체 MAPE가 가격만 쓰는 모형(15.1%)보다 나쁘지 않을 것.

## 데이터

- `data/raw/`: CABIS에서 복사한 스냅숏(2026-10-06). 여기서 직접 수정하지 않는다.
- 갱신: `python src/sync_from_cabis.py` (CABIS → data/raw 복사만 수행. CABIS 쪽 수집 실행은 하지 않음).
- `data/processed/`, `outputs/*.csv`는 산출물(gitignore). 그림(`outputs/*.png`)은 추적.
- 가격 단위 원/10kg 상품(가락). KREI 세미나자료는 원/kg 단위이므로 비교 시 10배.

## 실행 순서

```bash
python src/build_features.py
python src/backtest.py M          # 월말 원점
python src/backtest.py W          # 주간 원점
python src/anomaly_model.py M
python src/anomaly_model.py W
python src/plot_2024.py
```

콘솔 한글 깨짐 시 `PYTHONIOENCODING=utf-8` 지정. Python 3.13. 설치 라이브러리는 `requirements.txt` 참조:
sklearn, statsmodels, torch(CPU) 외에 2026-10-06 추가로 lightgbm·xgboost·catboost, shap, mapie(정합예측), quantile-forest,
skforecast, pmdarima, statsforecast(AutoETS/AutoARIMA/MSTL), prophet, chronos-forecasting(Chronos-2, CPU 추론 확인) 사용 가능.
node 없음(dataviz 팔레트 검증기 실행 불가).

## 규칙

- 누수 금지: 학습 표본의 타깃 창 종료일 ≤ 원점. 피처는 원점 당일까지 정보만.
- 모형 비교는 항상 같은 원점 집합·같은 타깃으로. 2024-09 한 점에 과적합하지 않도록 2020-09, 2022-09, 2025-08도 함께 본다.
- 결과 수치는 README의 결과표를 갱신하고 커밋한다.
- 보고는 한국어, 간결.

## 현재 상태 요약 (README 참조)

2024-08-31→9월: 가격만 ridge −37%, Chronos-2 제로샷 −18%, MSTL+ETS −13%, 기상 ridge −23%, 기상 HGB −27%, 계절편차 ridge −16%.
단변량 추세외삽 기준모형(`src/ts_baselines.py`)이 급등을 절반 이상 잡으므로 비교 기준선은 가격만 ridge가 아니라 Chronos-2·MSTL.
10월 하락 전환은 모든 모형이 과대. 다음 후보: KREI 단수·생산량 전망 + 탄력성 결합 레이어, 가을배추 전환 신호, 분위수 예측.
2026-10-07 전체 데이터 피팅(`src/all_data_model.py`): 두 성공 기준을 동시에 만족하는 모형 없음. all LightGBM 전체 MAPE 13.9%(최선)이나 2024-09 −31%,
all 계절편차 ridge 2024-09 −4%이나 MAPE 19.5%·2025-08 +35%(다중공선성 상쇄). 고랭지 90일 누적 폭염은 2025 > 2024인데 2025는 급등 없음 →
병목은 데이터 양이 아니라 2024·2025를 가르는 변수(산지 작황·정식 지연·재배면적)의 부재. 피처 추가 방향은 중단.
CABIS에 정부 비축 재고(잔량) 자료는 없음(aT 일별 수매·방출 흐름 2021~만). 보강은 aT 2010~2020 방출 자료.

## 아이디어 케이스 규칙 (`experiments/`)

- 메인(`src/`·README 결과표·평가 설계)을 바꾸지 않는 시도는 `experiments/caseNN_이름/`에 둔다. 메인은 import·읽기만 하고, 산출물은 케이스 폴더 `outputs/`에만 쓴다.
- 각 케이스는 `CASE.md`(아이디어·데이터 사실·결과·판정·승격 조건·재실행)를 갖고, README "아이디어 케이스" 표에 한 줄로 등록한다.
- 메인 편입(승격)은 CASE.md의 승격 조건을 채운 뒤 사용자 승인으로만.
- case01_release_trigger(2026-10-07): 비축 방출 급감 × 가격 상승 → Chronos-2 분위수 0.5→0.9 연속 전환. 월말 2024-09 +0.4%·MAPE 17.3%, 주간은 실패. 보류.

## 읽기 순서

1. `CLAUDE.md` (이 파일)
2. `AGENTS.md` — 권위 가드레일(2026-10-07 이식). 이 파일과 충돌하면 **`AGENTS.md`를 따른다**.
3. `docs/HANDOFF_20261006.md` — 모형 쪽 정본 인수인계서(배경·데이터·결과·실패 기록·다음 작업)
4. `wiki/handoff-next-session.md` — 그 이후 세션 블록(최신이 위)
5. `README.md`

논문화 작업 시 추가로:

6. `docs/PAPER_SPEC.md` · `docs/OUTLINE.md` · `docs/STYLE_GUIDE.md` — 사양·목차·표기(**전부 미승인 초안**, 2026-10-07)
7. `docs/PAPER_WRITING_PLAYBOOK.md` — `Desktop\논문`에서 정리한 작성·수정 규칙·절차·프롬프트
8. 원고 수정 전 `docs/EDIT_CHECKLIST.md`, 집필 전·투고 전 `wiki/risk-audit.md`, 방어 논리는 `wiki/presentation-defense.md`
9. 검산 `py tools/check_style_rules.py manuscript/main.md`(프로젝트 사전 `tools/style_rules_local.py`), 빌드는 `AGENTS.md` 검증 절의 pandoc 명령
