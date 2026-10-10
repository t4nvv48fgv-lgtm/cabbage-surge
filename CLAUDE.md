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
- `data/ext/`: CABIS 밖에서 사용자가 제공한 외부 자료 원본(2026-10-09~). 읽기 전용, 출처·기준일·인용 가능 확인은 `data/ext/README.md`에 기록. 정돈본은 케이스 `outputs/`에.
- `data/processed/`, `outputs/*.csv`, `experiments/*/outputs/*.csv`는 스크립트 산출물이지만 **2026-10-09부터 git 추적**(기기 간 재생성 비용 때문). 코드·원자료를 바꾸면 재실행해 바뀐 산출물을 같은 커밋에 넣는다. 로그·`.venv/`만 gitignore.
- 가격 단위 원/10kg 상품(가락). KREI 세미나자료는 원/kg 단위이므로 비교 시 10배.

## 실행 순서

```bash
python src/build_features.py
python src/backtest.py M          # 월말 원점
python src/backtest.py W          # 주간 원점
python src/anomaly_model.py M
python src/anomaly_model.py W
python src/plot_2024.py
python tools/make_report.py       # 쉬운 말 결과 보고서 PDF(docs/배추급등예측_결과보고서_YYYY-MM-DD.pdf). 수치는 outputs/*.csv에서 직접 읽음
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

2024-08-31→9월(2026-10-09 기준): 가격만 ridge −37%, Chronos-2 제로샷 −18%, MSTL+ETS −13%, 기상 ridge −24%, 기상 HGB −26%, **계절편차 희소(사전선택, 대표) −10%**, 계절편차 희소(여름, 사후) −6%.
정부 방출 피처 gov30/gov60은 2026-10-08 메인에서 제외(stock.csv 2022~만 있어 학습구간 0 → 2022-08 선형 외삽 사고. 기상 ridge MAPE 17.2→15.0%). 5개년짜리 자료는 학습 피처로 넣지 않는다.
**계절편차 대표 사양은 2026-10-09 `an_sparse_all`(SPARSE·전체 학습·alpha 10)로 교체(사용자 결정)** — case03에서 2016~2023 원점만으로 고른 사양이라 R-1(사후 설계)을 넘고, 학습 풀·추출 시작점·alpha에 안정(2024-09 −9.5~−12.4%, 2025-08 +20~+22%).
성공 기준: 대표 사양 2024-09 −10%(경계), 전체 MAPE 15.6%(기준 15.1% **미달**) → 두 기준 동시 충족 모형 없음.
2026-10-09 자료 갱신(가격 2026-10-08까지)으로 **2026-08-31→9월 표본외 원점** 추가: 서늘한 여름·급등 없음(12,843원), 대표 모형 −6%(거짓 경보 없음), 가격만 −21%, Chronos-2 +10%. 지표 창(2021~2025)은 고정 유지. 사후 사양(여름) −6%·14.9%는 학습 행 100개에 뒤집히므로 근거로 쓰지 않는다.
단 기상 피처는 2016~2023 평시 정확도를 높이지 않았고(가격만 ridge가 기상 ridge보다 나음), 사후 선택 이득은 검증 MAPE 1.5~2.4%p. "기상이 평시 정확도를 개선한다"고 쓰지 않는다.
단변량 추세외삽 기준모형(`src/ts_baselines.py`)이 급등을 절반 이상 잡으므로 비교 기준선은 가격만 ridge가 아니라 Chronos-2·MSTL.
10월 하락 전환은 모든 모형이 과대. 다음 후보: KREI 단수·생산량 전망 + 탄력성 결합 레이어, 가을배추 전환 신호, 분위수 예측.
병목은 데이터 양이 아니라 2024·2025를 가르는 변수(산지 작황·정식 지연·재배면적)의 부재(case02). 피처 추가 방향은 중단.
CABIS에 정부 비축 재고(잔량) 자료는 없음(aT 일별 수매·방출 흐름 2021~만). 보강은 aT 2010~2020 방출 자료.

## 아이디어 케이스 규칙 (`experiments/`)

- 메인(`src/`·README 결과표·평가 설계)을 바꾸지 않는 시도는 `experiments/caseNN_이름/`에 둔다. 메인은 import·읽기만 하고, 산출물은 케이스 폴더 `outputs/`에만 쓴다.
- 각 케이스는 `CASE.md`(아이디어·데이터 사실·결과·판정·승격 조건·재실행)를 갖고, README "아이디어 케이스" 표에 한 줄로 등록한다.
- 메인 편입(승격)은 CASE.md의 승격 조건을 채운 뒤 사용자 승인으로만.
- case01_release_trigger(2026-10-07): 비축 방출 급감 × 가격 상승 → Chronos-2 분위수 0.5→0.9 연속 전환. 월말 2024-09 +0.4%·MAPE 17.3%, 주간은 실패. 보류.
- case03_split_selection(2026-10-08): 48사양을 2016~2023으로 고르고 2024~2025 검증. 사전 선택 계절편차 희소 2024-09 −11%, ridge/HGB −28~−37%. 완료, 메인 사양 변경은 사용자 판단.
- **case06_supply_gate(2026-10-10)**: 대표 모형 위에 KREI 월보 여름배추 생산 전망(원점까지 발행 최신호) 게이트 → 2024-09 −10% 유지, 2025-08 +21→−9%, 2026-09 −6%, MAPE 15.6→**13.8%**(<15.1%). **성공 기준 두 항 첫 동시 충족(첫 항 경계).** 비용: 2021·2023 보통 상승 과소. 사후 설계·6개 해. 승격 점검 1(2017~2020 잔량 게이트)·3(주간) 통과, 2(평시 모형)는 가격만 ridge(거짓 경보 억제) vs 기상 HGB(평시 정확도, 2025 +13%)로 갈려 **사용자 결정 대기**.
- case05_gov_stock_krei(2026-10-09): 외부 자료 2건(`data/ext/`). aT 로트 실적으로 추정한 **8월 말 비축 잔량이 2024(762t)·2025(7,490t)를 가르는 첫 변수**(급등 3개 해 모두 <1,400t, 예외 2017·2019). KREI 전망치 모음 파일은 2025를 반대로 가리켰으나(기준 연도 재정의 아티팩트) **월보 8월호 호별 값은 2024 −7.2% vs 2025 +8.8%로 가름**(10-10 정정, 2021~2026 6개 해 모두 급등/비급등 분리). 후속 case06: 잔량 + 8월호 생산 전망 게이트.
- case04_normal_definition(2026-10-09): 평년치 정의 5종 비교. 2025-09 +21% 과대는 폭염 피처(≈8%p)보다 급등 해 셋이 든 평년치(≈12%p)가 주원인. 극단값 제외는 악화, 창 확대(mean10)는 2025 +7%·MAPE 15.2%이나 2024 −26%. 두 해를 따로 고칠 정의 없음 → mean5 유지, 종결.
- case02_all_data_fit(2026-10-07): 가용 데이터 전부를 피처로(`case_features.py`가 메인 피처 프레임을 확장). 두 기준 동시 충족 없음, all LightGBM MAPE 13.9%/2024-09 −31%, all 계절편차 ridge −4%/MAPE 19.5%·2025-08 +35%. 종결.

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
