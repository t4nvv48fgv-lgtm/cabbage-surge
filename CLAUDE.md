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

콘솔 한글 깨짐 시 `PYTHONIOENCODING=utf-8` 지정. 설치된 ML 라이브러리: sklearn, statsmodels, pandas, scipy, matplotlib, torch(CPU). lightgbm·xgboost·catboost 없음.

## 규칙

- 누수 금지: 학습 표본의 타깃 창 종료일 ≤ 원점. 피처는 원점 당일까지 정보만.
- 모형 비교는 항상 같은 원점 집합·같은 타깃으로. 2024-09 한 점에 과적합하지 않도록 2020-09, 2022-09, 2025-08도 함께 본다.
- 결과 수치는 README의 결과표를 갱신하고 커밋한다.
- 보고는 한국어, 간결.

## 현재 상태 요약 (README 참조)

2024-08-31→9월: 가격만 −37%, 기상 ridge −23%, 기상 HGB −30%, 계절편차 ridge −16%.
10월 하락 전환은 모든 모형이 과대. 다음 후보: KREI 단수·생산량 전망 + 탄력성 결합 레이어, 가을배추 전환 신호, 분위수 예측.

## 읽기 순서

1. `CLAUDE.md` (이 파일)
2. `docs/HANDOFF_20261006.md` — 최신 인수인계서(배경·데이터·결과·실패 기록·다음 작업)
3. `README.md`
