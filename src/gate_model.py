"""Supply-gated model (2026-10-12 메인 편입, experiments/case06 → case10) + 겨울 분기·잔차 β 층 (2026-10-12 2차 편입, case15·case16).

최종 예측(final) = 세 단계.

  1) 게이트(gate_hgb): 공급 게이트가 "조용함"이면 평시 모형, 아니면 대표 모형.
       대표 모형   an_sparse_all   (anomaly_model.py: 평년치 기준점 + 현재 가격 편차 지속 + 고랭지 기상 편차, 희소 ridge)
       평시 모형   wx_hgb_summer   (backtest.py: 기상 HGB 여름특화) — case10에서 선택(사용자 결정 2026-10-12)
       게이트 S1   KREI 농업관측 월보 여름배추(당해년산) 생산량 전망 전년비 — 원점까지 발행된 최신호 값.
                   여름 원점(6~9월)에서 S1 > 0 (공급이 전년보다 늘 것이라는 전망) 이면 calm → 평시 모형,
                   S1 ≤ 0 이거나 신호가 없으면(비여름·2021 이전) 대표 모형.
                   호별 값은 data/ext/krei_summer_issues.csv (2021~, 발행일은 호 월 5일로 보수적 가정; 8월호 실제 게시 8/1~8/5).
                   생산 전년비가 없는 호(4~5월호)는 면적 전년비로 대체.
  2) 겨울 분기(case15): 대상월(원점+15일의 달)이 1~4월이면 게이트 대신 **겨울 작형 학습 가격만 ridge**(winter_ridge).
       학습 행 = 대상월이 1~4월인 과거 행만(누수 차단 동일: 타깃 창 종료 ≤ 원점), 피처 = backtest.PRICE_FEATS, alpha 3.
       2016~2020 설계 창 9.5%·2021~2025 검증 창 10.4%로 두 창 모두 겨울 1~2위(현행 겨울 11.9%). 학습 행 < 120이면 게이트로 대체.
  3) 잔차 β 층(case16 사전 설계판): 게이트 예측 × (1 + β × S/100), |S| ≤ 20 절단.
       여름 경보 원점(게이트가 대표 모형을 쓴 여름 원점)  β −0.566, S = 여름배추 생산 전망 전년비(S ≤ 0 일 때만)
       대상월 11~12월(가을)                              β −0.216, S = 가을배추(당해년산) 생산 전망 전년비(없으면 면적), data/ext/krei_nonsummer_issues.csv
       β는 2016~2020 원점의 log(실제/게이트 예측) ~ β·S 회귀(절편 없음, 양수면 0)로 정한 값(겨울 −0.13은 겨울 분기와 겹쳐 제외, 봄 0).
       S의 가용 시점은 규약 B(m월호 내용은 (m−1)월 말일부터 가용 — 사용자 사실, CLAUDE.md). 게이트 상태 판정은 종전대로 호 월 5일(결과 동일).

  4) 9월 고온 층(case17, 2026-10-12 3차 편입, 사용자 "월말 9월 원점 한정으로"): **월말 9월 원점의 10월 예측에만**
       final = min(final_base × exp(γ·H), max(final_base, 최근 7일 가격)),  H = 가을배추 주산지 9곳의 9월 30°C 이상 일수(L_hot30_sum30).
       가을배추 생산 전망 전년비 > 0 이면 끈다. γ 는 상수가 아니라 그 해 이전 해들의 9월 말 원점 잔차로 매해 다시 추정(확장 창).
       백테스트에서 실제로 바뀌는 달은 2024-10 하나(−34→0%)이고, 그 γ(0.028)는 사실상 2023 한 해에서 배운 값이다. 주간 원점에는 적용하지 않는다.
       final_base = 3단계까지의 값(2차 편입의 최종).

  0) 대표 모형의 계절 구분(case27, 2026-10-13 4차 편입, 사용자 "고랭지 기상은 출하기에만 쓰는 걸로 메인에 넣어줘"):
       여름 원점(6~9월)은 종전 대표 an_sparse_all, **그 밖의 원점은 an_sparse_ship**(같은 사양, 고랭지 기상 4개를 여름 작형 출하기에만 켬).
       여름 원점에서는 고랭지가 출하기라 두 사양의 기상 재료가 사실상 같으므로 종전 값을 그대로 둔다(게이트·경보 조정·9월 고온 층·성공 기준 달 불변).
       열 head = 원점별로 실제 쓰인 대표 모형 값. 월말 2021~2025 MAPE 11.62 → 11.38, 2016~2020 18.54 → 17.37, 주간 15.10 → 15.09(30% 초과 23 → 26).

게이트 규칙·임계(0)는 case06에서 2024·2025를 본 뒤 정한 사후 설계이며 신호는 6개 해(급등 2회)뿐이다. 평시 모형 선택은 case10(2016~2020 설계
구간에서는 가격만 ridge가 더 나았고, 2021~ 검증 구간에서 HGB가 나음). 겨울 분기와 β는 각각 2016~2020에서 정했으나 둘을 함께 얹기로 한 결정은
2021~2025 결과를 본 뒤다(대상 달이 겹치지 않아 조합에서 새로 고른 값은 없음). 서술 상한은 README "성공 기준 점검" 참조.

Usage: python src/gate_model.py [M|W]   (outputs/backtest_predictions{tag}.csv, anomaly_predictions{tag}.csv, data/processed/daily_features.csv 가 먼저 있어야 함)
Outputs: outputs/gate_predictions{tag}.csv  (origin, actual, summer, krei_issue, S1, gate_calm, 대표·평시·gate_hgb, 참고 gate_ridge,
                                            tmonth, winter_ridge, adj_issue, S_adj, adj_pct, gate_wr(겨울 분기만), gate_beta(β 층만),
                                            final_base(3단계까지), heat_H, heat_F, heat_gamma_pct, heat_pct, final)
         outputs/gate_metrics{tag}.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
KREI_ISSUES = ROOT / "data" / "ext" / "krei_summer_issues.csv"
SEASON_ISSUES = ROOT / "data" / "ext" / "krei_nonsummer_issues.csv"
HEAD, CALM, CALM_REF = "an_sparse_all", "wx_hgb_summer", "price_only_ridge"
HEAD_OFF = "an_sparse_ship"      # 여름 원점 밖의 대표 모형: 고랭지 기상을 여름 작형 출하기에만 (case27, 2026-10-13)
FINAL = "final"
PUB_DAY = 5
WINTER_MONTHS, FALL_MONTHS = (1, 2, 3, 4), (11, 12)
BETA_SUMMER_ALERT, BETA_FALL = -0.566, -0.216   # case16 prereg_beta.py (2016~2020 잔차 회귀). 바꾸려면 그 케이스에서 다시 추정
S_CAP = 20.0
MIN_TRAIN_WINTER = 120
HEAT_ORIGIN_MONTH, HEAT_TARGET_MONTH = 9, 10   # 9월 고온 층: 월말 9월 원점 → 10월 대상에만 (case17)


def load_issues() -> pd.DataFrame:
    d = pd.read_csv(KREI_ISSUES, encoding="utf-8-sig")
    d["pub_date"] = pd.to_datetime(dict(year=d.year, month=d.issue_month, day=PUB_DAY))
    d["avail_b"] = pd.to_datetime(dict(year=d.year, month=d.issue_month, day=1)) - pd.Timedelta(days=1)   # 규약 B
    return d.sort_values("pub_date").reset_index(drop=True)


def load_fall_issues() -> pd.DataFrame:
    d = pd.read_csv(SEASON_ISSUES, encoding="utf-8-sig")
    d = d[(d.season == "가을") & (d.prod_yoy.notna() | d.area_yoy.notna())].copy()
    d["avail_b"] = pd.to_datetime(dict(year=d.year, month=d.issue_month, day=1)) - pd.Timedelta(days=1)   # 규약 B
    return d.sort_values(["year", "issue_month"]).reset_index(drop=True)


def latest_signal(issues: pd.DataFrame, origin: pd.Timestamp) -> tuple[str | float, float]:
    """원점 연도의 호 중 발행일 ≤ 원점인 최신호. (호 표기, S1). 없으면 (nan, nan)."""
    s = issues[(issues.year == origin.year) & (issues.pub_date <= origin)]
    if s.empty:
        return np.nan, np.nan
    r = s.iloc[-1]
    s1 = r.prod_yoy if pd.notna(r.prod_yoy) else r.area_yoy
    return f"{int(r.year)}-{int(r.issue_month):02d}", float(s1)


def adjust_signal(issues: pd.DataFrame, origin: pd.Timestamp, year: int, crop_col: str | None = None) -> tuple[str | float, float]:
    """규약 B로 원점에서 알 수 있는 최신호의 생산 전망 전년비(없으면 면적). 여름은 원점 연도 호, 가을은 대상 연도 작형 호."""
    s = issues[(issues[crop_col] == year) if crop_col else (issues.year == year)]
    s = s[s.avail_b <= origin]
    if s.empty:
        return np.nan, np.nan
    r = s.iloc[-1]
    v = r.prod_yoy if pd.notna(r.prod_yoy) else r.area_yoy
    return f"{int(r.year)}-{int(r.issue_month):02d}", float(v)


def winter_price_ridge(origins: pd.DatetimeIndex) -> pd.DataFrame:
    """겨울 작형(대상월 1~4월) 학습 행만으로 적합한 가격만 ridge 예측. case15 season_trained 의 s_price_ridge 와 같은 학습 풀·사양."""
    import anomaly_model as am
    import backtest as bt

    df = pd.read_csv(bt.PROC / "daily_features.csv", parse_dates=["date"]).set_index("date")
    df[bt.TARGET] = df["y_lp_fwd28"] - df[bt.BASE]
    df = am.add_normals(df)
    feats_all = bt.PRICE_FEATS + bt.SUPPLY_FEATS + bt.WX_FEATS
    pool = df[(df.is_trading == 1) & df[bt.TARGET].notna()].dropna(subset=feats_all + am.CORE + ["y_anom"]).iloc[::3]
    pool = pool[(pool.index + pd.Timedelta(days=15)).month.isin(WINTER_MONTHS)]
    rows = []
    for o in origins:
        if o not in df.index:
            continue
        te = df.loc[[o]]
        if te[bt.PRICE_FEATS].isna().any(axis=1).iloc[0]:
            continue
        tr = pool[pool.index + pd.Timedelta(days=28) <= o]      # 누수 차단: 학습 행의 타깃 창 종료 ≤ 원점
        if len(tr) < MIN_TRAIN_WINTER:
            continue
        m = make_pipeline(StandardScaler(), Ridge(alpha=3.0)).fit(tr[bt.PRICE_FEATS], tr[bt.TARGET])
        rows.append({"origin": o, "winter_ridge": float(np.exp(float(te[bt.BASE].iloc[0]) + m.predict(te[bt.PRICE_FEATS])[0]))})
    return pd.DataFrame(rows, columns=["origin", "winter_ridge"])


def run(freq: str = "M") -> pd.DataFrame:
    tag = "" if freq == "M" else f"_{freq}"
    b = pd.read_csv(OUT / f"backtest_predictions{tag}.csv", parse_dates=["origin"])
    a = pd.read_csv(OUT / f"anomaly_predictions{tag}.csv", parse_dates=["origin"])
    d = b[["origin", "actual", "summer", CALM_REF, CALM, "chronos2"]].merge(a[["origin", "normal_fwd", HEAD, HEAD_OFF]], on="origin")
    d["head"] = np.where(d.summer == 1, d[HEAD], d[HEAD_OFF])            # 0) 대표 모형: 여름 원점은 종전, 그 밖은 출하기 한정 기상
    issues = load_issues()
    sig = [latest_signal(issues, o) if s == 1 else (np.nan, np.nan) for o, s in zip(d.origin, d.summer)]
    d["krei_issue"] = [k for k, _ in sig]
    d["S1"] = [v for _, v in sig]
    d["gate_calm"] = (d.S1.notna() & (d.S1 > 0)).astype(int)
    d["gate_hgb"] = np.where(d.gate_calm == 1, d[CALM], d["head"])      # 1단계 게이트 (2026-10-12 1차 편입의 최종 모형)
    d["gate_ridge"] = np.where(d.gate_calm == 1, d[CALM_REF], d["head"])  # 참고: 평시 = 가격만 ridge (case06 G_K)

    # 2) 겨울 분기
    target = d.origin + pd.Timedelta(days=15)
    d["tmonth"] = target.dt.month
    winter = d.tmonth.isin(WINTER_MONTHS)
    d = d.merge(winter_price_ridge(pd.DatetimeIndex(d.origin[winter])), on="origin", how="left")
    use_wr = winter & d.winter_ridge.notna()

    # 3) 잔차 β 층
    fall = load_fall_issues()
    alert = (d.summer == 1) & (d.gate_calm == 0) & d.S1.notna()
    adj_issue, s_adj, factor = [], [], []
    for o, ty, tm, al in zip(d.origin, target.dt.year, d.tmonth, alert):
        k, s, beta = np.nan, np.nan, 0.0
        if al:
            k, s = adjust_signal(issues, o, o.year)
            beta = BETA_SUMMER_ALERT if (pd.notna(s) and s <= 0) else 0.0
        elif tm in FALL_MONTHS:
            k, s = adjust_signal(fall, o, ty, "crop_year")
            beta = BETA_FALL
        f = 1.0 if (pd.isna(s) or beta == 0.0) else 1.0 + beta * float(np.clip(s, -S_CAP, S_CAP)) / 100.0
        adj_issue.append(k if f != 1.0 else np.nan)
        s_adj.append(s if f != 1.0 else np.nan)
        factor.append(f)
    d["adj_issue"], d["S_adj"] = adj_issue, s_adj
    d["adj_pct"] = (np.array(factor) - 1.0) * 100
    d["gate_wr"] = np.where(use_wr, d.winter_ridge, d.gate_hgb)                      # 겨울 분기만
    d["gate_beta"] = d.gate_hgb * np.array(factor)                                  # β 층만
    d["final_base"] = np.where(use_wr, d.winter_ridge, d.gate_beta)                  # 2차 편입까지(게이트 + 겨울 분기 + 잔차 β)

    # 4) 9월 고온 층 (case17, 월말 9월 원점 한정)
    d = heat_hold(d, fall, monthly=(freq == "M"))
    return d


def heat_hold(d: pd.DataFrame, fall: pd.DataFrame, monthly: bool) -> pd.DataFrame:
    """9월 말 원점의 10월 예측에만: final = min(final_base × exp(γ·H), max(final_base, 최근 7일 가격)).

    H = 원점 직전 30일의 가을배추 주산지 30°C 이상 일수(daily_features.L_hot30_sum30; 9/30 원점에서는 9월 한 달).
    게이트: 가을배추 생산 전망 전년비 F > 0 이면 조정하지 않는다(F 없으면 연다).
    γ 는 고정 상수가 아니라 **그 해 이전 해들의 9월 말 원점 잔차만으로 매해 다시 추정**한다(확장 창):
      γ_y = max(0, Σ H·log(실제/final_base) / Σ H²),  게이트가 열린 과거 원점만. 과거 해가 없거나 음수면 0.
    주간 원점에는 적용하지 않는다(10월 원점에서 조정이 반대로 작용, case17).
    """
    import backtest as bt

    d["heat_H"], d["heat_F"], d["heat_gamma_pct"], d["heat_pct"] = np.nan, np.nan, np.nan, 0.0
    d[FINAL] = d.final_base
    if not monthly:
        return d
    rows = d.index[(d.origin.dt.month == HEAT_ORIGIN_MONTH) & (d.tmonth == HEAT_TARGET_MONTH)]
    if len(rows) == 0:
        return d
    df = pd.read_csv(bt.PROC / "daily_features.csv", parse_dates=["date"], usecols=["date", "L_hot30_sum30", "lp7"]).set_index("date")
    k = d.loc[rows, ["origin", "actual", "final_base"]].copy()
    k["H"] = df.L_hot30_sum30.reindex(k.origin).values
    k["p7"] = np.exp(df.lp7.reindex(k.origin).values)
    k["F"] = [adjust_signal(fall, o, o.year, "crop_year")[1] for o in k.origin]
    k["open"] = ~(k.F > 0) & k.H.notna()
    k["r"] = np.log(k.actual / k.final_base)
    k["year"] = k.origin.dt.year
    for i, r in k.iterrows():
        past = k[(k.year < r.year) & k.open & k.r.notna()]
        den = float((past.H ** 2).sum())
        gam = max(0.0, float((past.H * past.r).sum() / den)) if den > 0 else 0.0
        new = r.final_base
        if r.open and gam > 0:
            new = min(r.final_base * float(np.exp(gam * r.H)), max(r.final_base, r.p7))
        d.loc[i, ["heat_H", "heat_F", "heat_gamma_pct", "heat_pct"]] = [r.H, r.F, gam * 100, (new / r.final_base - 1.0) * 100]
        d.loc[i, FINAL] = new                                                       # 최종 모형 (2026-10-12 3차 편입)
    return d


def metrics(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    y = d.origin.dt.year
    masks = {"all_2021_2025": y.between(2021, 2025), "summer_origin_2021_2025": y.between(2021, 2025) & (d.summer == 1),
             "summer_origin_2021_2026": y.between(2021, 2026) & (d.summer == 1), "calm_origins_2021_2026": y.between(2021, 2026) & (d.gate_calm == 1),
             "all_2021_2026": y.between(2021, 2026), "all_2016_2020": y.between(2016, 2020),
             "winter_target_2021_2025": y.between(2021, 2025) & d.tmonth.isin(WINTER_MONTHS),
             "spring_target_2021_2025": y.between(2021, 2025) & d.tmonth.isin((5, 6)),
             "summer_target_2021_2025": y.between(2021, 2025) & d.tmonth.isin((7, 8, 9, 10)),
             "fall_target_2021_2025": y.between(2021, 2025) & d.tmonth.isin(FALL_MONTHS)}
    out = []
    for name, mk in masks.items():
        s = d[mk]
        for c in cols:
            e = s[c] - s.actual
            out.append({"subset": name, "model": c, "n": len(s), "MAE": e.abs().mean(), "MAPE%": (e.abs() / s.actual).mean() * 100, "bias": e.mean()})
    return pd.DataFrame(out)


if __name__ == "__main__":
    freq = sys.argv[1] if len(sys.argv) > 1 else "M"
    tag = "" if freq == "M" else f"_{freq}"
    d = run(freq)
    cols = [CALM_REF, "chronos2", CALM, HEAD, HEAD_OFF, "head", "gate_ridge", "gate_hgb", "gate_wr", "gate_beta", "final_base", FINAL]
    d.to_csv(OUT / f"gate_predictions{tag}.csv", index=False, float_format="%.1f", encoding="utf-8-sig")
    met = metrics(d, cols)
    met.to_csv(OUT / f"gate_metrics{tag}.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 320)
    print(f"[{freq}] 원점 {len(d)}개, calm 원점 {int(d.gate_calm.sum())}개, 겨울 분기 {int(d.winter_ridge.notna().sum())}개, β 조정 {int((d.adj_pct.abs() > 1e-9).sum())}개")
    print(met.pivot(index="model", columns="subset", values="MAPE%").loc[cols].round(2).T.to_string())
    if freq == "M":
        keys = pd.to_datetime([f"{y}-0{m}-{dd}" for y in range(2020, 2027) for m, dd in ((7, "31"), (8, "31"), (9, "30"))])
    else:
        keys = d.origin[(d.origin.dt.year >= 2020) & d.origin.dt.month.isin([8, 9])]
    sc = d[d.origin.isin(keys)].copy()
    for c in cols:
        sc[c + "%"] = ((sc[c] / sc.actual - 1) * 100).round(0)
    print(sc[["origin", "actual", "krei_issue", "S1", "gate_calm", "adj_issue", "adj_pct"] + [c + "%" for c in cols]].to_string(index=False))
