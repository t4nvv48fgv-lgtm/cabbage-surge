"""관측모형 점검계획서 양식(표 1 계량경제모형, 표 2 머신러닝 알고리즘, 표 4 보유 데이터)에 맞춘 배추 모형 점검표(엑셀).

양식 원본: 사용자 제공 「(모형점검)모형점검계획_정리예시.pdf」(관측모형팀). 표 1은 그 안의 표 3(실제 쌀 예시)처럼 비고를 용도·한계/개선·참고문헌·기타로 나눴다.
수치 출처
  계수·학습 설명력  기준 원점 2026-10-25 의 학습 줄(실시간 전망과 같은 줄)로 다시 적합. 직선형 부품은 ridge 원단위 계수,
                    p값은 같은 자료·같은 표준화 변수의 OLS(HAC 표준오차, 10시차) 참고값(ridge 자체는 p값이 없다)
  β·γ               case16 사전 설계 회귀(2016~2020 원점) 재계산, γ 는 gate_model.heat_hold 와 같은 무절편 회귀
  피처 중요도       평시 모형(여름 원점용, 2026-08-31 학습 줄) 세 시작점 각각의 순열 중요도(학습 줄, R² 감소) 평균
  표본외 예측력     outputs/gate_metrics{,_W}.csv (월말·주간 원점 2021~2025)
Usage: python tools/make_model_audit_xlsx.py
Output: docs/배추_모형점검표_<오늘>.xlsx
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.stattools import durbin_watson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import anomaly_model as am  # noqa: E402
import backtest as bt  # noqa: E402
import gate_model as gm  # noqa: E402

O_LIN, O_HGB, H = pd.Timestamp("2026-10-25"), pd.Timestamp("2026-08-31"), 28
FONT = "맑은 고딕"

# ------------------------------------------------------------------ 변수 설명(표 4 와 표 1·2 의 한글 이름)
DESC = {
    "lp7": ("최근 7일 가격", "최근 7일 로그가격 평균"), "lp28": ("최근 28일 가격", "최근 28일 로그가격 평균(평시·겨울 모형의 예측 기준점)"),
    "lp90": ("최근 90일 가격", "최근 90일 로그가격 평균"), "lp28_lag364": ("1년 전 가격", "1년 전 같은 시기 28일 로그가격 평균"),
    "lp28_lag728": ("2년 전 가격", "2년 전 같은 시기 28일 로그가격 평균"), "mom7_28": ("단기 모멘텀", "lp7 − lp28"),
    "mom28_90": ("중기 모멘텀", "lp28 − lp90"), "yoy28": ("가격 전년비", "lp28 − lp28_lag364"),
    "vol28": ("가격 변동성", "일별 로그가격 변화의 28일 표준편차"), "dlp7": ("7일 가격 변화", "7일 평균 로그가격의 7일 전 대비 변화"),
    "dlp14": ("14일 가격 변화", "7일 평균 로그가격의 14일 전 대비 변화"), "sin1": ("계절항 sin1", "연중 일자의 1차 사인항"),
    "cos1": ("계절항 cos1", "연중 일자의 1차 코사인항"), "sin2": ("계절항 sin2", "연중 일자의 2차 사인항"), "cos2": ("계절항 cos2", "연중 일자의 2차 코사인항"),
    "lq7": ("최근 7일 반입량", "가락시장 배추 반입량(톤) 로그의 7일 평균"), "lq28": ("최근 28일 반입량", "반입량 로그의 28일 평균"),
    "qyoy28": ("반입량 전년비", "lq28 − 1년 전 lq28"), "qmom7_28": ("반입량 모멘텀", "lq7 − lq28"),
    "cur_anom7": ("현재 가격 평년편차", "lp7 − 직전 5년 같은 날짜 lp7 평균(평년)"),
    "summer": ("여름 원점 여부", "원점이 6~9월이면 1"), "month": ("월", "원점의 달(1~12)"),
}
WX = {
    "H_hot30_sum14": "30°C 이상 일수 14일 합", "H_hot30_sum30": "30°C 이상 일수 30일 합", "H_hot30_sum60": "30°C 이상 일수 60일 합",
    "H_hot33_sum30": "33°C 이상 일수 30일 합", "H_heavy50_sum30": "일강수 50mm 이상 일수 30일 합", "H_heavy50_sum60": "일강수 50mm 이상 일수 60일 합",
    "H_tropical_sum30": "열대야(최저 25°C 이상) 일수 30일 합", "H_heat_sum30": "폭염일 30일 합", "H_heat_sum60": "폭염일 60일 합",
    "H_rain_sum14": "강수량 14일 합", "H_rain_sum30": "강수량 30일 합", "H_rain_sum60": "강수량 60일 합", "H_rain_day_sum30": "강수일수 30일 합",
    "H_h_tmax_anom_mean14": "최고기온 평년편차 14일 평균", "H_h_tmax_anom_mean30": "최고기온 평년편차 30일 평균",
    "H_h_tmax_anom_mean60": "최고기온 평년편차 60일 평균", "H_h_tmin_anom_mean30": "최저기온 평년편차 30일 평균",
    "H_h_tavg_anom_mean30": "평균기온 평년편차 30일 평균", "H_h_rain_anom_mean30": "강수량 평년편차 30일 평균", "H_h_rain_anom_mean60": "강수량 평년편차 60일 평균",
    "L_hot30_sum30": "30°C 이상 일수 30일 합", "L_heavy50_sum30": "일강수 50mm 이상 일수 30일 합", "L_rain_sum30": "강수량 30일 합",
    "L_l_tmax_anom_mean30": "최고기온 평년편차 30일 평균", "L_l_rain_anom_mean30": "강수량 평년편차 30일 평균", "L_l_tmin_anom_mean30": "최저기온 평년편차 30일 평균",
}


def kname(f: str) -> str:
    base = f[:-5] if f.endswith("_ship") else f
    if base in DESC:
        n = DESC[base][0]
    elif base.startswith("H_"):
        n = "고랭지 " + WX[base]
    else:
        n = "가을 산지 " + WX[base]
    return n + (" × 출하기 가중" if f.endswith("_ship") else "")


# ------------------------------------------------------------------ 통계
def stats() -> dict:
    df = pd.read_csv(bt.PROC / "daily_features.csv", parse_dates=["date"]).set_index("date")
    df[bt.TARGET] = df["y_lp_fwd28"] - df[bt.BASE]
    df = am.add_ship_weather(am.add_normals(df))
    cut = lambda p, o: p[p.index + pd.Timedelta(days=H) <= o]
    out = {}

    def linear(name, tr, feats, y, alpha):
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(tr[feats], tr[y])
        sc, rg = m.named_steps["standardscaler"], m.named_steps["ridge"]
        resid = tr[y] - m.predict(tr[feats])
        r2 = 1 - (resid ** 2).sum() / ((tr[y] - tr[y].mean()) ** 2).sum()
        X = sm.add_constant((tr[feats] - sc.mean_) / sc.scale_)
        ols = sm.OLS(tr[y].values, X.values).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
        n, k = len(tr), len(feats)
        out[name] = {"n": n, "start": tr.index.min(), "end": tr.index.max(), "r2": r2, "adj_r2": 1 - (1 - r2) * (n - 1) / (n - k - 1),
                     "dw": durbin_watson(resid), "f_p": ols.f_pvalue,
                     "coef": [(f, c, s, p) for f, c, s, p in zip(feats, rg.coef_ / sc.scale_, rg.coef_, ols.pvalues[1:])]}

    pool_a = cut(df[(df.is_trading == 1) & df.y_anom.notna()].dropna(subset=am.CORE).iloc[::3], O_LIN)
    linear("an_sparse_all", pool_a, am.SPARSE, "y_anom", 10.0)
    linear("an_sparse_ship", pool_a, am.SPARSE_SHIP, "y_anom", 10.0)
    wp = df[(df.is_trading == 1) & df[bt.TARGET].notna()].dropna(subset=bt.PRICE_FEATS + bt.SUPPLY_FEATS + bt.WX_FEATS + am.CORE + ["y_anom"]).iloc[::3]
    linear("winter_ridge", cut(wp[(wp.index + pd.Timedelta(days=15)).month.isin(gm.WINTER_MONTHS)], O_LIN), bt.PRICE_FEATS, bt.TARGET, 3.0)

    pp = pd.read_csv(ROOT / "experiments/case16_season_supply_adjust/outputs/prereg_predictions.csv", parse_dates=["origin"], encoding="utf-8-sig")
    alarm = (pp.season == "여름") & pp.S1_ext.notna() & (pp.S1_ext <= 0)
    for sea in ("여름", "가을"):
        s = pp[pp.origin.dt.year.between(2016, 2020) & (pp.season == sea) & pp.S_cap.notna() & ((pp.season != "여름") | alarm)]
        r = sm.OLS(np.log(s.actual / s.gate_hgb).values, (s.S_cap / 100).values).fit()
        out[f"beta_{sea}"] = {"n": len(s), "beta": r.params[0], "p": r.pvalues[0], "r2": r.rsquared}

    g = pd.read_csv(gm.OUT / "gate_predictions.csv", parse_dates=["origin"], encoding="utf-8-sig")
    k = g[(g.origin.dt.month == 9) & (g.tmonth == 10) & (g.origin < "2026-01-01")].copy()
    k["F"] = [gm.adjust_signal(gm.load_fall_issues(), o, o.year, "crop_year")[1] for o in k.origin]
    k = k[~(k.F > 0) & k.heat_H.notna()]
    r = sm.OLS(np.log(k.actual / k.final_base).values, k.heat_H.values).fit()
    out["gamma"] = {"n": len(k), "gamma": max(0.0, r.params[0]), "p": r.pvalues[0], "years": list(k.origin.dt.year)}

    bpool = df[(df.is_trading == 1) & df[bt.TARGET].notna()].dropna(subset=bt.PRICE_FEATS + bt.SUPPLY_FEATS + bt.WX_FEATS)
    feats = bt.PRICE_FEATS + bt.SUPPLY_FEATS + bt.WX_FEATS + ["summer", "month"]
    imps, ns, r2s = [], [], []
    for st in gm.THIN_STARTS:
        tr = bpool.iloc[st::3]
        tr = tr[(tr.index + pd.Timedelta(days=H) <= O_HGB) & (tr.summer == 1)]
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=1.0, random_state=0)
        m.fit(tr[feats], tr[bt.TARGET])
        imps.append(permutation_importance(m, tr[feats], tr[bt.TARGET], n_repeats=10, random_state=0, scoring="r2").importances_mean)
        ns.append(len(tr))
        r2s.append(m.score(tr[feats], tr[bt.TARGET]))
    out["hgb"] = {"n": ns, "r2": r2s, "start": bpool.index.min(), "end": O_HGB - pd.Timedelta(days=H),
                  "imp": pd.Series(np.mean(imps, axis=0), index=feats).sort_values(ascending=False)}

    mm = pd.read_csv(gm.OUT / "gate_metrics.csv").set_index(["subset", "model"])["MAPE%"]
    mw = pd.read_csv(gm.OUT / "gate_metrics_W.csv").set_index(["subset", "model"])["MAPE%"]
    out["mm"], out["mw"] = mm, mw
    return out


# ------------------------------------------------------------------ 서식
THIN = Side(style="thin", color="808080")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
HEAD_FILL = PatternFill("solid", fgColor="D9D9D9")
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)


def head(ws, row, labels, sub=None):
    """1~2줄 머리글. sub = {첫 열 번호: [하위 이름…]} 이면 그 열부터 하위 머리글을 둘째 줄에 쓰고 위 칸을 가로로 합친다."""
    c = 1
    for lab in labels:
        if sub and c in sub:
            n = len(sub[c])
            ws.cell(row, c, lab)
            ws.merge_cells(start_row=row, start_column=c, end_row=row, end_column=c + n - 1)
            for j, s in enumerate(sub[c]):
                ws.cell(row + 1, c + j, s)
            c += n
        else:
            ws.cell(row, c, lab)
            if sub:
                ws.merge_cells(start_row=row, start_column=c, end_row=row + 1, end_column=c)
            c += 1
    last = row + (1 if sub else 0)
    for r in range(row, last + 1):
        for cc in range(1, c):
            cell = ws.cell(r, cc)
            cell.font, cell.fill, cell.alignment, cell.border = Font(name=FONT, bold=True, size=10), HEAD_FILL, CENTER, BOX
    return last + 1


def put(ws, r, c, v, align=CENTER, bold=False, color=None):
    cell = ws.cell(r, c, v)
    cell.font = Font(name=FONT, size=10, bold=bold, color=color)
    cell.alignment = align
    cell.border = BOX
    return cell


def merge(ws, r0, r1, c, v, align=CENTER):
    put(ws, r0, c, v, align)
    for r in range(r0, r1 + 1):
        ws.cell(r, c).border = BOX
    if r1 > r0:
        ws.merge_cells(start_row=r0, start_column=c, end_row=r1, end_column=c)


def title(ws, text, ncol):
    ws.cell(1, 1, text).font = Font(name=FONT, bold=True, size=12)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncol)


def widths(ws, ws_w):
    for i, w in enumerate(ws_w, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def pfmt(p: float) -> str:
    return "<.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


# ------------------------------------------------------------------ 시트
def sheet_overview(wb, S):
    ws = wb.active
    ws.title = "개요"
    title(ws, "배추 관측모형 점검표 — 관측정보 결합 배추가격 머신러닝 모형(관측결합ML)", 2)
    mm, mw = S["mm"], S["mw"]
    rows = [
        ("품목", "배추"),
        ("모형명", "관측정보 결합 배추가격 머신러닝 모형(약칭 관측결합ML, 시험 운영)"),
        ("작성일", date.today().isoformat()),
        ("예측 대상", "가락시장 배추 상품(10kg) 도매가격의 앞으로 28일 평균(원점 다음 날부터). 월말 원점이면 다음 달 평균과 거의 같음"),
        ("예측 시점(원점)", "매달 말일, 매주 일요일. 원점마다 그날까지의 자료로 다시 학습(확장 창)"),
        ("구성", "① 머신러닝 부품: 기본 모형(Ridge, 평년 대비 편차) · 평시 모형(그래디언트 부스팅 나무) · 겨울 모형(Ridge, 1~4월 대상)\n"
                 "② 농업관측 전망정보 결합: 6~9월 말 여름배추 생산·출하량 전망으로 기본/평시 모형 전환, 생산 전망만큼 수준 보정(여름 공급 감소 시·11~12월)\n"
                 "③ 9월 말 원점의 10월 예측만 가을 산지 9월 고온 일수로 하락을 늦춤(2026년 10월 실적으로 존폐 판정 예정)"),
        ("표 구성", "표1_계량경제모형: 직선형 부품(Ridge 3개)과 보정 계수(β 2개, γ 1개)\n표2_머신러닝: 평시 모형\n"
                   "부록_조립규칙: 모형 전환·보정 규칙(회귀가 아닌 규칙)\n표4_보유데이터: 쓰는 변수와 쓰지 않는 보유 자료"),
        ("예측 성과(표본외)", f"월말 원점 2021~2025(60개) 평균 오차율 {mm[('all_2021_2025', 'final')]:.1f}% "
                         f"(가격 흐름만 쓰는 Ridge {mm[('all_2021_2025', 'price_only_ridge')]:.1f}%, Chronos-2 그대로 {mm[('all_2021_2025', 'chronos2')]:.1f}%)\n"
                         f"범위: 근거 사례가 1~2개인 보정(9월 고온·출하량 전망)을 빼면 11.9%, 모형 전환까지 빼면 13.8%\n"
                         f"주간 원점 2021~2025(261개) {mw[('all_2021_2025', 'final')]:.1f}%, 월말 2016~2020(관측 신호 미사용 구간) {mm[('all_2016_2020', 'final')]:.1f}%\n"
                         "2024년 9월 급등(24,714원): 8월 말 기준 −6%(가격만 Ridge −37%). 급등 없던 2025년 9월은 +20% 과대"),
        ("평가 방법", "롤링 원점 백테스트(확장 창). 학습 줄은 타깃 기간(28일)이 원점 이전에 끝난 줄만, 재료는 원점 당일까지의 정보만(누수 차단). "
                    "관측 월보 m월호는 (m−1)월 말 원점부터 사용(담당자 작성 시점 기준)"),
        ("모형 환경", "Python 3.13 (pandas 2.2, scikit-learn 1.7, statsmodels). 코드 src/build_features.py → backtest.py → anomaly_model.py → gate_model.py, 실시간 전망 forecast_live.py"),
        ("주요 한계", "모형 전환 기준(생산 전망 0)은 2024·2025년 결과를 본 뒤 정함(사후 설계). 관측 신호가 있는 해는 2021년 이후 6개 해(급등 2번)\n"
                    "작형 전환기(5·7·10·12월 대상)와 11월 대상의 오차가 큼(10월 하순 원점 2021~2025 10번 중 9번이 20% 넘게 빗나감)\n"
                    "보정 계수 β·γ는 표본이 8~14개라 통계적 유의성 미확보(표1)"),
        ("참고문헌 표기", "표의 caseNN은 모형 저장소(cabbage-surge) experiments 폴더의 검토 기록 번호. 각 폴더의 CASE.md에 설계·결과·판정이 있음"),
        ("p값 주의", "Ridge는 p값을 주지 않으므로, 같은 자료·같은 표준화 변수로 OLS를 적합해 HAC(Newey-West, 10시차) 표준오차의 p값을 참고로 적음. "
                   "학습 줄의 28일 타깃이 서로 겹쳐 잔차 자기상관이 큼(DW 0.3 안팎)"),
    ]
    for i, (k, v) in enumerate(rows, 3):
        put(ws, i, 1, k, CENTER, bold=True).fill = HEAD_FILL
        put(ws, i, 2, v, LEFT)
        ws.row_dimensions[i].height = max(20, 15 * (v.count("\n") + 1 + len(v) // 50))
    widths(ws, [18, 110])


def sheet_econ(wb, S):
    ws = wb.create_sheet("표1_계량경제모형")
    labels = ["품목", "모형 종류", "종속변수", "독립변수", "계수", "p값", "데이터 기간\n(예측 빈도)", "모형 설명력", "모형 환경", "비고"]
    title(ws, "〈표 1〉 계량경제모형 (관측결합ML의 직선형 부품과 보정 계수)", 13)
    r = head(ws, 2, labels, sub={10: ["용도", "한계/개선", "참고문헌", "기타"]})
    mm, mw = S["mm"], S["mw"]
    blocks = []

    def ridge_block(key, kind, dep, use, limit, ref, etc, freq):
        o = S[key]
        fit = (f"학습 R² {o['r2']:.2f}\nadj R² {o['adj_r2']:.2f}\nF검정 p {pfmt(o['f_p'])}\nDW {o['dw']:.2f}")
        per = f"{o['start']:%Y.%m.}–{o['end']:%Y.%m.}\n({freq})"
        blocks.append((kind, dep, [(kname(f), f"{c:+.4f}", pfmt(p)) for f, c, s, p in o["coef"]], per, fit, use, limit, ref, etc))

    ridge_block("an_sparse_all", "Ridge 회귀\n(α=10, 표준화)\n기본 모형\n[여름 원점]",
                "ln(앞으로 28일 평균가격)\n− ln(평년 같은 기간)", "6~9월 말 원점의 경보 상태 예측",
                f"급등을 끝까지 못 따라감(2022-09 −11%, 2024-09 −6%, 보정 포함).\n급등 해가 든 평년치 때문에 2025-09 +21% 과대.\n단독 표본외 오차 {mm[('all_2021_2025', 'an_sparse_all')]:.1f}%(월말)",
                "case03(2016~2023 원점으로 사양 선택)\ncase10(기여 분해)",
                "계수 원단위. 기상 순기여는 작고 현재 가격 편차(cur_anom7)가 대부분. 예측 = 평년 × exp(예측 편차)",
                "일별, 3거래일 간격 학습 / 월말·주간 원점, 28일 앞")
    ridge_block("an_sparse_ship", "Ridge 회귀\n(α=10, 표준화)\n기본 모형\n[그 밖 원점]",
                "ln(앞으로 28일 평균가격)\n− ln(평년 같은 기간)", "6~9월 말 밖 원점의 기본 예측(11월 전망 등)",
                f"단독 표본외 오차 {mm[('all_2021_2025', 'an_sparse_ship')]:.1f}%(월말).\n고랭지 기상은 여름배추 출하기(6/11~10/31)에만 켬",
                "case27(작형 달력)", "기상 4개 × 앞으로 28일 중 여름 작형 수확기 비율",
                "일별, 3거래일 간격 학습 / 월말·주간 원점, 28일 앞")
    ridge_block("winter_ridge", "Ridge 회귀\n(α=3, 표준화)\n겨울 모형",
                "ln(앞으로 28일 평균가격)\n− ln(최근 28일 평균가격)", "대상월 1~4월(겨울배추 시기) 예측",
                f"월말 겨울 대상 {mm[('winter_target_2021_2025', 'final')]:.1f}%로 이득이나 주간에서는 손해({mw[('winter_target_2021_2025', 'an_sparse_all')]:.1f}→{mw[('winter_target_2021_2025', 'final')]:.1f}%).\n가격 변수끼리 다중공선성 큼(계수 해석 주의)",
                "case15(작형별 모형)", "학습 줄은 대상월이 1~4월인 줄만",
                "일별, 3거래일 간격 학습 / 월말·주간 원점, 28일 앞")
    for sea, dep_lab, use, ref in (("여름", "여름배추 생산 전망 전년비/100\n(감소 전망일 때만)", "6~9월 말 경보 원점의 수준 보정", "case16 사전 설계"),
                                   ("가을", "가을배추 생산 전망 전년비/100\n(없으면 재배면적)", "대상월 11~12월 수준 보정", "case16 사전 설계")):
        b = S[f"beta_{sea}"]
        blocks.append(("무절편 선형회귀\n(잔차 회귀)\n수준 보정 β", "ln(실제 ÷ 1단계 예측)", [(dep_lab, f"{b['beta']:+.3f}", pfmt(b["p"]))],
                       f"2016.–2020.\n(월말 원점, n={b['n']})", f"R²(무절편) {b['r2']:.2f}", use,
                       "변수 통계적 유의성 미확보(표본 작음). 계수는 고정값(매번 재추정 안 함)", ref,
                       f"예측 × (1 + β × 전망 전년비/100), 전년비 ±20% 절단. 관측 m월호는 (m−1)월 말부터 사용"))
    gmm = S["gamma"]
    blocks.append(("무절편 선형회귀\n9월 고온 보정 γ", "ln(실제 ÷ 앞 단계 예측)\n(9월 말 원점 → 10월)", [("가을 산지 9월 30°C 이상 일수", f"{gmm['gamma']:+.4f}", pfmt(gmm["p"]))],
                   f"{min(gmm['years'])}.–{max(gmm['years'])}.\n(9월 말 원점, n={gmm['n']}, 매해 재추정)", "—",
                   "9월 말 원점의 10월 예측", "근거가 사실상 2023·2024 두 해. 2026년 10월 실적으로 존폐 판정 예정(사전 규칙)",
                   "case17", "가을배추 생산 전망 > 0이면 끔. 상한 = 최근 7일 가격"))

    r0 = r
    for kind, dep, vars_, per, fit, use, limit, ref, etc in blocks:
        r1 = r + len(vars_) - 1
        merge(ws, r, r1, 2, kind)
        merge(ws, r, r1, 3, dep)
        for i, (v, c, p) in enumerate(vars_):
            put(ws, r + i, 4, v, LEFT)
            put(ws, r + i, 5, c)
            put(ws, r + i, 6, p)
        for col, val in ((7, per), (8, fit), (9, "Python\n(scikit-learn,\nstatsmodels)"), (10, use), (11, limit), (12, ref), (13, etc)):
            merge(ws, r, r1, col, val, LEFT if col >= 10 or col == 8 else CENTER)
        for rr in range(r, r1 + 1):
            ws.row_dimensions[rr].height = 30 if r1 > r else 80                   # 한 줄짜리 블록(β·γ)은 긴 비고가 보이도록
        r = r1 + 1
    merge(ws, r0, r - 1, 1, "배추")
    note = r + 1
    ws.cell(note, 1, "주: 계수는 2026-10-25 원점의 학습 줄로 적합한 Ridge 원단위 계수(β·γ는 표기 기간의 회귀). p값은 같은 자료의 OLS(HAC 10시차) 참고값. "
                     "모형 설명력은 학습 자료 기준이며 표본외 성과는 개요 시트.").font = Font(name=FONT, size=9, color="555555")
    widths(ws, [7, 16, 22, 34, 10, 8, 18, 15, 13, 20, 34, 16, 34])


def sheet_ml(wb, S):
    ws = wb.create_sheet("표2_머신러닝")
    title(ws, "〈표 2〉 머신러닝 알고리즘 (관측결합ML의 평시 모형)", 10)
    labels = ["품목", "모형 종류", "타겟변수", "피처변수", "피처중요도\n순위", "재학습\n주기", "데이터 기간\n(예측 빈도)", "모형 예측력", "모형 환경", "비고"]
    r = head(ws, 2, labels)
    h, mm, mw = S["hgb"], S["mm"], S["mw"]
    top = h["imp"].head(10)
    rows = [(kname(f), str(i)) for i, f in enumerate(top.index, 1)] + [(f"그 외 {len(h['imp']) - 10}개(가격·반입량·고랭지/가을 산지 기상, 계 47개)", "11~")]
    r1 = r + len(rows) - 1
    merge(ws, r, r1, 1, "배추")
    merge(ws, r, r1, 2, "HistGradient\nBoosting\n(나무 300개,\n깊이 4, 학습률 0.04,\n잎 최소 25줄)\n평시 모형")
    merge(ws, r, r1, 3, "ln(앞으로 28일 평균가격)\n− ln(최근 28일 평균가격)")
    for i, (f, rank) in enumerate(rows):
        put(ws, r + i, 4, f, LEFT)
        put(ws, r + i, 5, rank)
    merge(ws, r, r1, 6, "원점마다\n(월말·매주)\n확장 창")
    merge(ws, r, r1, 7, f"{h['start']:%Y.%m.}–\n(원점까지)\n(일별, 3거래일 간격 학습 / 월말·주간 원점, 28일 앞)")
    merge(ws, r, r1, 8, f"표본외 MAPE(월말 2021~2025)\n단독 {mm[('all_2021_2025', 'calm_avg')]:.1f}%\n평시 판정 원점 {mm[('calm_origins_2021_2026', 'calm_avg')]:.1f}%(14개)\n"
                        f"주간 단독 {mw[('all_2021_2025', 'calm_avg')]:.1f}%\n학습 R² {min(h['r2']):.2f}~{max(h['r2']):.2f}", LEFT)
    merge(ws, r, r1, 9, "Python\n(scikit-learn\n1.7)")
    merge(ws, r, r1, 10,
          "용도: 6~9월 말 원점에서 관측 월보 여름배추 생산 전망이 '증가'이거나 그 달 출하량 전망이 '증가'면 이 모형으로 예측(평시).\n"
          "여름 원점은 6~9월 줄만 학습.\n"
          "과적합 진단: 학습 R² 0.98 대 표본외 오차 9~13% → 학습 자료에 강하게 맞음. 학습곡선은 미작성.\n"
          "학습 줄을 3일 간격으로 뽑는 시작점(0·1·2)에 따라 예측이 평균 4%·최대 10% 달라져, 세 벌을 따로 학습해 로그 평균함(2026-10-14 반영).\n"
          "경험 범위 밖 급등은 늘려 잡지 못함(2024-09 단독 −32%). 2025-09 +20% 과대.\n"
          "피처 중요도: 2026-08-31 학습 줄의 순열 중요도(R² 감소, 10회 반복) 세 벌 평균. 학습 줄 기준이라 표본외 기여와 다를 수 있음.\n"
          "참고문헌: case10(평시 모형 선택), case31(시작점 민감도)", LEFT)
    for rr in range(r, r1 + 1):
        ws.row_dimensions[rr].height = 30
    widths(ws, [7, 18, 22, 46, 10, 11, 20, 26, 12, 70])


def sheet_rules(wb, S):
    ws = wb.create_sheet("부록_조립규칙")
    title(ws, "〈부록〉 조립 규칙 — 농업관측 전망정보 결합(회귀가 아닌 규칙)", 7)
    r = head(ws, 2, ["단계", "규칙", "입력", "적용 범위", "정한 방법·근거", "성과 몫\n(월말 2021~2025, %p)", "한계"])
    rows = [
        ("1. 모형 전환\n(여름 게이트)", "생산 전망 전년비 > 0 또는 그 달 출하량 전망 '증가' → 평시 모형, 그 밖 → 기본 모형",
         "관측 월보 여름배추 생산 전망 전년비(호별), 그 달 배추 출하량 전망 방향(6~10월호)", "6~9월 말 원점, 2021년 이후",
         "case06·case10(2024·2025 결과를 본 뒤 기준 0 설정), case30(출하량 전망은 경보를 끄는 데만)", "2.40",
         "사후 설계. 기준값을 −4.3~+4.5 어디에 둬도 결과 같음(자료가 못 정함). 2016~2020 소급 시 악화. 몫의 1.68이 2021년"),
        ("2. 겨울 분기", "대상월 1~4월은 겨울 모형 예측으로 대체", "—", "대상월 1~4월", "case15(두 구간 모두 겨울 1~2위)", "1.05",
         "주간 원점에서는 손해. 달 경계(1~4월)는 평가 구간 최선"),
        ("3. 수준 보정 β", "예측 × (1 + β × 생산 전망 전년비/100), ±20% 절단", "여름(감소 전망 시) β −0.566, 가을 β −0.216",
         "여름 경보 원점, 대상월 11~12월", "case16 사전 설계(2016~2020 잔차 회귀)", "0.39 / 0.15",
         "β 통계적 유의성 미확보. 봉우리 뒤 달은 악화"),
        ("4. 9월 고온 보정", "min(앞 단계 × exp(γ·H), max(앞 단계, 최근 7일 가격)), 가을 생산 전망 > 0이면 끔", "가을 산지 9월 30°C 이상 일수 H, γ 매해 재추정",
         "월말 9월 원점 → 10월", "case17(사용자 결정으로 편입)", "0.57",
         "바뀐 달은 2024-10 하나. 2026년 10월 실적으로 존폐 판정(조정 전 7,872원과 후 8,766원 중 가까운 쪽, 경계 8,319원)"),
        ("5. 평시 모형 안정화", "평시 모형을 학습 줄 시작점 0·1·2로 따로 학습해 로그 평균", "—", "평시 모형 전부", "case31", "−0.01",
         "정확도 개선이 아니라 근거 없는 선택(시작점) 제거"),
        ("관측정보 가용 시점", "관측 m월호 내용은 (m−1)월 말 원점부터 사용", "—", "전환·보정 모두", "월보 담당자가 다음 달 호를 전달 말에 작성", "—",
         "외부 사용자 기준이면 게시일(호 월 초) 이후로 늦춰야 함"),
    ]
    for i, row in enumerate(rows):
        for c, v in enumerate(row, 1):
            put(ws, r + i, c, v, LEFT if c in (2, 3, 5, 7) else CENTER)
        ws.row_dimensions[r + i].height = 48
    ws.cell(r + len(rows) + 1, 1, "성과 몫: 해당 규칙을 뺐을 때 늘어나는 평균 오차율(case31 점검, 5차 편입 시점). 관측정보(1·3·4번의 관측 입력)를 모두 빼면 13.9%, 결합 시 11.4%.").font = Font(name=FONT, size=9, color="555555")
    widths(ws, [16, 40, 36, 20, 34, 14, 46])


def sheet_data(wb):
    ws = wb.create_sheet("표4_보유데이터")
    title(ws, "〈표 4〉 배추 보유 데이터 구성 및 설명", 4)
    r = head(ws, 2, ["품목", "종류", "변수", "변수 설명"])
    groups = [
        ("가격", [("price", "가락시장 배추 상품 일별 평균가격(원/10kg). CABIS 수집 스냅숏(clean_daily), 2010.1~"),
                  *[(f, f"{DESC[f][1]}") for f in ("lp7", "lp28", "lp90", "lp28_lag364", "lp28_lag728", "mom7_28", "mom28_90", "yoy28", "vol28", "dlp7", "dlp14")]]),
        ("반입량", [("quantity", "가락시장 배추 일별 반입량(톤), clean_daily, 2010.1~"), *[(f, DESC[f][1]) for f in ("lq7", "lq28", "qyoy28", "qmom7_28")]]),
        ("기상\n(고랭지)", [("관측소", "평창·태백·정선군·강릉 4곳 일별 평균(CABIS 자동기상, 2010.1~). 평년편차는 2010~2019 일자별 평년(15일 평활) 대비"),
                         *[(f, WX[f]) for f in WX if f.startswith("H_")],
                         ("*_ship", "고랭지 기상 4개 × 앞으로 28일 중 여름 작형 수확기(준고랭지 1기작 6/11~7/20, 고랭지 7/11~9/30, 준고랭지 2기작 9/21~10/31) 비율")]),
        ("기상\n(가을 산지)", [("관측소", "해남·진도군·고창·아산·괴산·제천·영양·문경·춘천 9곳 일별 평균"), *[(f, WX[f]) for f in WX if f.startswith("L_")]]),
        ("평년·편차", [("norm7, norm28", "직전 5년 같은 날짜 lp7·lp28 평균(평년)"), ("norm_fwd", "직전 5년 같은 기간 앞으로 28일 로그가격 평균(평년, 기본 모형 기준점)"),
                       ("cur_anom7", DESC["cur_anom7"][1])]),
        ("달력", [(f, DESC[f][1]) for f in ("sin1", "cos1", "sin2", "cos2", "summer", "month")]),
        ("타깃", [("y_lp_fwd28", "원점 다음 날부터 28일 로그가격 평균(거래일)"), ("y_anom", "y_lp_fwd28 − norm_fwd (기본 모형 타깃)"),
                  ("y_chg", "y_lp_fwd28 − lp28 (평시·겨울 모형 타깃)"), ("y_price_fwd28_true", "같은 28일 가격 평균(평가용 실제값)")]),
        ("농업관측\n전망정보", [("S1", "관측 월보 호별 여름배추(당해년산) 생산량 전망 전년비(%), 없으면 재배면적 전년비. data/ext/krei_summer_issues.csv, 2021.3호~"),
                            ("ship_q", "관측 월보 6~10월호의 그 달 배추 출하량 전망 방향(증가 +1, 비슷 0, 감소 −1). 표지·요약 문장 판독. data/ext/krei_shipment_outlook.csv, 2021~"),
                            ("가을 S", "관측 월보 호별 가을배추 생산량 전망 전년비(없으면 면적). data/ext/krei_nonsummer_issues.csv"),
                            ("가용 시점", "m월호 내용은 (m−1)월 말 원점부터 사용")]),
        ("보유하나\n미사용", [("gov30, gov60", "정부 비축 방출량 30·60일 합(stock.csv). 2022년부터만 있어 학습 구간이 0으로 채워져 제외"),
                           ("등급 가격비", "가락 상품/중품 가격비(garak_daily). 선행성 없어 제외"),
                           ("농넷 거래 원장", "aT 농넷 시장·산지·품종별 거래(2011~). 반입량으로 공급을 재는 시도 4건 모두 개선 없어 미사용"),
                           ("비축 실적", "aT 정부비축 로트별 수매·방출(2017~). 8월 말 잔량 추정은 참고만"),
                           ("관측 가격 전망·포전가격", "순환 참조·수집 자료라 입력으로 쓰지 않음")]),
    ]
    r0 = r
    for kind, items in groups:
        r1 = r + len(items) - 1
        merge(ws, r, r1, 2, kind)
        for i, (v, d) in enumerate(items):
            put(ws, r + i, 3, v, LEFT)
            put(ws, r + i, 4, d, LEFT)
            ws.row_dimensions[r + i].height = 15 * (1 + len(d) // 48) + 4
        r = r1 + 1
    merge(ws, r0, r - 1, 1, "배추")
    widths(ws, [7, 12, 26, 100])


if __name__ == "__main__":
    S = stats()
    wb = Workbook()
    sheet_overview(wb, S)
    sheet_econ(wb, S)
    sheet_ml(wb, S)
    sheet_rules(wb, S)
    sheet_data(wb)
    for ws in wb.worksheets:
        ws.sheet_view.showGridLines = False
        ws.freeze_panes = None
    out = ROOT / "docs" / f"배추_모형점검표_{date.today().isoformat()}.xlsx"
    wb.save(out)
    print("wrote", out)
