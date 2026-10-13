"""Case 30 — 월보의 '대상월 출하량 전망'을 생산 전망과 엮어 평시·경보를 가르기.

사용자 제안(2026-10-13): "해당 월보에 그럼에도 출하량은 증가할 것이라는 전망도 있잖아? 그걸 엮어서 사인으로 만들 수 있나?"
계기: 2026년 8월호는 여름배추 생산 전망이 −0.9%인데 8월 출하량은 전년 대비 증가로 봤다. 현행 게이트는 생산 전망만 보고
      0 이하이면 경보로 분류해 8월 가격을 +19% 과대 예측했다(평시 모형이면 −3%).

신호: Q = 그 호가 전망한 '그 달 출하량의 전년 대비 방향'(+1 증가, 0 비슷, −1 감소). krei_shipment_outlook_2021_2026.csv
      (월보 본문·표지·요약, 2021년은 자문회의 원고에서 판독. 발간물에 실린 문장이다.)
      S = 여름배추 생산 전망 전년비(메인과 같음). 둘 다 그 달 말에 이미 쓰여 있는 다음 달 호를 본다(규약 B).
주의: case14 에서 월보의 출하량 전망은 쓰지 않기로 했었다(월보와 모형이 서로 맞물림). 이번에는 사용자가 직접 제안해 시험한다.

규칙(여름 원점 6~9월에만 적용. 평시 → wx_hgb_summer, 경보 → 대표 × (1 − 0.566·S/100) [S ≤ 0 일 때만]):
  현행        S > 0 이면 평시
  둘 다 감소  S ≤ 0 이고 Q ≤ 0 일 때만 경보 (생산이 줄어도 출하가 늘 전망이면 평시)      ← 사용자가 말한 엮는 방식
  하나라도    S ≤ 0 이거나 Q < 0 이면 경보
  출하 우선   Q > 0 평시, Q < 0 경보, Q 가 0 이거나 없으면 S 로
그 뒤 단계(겨울 분기·가을 조정·9월 고온 층)는 메인 함수로 다시 계산한다.

Usage: python experiments/case30_shipment_sign/shipment_sign.py [M|W]
Outputs: outputs/ship_states{tag}.csv, ship_metrics{tag}.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parents[1]
OUT = CASE_DIR / "outputs"
sys.path.insert(0, str(ROOT / "src"))
import gate_model as gm  # noqa: E402

RULES = ["현행", "둘 다 감소", "하나라도", "출하 우선"]


def assemble(d: pd.DataFrame, calm: pd.Series, fall: pd.DataFrame, monthly: bool) -> pd.DataFrame:
    """주어진 평시 판정으로 메인의 1·3·4단계를 다시 계산(겨울 분기는 그대로)."""
    x = d.copy()
    summer = x.summer == 1
    gate = np.where(summer & calm, x[gm.CALM], x["head"])
    alert = summer & ~calm & x.S1.notna()
    f = np.where(alert & (x.S1 <= 0), 1.0 + gm.BETA_SUMMER_ALERT * x.S1.clip(-gm.S_CAP, gm.S_CAP) / 100.0, 1.0)
    base = np.where(summer, gate * f, x.gate_beta)                 # 여름 원점 밖은 메인 값(가을 조정 포함) 그대로
    x["final_base"] = np.where(x.winter_ridge.notna() & x.tmonth.isin(gm.WINTER_MONTHS), x.winter_ridge, base)
    return gm.heat_hold(x, fall, monthly)


def run(freq: str) -> None:
    tag = "" if freq == "M" else f"_{freq}"
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT / "outputs" / f"gate_predictions{tag}.csv", parse_dates=["origin"], encoding="utf-8-sig")
    q = pd.read_csv(CASE_DIR / "krei_shipment_outlook_2021_2026.csv")
    q["krei_issue"] = q.year.astype(str) + "-" + q.issue_month.astype(str).str.zfill(2)
    d = d.merge(q[["krei_issue", "q_sign"]], on="krei_issue", how="left")
    fall = gm.load_fall_issues()
    monthly = freq == "M"
    S, Q = d.S1, d.q_sign
    calm = {"현행": S > 0,
            "둘 다 감소": (S > 0) | (Q > 0),
            "하나라도": (S > 0) & ~(Q < 0),
            "출하 우선": np.where(Q > 0, True, np.where(Q < 0, False, S > 0)).astype(bool)}
    chk = assemble(d, pd.Series(calm["현행"], index=d.index), fall, monthly)
    assert np.allclose(chk[gm.FINAL], d[gm.FINAL], atol=0.11), "현행 재현 실패"
    res = {k: assemble(d, pd.Series(v, index=d.index), fall, monthly)[gm.FINAL].values for k, v in calm.items()}
    yr = d.origin.dt.year
    s = d[(d.summer == 1) & (yr >= 2021)].copy()
    out = s[["origin", "krei_issue", "S1", "q_sign", "actual"]].copy()
    for k in RULES:
        out[k + " 상태"] = np.where(pd.Series(calm[k], index=d.index)[s.index], "평시", "경보")
        out[k + "%"] = ((pd.Series(res[k], index=d.index)[s.index] / s.actual - 1) * 100).round(1)
    out.to_csv(OUT / f"ship_states{tag}.csv", index=False, encoding="utf-8-sig")
    rows = []
    for name, mk in (("여름 원점 2021~2025", (d.summer == 1) & yr.between(2021, 2025)), ("여름 원점 2021~2026", (d.summer == 1) & yr.between(2021, 2026)),
                     ("전체 2021~2025", yr.between(2021, 2025)), ("전체 2021~2026", yr.between(2021, 2026))):
        for k in RULES:
            e = (pd.Series(res[k], index=d.index)[mk] / d.actual[mk] - 1) * 100
            rows.append({"subset": name, "rule": k, "n": int(mk.sum()), "MAPE%": e.abs().mean(), "n_gt20": int((e.abs() > 20).sum()), "n_gt30": int((e.abs() > 30).sum())})
    met = pd.DataFrame(rows)
    met.to_csv(OUT / f"ship_metrics{tag}.csv", index=False, float_format="%.2f", encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    print(f"===== {freq} =====")
    print(met.pivot_table(index="subset", columns="rule", values="MAPE%")[RULES].round(2).to_string())
    print(met.pivot_table(index="subset", columns="rule", values="n_gt30")[RULES].to_string())
    ch = out[(out["둘 다 감소 상태"] != out["현행 상태"]) | (out["하나라도 상태"] != out["현행 상태"]) | (out["출하 우선 상태"] != out["현행 상태"])]
    print(f"\n[판정이 현행과 달라지는 여름 원점: {len(ch)}개 / {len(out)}개]")
    print(ch[["origin", "krei_issue", "S1", "q_sign", "현행 상태", "현행%", "둘 다 감소 상태", "둘 다 감소%", "하나라도 상태", "하나라도%", "출하 우선 상태", "출하 우선%"]].to_string(index=False))
    if monthly:
        print("\n[여름 월말 원점 전체]")
        print(out[["origin", "krei_issue", "S1", "q_sign", "현행 상태", "현행%", "둘 다 감소%", "하나라도%", "출하 우선%"]].to_string(index=False))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "M")
