"""005: 核の冬後の食料・人口ボトルネック → 脆弱化 δ と回復時間 r の下からの推定

再現: python3 sim.py > stdout.txt  (seed=20261014) -> results.json, fig_*.png

事前登録仮説（本回の実行前に本ファイルへ記載。第三者のタイムスタンプはない）:
 H11  Xia 2022 の煤量シナリオから推定した人口・食料の回復時間 r は、研究004の
     「脆弱化の符号反転点」（50〜200年）より短く、かつ Ord 減衰モデルの τ=200年より短い。
     したがって、核の冬規模のショックを研究004の回復つき脆弱化モデルに入れると、
     長期生存 S(1万年) はショックなしからほとんど下がらない（相対差 < 5%）。
 H12  ピーク時のカロリー不足から作る δ の下界（δ_cap = (1-y_min)/y_min）は、
     研究004で「100年合計を1/6付近にする」のに必要な δ≈2（λ0≈1）と同程度かそれ以上になる。
     ただし回復が速いと、大きな δ でも長期Sへの害は小さい（H11と整合）。
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20261014
OUT = Path(__file__).resolve().parent
t0 = time.time()
rng = np.random.default_rng(SEED)
res: dict = {"seed": SEED, "study": "005-nuclear-winter-bottleneck"}

# =============================================================================
# 0. 文献入力（実際に開いた Xia 2022 / ALLFED preprint の数値のみ）
# =============================================================================
# 【事実（Xia et al. 2022 Nature Food Table 1）】
# Partial Livestock・貿易停止・Year 2 末に最小カロリー(1911 kcal/人/日)を配った場合の
# 「食料なし＝飢餓死の危険」人口。2010年対象国人口合計 6.7e9。
XIA_TABLE1 = {
    # soot_Tg: (weapons, yield_kt, direct_fatalities, people_without_food_Y2)
    5:   (100, 15, 27_000_000, 255_000_000),
    16:  (250, 15, 52_000_000, 926_000_000),
    27:  (250, 50, 97_000_000, 1_426_000_000),
    37:  (250, 100, 127_000_000, 2_081_000_000),
    47:  (500, 100, 164_000_000, 2_512_000_000),
    150: (4400, 100, 360_000_000, 5_341_000_000),
}
P0 = 6_700_000_000  # 【事実】Xia Table 1 注: 2010年対象国人口
KCAL_SURVIVE = 1911  # 【事実】Xia Table 1 注: 生存最小カロリー
KCAL_BASE = 2855  # 【事実】Xia Methods: 2010年平均供給カロリー（摂取+廃棄）

# 【事実（Xia Table 2）】Year 2 グローバル平均カロリー可用性の変化率（貿易停止前提）
# Partial Livestock（本研究の基準ケース = Table 1 と対応）
XIA_T2_GLOBAL_PL = {5: 0.135, 16: -0.057, 27: -0.174, 37: -0.281, 47: -0.367, 150: -0.771}
XIA_T2_GLOBAL_L = {5: -0.082, 16: -0.238, 27: -0.334, 37: -0.421, 47: -0.492, 150: -0.813}

# 【事実（Xia 本文）】作物カロリー: 5 Tg で years 1–5 に −7%; 47 Tg で最大〜−50%;
# 150 Tg で years 3–4 に〜−90%。気候擾乱は約10年、ピークは最初の数年。15年シミュレーション。
# 【事実（ALLFED preprint, allfed.info PDF）】150 Tg・レジリエント食なしで 1600 kcal 超まで 6.5年(78月)。
# レジリエント食ありで約2700 kcal を維持可能。月48でレジリエント食がフルスケール、
# 従来作物は最低点を過ぎ回復開始。

res["literature_inputs"] = {
    "xia_table1_partial_livestock_no_trade": {
        str(k): {"direct_fatalities": v[2], "people_without_food_Y2": v[3]}
        for k, v in XIA_TABLE1.items()
    },
    "P0_2010": P0,
    "kcal_survive": KCAL_SURVIVE,
    "kcal_baseline_2010": KCAL_BASE,
    "xia_table2_global_partial_livestock_Y2_frac_change": XIA_T2_GLOBAL_PL,
    "xia_table2_global_livestock_Y2_frac_change": XIA_T2_GLOBAL_L,
    "xia_crop_notes": {
        "5Tg_years1to5": -0.07,
        "47Tg_approx_peak": -0.50,
        "150Tg_years3to4": -0.90,
        "climate_duration_years": "~10+",
        "simulation_horizon_years": 15,
    },
    "allfed_150Tg": {
        "years_to_above_1600kcal_no_resilient": 6.5,
        "kcal_with_resilient_approx": 2700,
        "resilient_full_scale_months": 48,
        "source": "https://allfed.info/images/pdfs/PREPRINT%201%20-%20integrated%20assessment%20paper%20ALLFED.pdf",
    },
    "sources_opened": [
        "https://www.nature.com/articles/s43016-022-00573-0",
        "https://opensky.ucar.edu/system/files/2024-08/articles_25657.pdf",
        "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs43016-022-00573-0/MediaObjects/43016_2022_573_MOESM1_ESM.pdf",
        "https://allfed.info/images/pdfs/PREPRINT%201%20-%20integrated%20assessment%20paper%20ALLFED.pdf",
    ],
}

# =============================================================================
# 1. 単純人口動態モデル
# =============================================================================
# 【独自仮説・モデル仮定】
# - Year 0: 戦争。Year 1: 備蓄でしのぐ（Xia仮定）。Year 2末: Table1の飢餓死を適用。
# - その後、グローバル単一プール（貿易再開相当の楽観ケース）と
#   貿易なし継続（悲観・Xia準拠）の2ケース。
# - 相対カロリー生産 y(t) は「ピーク欠損 → 指数回復」:
#     y(t) = 1 - D * exp(-(t - t_peak)/tau_y)  for t >= t_peak
#     t < t_peak では線形にピークへ下降（Year2の観測点を通る）。
# - D, t_peak, tau_y はシナリオごとに Xia / ALLFED から較正（下記）。
# - 人口: 年次で K(t)=P0*y(t) を目標とするロジスティック型
#     dP/dt ≈ g * P * (1 - P/K)  （K>0）。K < P なら減少。
# - 終端: E_extinct = P < 5000（XPT定義）; E_collapse = P < 0.1*P0 を「深刻な文明崩壊」代理。

def calibrate_yield_params(soot: float) -> dict:
    """シナリオごとの y(t) パラメータ。【独自仮説】の較正ルールを明示。"""
    y2 = 1.0 + XIA_T2_GLOBAL_PL[soot]  # Year 2 remaining fraction (global PL)
    # ピーク欠損: Xia作物ノートから。5TgはY2が余剰でも局所飢餓あり → ピークは作物−7%付近。
    if soot <= 5:
        y_peak = 0.93  # −7% years 1–5
        t_peak = 3.0
        tau_y = 4.0  # 軽い冷却は数年で緩和【推測】
    elif soot <= 16:
        y_peak = min(y2, 0.85)
        t_peak = 3.0
        tau_y = 5.0
    elif soot <= 27:
        y_peak = min(y2, 0.70)
        t_peak = 3.0
        tau_y = 6.0
    elif soot <= 37:
        y_peak = min(y2, 0.55)
        t_peak = 3.5
        tau_y = 7.0
    elif soot <= 47:
        y_peak = min(y2, 0.50)  # Xia: 〜−50%
        t_peak = 3.5
        tau_y = 8.0
    else:  # 150
        y_peak = 0.10  # Xia: years 3–4 〜−90% crops; ALLFED min ~11%
        t_peak = 3.5
        # ALLFED: 6.5年で1600kcal超。1600/2855≈0.56。ピーク0.10から0.56まで3年 → tau≈3/ln(0.9/0.44)≈3.8
        # ただし気候は10年超。食料回復をやや遅めに: tau_y=5（ピーク後）
        tau_y = 5.0
    return {"y2": y2, "y_peak": y_peak, "t_peak": t_peak, "tau_y": tau_y}


def yield_frac(t: float, p: dict) -> float:
    """t は戦争後の年（連続）。"""
    y2, yp, tp, tau = p["y2"], p["y_peak"], p["t_peak"], p["tau_y"]
    if t <= 0:
        return 1.0
    if t <= 2.0:
        # Year0→Year2: 1 → y2
        return 1.0 + (y2 - 1.0) * (t / 2.0)
    if t <= tp:
        # Year2 → peak
        if tp == 2.0:
            return yp
        return y2 + (yp - y2) * ((t - 2.0) / (tp - 2.0))
    # exponential recovery toward 1
    return 1.0 - (1.0 - yp) * math.exp(-(t - tp) / tau)


def simulate_population(soot: float, years: int = 200, g: float = 0.01,
                        apply_direct_blast: bool = False) -> dict:
    p = calibrate_yield_params(soot)
    without = XIA_TABLE1[soot][3]
    direct = XIA_TABLE1[soot][2]
    # Year 2 末の人口
    P = float(P0 - without)
    if apply_direct_blast:
        # 【推測】直接死は Year0–1。飢餓死と一部重複しうるが、Xiaは別掲なので減算は上限側。
        P = float(P0 - without - direct)
        P = max(P, 0.0)

    ts = np.arange(0, years + 1, dtype=float)
    Ps = np.zeros_like(ts)
    ys = np.zeros_like(ts)
    # t=0..2: 備蓄期。人口は直接死のみ（オプション）反映し、飢餓死は t=2 で一括。
    P_pre_famine = float(P0 - direct) if apply_direct_blast else float(P0)
    for i, t in enumerate(ts):
        ys[i] = yield_frac(float(t), p)
        if t < 2:
            Ps[i] = P_pre_famine
        elif t == 2:
            Ps[i] = float(P0 - without - (direct if apply_direct_blast else 0))
            P = Ps[i]
        else:
            K = P0 * ys[i]
            # ロジスティック・オイラー（年次）
            if K <= 1:
                P = max(P * 0.5, 0.0)  # 壊滅的
            else:
                P = P + g * P * (1.0 - P / K)
                P = max(P, 0.0)
            Ps[i] = P

    Pmin = float(np.min(Ps[2:]))  # after famine applied
    t_min = float(ts[2 + int(np.argmin(Ps[2:]))])
    # 回復時間定義
    # r_yield_95: いったん 0.95 未満に落ちたあと、再び 0.95 以上に戻る最初の時刻
    r_y95 = None
    dropped = False
    for t in np.linspace(0, years, years * 20 + 1):
        yv = yield_frac(float(t), p)
        if yv < 0.95:
            dropped = True
        elif dropped and yv >= 0.95:
            r_y95 = float(t)
            break
    if not dropped:
        r_y95 = 0.0  # 一度も 5% 以上落ちない
    # r_pop_90_of_P2: 飢餓直後人口の90%まで戻る（減少後）
    P2 = float(Ps[2])
    target = 0.90 * P2 if Pmin < 0.90 * P2 else P2  # もし底が浅いなら P2 自体
    r_pop90 = None
    if Pmin < 0.90 * P2:
        for i in range(int(t_min), years + 1):
            if Ps[i] >= 0.90 * P2:
                r_pop90 = float(ts[i])
                break
    else:
        r_pop90 = float(t_min)  # ほぼ即回復
    # r_pop_half_P0: P0 の半分まで戻る
    r_half = None
    for i in range(2, years + 1):
        if Ps[i] >= 0.5 * P0:
            r_half = float(ts[i])
            break
    # ALLFED整合チェック（150Tg）: y が 1600/2855 を超える時刻
    r_1600 = None
    thr = 1600 / KCAL_BASE
    for t in np.linspace(0, years, years * 4 + 1):
        if t >= 2 and yield_frac(float(t), p) >= thr:
            r_1600 = float(t)
            break

    extinct = bool(Pmin < 5000)
    collapse = bool(Pmin < 0.1 * P0)
    survivors_frac = P2 / P0
    dead_frac_famine = without / P0

    # δ 代理 【独自仮説】
    delta_pop = (P0 - Pmin) / max(Pmin, 1.0)  # 人口損失倍率
    delta_cap = (1.0 - p["y_peak"]) / max(p["y_peak"], 1e-6)  # キャパ損失倍率
    # 制度的脆弱化の粗い下界: 死者割合 / 生存者割合（研究004の乗法 δ と同型の解釈）
    delta_frag = dead_frac_famine / max(1.0 - dead_frac_famine, 1e-6)

    return {
        "soot_Tg": soot,
        "params": p,
        "P2": P2,
        "Pmin": Pmin,
        "t_min": t_min,
        "survivors_frac_Y2": survivors_frac,
        "dead_frac_famine": dead_frac_famine,
        "r_yield_95": r_y95,
        "r_pop_90_of_P2": r_pop90,
        "r_pop_half_P0": r_half,
        "r_above_1600kcal_proxy": r_1600,
        "E_extinct_5000": extinct,
        "E_collapse_10pct": collapse,
        "delta_pop": delta_pop,
        "delta_cap": delta_cap,
        "delta_frag": delta_frag,
        "ts": ts,
        "Ps": Ps,
        "ys": ys,
    }


scenarios = [5, 16, 27, 37, 47, 150]
runs = {s: simulate_population(s) for s in scenarios}
runs_blast = {s: simulate_population(s, apply_direct_blast=True) for s in scenarios}

# 感度: tau_y を ±50%、g∈{0.005,0.01,0.02}
sensitivity = []
for soot in [47, 150]:
    base = calibrate_yield_params(soot)
    for tau_mult in [0.5, 1.0, 1.5]:
        for g in [0.005, 0.01, 0.02]:
            # monkey via temporary override
            def sim_override(soot=soot, tau_mult=tau_mult, g=g, base=base):
                # inline mini-sim
                p = dict(base)
                p["tau_y"] = base["tau_y"] * tau_mult
                without = XIA_TABLE1[soot][3]
                years = 200
                ts = np.arange(0, years + 1, dtype=float)
                Ps = np.zeros_like(ts)
                P = float(P0 - without)
                for i, t in enumerate(ts):
                    y = yield_frac(float(t), p)
                    if t < 2:
                        Ps[i] = float(P0)
                    elif t == 2:
                        Ps[i] = P
                    else:
                        K = P0 * y
                        P = P + g * P * (1.0 - P / max(K, 1e-12))
                        P = max(P, 0.0)
                        Ps[i] = P
                Pmin = float(np.min(Ps[2:]))
                r_y95 = next((float(t) for t in np.linspace(0, years, years * 4 + 1)
                              if yield_frac(float(t), p) >= 0.95), None)
                return {"soot": soot, "tau_mult": tau_mult, "g": g,
                        "Pmin": Pmin, "r_yield_95": r_y95,
                        "delta_cap": (1 - p["y_peak"]) / p["y_peak"]}
            sensitivity.append(sim_override())

res["population_runs"] = {
    str(s): {k: (float(v) if isinstance(v, (float, np.floating)) else
                 bool(v) if isinstance(v, (bool, np.bool_)) else
                 {kk: float(vv) for kk, vv in v.items()} if k == "params" else v)
             for k, v in runs[s].items() if k not in ("ts", "Ps", "ys")}
    for s in scenarios
}
res["population_runs_with_blast"] = {
    str(s): {"P2": runs_blast[s]["P2"], "Pmin": runs_blast[s]["Pmin"],
             "E_extinct_5000": runs_blast[s]["E_extinct_5000"],
             "E_collapse_10pct": runs_blast[s]["E_collapse_10pct"]}
    for s in scenarios
}
res["sensitivity"] = sensitivity

# =============================================================================
# 2. 研究004接続: 回復つき脆弱化で r スキャン → r* と H11
# =============================================================================
# 研究004と同じ厳密積分 I_add（年次近似は使わない）
TAU = 200.0
H_NAT = 1 / 87_000
ORD_TOTAL = 1 / 6
N2 = 50_000


def draw_shocks_decay(n, lam0_per_century, T, rng):
    lam0 = lam0_per_century / 100.0
    Lam = lam0 * TAU * (-math.expm1(-T / TAU))
    k = rng.poisson(Lam, size=n)
    m = int(k.max()) if k.max() > 0 else 0
    s = np.full((n, max(m, 1)), np.inf)
    if m > 0:
        u = rng.uniform(size=(n, m))
        t = -TAU * np.log1p(-u * (-math.expm1(-T / TAU)))
        mask = np.arange(m)[None, :] < k[:, None]
        t = np.where(mask, t, np.inf)
        s = np.sort(t, axis=1)
    return s


def I_add(s, delta, r, T):
    base = TAU * (-math.expm1(-T / TAU))
    sj = np.where(np.isfinite(s) & (s < T), s, np.nan)
    if math.isinf(r):
        c = TAU * (np.exp(-sj / TAU) - math.exp(-T / TAU))
    else:
        k = 1 / TAU + 1 / r
        c = np.exp(sj / r) * (np.exp(-k * sj) - math.exp(-k * T)) / k
    return base + delta * np.nansum(c, axis=1)


def solve_b(I, target_p, extra=0.0):
    lo, hi = 0.0, 1.0
    while 1 - np.mean(np.exp(-hi * I - extra)) < target_p:
        hi *= 2
        if hi > 1e6:
            break
    for _ in range(60):
        mid = (lo + hi) / 2
        if 1 - np.mean(np.exp(-mid * I - extra)) < target_p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def longterm_add(lam, d, r, n=N2):
    rng_l = np.random.default_rng([SEED, int(lam * 1000), int(d * 100), int(min(r, 1e6) if not math.isinf(r) else 0), 5])
    s = draw_shocks_decay(n, lam, 10_000.0, rng_l)
    f = lambda T: I_add(s, d, r, T)
    B = solve_b(f(100.0), ORD_TOTAL, extra=100 * H_NAT)
    x = np.exp(-B * f(10_000.0) - H_NAT * 10_000)
    return {"B": float(B), "S_10000": float(x.mean()), "SE": float(x.std(ddof=1) / math.sqrt(n))}


I_target = -math.log(5 / 6) - H_NAT * 100
B_noshock = I_target / (TAU * (1 - math.exp(-100 / TAU)))
S_noshock_1e4 = math.exp(-B_noshock * TAU * (1 - math.exp(-10_000 / TAU)) - H_NAT * 10_000)

r_grid = [5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0, 500.0, math.inf]
delta_cap_150 = runs[150]["delta_cap"]
delta_frag_47 = runs[47]["delta_frag"]
delta_frag_150 = runs[150]["delta_frag"]

scan_cases = [
    ("lam0.3_d1", 0.3, 1.0),
    ("lam1_d1", 1.0, 1.0),
    ("lam0.3_dCap150", 0.3, float(min(delta_cap_150, 20.0))),  # 数値安定のため上限
    ("lam0.3_dFrag47", 0.3, float(delta_frag_47)),
    ("lam0.3_dFrag150", 0.3, float(min(delta_frag_150, 20.0))),
]

r_scan = {}
for name, lam0, dlt in scan_cases:
    rows = []
    for r_rec in r_grid:
        out = longterm_add(lam0, dlt, r_rec)
        rows.append({"r": None if math.isinf(r_rec) else r_rec, "r_inf": math.isinf(r_rec),
                     "S1e4": out["S_10000"], "SE": out["SE"],
                     "dS": out["S_10000"] - S_noshock_1e4, "delta_used": dlt})
    r_scan[name] = rows

r_star_est = {}
for name, rows in r_scan.items():
    crossed = None
    for a, b in zip(rows[:-1], rows[1:]):
        # r 小→大で dS が正→負に変わる点（回復が遅くなると害が出る）
        if a["dS"] > 0 and b["dS"] < 0:
            crossed = (a["r"], b["r"])
            break
    r_star_est[name] = crossed

res["connection_004"] = {
    "S_noshock_1e4": S_noshock_1e4,
    "N": N2,
    "delta_cap_150": delta_cap_150,
    "delta_frag_47": delta_frag_47,
    "delta_frag_150": delta_frag_150,
    "r_scan": r_scan,
    "r_star_bracket": r_star_est,
    "note": "研究004と同じ I_add 厳密積分。S(100)=5/6 に再較正。",
}

r_yields = [runs[s]["r_yield_95"] for s in scenarios]
r_pops = [runs[s]["r_pop_90_of_P2"] for s in scenarios]
max_r_yield = max(x for x in r_yields if x is not None)
max_r_pop = max((x for x in r_pops if x is not None), default=None)

h11_r_short = max_r_yield < 50 and (max_r_pop is None or max_r_pop < 50)
h11_S_effects = {}
for s in scenarios:
    r_use = max(float(runs[s]["r_yield_95"] if runs[s]["r_yield_95"] is not None else 20.0), 0.5)
    dlt = float(min(runs[s]["delta_frag"], 20.0))
    out = longterm_add(0.3, dlt, r_use)
    rel = (out["S_10000"] - S_noshock_1e4) / S_noshock_1e4
    h11_S_effects[str(s)] = {"r_used": r_use, "delta_frag": float(runs[s]["delta_frag"]),
                             "delta_used_capped": dlt, "S1e4": out["S_10000"],
                             "rel_vs_noshock": rel}

h11_supported = h11_r_short and all(abs(v["rel_vs_noshock"]) < 0.05 for v in h11_S_effects.values())
h12_caps = {str(s): runs[s]["delta_cap"] for s in scenarios}
h12_supported = h12_caps["150"] >= 2.0 and h12_caps["47"] >= 1.0

res["hypotheses"] = {
    "H11": {
        "statement": "核の冬シナリオの回復時間は符号反転点(50-200y)とτ=200より短く、長期Sをほとんど下げない",
        "max_r_yield_95": max_r_yield,
        "max_r_pop_90": max_r_pop,
        "S_effects_lam0.3_delta_frag": h11_S_effects,
        "verdict": "支持" if h11_supported else ("部分支持" if h11_r_short else "棄却"),
        "h11_r_short": h11_r_short,
        "h11_all_rel_lt_5pct": all(abs(v["rel_vs_noshock"]) < 0.05 for v in h11_S_effects.values()),
    },
    "H12": {
        "statement": "ピークカロリー不足由来のδ_capは、004で1/6再現に要るδ≈2と同程度以上",
        "delta_cap_by_soot": h12_caps,
        "verdict": "支持" if h12_supported else "棄却",
    },
}

# =============================================================================
# 3. 図
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for s in scenarios:
    t = runs[s]["ts"][:81]
    axes[0].plot(t, runs[s]["ys"][:81], label=f"{s} Tg")
axes[0].axhline(1600 / KCAL_BASE, color="k", ls="--", lw=0.8, label="1600/2855")
axes[0].set_xlabel("years after war")
axes[0].set_ylabel("relative calorie production y(t)")
axes[0].set_title("Calorie recovery trajectories (model)")
axes[0].legend(fontsize=7, ncol=2)
axes[0].set_ylim(0, 1.2)
axes[0].grid(True, alpha=0.3)

for s in scenarios:
    t = runs[s]["ts"][:81]
    axes[1].plot(t, runs[s]["Ps"][:81] / 1e9, label=f"{s} Tg")
axes[1].axhline(5000 / 1e9, color="r", ls=":", lw=0.8, label="extinction 5000")
axes[1].axhline(0.1 * P0 / 1e9, color="orange", ls="--", lw=0.8, label="10% P0 collapse")
axes[1].set_xlabel("years after war")
axes[1].set_ylabel("population (billions)")
axes[1].set_title("Population (global pool after Y2 famine)")
axes[1].legend(fontsize=7, ncol=2)
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig_population_calories.png", dpi=120)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
soots = np.array(scenarios, dtype=float)
axes[0].plot(soots, [runs[s]["r_yield_95"] for s in scenarios], "o-", label="r_yield_95")
axes[0].plot(soots, [runs[s]["r_pop_90_of_P2"] for s in scenarios], "s-", label="r_pop_90_of_P2")
axes[0].plot(soots, [runs[s]["r_above_1600kcal_proxy"] for s in scenarios], "^-", label="r_above_1600")
axes[0].axhline(50, color="g", ls="--", label="004 sign-flip lower (50y)")
axes[0].axhline(200, color="r", ls="--", label="τ=200y / upper band")
axes[0].set_xlabel("soot (Tg)")
axes[0].set_ylabel("recovery time (years)")
axes[0].set_title("Recovery time proxies vs soot")
axes[0].legend(fontsize=7)
axes[0].grid(True, alpha=0.3)

axes[1].plot(soots, [runs[s]["delta_cap"] for s in scenarios], "o-", label="δ_cap=(1-y_peak)/y_peak")
axes[1].plot(soots, [runs[s]["delta_frag"] for s in scenarios], "s-", label="δ_frag=dead/(1-dead)")
axes[1].axhline(2.0, color="k", ls="--", label="δ≈2 (004 1/6 fit)")
axes[1].set_xlabel("soot (Tg)")
axes[1].set_ylabel("δ proxy")
axes[1].set_title("Vulnerability δ proxies (hypothesis mapping)")
axes[1].legend(fontsize=7)
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig_r_and_delta.png", dpi=120)
plt.close()

# 004接続図
fig, ax = plt.subplots(figsize=(7, 4.5))
for name, rows in r_scan.items():
    rs = [row["r"] if row["r"] is not None else 1000 for row in rows]
    dS = [row["dS"] for row in rows]
    ax.plot(rs, dS, "o-", label=name)
ax.axhline(0, color="k", lw=0.8)
ax.axvline(50, color="g", ls="--", lw=0.8)
ax.axvline(200, color="r", ls="--", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("recovery time r (years)")
ax.set_ylabel("S(1e4) − S_noshock")
ax.set_title("Sign of vulnerability effect vs r (004-style)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig_r_star_scan.png", dpi=120)
plt.close()

res["runtime_seconds"] = time.time() - t0
with open(OUT / "results.json", "w") as f:
    json.dump(res, f, indent=1, allow_nan=False)

# stdout summary
print("=== 005 nuclear-winter-bottleneck ===")
print(f"seed={SEED} runtime={res['runtime_seconds']:.1f}s")
print(f"S_noshock_1e4={S_noshock_1e4:.4f}")
print("\nScenario summary (no blast subtraction):")
print(f"{'Tg':>5} {'dead%':>7} {'P2/P0':>7} {'y_peak':>7} {'r_y95':>7} {'r_pop90':>8} {'r_1600':>7} {'δ_cap':>7} {'δ_frag':>7} {'extinct':>7} {'coll10%':>7}")
for s in scenarios:
    r = runs[s]
    print(f"{s:5d} {100*r['dead_frac_famine']:6.1f}% {r['survivors_frac_Y2']:7.3f} {r['params']['y_peak']:7.3f} "
          f"{r['r_yield_95']:7.1f} {r['r_pop_90_of_P2']:8.1f} {r['r_above_1600kcal_proxy']:7.1f} "
          f"{r['delta_cap']:7.2f} {r['delta_frag']:7.2f} {str(r['E_extinct_5000']):>7} {str(r['E_collapse_10pct']):>7}")
print("\nH11:", res["hypotheses"]["H11"]["verdict"],
      "max_r_yield=", max_r_yield, "all_|rel|<5%=", res["hypotheses"]["H11"]["h11_all_rel_lt_5pct"])
print("H12:", res["hypotheses"]["H12"]["verdict"], "delta_caps=", h12_caps)
print("r* brackets:", r_star_est)
for s, v in h11_S_effects.items():
    print(f"  soot={s}: rel_S={v['rel_vs_noshock']:+.4f} (δ_frag={v['delta_frag']:.2f}, r={v['r_used']:.1f})")
