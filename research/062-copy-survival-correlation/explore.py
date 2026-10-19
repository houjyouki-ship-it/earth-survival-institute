"""研究062 探索（初回の sim.py の結果を見た後に追加、事前登録の外）。
sim.py の主の模型は、戦乱にあった地域の中でも施設ごと・写しごとに独立に焼け・見つかるので、含意の c が小さく出た。
相関を強める2つの仮定を足して、結論がどれだけ動くかを見る：
 E1 fp_common：見つけて読む社会がいるかは写しごとでなく標本ごとに1回（fp を全写しに共通に掛ける）。
 E2 region_burn：焼かれるかを施設ごとでなく地域ごとに決める（地域ごと確率 b で全施設が焼かれる）。
 E3 both。乱数は sim.py と同じ（sim.py を「# ---------- 版 ----------」の前まで exec）。"""
import json, os, time
import numpy as np
t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "sim.py"); src = open(P, encoding="utf8").read(); cut = src.index("\n# ---------- 版 ----------\n")
G = {"__file__": P, "__name__": "sim062"}; exec(compile(src[:cut], P, "exec"), G)
N = G["N"]; deltas_t = G["deltas_t"]; ver = G["ver"]; get = G["get"]; rho1 = G["rho1"]; r_reach = G["r_reach"]
taup = G["taup_061"]; fp = G["fp"]; h_R = G["h_R"]; b_R = G["b_R"]; s_b = G["s_b"]; s_u = G["s_u"]; q_h = G["q_h"]; C_T = G["C_T"]; C_P = G["C_P"]; qf = G["qf"]
M1 = "A_Ord/U3"
def P_var(T, n_p, K=20, fp_common=False, region_burn=False):
    Ke = min(n_p, K); m = n_p / Ke; a = np.exp(-T / taup); f = 1.0 if fp_common else fp
    if region_burn: hit = b_R * (1 - s_b * a * f) ** m + (1 - b_R) * (1 - s_u * a * f) ** m
    else: hit = (1 - q_h * a * f) ** m
    S = 1 - ((1 - h_R) * (1 - a * f) ** m + h_R * hit) ** Ke
    return fp * S if fp_common else S
def rho_var(**kw): return lambda T: 1 - (1 - rho1) * (1 - r_reach * P_var(T, **kw))
def rat(o, p): return deltas_t(o)["A_Ord"]["U3"]["TC"][p] / get("V0", M1)["TC"][p]
m_heavy = get("main", M1)["TC"]["heavy"] / get("V0", M1)["TC"]["heavy"]
OUT = {"main_heavy": m_heavy, "versions": {}, "scan": []}
VARS = {"E1_fp_common": dict(fp_common=True), "E2_region_burn": dict(region_burn=True), "E3_both": dict(fp_common=True, region_burn=True)}
s_ce = G["G61"]["R"]["CE_STC_060_main"][M1]["total"]["S_TC_given_TC"]
for nm, kw in VARS.items():
    d = deltas_t(ver(rho=rho_var(n_p=100, K=20, **kw)))["A_Ord"]["U3"]
    vt = get("main", M1)["TAB"]["total"]; dv = d["TC"]["total"] - get("main", M1)["TC"]["total"]
    OUT["versions"][nm] = {p: d["TC"][p] / get("V0", M1)["TC"][p] for p in ("total", "heavy", "light")}
    OUT["versions"][nm]["CE_total_q"] = qf(dv / vt * C_T / C_P(100))
for nm in ("E1_fp_common", "E3_both"):
    for n_p, K in ((10, 10), (1000, 20), (100, 1), (100, 5)):
        OUT["scan"].append({"v": nm, "n_p": n_p, "K": K, "heavy": rat(ver(rho=rho_var(n_p=n_p, K=K, **VARS[nm])), "heavy")})
def implied(T, n_p, K, kw):
    Pn = P_var(np.full(N, float(T)), n_p, K, **kw); pbar = (1 - h_R + h_R * q_h) * np.exp(-T / taup) * fp
    neff = np.log(np.maximum(1 - Pn, 1e-300)) / np.log(1 - pbar); c = 1 - (neff - 1) / (n_p - 1)
    return [float(x) for x in np.quantile(c, [0.05, 0.5, 0.95])]
OUT["implied_c_T300_n100_K20"] = {nm: implied(300, 100, 20, kw) for nm, kw in VARS.items()}
for nm in ("E1_fp_common", "E3_both"):
    rows = {(r["n_p"], r["K"]): r["heavy"] for r in OUT["scan"] if r["v"] == nm}; full = OUT["versions"][nm]["heavy"]
    OUT[nm + "_sat10_of_1000"] = (rows[(10, 10)] - m_heavy) / (rows[(1000, 20)] - m_heavy)
    OUT[nm + "_K1_over_K20"] = (rows[(100, 1)] - m_heavy) / (full - m_heavy)
OUT["S_TC_CE"] = s_ce; OUT["runtime_s"] = time.time() - t0
json.dump(OUT, open(os.path.join(HERE, "explore_results.json"), "w"), indent=1, default=float)
print(json.dumps(OUT, indent=1, default=float))
