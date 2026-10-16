"""研究024：手読み層を「内容」で測る — T3 読取装置の製作手引きの価値（open_problems 73）
事前登録：PREREGISTRATION.md。seed 固定、numpy＋matplotlib のみ。決定論的に再現する。
"""
import json, time, sys, platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
SEED = 20261024
rng = np.random.default_rng(SEED)
N = 200_000
H = 1000.0
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."

def logu(lo, hi, n=N):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))
def q(x, ps=(5, 50, 95)):
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}

SIZES = json.load(open(f"{HERE}/sources/text_sizes.json"))
G_LO = SIZES["wp_total"] / 1e6                 # 0.725 MB：Wikipedia 18記事（説明のみ）
G_RDH4 = SIZES["rdh4"]["utf8"] / 1e6            # 4.45 MB：Radiotron Designer's Handbook 4版 OCR 本文
G_HI = 2 * G_RDH4                               # 製作工程（管・ポンプ・ガラス）を足す余裕【推測】
P_H = 0.9305                                    # 023 results: 線幅7倍の読取確率（焦点込み c）
D_MAIN, D_STAGED = 1.9, 3.5                     # 021 の D_max（1e-6）、022 段階書式 P_g=0.7 の値

R = {"seed": SEED, "N": N, "python": sys.version.split()[0], "platform": platform.platform(),
     "inputs": {"G_LO_MB": G_LO, "G_RDH4_MB": G_RDH4, "G_HI_MB": G_HI, "P_H": P_H, "D_main": D_MAIN, "D_staged": D_STAGED, "H": H}}

# ---- 共通乱数 ----
Z = rng.standard_normal((3, N))
T1 = 20 * np.exp(0.7 * Z[0]); T2 = T1 + 30 * np.exp(0.7 * Z[1])     # 012 と同じ d01, d12
claim = logu(500, 2000); f_store = logu(0.3, 1.0)                     # Piql、金庫
life = claim * f_store / np.log(2) ** 0.5 * (-np.log(rng.uniform(size=N))) ** 0.5
g_req = logu(G_LO, G_HI)
P_max = rng.uniform(0.3, 0.9, N)
kappa = logu(0.5, 2.0)
t_build = 20 * np.exp(0.7 * rng.standard_normal(N))
U_t = rng.uniform(size=N); U_h = rng.uniform(size=N); U_sep = rng.uniform(size=N)
p_sep = rng.uniform(0.1, 0.5, N)
# 018 の電子経路最小人口（同じ式・同じ事前分布）
L_core = 50 * 11 / 12 * logu(1, 10); V = logu(550, 2400); t_tube = logu(0.5, 5); Uup = logu(3, 30)
N_min = Uup * (L_core + V * 3 * t_tube / 250) / (10.0 * logu(0.1, 0.4) * logu(0.005, 0.05))
N_T2 = logu(0.5e6, 5e6)
pop_ok = N_min < N_T2

def V_of(m23, qv, g_hand, tablet=False, D=None, greq=None):
    greq = g_req if greq is None else greq
    T3 = T2 + m23 * np.exp(0.8 * Z[2])
    gen = (T3 < life) & (T3 <= H)
    # 手引きの入手：手読み層（T2、P_h）または板（T1、分離なし）。着手は T2 以降
    cov = np.zeros(N); avail = np.zeros(N, bool)
    if g_hand > 0:
        hand_ok = U_h < P_H
        c_h = np.minimum(1.0, g_hand / greq)
        cov = np.where(hand_ok, c_h, cov); avail |= hand_ok
    if tablet:
        tab_ok = U_sep > p_sep
        cov = np.where(tab_ok, 1.0, cov); avail |= tab_ok
    Tt = T2 + t_build
    tgt = avail & pop_ok & (U_t < P_max * cov ** kappa) & (Tt < life) & (Tt <= H)
    return np.maximum(qv * tgt, gen.astype(float)), gen, tgt

def policies(m23, qv, D=D_MAIN, greq=None):
    out = {}
    v0, gen, _ = V_of(m23, qv, 0, greq=greq); out["P0"] = v0
    for f in (0.25, 0.5, 1.0):
        out[f"P1_f{f}"] = V_of(m23, qv, f * D, greq=greq)[0]
    out["P2_tablet"] = V_of(m23, qv, 0, tablet=True, greq=greq)[0]
    out["P3_tablet_plus_f0.25"] = V_of(m23, qv, 0.25 * D, tablet=True, greq=greq)[0]
    return out, gen

def summarize(pol):
    m = {k: float(np.mean(v)) for k, v in pol.items()}
    d = {k: m[k] - m["P0"] for k in m if k != "P0"}
    Rstar = {f"f{f}": d[f"P1_f{f}"] / (f * P_H) for f in (0.25, 0.5, 1.0)}
    return {"V": m, "dV": d, "R_star": Rstar, "dV_P3_minus_P2": m["P3_tablet_plus_f0.25"] - m["P2_tablet"]}

M_GRID = [150, 300, 600, 943, 1600]; Q_GRID = [0.1, 0.3, 1.0]
grid = {}
for M in M_GRID:
    for qv in Q_GRID:
        pol, gen = policies(M, qv)
        s = summarize(pol); s["P_general"] = float(np.mean(gen))
        grid[f"m{M}_q{qv}"] = s
R["grid_D1.9"] = grid
R["grid_D3.5_main"] = {f"q{qv}": summarize(policies(943, qv, D=D_STAGED)[0]) for qv in Q_GRID}
R["greq_range_sensitivity"] = {
    "lo_to_RDH4": summarize(policies(943, 0.3, greq=logu(G_LO, G_RDH4))[0]),
    "2MB_to_hi": summarize(policies(943, 0.3, greq=logu(2.0, G_HI))[0])}
R["bytes_proportional_value_v_h"] = D_MAIN * 1e6 / 120e9

# ---- 主設定の分解・感度（五分位） ----
pol, gen = policies(943, 0.3)
dv = pol["P1_f1.0"] - pol["P0"]
_, _, tgt = V_of(943, 0.3, D_MAIN)
cov_main = np.minimum(1, D_MAIN / g_req)
R["main_breakdown"] = {"P_general": float(np.mean(gen)), "P_targeted_ok": float(np.mean(tgt)),
    "P_targeted_ok_and_not_general": float(np.mean(tgt & ~gen)), "P_pop_ok": float(np.mean(pop_ok)),
    "P_greq_le_D": float(np.mean(g_req <= D_MAIN)), "coverage_main": q(cov_main),
    "T2": q(T2), "t_build": q(t_build), "life": q(life), "N_min": q(N_min)}
params = {"g_req": g_req, "P_max": P_max, "kappa": kappa, "t_build": t_build,
          "log_Nmin_over_NT2": np.log(N_min / N_T2), "life": life}
torn = {}
for k, x in params.items():
    lo, hi = np.percentile(x, [20, 80])
    a, b = float(np.mean(dv[x <= lo])), float(np.mean(dv[x >= hi]))
    torn[k] = {"dV_bottom20": a, "dV_top20": b, "abs_diff": abs(a - b)}
R["tornado_main"] = torn
top = max(torn, key=lambda k: torn[k]["abs_diff"])

S = R["grid_D1.9"]["m943_q0.3"]; S300 = R["grid_D1.9"]["m300_q0.3"]
H36 = {
 "H36a": {"value": S["dV"]["P1_f1.0"], "rule": ">=0.05", "supported": S["dV"]["P1_f1.0"] >= 0.05},
 "H36b": {"value": [S["V"]["P2_tablet"], S["V"]["P1_f1.0"]], "rule": "V(P2)>=V(P1 f=1)", "supported": S["V"]["P2_tablet"] >= S["V"]["P1_f1.0"]},
 "H36c": {"value": S["R_star"]["f0.5"], "rule": ">=0.05", "supported": S["R_star"]["f0.5"] >= 0.05},
 "H36d": {"value": S300["dV"]["P1_f1.0"], "rule": "<0.02", "supported": S300["dV"]["P1_f1.0"] < 0.02},
 "H36e": {"value": top, "rule": "largest quintile diff is g_req", "supported": top == "g_req"},
 "H36f": {"value": S["dV_P3_minus_P2"], "rule": "<0.01", "supported": S["dV_P3_minus_P2"] < 0.01},
}
R["hypotheses"] = H36

# ---- 図 ----
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
lab = {"P1_f0.25": "hand layer 25% guide", "P1_f0.5": "hand 50%", "P1_f1.0": "hand 100%", "P2_tablet": "ceramic tablet (same vault)", "P3_tablet_plus_f0.25": "tablet + hand 25%"}
for i, qv in enumerate(Q_GRID):
    for k, l in lab.items():
        ax[i].plot(M_GRID, [grid[f"m{M}_q{qv}"]["dV"][k] for M in M_GRID], "o-", label=l)
    ax[i].set_xscale("log"); ax[i].set_title(f"q = {qv} (value share a targeted reader delivers)")
    ax[i].set_xlabel("median general T2->T3 years, m23"); ax[i].axhline(0, color="k", lw=0.5)
ax[0].set_ylabel("gain in expected machine-layer value read, dV"); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_gain_vs_m23.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
fs = [0.25, 0.5, 1.0]
for qv in Q_GRID:
    ax[0].plot(fs, [grid[f"m943_q{qv}"]["R_star"][f"f{f}"] for f in fs], "o-", label=f"q={qv}")
ax[0].axhline(R["bytes_proportional_value_v_h"], color="gray", ls=":", label="v_h if value ∝ bytes")
ax[0].set_yscale("log"); ax[0].set_xlabel("share of hand layer used for the reader guide, f")
ax[0].set_ylabel("break-even v_h / V_machine (guide wins below)"); ax[0].legend(fontsize=8); ax[0].set_title("m23=943, D=1.9 MB")
ks = list(torn); y = np.arange(len(ks))
ax[1].barh(y - 0.2, [torn[k]["dV_bottom20"] for k in ks], 0.4, label="bottom 20%")
ax[1].barh(y + 0.2, [torn[k]["dV_top20"] for k in ks], 0.4, label="top 20%")
ax[1].set_yticks(y); ax[1].set_yticklabels(ks); ax[1].set_xlabel("dV (hand 100% guide), m23=943, q=0.3"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_breakeven_tornado.png", dpi=120); plt.close()

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({"main": S, "m300": S300, "breakdown": R["main_breakdown"], "tornado": torn, "H36": H36,
                  "staged": R["grid_D3.5_main"]["q0.3"], "greq_sens": R["greq_range_sensitivity"]}, ensure_ascii=False, indent=1))
print("runtime", round(R["runtime_seconds"], 1))
