"""研究033：シナリオに依存する読者の RAM（q）と、提言4つの Shapley 値（open_problems 99）
事前登録：PREREGISTRATION.md。seed 固定、numpy＋matplotlib のみ。
先頭の乱数生成は 024 と同じ seed・同じ順序にし、GEN0（T0 出発・金庫喪失0・発見1）で 024 の P2 を再現する。
"""
import json, time, sys, platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
# ---------- 024 と同一の乱数列（seed 20261024） ----------
rng = np.random.default_rng(20261024)
N = 200_000; H = 1000.0
def logu(lo, hi, n=N, g=None):
    g = rng if g is None else g
    return np.exp(g.uniform(np.log(lo), np.log(hi), n))
G_LO, G_RDH4 = 0.725, 4.45   # 024 sources/text_sizes.json の値（0.725 MB, 4.45 MB）
S24 = json.load(open(f"{HERE}/../024-content-value-reader-guide/sources/text_sizes.json"))
G_LO = S24["wp_total"] / 1e6; G_RDH4 = S24["rdh4"]["utf8"] / 1e6; G_HI = 2 * G_RDH4
P_H = 0.9305; D_MAIN = 1.9
Z = rng.standard_normal((3, N))
T1 = 20 * np.exp(0.7 * Z[0]); T2 = T1 + 30 * np.exp(0.7 * Z[1])
claim = logu(500, 2000); f_store = logu(0.3, 1.0)
life = claim * f_store / np.log(2) ** 0.5 * (-np.log(rng.uniform(size=N))) ** 0.5
g_req = logu(G_LO, G_HI); P_max = rng.uniform(0.3, 0.9, N); kappa = logu(0.5, 2.0)
t_build = 20 * np.exp(0.7 * rng.standard_normal(N))
U_t = rng.uniform(size=N); U_h = rng.uniform(size=N); U_sep = rng.uniform(size=N)
p_sep = rng.uniform(0.1, 0.5, N)
L_core = 50 * 11 / 12 * logu(1, 10); Vv = logu(550, 2400); t_tube = logu(0.5, 5); Uup = logu(3, 30)
N_min = Uup * (L_core + Vv * 3 * t_tube / 250) / (10.0 * logu(0.1, 0.4) * logu(0.005, 0.05))
N_T2 = logu(0.5e6, 5e6); pop_ok = N_min < N_T2
d01, d12 = T1, T2 - T1

# ---------- 本回の乱数（別の生成器、seed 20261033） ----------
SEED = 20261033
g = np.random.default_rng(SEED)
STATES = ["extinct", "T3", "T3_fragile", "T2", "T1", "T0"]
S32 = {}  # 032 の SCEN をそのまま使う（平均の状態割合・喪失・発見・M・識字喪失：すべて【推測】）
exec(open(f"{HERE}/../032-collapse-scenarios/sim.py").read().split("SCEN = ")[1].split("KAPPA_DIR")[0].join(["S32['SCEN'] = ", ""]))
SCEN = S32["SCEN"]
P_SALV = {"NUC_G": 0.3, "NUC_R": 0.7, "BIO": 0.8, "AI": 0.3, "CASC": 0.5, "GEN": 0.0}   # 【推測】
KAPPA_DIR = 20.0
U_state = g.uniform(size=N); U_lost = g.uniform(size=N); U_find = g.uniform(size=N); U_lit = g.uniform(size=N)
U_acc = g.uniform(size=N); W = g.uniform(20, 80, N); p_acc = g.uniform(0.1, 0.6, N)
lit_mult = logu(1, 30, g=g); Zm = g.uniform(size=N)
U_lost2 = g.uniform(size=N); U_find2 = g.uniform(size=N)
SITE2 = {"lost": 0.06, "find": 0.75}
# 読者の RAM
M_B = logu(4 * 1024, 144 * 1024, g=g)                     # 手引きで作る読者（025 の B 級）
f_psram = g.uniform(0.1, 0.4, N); U_ps = g.uniform(size=N)
M_salv = np.where(U_ps < f_psram, logu(4 * 2**20, 16 * 2**20, g=g), logu(2 * 1024, 520 * 1024, g=g))
f_mcu = g.uniform(0.0, 0.5, N); U_mcu = g.uniform(size=N)
U_salv = g.uniform(size=N); L_e = logu(10, 60, g=g); sp_salv = g.uniform(0.2, 0.6, N)
M_C = logu(144 * 1024, 2**20, g=g)                        # 感度：C 級の読者

ANCH_M = np.log([4096, 18432, 65536, 147456, 2**20 + 16384, 9 * 2**20])
ANCH_Q = np.array([0.0, 0.004583300352870422, 0.05644917259355787, 0.17015031414137002, 0.4316764473145267, 1.0])
def q_of(Mr):
    return np.interp(np.log(Mr), ANCH_M, ANCH_Q)
OPT = {"c4_rule": "noisy_or", "reader_B": "B", "f_mcu_on": True, "salv_on": True}
def q_read(Mr, C, mode="star"):
    q = q_of(Mr)
    if mode == "star":
        if not C: return q
        return 1 - (1 - 0.76) * (1 - q) if OPT["c4_rule"] == "noisy_or" else np.maximum(0.76, q)
    qd = np.maximum(0.101, q)                       # 依存重み【推測】
    return 1 - (1 - 0.704) * (1 - qd) if C else qd  # mode=="dep": C は依存枠の有無

def draw_scenario(name):
    s = SCEN[name]; mean = np.array(s["mean"], float); mean /= mean.sum()
    probs = g.dirichlet(KAPPA_DIR * mean, size=N)
    st = (U_state[:, None] > np.cumsum(probs, axis=1)).sum(axis=1).clip(0, 5)
    p_lost = g.uniform(*s["lost"], N); p_find = g.uniform(*s["find"], N)
    M = np.exp(np.log(s["M"][0]) + Zm * (np.log(s["M"][1]) - np.log(s["M"][0])))
    return {"name": name, "probs": probs, "st": st, "p_lost": p_lost, "p_find": p_find, "M": M,
            "lit": s["lit"], "p_salv": P_SALV[name]}

def V(sc, T=False, Hh=False, C=False, S=False, ideal=False, qconst=None, mode="star", detail=False):
    if ideal:
        st = np.full(N, 5); alive = np.ones(N, bool); intact = np.ones(N, bool); Mm = np.ones(N)
        T2s = T2.copy(); salv_ok = np.zeros(N, bool)
    else:
        st = sc["st"]; alive = st != 0; Mm = sc["M"]
        intact = (U_lost > sc["p_lost"]) & (U_find < sc["p_find"])
        if S: intact = intact | ((U_lost2 > SITE2["lost"]) & (U_find2 < SITE2["find"]))
        lit_lost = U_lit < sc["lit"]
        T0s = np.where(lit_lost, d01 * lit_mult, d01) + d12
        acc = (st == 2) & (U_acc < p_acc)
        T2s = np.select([st == 3, st == 4, st == 5, st == 2], [np.zeros(N), d12, T0s, W], default=np.zeros(N))
        salv_ok = (U_salv < sc["p_salv"]) & np.isin(st, [2, 3, 4, 5]) & ~acc & OPT["salv_on"]
        salv_ok &= (T2s + t_build * sp_salv) < L_e
    T3t = T2s + 943 * Mm * np.exp(0.8 * Z[2])
    gen = (T3t < life) & (T3t <= H)
    cov = np.zeros(N); avail = np.zeros(N, bool)
    if Hh:
        hand_ok = U_h < P_H; cov = np.where(hand_ok, np.minimum(1.0, 0.25 * D_MAIN / g_req), cov); avail |= hand_ok
    if T:
        tab_ok = U_sep > p_sep; cov = np.where(tab_ok, 1.0, cov); avail |= tab_ok
    Tt = T2s + np.where(salv_ok, t_build * sp_salv, t_build)
    tgt = avail & pop_ok & (U_t < P_max * cov ** kappa) & (Tt < life) & (Tt <= H)
    MB = M_C if OPT["reader_B"] == "C" else M_B
    Mr = np.where(salv_ok, M_salv, MB)
    qv = np.full(N, qconst) if qconst is not None else q_read(Mr, C, mode)
    v = np.maximum(qv * tgt, gen.astype(float))
    mcu_now = np.zeros(N, bool)
    if not ideal:
        mcu_now = acc & (U_mcu < f_mcu) & OPT["f_mcu_on"]
        v = np.where(acc & ~mcu_now, 1.0, v)
        if qconst is None:
            v = np.where(mcu_now, np.maximum(q_read(M_salv, C, mode), gen), v)
        else:
            v = np.where(mcu_now, 1.0, v)
        v = np.where(st == 1, 1.0, v)
        v = np.where(alive & intact, v, 0.0)
    if detail:
        return v, {"salv_reader": salv_ok & tgt, "mcu_now": mcu_now}
    return v

PL = ["T", "H", "C", "S"]
from itertools import combinations
from math import factorial
def shapley(sc, ideal=False):
    vals = {}
    for k in range(5):
        for sub in combinations(PL, k):
            vals[frozenset(sub)] = V(sc, T="T" in sub, Hh="H" in sub, C="C" in sub, S="S" in sub, ideal=ideal)
    n = len(PL); sh = {}
    for p in PL:
        acc_ = np.zeros(N)
        others = [x for x in PL if x != p]
        for k in range(n):
            for sub in combinations(others, k):
                wgt = factorial(k) * factorial(n - k - 1) / factorial(n)
                acc_ += wgt * (vals[frozenset(sub) | {p}] - vals[frozenset(sub)])
        sh[p] = acc_
    return sh, vals

R = {"seed_024_stream": 20261024, "seed": SEED, "N": N, "python": sys.version.split()[0], "numpy": np.__version__,
     "platform": platform.platform(), "P_SALV": P_SALV, "anchors_q": ANCH_Q.tolist()}

# ---------- 検査 ----------
r24 = json.load(open(f"{HERE}/../024-content-value-reader-guide/results.json"))["grid_D1.9"]["m943_q0.3"]
chk = float((V(None, T=True, ideal=True, qconst=0.3) - V(None, ideal=True, qconst=0.3)).mean())
R["check_024"] = {"dV_P2_q0.3": chk, "024": r24["dV"]["P2_tablet"], "reproduced_exactly": abs(chk - r24["dV"]["P2_tablet"]) < 1e-12,
                  "mean_q_readerB": float(q_of(M_B).mean()), "025_readerB_mean_q": 0.03743933823833786}

# ---------- 重み（032 と同じ） ----------
r_g = g.uniform(0.3, 0.7, N)
def weights(anchor):
    if anchor == "A_Ord":
        base = {"AI": 1/10, "BIO": 1/30, "NUC": 1/1000, "CASC": 2/1000, "GEN": 1/30 + 1/50}
    elif anchor == "A_XPT_S":
        bio = 2.13 * 0.01 / 0.38; rest = max(0.0, 9.05 - 2.13 - 4.0 - bio)
        base = {"AI": 2.13, "BIO": bio, "NUC": 4.0, "CASC": rest / 2, "GEN": rest / 2}
    else:
        bio = 12 * 1 / 3; rest = max(0.0, 20 - 12 - 8 - bio)
        base = {"AI": 12, "BIO": bio, "NUC": 8, "CASC": rest / 2, "GEN": rest / 2}
    tot = sum(base.values())
    w = {k: v / tot for k, v in base.items() if k != "NUC"}
    w["NUC_G"] = base["NUC"] / tot * float(np.mean(r_g)); w["NUC_R"] = base["NUC"] / tot * (1 - float(np.mean(r_g)))
    return w
WTS = {a: weights(a) for a in ("A_Ord", "A_XPT_S", "A_XPT_E")}

DRAWS = {name: draw_scenario(name) for name in SCEN}

def run_all(tag):
    out = {"GEN0": {}, "scen": {}, "weighted": {}}
    sh0, vals0 = shapley(None, ideal=True)
    out["GEN0"] = {"shapley": {p: float(sh0[p].mean()) for p in PL},
                   "v_T_alone_minus_none": float((vals0[frozenset("T")] - vals0[frozenset()]).mean()),
                   "v_C_given_T": float((vals0[frozenset("TC")] - vals0[frozenset("T")]).mean())}
    attrC = {}
    for name, sc in DRAWS.items():
        sh, vals = shapley(sc)
        _, det = V(sc, T=True, Hh=True, C=True, S=True, detail=True)
        mask = det["salv_reader"] | det["mcu_now"]
        o = {"shapley": {p: float(sh[p].mean()) for p in PL}}
        o["rank"] = sorted(PL, key=lambda p: -o["shapley"][p])
        o["total"] = float((vals[frozenset(PL)] - vals[frozenset()]).mean())
        o["ratio_C_over_T"] = o["shapley"]["C"] / o["shapley"]["T"] if o["shapley"]["T"] > 0 else None
        o["C_share_from_mcu_salv"] = float(sh["C"][mask].sum() / sh["C"].sum()) if sh["C"].sum() > 0 else None
        o["frac_mcu_salv_readers"] = float(mask.mean())
        o["marg_T_alone"] = float((vals[frozenset("T")] - vals[frozenset()]).mean())
        o["marg_C_given_T"] = float((vals[frozenset("TC")] - vals[frozenset("T")]).mean())
        o["marg_C_alone"] = float((vals[frozenset("C")] - vals[frozenset()]).mean())
        o["marg_S_given_TC"] = float((vals[frozenset("TCS")] - vals[frozenset("TC")]).mean())
        o["marg_dep_given_T"] = float((V(sc, T=True, C=True, mode="dep") - V(sc, T=True, mode="dep")).mean())
        # 探索（事前登録外、初回実行後に追加）：T3 がそのまま残る結果（st==1）を除いた「再建・衰退する社会だけ」の Shapley と、T＋C の束の価値
        keep_ = sc["st"] != 1
        o["explore_shapley_excl_T3"] = {p: float((sh[p] * keep_).mean()) for p in PL}
        o["explore_bundle_TC"] = o["marg_T_alone"] + o["marg_C_given_T"]
        o["state_shares"] = {STATES[i]: float(np.mean(sc["st"] == i)) for i in range(6)}
        attrC[name] = (float(sh["C"][mask].sum() / N), float(sh["C"].mean()))
        out["scen"][name] = o
    for a, w in WTS.items():
        ws = {p: sum(w[s] * out["scen"][s]["shapley"][p] for s in w) for p in PL}
        share_mcu = sum(w[s] * attrC[s][0] for s in w) / max(1e-300, sum(w[s] * attrC[s][1] for s in w))
        out["weighted"][a] = {"shapley": ws, "rank": sorted(PL, key=lambda p: -ws[p]),
                              "T_ratio_vs_GEN0": ws["T"] / out["GEN0"]["shapley"]["T"],
                              "C_share_from_mcu_salv": share_mcu,
                              "contrib_share_by_scen": {p: {s: w[s] * out["scen"][s]["shapley"][p] / ws[p] for s in w} for p in PL},
                              "marg_C_given_T": sum(w[s] * out["scen"][s]["marg_C_given_T"] for s in w),
                              "marg_T_alone": sum(w[s] * out["scen"][s]["marg_T_alone"] for s in w),
                              "explore_shapley_excl_T3": {p: sum(w[s] * out["scen"][s]["explore_shapley_excl_T3"][p] for s in w) for p in PL},
                              "explore_bundle_TC": sum(w[s] * out["scen"][s]["explore_bundle_TC"] for s in w),
                              "marg_dep_given_T": sum(w[s] * out["scen"][s]["marg_dep_given_T"] for s in w)}
    return out

MAIN = run_all("main")
R["main"] = MAIN
# ---------- 感度 ----------
SENS = {}
for tag, opt in {"c4_max_rule": {"c4_rule": "max"}, "readerC_144K_1M": {"reader_B": "C"},
                 "no_mcu_in_fragile": {"f_mcu_on": False}, "no_salvage": {"salv_on": False},
                 "no_mcu_no_salv": {"f_mcu_on": False, "salv_on": False}}.items():
    keep = dict(OPT); OPT.update(opt); o = run_all(tag); OPT.clear(); OPT.update(keep)
    SENS[tag] = {"GEN0_shapley": o["GEN0"]["shapley"],
                 "weighted": {a: {"shapley": o["weighted"][a]["shapley"], "rank": o["weighted"][a]["rank"],
                                  "T_ratio_vs_GEN0": o["weighted"][a]["T_ratio_vs_GEN0"]} for a in WTS},
                 "ranks": {s: o["scen"][s]["rank"] for s in SCEN}, "ratio_C_over_T": {s: o["scen"][s]["ratio_C_over_T"] for s in SCEN}}
R["sensitivity"] = SENS

# ---------- 仮説 ----------
G0 = MAIN["GEN0"]["shapley"]; SC = MAIN["scen"]; WW = MAIN["weighted"]
elig = [s for s in SCEN if s != "GEN" and SC[s]["total"] > 0.001]
ranks = {s: tuple(SC[s]["rank"]) for s in elig}
rC = lambda s: SC[s]["ratio_C_over_T"]
R["hypotheses"] = {
 "H45a": {"value": G0, "rule": "GEN0 Shapley C > T", "supported": G0["C"] > G0["T"]},
 "H45b": {"value": ranks, "rule": ">=2 distinct rankings among eligible non-GEN", "supported": len(set(ranks.values())) >= 2},
 "H45c": {"value": {"BIO": rC("BIO"), "GEN": rC("GEN")}, "rule": "BIO ratio >= 1.5 x GEN ratio", "supported": rC("BIO") >= 1.5 * rC("GEN")},
 "H45d": {"value": {a: WW[a]["rank"] for a in WW}, "rule": "S top in all weightings", "supported": all(WW[a]["rank"][0] == "S" for a in WW)},
 "H45e": {"value": {a: WW[a]["T_ratio_vs_GEN0"] for a in WW}, "rule": "all in [0.04,0.38]", "supported": all(0.04 <= WW[a]["T_ratio_vs_GEN0"] <= 0.38 for a in WW)},
 "H45f": {"value": WW["A_XPT_S"]["C_share_from_mcu_salv"], "rule": ">= 0.25", "supported": WW["A_XPT_S"]["C_share_from_mcu_salv"] >= 0.25},
}

# ---------- 図 ----------
names = list(SCEN)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
x = np.arange(len(names) + 1); wdt = 0.2
for i, p in enumerate(PL):
    ax[0].bar(x + (i - 1.5) * wdt, [SC[n]["shapley"][p] for n in names] + [G0[p]], wdt, label={"T": "T tablet guide", "H": "H hand-layer 25%", "C": "C c4 small-window", "S": "S 2nd site (south)"}[p])
ax[0].set_xticks(x); ax[0].set_xticklabels(names + ["GEN0 ideal"]); ax[0].set_ylabel("Shapley value (dV per occurrence)")
ax[0].set_title("Shapley value of each recommendation, reader-RAM-dependent q"); ax[0].legend(fontsize=8)
anc = list(WW); xx = np.arange(len(anc))
for i, p in enumerate(PL):
    ax[1].bar(xx + (i - 1.5) * wdt, [WW[a]["shapley"][p] for a in anc], wdt, label=p)
ax[1].set_xticks(xx); ax[1].set_xticklabels(anc); ax[1].set_ylabel("weighted Shapley value per catastrophe"); ax[1].legend(fontsize=8)
ax[1].set_title("weighted by literature likelihood (032 weights)")
plt.tight_layout(); plt.savefig(f"{HERE}/fig_shapley_by_scenario.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
Ms = np.exp(np.linspace(np.log(2048), np.log(16 * 2**20), 300))
ax[0].semilogx(Ms / 1024, q_of(Ms), label="q(M), 025 high-tier star weight")
ax[0].semilogx(Ms / 1024, 1 - 0.24 * (1 - q_of(Ms)), label="with c4 (noisy-or)")
ax[0].semilogx(Ms / 1024, np.maximum(0.76, q_of(Ms)), ls="--", label="with c4 (max rule)")
for lo, hi, lab, c in [(4, 144, "reader B", "C0"), (2, 520, "MCU SRAM", "C1"), (4096, 16384, "PSRAM", "C2")]:
    ax[0].axvspan(lo, hi, alpha=0.12, color=c, label=lab)
ax[0].set_xlabel("reader RAM (KiB)"); ax[0].set_ylabel("q"); ax[0].legend(fontsize=8); ax[0].set_title("q by reader RAM")
tags = ["main"] + list(SENS)
vals_ = [[WW["A_XPT_S"]["shapley"][p] for p in PL]] + [[SENS[t]["weighted"]["A_XPT_S"]["shapley"][p] for p in PL] for t in SENS]
for i, p in enumerate(PL):
    ax[1].bar(np.arange(len(tags)) + (i - 1.5) * wdt, [v[i] for v in vals_], wdt, label=p)
ax[1].set_xticks(np.arange(len(tags))); ax[1].set_xticklabels(tags, rotation=20, fontsize=8); ax[1].set_ylabel("weighted Shapley (A_XPT_S)")
ax[1].legend(fontsize=8); ax[1].set_title("sensitivity")
plt.tight_layout(); plt.savefig(f"{HERE}/fig_q_ram_and_sensitivity.png", dpi=120); plt.close()

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
print(json.dumps({"check": R["check_024"], "GEN0": MAIN["GEN0"], "scen": {s: {k: SC[s][k] for k in ("shapley", "rank", "total", "ratio_C_over_T", "C_share_from_mcu_salv", "frac_mcu_salv_readers", "marg_T_alone", "marg_C_given_T", "marg_C_alone", "marg_dep_given_T")} for s in SC},
                  "weighted": {a: {k: WW[a][k] for k in ("shapley", "rank", "T_ratio_vs_GEN0", "C_share_from_mcu_salv", "marg_C_given_T", "marg_T_alone", "marg_dep_given_T", "explore_shapley_excl_T3", "explore_bundle_TC")} for a in WW},
                  "sens": SENS, "hyp": R["hypotheses"]}, ensure_ascii=False, indent=1, default=str))
print("runtime", round(R["runtime_seconds"], 1))
