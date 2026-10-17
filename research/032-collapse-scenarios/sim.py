"""研究032：崩壊シナリオ別に、024〜031 の提言の価値を評価し直す（next_plan 最優先課題）
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

# ---------- 本回の乱数（別の生成器、seed 20261032） ----------
SEED = 20261032
g = np.random.default_rng(SEED)
STATES = ["extinct", "T3", "T3_fragile", "T2", "T1", "T0"]
# 平均の状態確率・金庫喪失・発見・m23 倍率・T0 での識字喪失確率：すべて【推測】
SCEN = {
 "NUC_G": {"mean": [0.01, 0.65, 0.15, 0.14, 0.04, 0.01], "lost": (0.05, 0.30), "find": (0.2, 0.6), "M": (1, 2), "lit": 0.2,
           "note": "Xia 2022: >5e9 deaths, 150 Tg, crops -90%, Australia/NZ least affected; vault in Svalbard (high N latitude)"},
 "NUC_R": {"mean": [0.001, 0.93, 0.05, 0.02, 0.001, 0.001], "lost": (0.01, 0.05), "find": (0.5, 0.9), "M": (1, 1), "lit": 0.1,
           "note": "Xia 2022: >2e9 deaths, 5-47 Tg"},
 "BIO":   {"mean": [0.15, 0.40, 0.20, 0.12, 0.08, 0.05], "lost": (0.01, 0.05), "find": (0.3, 0.9), "M": (1, 3), "lit": 0.5,
           "note": "infrastructure intact, people lost; small survivor population slows rebuild"},
 "AI":    {"mean": [0.80, 0.08, 0.04, 0.05, 0.02, 0.01], "lost": (0.2, 0.8), "find": (0.3, 0.9), "M": (1, 1), "lit": 0.3,
           "note": "mostly extinction / permanent disempowerment; vault may be destroyed"},
 "CASC":  {"mean": [0.02, 0.35, 0.20, 0.23, 0.12, 0.08], "lost": (0.05, 0.30), "find": (0.3, 0.9), "M": (1, 3), "lit": 0.5,
           "note": "compound / slow collapse (climate, war, pandemics); easy resources depleted"},
 "GEN":   {"mean": [0.05, 0.001, 0.001, 0.001, 0.001, 0.946], "lost": (0.05, 0.30), "find": (0.3, 0.9), "M": (1, 1), "lit": 0.3,
           "note": "unknown / other risks: the old single assumption (restart from T0)"},
}
KAPPA_DIR = 20.0
U_state = g.uniform(size=N); U_lost = g.uniform(size=N); U_find = g.uniform(size=N); U_lit = g.uniform(size=N)
U_acc = g.uniform(size=N); W = g.uniform(20, 80, N); p_acc = g.uniform(0.1, 0.6, N)
lit_mult = logu(1, 30, g=g); Zm = g.uniform(size=N)
# 第2拠点（南半球）の独立な乱数（探索用）
U_lost2 = g.uniform(size=N); U_find2 = g.uniform(size=N)

def draw_scenario(name, mean_override=None):
    s = SCEN[name]; mean = np.array(mean_override if mean_override is not None else s["mean"], float); mean /= mean.sum()
    probs = g.dirichlet(KAPPA_DIR * mean, size=N)
    cum = np.cumsum(probs, axis=1)
    st = (U_state[:, None] > cum).sum(axis=1).clip(0, 5)
    p_lost = g.uniform(*s["lost"], N); p_find = g.uniform(*s["find"], N)
    M = np.exp(np.log(s["M"][0]) + Zm * (np.log(s["M"][1]) - np.log(s["M"][0])))
    return {"probs": probs, "st": st, "p_lost": p_lost, "p_find": p_find, "M": M}

def V_s(sc, qv, g_hand=0.0, tablet=False, site2=False, ideal=False):
    if ideal:  # 024 の想定：T0 出発・金庫は残り必ず見つかる・識字あり
        alive = np.ones(N, bool); T2s = T2.copy(); M = np.ones(N); st = np.full(N, 5)
        intact = np.ones(N, bool)
    else:
        st = sc["st"]; alive = st != 0
        intact = (U_lost > sc["p_lost"]) & (U_find < sc["p_find"])
        if site2:
            intact = intact | ((U_lost2 > SITE2["lost"]) & (U_find2 < SITE2["find"]))
        lit_lost = U_lit < SCEN_LIT
        T0s = np.where(lit_lost, d01 * lit_mult, d01) + d12
        T2s = np.select([st == 3, st == 4, st == 5, st == 2], [np.zeros(N), d12, T0s, W], default=np.zeros(N))
        M = sc["M"]
    T3t = T2s + 943 * M * np.exp(0.8 * Z[2])
    gen = (T3t < life) & (T3t <= H)
    cov = np.zeros(N); avail = np.zeros(N, bool)
    if g_hand > 0:
        hand_ok = U_h < P_H; cov = np.where(hand_ok, np.minimum(1.0, g_hand / g_req), cov); avail |= hand_ok
    if tablet:
        tab_ok = U_sep > p_sep; cov = np.where(tab_ok, 1.0, cov); avail |= tab_ok
    Tt = T2s + t_build
    tgt = avail & pop_ok & (U_t < P_max * cov ** kappa) & (Tt < life) & (Tt <= H)
    v = np.maximum(qv * tgt, gen.astype(float))
    if not ideal:
        t3_now = (st == 1) | ((st == 2) & (U_acc < p_acc))
        v = np.where(t3_now, 1.0, v)
        v = np.where(alive & intact, v, 0.0)
    return v, gen, tgt

def recs(sc, ideal=False):
    P0 = V_s(sc, 0.3, ideal=ideal)[0]; P2 = V_s(sc, 0.3, tablet=True, ideal=ideal)[0]
    P3 = V_s(sc, 0.3, g_hand=0.25 * D_MAIN, tablet=True, ideal=ideal)[0]
    c4 = V_s(sc, 0.76, tablet=True, ideal=ideal)[0] - V_s(sc, 0.031, tablet=True, ideal=ideal)[0]
    dep = V_s(sc, 0.704, tablet=True, ideal=ideal)[0] - V_s(sc, 0.101, tablet=True, ideal=ideal)[0]
    per = {"R_tab": P2 - P0, "R_hand25": P3 - P2, "R_c4": c4, "R_dep": dep, "V_P0": P0}
    return per

R = {"seed_024_stream": 20261024, "seed": SEED, "N": N, "python": sys.version.split()[0], "platform": platform.platform(),
     "SCEN": {k: {kk: vv for kk, vv in v.items()} for k, v in SCEN.items()}, "states": STATES}

# ---------- 検査：024 の P2 を再現 ----------
SCEN_LIT = np.zeros(N)
chk = recs(None, ideal=True)
r24 = json.load(open(f"{HERE}/../024-content-value-reader-guide/results.json"))["grid_D1.9"]["m943_q0.3"]
R["check_024"] = {"dV_P2": float(chk["R_tab"].mean()), "024_dV_P2": r24["dV"]["P2_tablet"],
                  "dV_P3_minus_P2": float(chk["R_hand25"].mean()), "024_P3_minus_P2": r24["dV_P3_minus_P2"],
                  "reproduced_exactly": bool(abs(chk["R_tab"].mean() - r24["dV"]["P2_tablet"]) < 1e-12 and abs(chk["R_hand25"].mean() - r24["dV_P3_minus_P2"]) < 1e-12),
                  "dV_c4": float(chk["R_c4"].mean()), "dV_dep": float(chk["R_dep"].mean())}

# ---------- シナリオ別 ----------
SITE2 = {"lost": 0.06, "find": 0.75}   # 探索用の第2拠点（南半球）【推測】
DRAWS = {}; PER = {}; SC_OUT = {}
for name in SCEN:
    sc = draw_scenario(name); DRAWS[name] = sc
    SCEN_LIT = np.full(N, SCEN[name]["lit"])
    per = recs(sc); PER[name] = per
    # 探索：南半球の第2拠点（NUC_G では南の生存者が見つけやすい：find 0.75；他も同じ値で置く）
    v2 = V_s(sc, 0.3, site2=True)[0]
    st = sc["st"]
    SC_OUT[name] = {k: float(v.mean()) for k, v in per.items()}
    SC_OUT[name]["state_shares"] = {STATES[i]: float(np.mean(st == i)) for i in range(6)}
    SC_OUT[name]["explore_site2_dV_P0"] = float(v2.mean() - per["V_P0"].mean())
    # 探索（事前登録外、初回実行後に追加）：T3 が残る結果を除いた（再建する社会だけの）第2拠点の価値
    rebuild = np.isin(st, [3, 4, 5]) | ((st == 2) & (U_acc >= p_acc))
    SC_OUT[name]["explore_site2_dV_P0_rebuild_only"] = float(np.mean((v2 - per["V_P0"]) * rebuild))
    SC_OUT[name]["explore_tab_dV_rebuild_only"] = float(np.mean(per["R_tab"] * rebuild))
    SC_OUT[name]["ratio_c4_over_tab"] = SC_OUT[name]["R_c4"] / SC_OUT[name]["R_tab"] if SC_OUT[name]["R_tab"] > 0 else None
    SC_OUT[name]["ratio_tab_over_GEN0"] = SC_OUT[name]["R_tab"] / R["check_024"]["dV_P2"]
R["scenarios"] = SC_OUT

# ---------- 重み ----------
r_g = g.uniform(0.3, 0.7, N)   # 核のうち全面戦争の割合【推測】
def weights(anchor):
    if anchor == "A_Ord":
        base = {"AI": 1/10, "BIO": 1/30, "NUC": 1/1000, "CASC": 2/1000, "GEN": 1/30 + 1/50}
    elif anchor == "A_XPT_S":
        bio = 2.13 * 0.01 / 0.38; rest = max(0.0, 9.05 - 2.13 - 4.0 - bio)
        base = {"AI": 2.13, "BIO": bio, "NUC": 4.0, "CASC": rest / 2, "GEN": rest / 2}
    elif anchor == "A_XPT_E":
        bio = 12 * 1 / 3; rest = max(0.0, 20 - 12 - 8 - bio)
        base = {"AI": 12, "BIO": bio, "NUC": 8, "CASC": rest / 2, "GEN": rest / 2}
    tot = sum(base.values())
    w = {k: v / tot for k, v in base.items() if k != "NUC"}
    w["NUC_G"] = base["NUC"] / tot * r_g; w["NUC_R"] = base["NUC"] / tot * (1 - r_g)
    return w, base
WT = {}
for a in ("A_Ord", "A_XPT_S", "A_XPT_E"):
    w, base = weights(a)
    out = {"base": base, "mean_weights": {k: float(np.mean(v)) for k, v in w.items()}}
    for rk in ("R_tab", "R_hand25", "R_c4", "R_dep"):
        contrib = {s: w[s] * PER[s][rk] for s in w}
        tot = sum(contrib.values())
        out[rk] = float(np.mean(tot))
        out[rk + "_share_by_scen"] = {s: float(np.mean(c) / np.mean(tot)) for s, c in contrib.items()}
        out[rk + "_ratio_vs_GEN0"] = float(np.mean(tot) / chk[rk].mean())
    out["explore_site2_dV_P0"] = float(sum(np.mean(w[s]) * SC_OUT[s]["explore_site2_dV_P0"] for s in w))
    out["explore_site2_dV_P0_rebuild_only"] = float(sum(np.mean(w[s]) * SC_OUT[s]["explore_site2_dV_P0_rebuild_only"] for s in w))
    out["decomp_ratio_vs_GEN"] = out["R_tab"] / SC_OUT["GEN"]["R_tab"]
    out["_per_sample_tab"] = sum(w[s] * PER[s]["R_tab"] for s in w)
    out["_w"] = w
    WT[a] = out

# ---------- 感度（A_Ord、重み付き R_tab の五分位差） ----------
def tornado(a):
    w = WT[a]["_w"]; y = WT[a]["_per_sample_tab"]
    ss = list(w)
    feats = {
      "T3_share_incl_fragile": sum(w[s] * (DRAWS[s]["probs"][:, 1] + DRAWS[s]["probs"][:, 2]) for s in ss),
      "extinct_share": sum(w[s] * DRAWS[s]["probs"][:, 0] for s in ss),
      "p_lost": sum(w[s] * DRAWS[s]["p_lost"] for s in ss),
      "p_find": sum(w[s] * DRAWS[s]["p_find"] for s in ss),
      "m23_multiplier": sum(w[s] * DRAWS[s]["M"] for s in ss),
      "T0_share": sum(w[s] * DRAWS[s]["probs"][:, 5] for s in ss),
      "lit_mult": lit_mult, "g_req": g_req, "P_max": P_max, "life": life}
    out = {}
    for k, x in feats.items():
        lo, hi = np.percentile(x, [20, 80])
        a_, b_ = float(np.mean(y[x <= lo])), float(np.mean(y[x >= hi]))
        out[k] = {"bottom20": a_, "top20": b_, "abs_diff": abs(a_ - b_)}
    return out
R["tornado"] = {a: tornado(a) for a in WT}

# ---------- 感度：T3 保持を半分にした悲観表 ----------
pess = {}
for name in SCEN:
    m = np.array(SCEN[name]["mean"], float)
    moved = 0.5 * (m[1] + m[2]); m[1] *= 0.5; m[2] *= 0.5; m[3] += moved * 0.5; m[4] += moved * 0.3; m[5] += moved * 0.2
    sc = draw_scenario(name, m); SCEN_LIT = np.full(N, SCEN[name]["lit"])
    pess[name] = {k: float(v.mean()) for k, v in recs(sc).items()}
R["pessimistic_T3_halved"] = pess
for a in WT:
    w = WT[a]["_w"]
    R.setdefault("pessimistic_weighted", {})[a] = {rk: float(sum(np.mean(w[s]) * pess[s][rk] for s in w) / chk[rk].mean())
                                                    for rk in ("R_tab", "R_c4")}

# ---------- 仮説 ----------
nonGEN = [s for s in SCEN if s != "GEN"]
best = max(nonGEN, key=lambda s: SC_OUT[s]["R_tab"])
ratios = [SC_OUT[s]["ratio_c4_over_tab"] for s in SCEN if SC_OUT[s]["R_tab"] > 0.001]
torO = R["tornado"]["A_Ord"]; topO = max(torO, key=lambda k: torO[k]["abs_diff"])
nucS = WT["A_XPT_S"]["R_tab_share_by_scen"]["NUC_G"] + WT["A_XPT_S"]["R_tab_share_by_scen"]["NUC_R"]
R["hypotheses"] = {
 "H44a": {"value": {a: WT[a]["R_tab_ratio_vs_GEN0"] for a in WT}, "rule": "all < 0.30", "supported": all(WT[a]["R_tab_ratio_vs_GEN0"] < 0.30 for a in WT)},
 "H44b": {"value": WT["A_Ord"]["R_tab_share_by_scen"]["AI"], "rule": "AI share < 0.10 (A_Ord)", "supported": WT["A_Ord"]["R_tab_share_by_scen"]["AI"] < 0.10},
 "H44c": {"value": best, "rule": "argmax R_tab over non-GEN is CASC", "supported": best == "CASC"},
 "H44d": {"value": [min(ratios), max(ratios)], "rule": "max/min < 1.5", "supported": max(ratios) / min(ratios) < 1.5},
 "H44e": {"value": topO, "rule": "top tornado input is T3_share_incl_fragile", "supported": topO == "T3_share_incl_fragile"},
 "H44f": {"value": nucS, "rule": "NUC share >= 0.5 (A_XPT_S)", "supported": nucS >= 0.5},
}
for a in WT:
    WT[a].pop("_per_sample_tab"); WT[a].pop("_w")
R["weighted"] = WT

# ---------- 図 ----------
names = list(SCEN); rk_l = {"R_tab": "tablet guide", "R_hand25": "hand-layer 25% backup", "R_c4": "c4 small-window dual-write", "R_dep": "0.003-reel dependency slot"}
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
x = np.arange(len(names)); wdt = 0.2
for i, rk in enumerate(rk_l):
    ax[0].bar(x + (i - 1.5) * wdt, [SC_OUT[n][rk] for n in names], wdt, label=rk_l[rk])
for i, rk in enumerate(rk_l):
    ax[0].plot([len(names) - 0.5 + 0.5], [chk[rk].mean()], marker="*", ms=12, color=f"C{i}")
ax[0].set_xticks(list(x) + [len(names)]); ax[0].set_xticklabels(names + ["024 ideal"])
ax[0].set_ylabel("dV per occurrence (machine-layer value read)"); ax[0].legend(fontsize=8); ax[0].set_title("value of each recommendation by collapse scenario")
bottom = np.zeros(len(names))
for i, stn in enumerate(STATES):
    vals = np.array([SC_OUT[n]["state_shares"][stn] for n in names])
    ax[1].bar(names, vals, bottom=bottom, label=stn); bottom += vals
ax[1].set_ylabel("share of outcomes"); ax[1].legend(fontsize=8, ncol=3); ax[1].set_title("restart state (assumed, Dirichlet)")
plt.tight_layout(); plt.savefig(f"{HERE}/fig_dV_by_scenario.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
anc = list(WT)
bottom = np.zeros(len(anc))
for s in ["NUC_G", "NUC_R", "BIO", "AI", "CASC", "GEN"]:
    vals = np.array([WT[a]["R_tab"] * WT[a]["R_tab_share_by_scen"][s] for a in anc])
    ax[0].bar(anc, vals, bottom=bottom, label=s); bottom += vals
ax[0].axhline(chk["R_tab"].mean(), color="k", ls="--", label="024 single assumption")
ax[0].set_ylabel("weighted dV (tablet guide) per catastrophe"); ax[0].legend(fontsize=8); ax[0].set_title("weighted by literature likelihood")
tor = R["tornado"]["A_Ord"]; ks = list(tor); yy = np.arange(len(ks))
ax[1].barh(yy - 0.2, [tor[k]["bottom20"] for k in ks], 0.4, label="bottom 20%")
ax[1].barh(yy + 0.2, [tor[k]["top20"] for k in ks], 0.4, label="top 20%")
ax[1].set_yticks(yy); ax[1].set_yticklabels(ks); ax[1].set_xlabel("weighted dV (tablet guide), A_Ord"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_weighted_and_tornado.png", dpi=120); plt.close()

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({"check_024": R["check_024"], "scenarios": SC_OUT, "weighted": WT, "hyp": R["hypotheses"],
                  "pess": R["pessimistic_weighted"], "tornado_A_Ord": R["tornado"]["A_Ord"]}, ensure_ascii=False, indent=1))
print("runtime", round(R["runtime_seconds"], 1))
