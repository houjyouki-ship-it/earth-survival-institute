"""研究071：T2 より前の社会が文書を求めて届いた範囲（open_problems 185）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md。
070 の sim.py を行頭の「# ---------- 版 ----------」の前まで exec し、070 の主を差0で再現してから、
重い崩壊の標本の「届く範囲」を文書を求めた旅の事例 19件で置き換える。新しい乱数は seed 20261071。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P70 = os.path.abspath(os.path.join(HERE, "..", "070-group-deposit-overlap", "sim.py"))
src = open(P70, encoding="utf8").read(); cut = src.index("\n# ---------- 版 ----------\n")
G70 = {"__file__": P70, "__name__": "sim070"}
exec(compile(src[:cut], P70, "exec"), G70)
r70 = json.load(open(os.path.join(HERE, "..", "070-group-deposit-overlap", "results.json"), encoding="utf8"))
N = G70["N"]; ver = G70["ver"]; ratio_all = G70["ratio_all"]; rho70 = G70["rho70"]; place = G70["place"]; reach_units = G70["reach_units"]
DEP = G70["DEP"]; REG = G70["REG"]; NR = G70["NR"]; U_geo = G70["U_geo"]; G_main = G70["G_main"]; U_reach = G70["U_reach"]; KU = G70["KU"]; GMAX = G70["GMAX"]
KEYS = G70["KEYS"]; PARTS = G70["PARTS"]; M1 = G70["M1"]; get = G70["get"]; qf = G70["qf"]; C_T = G70["C_T"]; C_P = G70["C_P"]; C_pr = G70["C_pr"]
R = {"study": "071-text-seeking-reach", "seed": 20261071, "N": int(N)}
RT = {}; DV = {}

# ---------- 検査1：070 の主を差0で再現 ----------
POS_main = place(U_geo, G_main)
RT["chk070"], DV["chk070"] = ratio_all(ver(rho=rho70(reach_units(POS_main, DEP["D_nat"], U_reach))))
d70 = max(abs(RT["chk070"][k][p] - r70["ratio_to_V0"]["main"][k][p]) for k in KEYS for p in PARTS)
R["check_070_main"] = {"max_abs_diff": d70, "reproduced": d70 < 1e-12}

# ---------- 資料 ----------
CJ = json.load(open(os.path.join(HERE, "sources", "cases.json"), encoding="utf8"))
CS = CJ["cases"]; CAT = {"sub": 0, "cont": 1, "global": 2}
def counts(cases, key="reach_adj"):
    c = np.zeros(3)
    for x in cases: c[CAT[x[key]]] += 1
    return c
EUR = {"Northern Europe", "Southern Europe", "Western Europe"}
CNT = {"main": counts(CS), "literal": counts(CS, "reach_literal"),
       "asia_only": counts([x for x in CS if x["origin_sub"] not in EUR]), "europe_only": counts([x for x in CS if x["origin_sub"] in EUR])}
dist = np.array([x["dist_max_km"] for x in CS]); durs = np.array([x["dur_years"] for x in CS if x["dur_years"] is not None], float)
R["cases"] = {"n": len(CS), "counts": {k: v.tolist() for k, v in CNT.items()}, "share_beyond_sub_adj": float(1 - CNT["main"][0] / len(CS)),
              "dist_km": {"min": float(dist.min()), "median": float(np.median(dist)), "max": float(dist.max()),
                          "median_sub": float(np.median([x["dist_max_km"] for x in CS if x["reach_adj"] == "sub"])),
                          "median_cont": float(np.median([x["dist_max_km"] for x in CS if x["reach_adj"] == "cont"]))},
              "dur_years": {"n": int(len(durs)), "median": float(np.median(durs)), "min": float(durs.min()), "max": float(durs.max())}}
CEN = json.load(open(os.path.join(HERE, "sources", "centroids.json"), encoding="utf8"))
ci = {r: i for i, r in enumerate(CEN["regions"])}
DM = np.array([[CEN["dist_km"][ci[a]][ci[b]] for b in REG] for a in REG])
R["centroid_dist_examples_km"] = {f"{a}|{b}": float(DM[REG.index(a), REG.index(b)]) for a, b in
    (("Eastern Asia", "Southern Asia"), ("Eastern Asia", "South-eastern Asia"), ("Southern Europe", "Western Asia"), ("Northern Europe", "Southern Europe"),
     ("Western Europe", "Southern Europe"), ("South America", "Central America"), ("Australia and New Zealand", "Melanesia"), ("Southern Asia", "South-eastern Asia"))}
R["share_region_pairs_within_median_case_dist"] = float((DM[~np.eye(NR, dtype=bool)] <= np.median(dist)).mean())

# ---------- 新しい乱数 ----------
g = np.random.default_rng(20261071)
A0 = 0.5
UC = g.uniform(size=N); UL = g.uniform(size=N); IB = g.integers(0, len(CS), N)
def draw_cat(c, alpha0=A0, glob_zero=False):
    a = c + alpha0
    if glob_zero: a = a.copy(); a[2] = 0.0
    # Dirichlet(a) は a ごとに固定の生成器で引く（seed は版名に依らず a だけで決まる）。範囲の選択は版の間で共通の一様乱数 UC を使う。
    rg = np.random.default_rng([20261071] + [int(round(x * 2)) for x in a])
    gm = np.stack([rg.gamma(x, size=N) if x > 0 else np.zeros(N) for x in a], 1)
    p = gm / gm.sum(1, keepdims=True)
    cp = np.cumsum(p, 1)
    return (UC[:, None] > cp[:, :2]).sum(1), p
def reach_cat(pos, dep, cat):
    sub = reach_units(pos, dep, "sub"); con = reach_units(pos, dep, "cont"); glo = reach_units(pos, dep, "global")
    return np.where(cat[:, None] == 0, sub, np.where(cat[:, None] == 1, con, glo))
def reach_dist(pos, dep, d):
    ok = DM[np.maximum(pos, 0)] <= d[:, None, None]          # N×GMAX×NR
    units = (ok * dep[None, None, :]).sum(2)
    return np.where(pos >= 0, units, -1)

CATS = {}; PS = {}
CATS["main"], PS["main"] = draw_cat(CNT["main"])
CATS["literal"], PS["literal"] = draw_cat(CNT["literal"])
CATS["bias3"], PS["bias3"] = draw_cat(CNT["main"] * np.array([3.0, 1.0, 1.0]))
CATS["no_global"], PS["no_global"] = draw_cat(CNT["main"], glob_zero=True)
CATS["asia_only"], PS["asia_only"] = draw_cat(CNT["asia_only"])
CATS["europe_only"], PS["europe_only"] = draw_cat(CNT["europe_only"])
loss = CATS["main"].copy(); loss[(loss > 0) & (UL < 1 / 3)] = 0; CATS["loss_once"] = loss
R["realised_reach_share"] = {k: np.bincount(v, minlength=3).astype(float).__truediv__(N).tolist() for k, v in CATS.items()}
R["070_reach_share"] = [float(U_reach.mean()), float(1 - U_reach.mean()), 0.0]
d_s = dist[IB]

# ---------- 版 ----------
POS_G3 = place(U_geo, 3)
VS = {}
for k in ("main", "literal", "bias3", "no_global", "loss_once", "asia_only", "europe_only"):
    VS[k] = lambda k=k: reach_cat(POS_main, DEP["D_nat"], CATS[k])
VS["dist"] = lambda: reach_dist(POS_main, DEP["D_nat"], d_s)
VS["dist_x2"] = lambda: reach_dist(POS_main, DEP["D_nat"], 2 * d_s)
VS["dist_max"] = lambda: reach_dist(POS_main, DEP["D_nat"], np.full(N, dist.max()))
for dk in ("D_big", "D_pop", "D_surv", "D_even"): VS["main_" + dk] = lambda dk=dk: reach_cat(POS_main, DEP[dk], CATS["main"])
VS["dist_D_even"] = lambda: reach_dist(POS_main, DEP["D_even"], d_s)
VS["G3"] = lambda: reach_cat(POS_G3, DEP["D_nat"], CATS["main"])
SPREAD = {"spread"}
VS["spread"] = lambda: reach_cat(POS_main, DEP["D_nat"], CATS["main"])
for nm, fn in VS.items():
    RT[nm], DV[nm] = ratio_all(ver(rho=rho70(fn(), spread=(nm in SPREAD))))
RT["070_main"] = r70["ratio_to_V0"]["main"]
for k in ("reach_sub", "reach_cont", "reach_global", "D_even", "D_big"): RT["070_" + k] = r70["ratio_to_V0"][k]
R["ratio_to_V0"] = RT
R["scan_psub"] = []
for ps in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
    cat = np.where(UC < ps, 0, 1)
    rr, _ = ratio_all(ver(rho=rho70(reach_cat(POS_main, DEP["D_nat"], cat))))
    R["scan_psub"].append({"p_sub": ps, "heavy": rr[M1]["heavy"], "heavy_XPT": rr["A_XPT_S/U2"]["heavy"]})
R["scan_dist"] = []
for dd in (0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000, 10000, 20000):
    rr, _ = ratio_all(ver(rho=rho70(reach_dist(POS_main, DEP["D_nat"], np.full(N, float(dd))))))
    R["scan_dist"].append({"d_km": dd, "heavy": rr[M1]["heavy"]})

# ---------- 費用あたり（TAB=1、069・070 と同じ式） ----------
C_map = C_pr; R["CE"] = {}
for key in KEYS:
    a, u = key.split("/"); R["CE"][key] = {}
    for p in ("total", "heavy"):
        vt = get("main", key)["TAB"][p]; vtc = get("main", key)["TC"][p]; R["CE"][key][p] = {}
        for nm in ["chk070", "main", "dist", "bias3", "main_D_big", "main_D_even"]:
            dall = DV[nm][a][u]["TC"][p] - vtc
            R["CE"][key][p][nm] = qf(dall / vt * C_T / (C_P(100) + C_pr + C_map)) if dall > 0 else [0.0, 0.0, 0.0]

# ---------- 仮説 ----------
H = {}; h = lambda nm, k=M1: RT[nm][k]["heavy"]; h70 = r70["ratio_to_V0"]["main"][M1]["heavy"]
v = R["cases"]["share_beyond_sub_adj"]; H["H83a"] = {"value": v, "rule": "share beyond own subregion >= 0.5 (descriptive)", "supported": v >= 0.5}
v = [h("main"), h70, h("main") - h70]; H["H83b"] = {"value": v, "rule": "|main - 070 main| <= 0.010", "supported": abs(v[2]) <= 0.010}
v = int(CNT["main"][2]); H["H83c"] = {"value": v, "rule": "adjusted global cases == 0 (descriptive)", "supported": v == 0}
v = [h("bias3"), h70]; H["H83d"] = {"value": v, "rule": "bias3 heavy <= 070 main", "supported": v[0] <= v[1]}
v = [h("dist"), h("main"), h("dist") - h("main")]; H["H83e"] = {"value": v, "rule": "dist heavy <= main - 0.010", "supported": v[2] <= -0.010}
v = len(CS); H["H83f"] = {"value": v, "rule": "cases >= 8 (descriptive)", "supported": v >= 8}
vn = ["main", "literal", "bias3", "no_global", "loss_once", "dist", "asia_only", "europe_only"]; vs = [h(x) for x in vn]
H["H83g"] = {"value": [min(vs), max(vs), max(vs) - min(vs)], "rule": "range of 8 versions <= 0.04", "supported": max(vs) - min(vs) <= 0.04}
v = R["cases"]["dur_years"]["median"]; H["H83h"] = {"value": v, "rule": "median duration >= 5 y (descriptive)", "supported": v >= 5}
for xx in H.values(): xx["supported"] = bool(xx["supported"])
R["hypotheses"] = H

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
order = np.argsort(dist); col = {"sub": "C0", "cont": "C1", "global": "C3"}
for j, i in enumerate(order):
    x = CS[i]; axs[0].barh(j, x["dist_max_km"], color=col[x["reach_adj"]], hatch="//" if x["reach_literal"] == "global" else None)
axs[0].set_yticks(range(len(CS))); axs[0].set_yticklabels([f'{CS[i]["id"]} ({CS[i]["y0"] or "?"})' for i in order], fontsize=7)
for nm in ("Eastern Asia|Southern Asia", "Northern Europe|Southern Europe", "Southern Europe|Western Asia"):
    axs[0].axvline(R["centroid_dist_examples_km"][nm], ls=":", lw=0.8, color="k"); axs[0].text(R["centroid_dist_examples_km"][nm], len(CS) - 0.5, nm.replace("|", "-"), rotation=90, fontsize=6, va="top")
axs[0].set_xlabel("great-circle distance origin -> destination (km); blue=same subregion, orange=same continent, hatched=M49 cross-continent")
axs[0].xaxis.label.set_fontsize(7)
sp = R["scan_psub"]; axs[1].plot([r["p_sub"] for r in sp], [r["heavy"] for r in sp], "o-", label="A_Ord/U3 (sub vs continent mix)")
axs[1].axvline(0.5, color="gray", ls=":", lw=0.8, label="070 prior p_sub=0.5")
axs[1].axvline(float(np.mean(PS["main"][:, 0])), color="C1", ls="--", lw=0.8, label="071 cases: mean p_sub")
axs[1].axvline(float(np.mean(PS["bias3"][:, 0])), color="C3", ls="--", lw=0.8, label="bias3: mean p_sub")
axs[1].set_xlabel("share of samples whose groups reach only their own subregion"); axs[1].set_ylabel("heavy-collapse TC / V0"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_cases_scan.png"), dpi=130); plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 6.5))
names = ["070_main", "070_reach_sub", "070_reach_cont", "070_reach_global", "main", "literal", "bias3", "no_global", "loss_once", "asia_only", "europe_only",
         "dist", "dist_x2", "dist_max", "main_D_big", "main_D_even", "dist_D_even", "G3", "spread"]
yy = np.arange(len(names))
ax.barh(yy - 0.2, [RT[n][M1]["heavy"] for n in names], height=0.4, label="A_Ord/U3")
ax.barh(yy + 0.2, [RT[n]["A_XPT_S/U2"]["heavy"] for n in names], height=0.4, color="C1", alpha=0.7, label="A_XPT_S/U2")
ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=7); ax.axvline(h70, color="k", ls=":", lw=0.7)
lo = min(min(RT[n][M1]["heavy"] for n in names), min(RT[n]["A_XPT_S/U2"]["heavy"] for n in names))
ax.set_xlim(max(0.0, lo - 0.03), 1.0); ax.set_xlabel("heavy-collapse TC / V0"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_versions.png"), dpi=130); plt.close(fig)

R["runtime_s"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float, ensure_ascii=False)
print("check070", R["check_070_main"])
print("cases", json.dumps(R["cases"]))
print("centroid", json.dumps(R["centroid_dist_examples_km"]), "pairs_within_median", R["share_region_pairs_within_median_case_dist"])
print("reach_share", json.dumps(R["realised_reach_share"]), "070", R["070_reach_share"])
for nm in RT: print(nm, {k: {p: round(RT[nm][k][p], 4) for p in PARTS} for k in KEYS})
print("scan_psub", json.dumps(R["scan_psub"])); print("scan_dist", json.dumps(R["scan_dist"]))
print("CE total", json.dumps(R["CE"][M1]["total"])); print("CE heavy", json.dumps(R["CE"][M1]["heavy"]))
for k, xx in H.items(): print(k, xx["supported"], xx["value"])
print("runtime", R["runtime_s"])
