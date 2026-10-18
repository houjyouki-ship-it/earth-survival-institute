"""研究041：崩壊後の人口の増え方 r の外部根拠（116）と、r は知識で上がるか（121）
事前登録：PREREGISTRATION.md。040 の sim.py を「A = all_arrays()」の後・感度の前まで exec し、038〜040 の乱数・関数をそのまま使う。
新しい乱数は seed 20261041。numpy＋matplotlib のみ、ネット接続なし。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
R = {"study": "041-postcollapse-growth-rate", "seed": 20261041}

# ---------- 1. 事例の r（sources/excerpts.json の数値から計算） ----------
def rate(p0, p1, y0, y1): return float(np.log(p1 / p0) / (y1 - y0))
CASES = [
 # (名前, 区分, r, 根拠の数値)
 ("Europe 1350-1500", "E", rate(70.7, 90.7, 1350, 1500), "70.7M->90.7M"),
 ("England&Wales 1350-1400", "E", rate(2.4, 3.0, 1350, 1400), "2.4M->3.0M"),
 ("England&Wales 1350-1500", "E", rate(2.4, 3.6, 1350, 1500), "2.4M->3.6M"),
 ("France 1350-1500", "E", rate(15, 15.5, 1350, 1500), "15M->15.5M"),
 ("Germany 1650-1750 (mid)", "E", rate(12, 19, 1650, 1750), "11-13M -> 18-20M"),
 ("Iceland 1703-1800", "E", rate(50358, 46176, 1703, 1800), "50,358->46,176 (harsh, reference)"),
 ("Quebec 1706-1754", "F", rate(16745, 55009, 1706, 1754), "16,745->55,009 (some immigration)"),
 ("Quebec 1765-1784", "F", rate(69810, 113012, 1765, 1784), "69,810->113,012 (French immigration stopped; some British/Loyalist)"),
 ("Paraguay pre-war standard (low)", "F", 0.017, "1.7%/yr"),
 ("Paraguay pre-war standard (high)", "F", 0.025, "2.5%/yr"),
 ("USA 1750-1900 (25-yr doubling, incl. immigration)", "F", float(np.log(2) / 25), "doubling every 25 years"),
 ("Pitcairn 1790-1856", "M", rate(27, 193, 1790, 1856), "27->193"),
 ("Tristan da Cunha 1909-1961", "M", rate(95, 268, 1909, 1961), "95->268"),
 ("Hutterites 140 yr", "M", rate(400, 50000, 0, 140), "~400->~50,000"),
]
R["cases"] = [{"name": n, "class": c, "r": r, "basis": b} for n, c, r, b in CASES]
med = {c: float(np.median([r for _, cc, r, _ in CASES if cc == c])) for c in "EFM"}
R["class_median_r"] = med
R["germany_range"] = [rate(13, 18, 1650, 1750), rate(11, 20, 1650, 1750)]
# 近代医療なしで高い r が長く続いた年数（F の事例）
R["sustained_years_F"] = {"USA": 150, "Quebec_1706_1844": 138, "Quebec_1706_1844_r": rate(16745, 697084, 1706, 1844)}

# ---------- 2. 040 の模型を読み込む ----------
P40 = os.path.abspath(os.path.join(HERE, "..", "040-cohesion-vs-knowledge", "sim.py"))
src = open(P40, encoding="utf8").read()
cut = src.index("# ---------- 感度")
G = {"__file__": P40, "__name__": "sim040"}
exec(compile(src[:cut], P40, "exec"), G)
nstart = src.index("# ---------- n* の走査"); nend = src.index("SW = {")
exec(compile(src[nstart:nend], P40, "exec"), G)
N = G["N"]; SRC = G["SRC"]; r_pop = G["r_pop"]; p_grow = G["p_grow"]; U5 = G["U5"]; lu = G["lu"]
assign = G["assign"]; times = G["times"]; V_gen = G["V_gen"]; fixed_src = G["fixed_src"]
deltas = G["deltas"]; weighted = G["weighted"]; per_heavy = G["per_heavy"]; sweep = G["sweep"]; HEAVY = G["HEAVY"]
r40 = json.load(open(os.path.join(HERE, "..", "040-cohesion-vs-knowledge", "results.json")))
PH40 = G["per_heavy"](G["D"], G["HV"])
R["check_040"] = {"U3_TAB": PH40["U3"]["TAB"], "U3_TAB_040": r40["per_heavy_event"]["U3"]["TAB"],
                  "reproduced": bool(abs(PH40["U3"]["TAB"] - r40["per_heavy_event"]["U3"]["TAB"]) < 1e-12)}

# ---------- 3. 新しい乱数と経験的な r ----------
g6 = np.random.default_rng(20261041)
U6 = g6.uniform(size=(5, N))
r_F = lu(U6[0], 0.017, 0.028)            # F の水準【事実の事例から：Paraguay 1.7〜2.5%、Quebec 2.5%、米国 2.77%（移民込み）】
r_F_noUS = lu(U6[0], 0.017, 0.0255)      # 感度：米国を外す
dK = g6.uniform(0.0, 0.010, N)           # 近代の知識で上がる分 Δ_K【推測：M−F の差の目安】
qF_main = g6.uniform(0, 1, N)            # 主の設定：q_F ~ U(0,1)

def r_world(qF, rF=r_F):
    isF = U6[1] < qF
    return np.where(isF, rF, r_pop), isF

def run_src(d, r_arr):
    sv = dict(d["sv"]); st = assign(sv, 1.0)
    if d["ext"] > 0: st = np.where(d["uext"] < d["ext"], 0, st)
    T2s = times(st, sv, d["lit"], r_mult=r_arr / r_pop)[0]
    return {"V0": V_gen(st, T2s, d["sc"], 0.3), "V2": V_gen(st, T2s, d["sc"], 0.3, tablet=True), "st": st, "T2s": T2s}

PCONV = [0.0, 0.1, 0.25, 0.5, 1.0]
def evaluate(qF, rF=r_F, dk=dK):
    rw, isF = r_world(qF, rF)
    A = {"BASE": {k: run_src(SRC[k], rw) for k in SRC}}
    used = U5[1] < p_grow                  # 040 と同じ乱数で「知識が使われる」
    for pc in PCONV:
        conv = (~isF) & (U6[2] < pc)
        rg = np.where(used & isF, rw + dk, np.where(used & conv, rF + dk, rw))
        A[f"GROW_{pc}"] = {k: run_src(SRC[k], rg) for k in SRC}
    for iv in A:
        for k in ("AI", "GEN"): A[iv][k] = fixed_src(k, "BASE")
    B = A["BASE"]; D = {"TAB": {k: B[k]["V2"] - B[k]["V0"] for k in B}}
    for iv in A:
        if iv != "BASE": D[iv] = {k: A[iv][k]["V2"] - B[k]["V2"] for k in B}
    hv = {k: np.isin(B[k]["st"], HEAVY).astype(float) for k in B}
    ph = per_heavy(D, hv, keys=("U3", "CASC", "BIO"))
    return D, hv, ph, rw

def breakeven(ph, key="U3"):
    tab = ph[key]["TAB"]; g = [ph[key][f"GROW_{pc}"] for pc in PCONV]
    for i in range(1, len(PCONV)):
        if g[i - 1] < tab <= g[i]:
            return float(PCONV[i - 1] + (tab - g[i - 1]) / (g[i] - g[i - 1]) * (PCONV[i] - PCONV[i - 1]))
    return 0.0 if g[0] >= tab else None

D, HV, PH, RW = evaluate(qF_main)
R["main"] = {"per_heavy_event": PH, "breakeven_pconv_U3": breakeven(PH), "breakeven_pconv_named": breakeven(PH, "named_pooled"),
             "r_quantiles": [float(x) for x in np.percentile(RW, [5, 25, 50, 75, 95])]}
W = weighted(D, HV, anchors=("A_Ord", "A_XPT_S"))
R["main"]["weighted"] = {a: {u: {x: W[a][u][x]["total"] for x in W[a][u]} for u in ("U2", "U3")} for a in W}

# ---------- 4. n* ----------
SWm = sweep(RW)
R["n_star"] = {"empirical_qF_U01": SWm["n_star"], "max_R_tab": SWm["max_R_tab"], "n_star_040_blackdeath": r40["n_star"]["r_blackdeath"]["n_star"]}
R["n_star_rows_main"] = SWm["rows"]
# ---------- 5. 感度（判定に使わない） ----------
R["qF_scan"] = {}
for q in (0.0, 0.25, 0.5, 0.75, 1.0):
    Dq, Hq, Pq, Rq = evaluate(np.full(N, q))
    R["qF_scan"][str(q)] = {"U3": Pq["U3"], "named_pooled": Pq["named_pooled"], "breakeven_pconv_U3": breakeven(Pq), "n_star": sweep(Rq)["n_star"]}
Dn, Hn, Pn, Rn = evaluate(qF_main, rF=r_F_noUS)
R["sens_noUS"] = {"U3": Pn["U3"], "breakeven": breakeven(Pn), "n_star": sweep(Rn)["n_star"]}
D0, H0, P0, R0 = evaluate(qF_main, dk=np.zeros(N))
R["sens_dK0"] = {"U3": P0["U3"], "breakeven": breakeven(P0)}

# ---------- 6. 仮説 ----------
U3 = PH["U3"]; be = R["main"]["breakeven_pconv_U3"]; ns_ = R["n_star"]["empirical_qF_U01"]
R["hypotheses"] = {
 "H53a": {"value": [med["F"], med["E"], med["F"] / med["E"]], "rule": "median F >= 5 x median E", "supported": bool(med["F"] >= 5 * med["E"])},
 "H53b": {"value": [med["M"] - med["F"], med["F"] - med["E"]], "rule": "M-F < 0.25 (F-E)", "supported": bool(med["M"] - med["F"] < 0.25 * (med["F"] - med["E"]))},
 "H53c": {"value": ns_, "rule": "n*(empirical) <= 1e5", "supported": bool(ns_ is not None and ns_ <= 1e5)},
 "H53d": {"value": [U3["TAB"], r40["per_heavy_event"]["U3"]["TAB"]], "rule": "U3 heavy TAB >= 2 x 040", "supported": bool(U3["TAB"] >= 2 * r40["per_heavy_event"]["U3"]["TAB"])},
 "H53e": {"value": [U3["GROW_0.0"], U3["TAB"]], "rule": "GROW(p_conv=0) < 0.25 TAB", "supported": bool(U3["GROW_0.0"] < 0.25 * U3["TAB"])},
 "H53f": {"value": be, "rule": "breakeven p_conv >= 0.3 (None = never reaches)", "supported": bool(be is None or be >= 0.3)},
}

# ---------- 7. 図 ----------
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
cols = {"E": "C3", "F": "C2", "M": "C0"}
for i, c in enumerate(R["cases"]):
    ax[0].barh(i, 100 * c["r"], color=cols[c["class"]])
ax[0].set_yticks(range(len(R["cases"]))); ax[0].set_yticklabels([f'[{c["class"]}] {c["name"]}' for c in R["cases"]], fontsize=7)
for c, v in med.items(): ax[0].axvline(100 * v, color=cols[c], ls=":", lw=1)
ax[0].axvspan(0.14, 0.61, color="C3", alpha=0.1, label="038 r (Black Death calib.)")
ax[0].set_xlabel("annual growth rate r (%/yr)"); ax[0].set_title("116: recovery / growth cases (E: settled+epidemics, F: frontier pre-modern, M: modern)")
ax[0].legend(fontsize=7); ax[0].invert_yaxis()
for k, cc, lab in (("r_blackdeath", "C3", "040: Black Death r"), ("r_frontier", "C2", "040: frontier r")):
    rows = r40["n_star_rows"][k]; ax[1].semilogx([r["n"] for r in rows], [r["R_tab"] for r in rows], cc, lw=1, label=lab)
rows = SWm["rows"]; ax[1].semilogx([r["n"] for r in rows], [r["R_tab"] for r in rows], "k", lw=2, label="041: empirical mixture (q_F~U(0,1))")
for q, cc in (("0.25", "C1"), ("0.75", "C4")):
    pass
ax[1].axhline(0.5 * G["REF_U1"], color="gray", ls="--", lw=0.8, label="half of U1 value")
if ns_: ax[1].axvline(ns_, color="k", ls=":")
ax[1].set_xlabel("single connected surviving group n"); ax[1].set_ylabel("tablet-guide value per event"); ax[1].legend(fontsize=8)
ax[1].set_title("119 revisited: n* under the empirical r prior")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_cases_and_nstar.png"), dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
qs = sorted(R["qF_scan"], key=float)
ax[0].plot([float(q) for q in qs], [R["qF_scan"][q]["U3"]["TAB"] for q in qs], "k-o", label="TAB (tablet guide)")
for pc, cc in zip(PCONV, ("C0", "C1", "C2", "C3", "C4")):
    ax[0].plot([float(q) for q in qs], [R["qF_scan"][q]["U3"][f"GROW_{pc}"] for q in qs], color=cc, marker=".", label=f"GROW p_conv={pc}")
ax[0].axhline(r40["per_heavy_event"]["U3"]["TAB"], color="gray", ls=":", label="040 TAB")
ax[0].set_xlabel("q_F (share of post-collapse groups in frontier regime)"); ax[0].set_ylabel("gain per U3 heavy-collapse event"); ax[0].legend(fontsize=7)
ax[0].set_title("121: knowledge-driven growth vs tablet guide")
ax[1].bar(range(len(PCONV)), [U3[f"GROW_{pc}"] / U3["TAB"] for pc in PCONV], color="C2")
ax[1].axhline(1, color="k", ls="--"); ax[1].set_xticks(range(len(PCONV))); ax[1].set_xticklabels([f"p_conv={p}" for p in PCONV])
ax[1].set_ylabel("GROW / TAB (U3 heavy, q_F~U(0,1))"); ax[1].set_title("break-even: knowledge must turn epidemic regime into frontier regime")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_grow_vs_tab.png"), dpi=120); plt.close()

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), ensure_ascii=False, indent=1, default=float)
out = {k: R[k] for k in ("check_040", "class_median_r", "germany_range", "sustained_years_F", "n_star", "hypotheses", "sens_noUS", "sens_dK0")}
out["main_U3"] = PH["U3"]; out["main_named"] = PH["named_pooled"]; out["main_CASC"] = PH.get("CASC"); out["breakeven"] = R["main"]["breakeven_pconv_U3"]
out["breakeven_named"] = R["main"]["breakeven_pconv_named"]; out["r_q"] = R["main"]["r_quantiles"]; out["weighted"] = R["main"]["weighted"]
out["qF_scan"] = {q: {"TAB": v["U3"]["TAB"], "G0": v["U3"]["GROW_0.0"], "G1": v["U3"]["GROW_1.0"], "be": v["breakeven_pconv_U3"], "n_star": v["n_star"]} for q, v in R["qF_scan"].items()}
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
print("runtime", round(R["runtime_seconds"], 1))
