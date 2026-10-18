"""研究046：技術段の最小人口の根拠（open_problems 114）。事前登録：PREREGISTRATION.md。
045 の sim.py を「# 確認：044 の模型を新しい関数で再現」の手前まで exec し、sweep の定義だけ追加で exec する。
閾値は 038 の名前空間（T）の N_Fe・N_lens・N_T2e・N_3f・N_3・pop_ok を差し替える。新しい乱数は使わない。numpy＋matplotlib のみ、ネット接続なし。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
R = {"study": "046-tech-stage-thresholds", "seed_record_only": 20261046}
EX = json.load(open(os.path.join(HERE, "sources", "excerpts.json"), encoding="utf8"))
assert all(e["found_in_fetched_text"] for e in EX["excerpts"])
POP = [e for e in EX["excerpts"] if "table_values" in e][0]["table_values"]["Total"]
MASS = [e for e in EX["excerpts"] if "table_values" in e][0]["table_values"]["Massachusetts"]
VIRG = [e for e in EX["excerpts"] if "table_values" in e][0]["table_values"]["Virginia"]

# ================= A. 例の表（人口の区間は sources の値） =================
CASES = [  # 段, 名前, 人口下, 人口上, 結果（trial / fail / partial / sustained）, 開いているか
 ("Fe", "L'Anse aux Meadows (c.1000)", 30, 160, "trial", "open"),
 ("Fe", "Norse Greenland", 2000, 3000, "fail", "open"),
 ("Fe", "Saugus 1646-c.1670 (colonies total)", POP["1640"], POP["1670"], "partial", "open"),
 ("Fe", "Iceland (bog iron)", 30000, 80000, "sustained", "semi"),
 ("Fe", "Edo Japan (tatara)", 30e6, 32e6, "sustained", "semi"),
 ("glass", "Jamestown 1608-09", 60, 500, "trial", "open"),
 ("glass", "Jamestown 1621-24 (Virginia)", VIRG["1620"], VIRG["1630"], "fail", "open"),
 ("glass", "Salem 1639- (Massachusetts)", MASS["1640"], MASS["1660"], "partial", "open"),
 ("glass", "Wistarburg 1739 (colonies)", POP["1740"], POP["1740"], "sustained", "open"),
 ("optics", "Rittenhouse telescope (<1769, colonies)", POP["1760"], POP["1770"], "sustained", "open"),
 ("optics", "Iwahashi Zenbei 1793 (Japan)", 30e6, 32e6, "sustained", "semi"),
]
R["A_cases"] = [dict(zip(("stage", "name", "pop_lo", "pop_hi", "outcome", "openness"), c)) for c in CASES]
# Saugus の能力と市場：年 30週 × 6〜7日 × 1 t
cap = {"6d": 30 * 6 * 1.0, "7d": 30 * 7 * 1.0}
pop1646 = float(np.exp(np.interp(1646, [1640, 1650], np.log([POP["1640"], POP["1650"]]))))
R["A_saugus"] = {"capacity_t_per_yr": cap, "colonies_pop_1646_loginterp": pop1646, "colonies_pop_1670": POP["1670"],
                 "kg_per_colonist_yr_1646": {k: 1000 * v / pop1646 for k, v in cap.items()},
                 "kg_per_colonist_yr_1670": {k: 1000 * v / POP["1670"] for k, v in cap.items()},
                 "massachusetts_pop_1650": MASS["1650"], "kg_per_mass_resident_1650": {k: 1000 * v / MASS["1650"] for k, v in cap.items()}}
R["A_tatara"] = {"charcoal_per_steel": 11 / 2.3, "satetsu_per_steel": 9.1 / 2.3, "crew": [4, 5], "hours": 72}

# ================= B. 045 の模型 =================
P45 = os.path.abspath(os.path.join(HERE, "..", "045-local-foci-endemic-drag", "sim.py"))
src = open(P45, encoding="utf8").read()
G = {"__file__": P45, "__name__": "sim045"}
exec(compile(src[:src.index("# 確認：044 の模型を新しい関数で再現")], P45, "exec"), G)
exec(compile(src[src.index("def sweep(cfg=None):"):src.index("SW44 = sweep(")], P45, "exec"), G)
r45 = json.load(open(os.path.join(HERE, "..", "045-local-foci-endemic-drag", "results.json")))
T = G["T"]; N = G["N"]; lu = G["lu"]; evaluate = G["evaluate"]; weighted = G["weighted"]; sweep = G["sweep"]

# 元の配列と一様乱数の復元（038 の g2 の最初の4回の引き、024 の N_T2）
ORIG = {k: T[k].copy() for k in ("N_Fe", "N_lens", "N_T2", "N_T2e", "N_3f", "N_3", "pop_ok")}
g2 = np.random.default_rng(20261038)
RANGES38 = {"Fe": (2e3, 1e5), "lens": (3e4, 3e6), "f3": (1e5, 1e7), "t3": (1e7, 5e8)}
U = {}
for k in ("Fe", "lens", "f3", "t3"):
    lo, hi = RANGES38[k]; raw = g2.uniform(np.log(lo), np.log(hi), N); U[k] = (raw - np.log(lo)) / (np.log(hi) - np.log(lo))
    if k == "Fe": assert np.array_equal(np.exp(raw), ORIG["N_Fe"])
U["T2"] = np.log(ORIG["N_T2"] / 0.5e6) / np.log(10.0)
R["check_uniform_recovery"] = {"Fe_max_rel": float(np.max(np.abs(lu(U["Fe"], *RANGES38["Fe"]) / ORIG["N_Fe"] - 1))),
                               "T2_max_rel": float(np.max(np.abs(lu(U["T2"], 0.5e6, 5e6) / ORIG["N_T2"] - 1)))}
DEF = {"Fe": RANGES38["Fe"], "lens": RANGES38["lens"], "T2": (0.5e6, 5e6), "f3": RANGES38["f3"], "t3": RANGES38["t3"]}

def set_thr(rng=None, M=1.0, fix=None):
    """rng：段ごとの (lo, hi)。fix：段ごとの固定値。M：Fe・lens・T2 に掛ける閉じた集団の補正。何も変えなければ元の配列。"""
    rng = {**DEF, **(rng or {})}; fix = fix or {}
    if all(rng[k] == DEF[k] for k in DEF) and not fix and M == 1.0:
        for k, v in ORIG.items(): T[k] = v.copy()
        return
    def draw(k):
        if k in fix: return np.full(N, float(fix[k]))
        if rng[k] == DEF[k] and k != "T2": return None
        return lu(U[k], *rng[k])
    nfe = draw("Fe"); nfe = ORIG["N_Fe"].copy() if nfe is None else nfe
    nl = draw("lens")
    if nl is None: nl = lu(U["lens"], *DEF["lens"])
    nt2 = draw("T2")
    nfe = nfe * M; nl = np.maximum(nl * M, nfe); nt2 = nt2 * M
    nt2e = np.maximum(nt2, nl)
    n3f = draw("f3"); n3f = lu(U["f3"], *DEF["f3"]) if n3f is None else n3f
    n3f = np.maximum(n3f, nt2e)
    n3 = draw("t3"); n3 = lu(U["t3"], *DEF["t3"]) if n3 is None else n3
    n3 = np.maximum(n3, n3f)
    T["N_Fe"] = nfe; T["N_lens"] = nl; T["N_T2"] = nt2; T["N_T2e"] = nt2e; T["N_3f"] = n3f; T["N_3"] = n3
    T["pop_ok"] = T["N_min"] < nt2

KEYS = ("U3", "CASC", "BIO", "NUC_G", "NUC_R", "NAT", "AS")
def run(label, rng=None, M=1.0, fix=None, nstar=False):
    set_thr(rng, M, fix)
    D, hv, ph, info = evaluate({}, keys=("U3", "CASC", "BIO"))
    W = weighted(D, hv, anchors=("A_Ord", "A_XPT_S", "A_XPT_E"))
    out = {"U3_TAB": ph["U3"]["TAB"], "CASC_TAB": ph["CASC"]["TAB"], "BIO_TAB": ph["BIO"]["TAB"], "named_TAB": ph["named_pooled"]["TAB"],
           "heavy_share": {k: float(hv[k].mean()) for k in KEYS},
           "w_TAB": {a: {u: W[a][u]["TAB"]["total"] for u in ("U1", "U2", "U3")} for a in W},
           "w_TAB_from_heavy": {a: {u: W[a][u]["TAB"]["from_heavy"] for u in ("U1", "U2", "U3")} for a in W},
           "pop_ok_share": float(T["pop_ok"].mean()),
           "thr_quantiles": {k: [float(x) for x in np.percentile(T[k], [5, 50, 95])] for k in ("N_Fe", "N_lens", "N_T2e", "N_3f")}}
    if nstar: out["n_star"] = sweep({})["n_star"]
    set_thr()
    return out

base = run("BASE", nstar=True)
R["check_045"] = {"U3_TAB": base["U3_TAB"], "ref": r45["B"]["B_main"]["U3_TAB"], "n_star": base["n_star"], "n_star_ref": r45["n_star"]["main"],
                  "w_AOrd_U3": base["w_TAB"]["A_Ord"]["U3"], "w_ref": r45["B"]["B_main"]["w_AOrd_U3_TAB"],
                  "reproduced": bool(base["U3_TAB"] == r45["B"]["B_main"]["U3_TAB"] and base["n_star"] == r45["n_star"]["main"]
                                     and base["w_TAB"]["A_Ord"]["U3"] == r45["B"]["B_main"]["w_AOrd_U3_TAB"])}
E1 = {"Fe": (2500, 5e4), "lens": (1e4, 1e6)}
E2 = {**E1, "T2": (0.5e6, float(POP["1770"]))}
V = {"BASE_045": base,
     "E1": run("E1", E1, nstar=True), "E1_M3": run("E1_M3", E1, 3.0), "E1_M10": run("E1_M10", E1, 10.0),
     "E2": run("E2", E2, nstar=True), "E2_M3": run("E2_M3", E2, 3.0, nstar=True), "E2_M10": run("E2_M10", E2, 10.0, nstar=True),
     "BASE_M3": run("BASE_M3", None, 3.0), "BASE_M10": run("BASE_M10", None, 10.0)}
R["B_variants"] = V

# ================= C. トルネード =================
TOR = {}
for k, (lo, hi) in (("N_Fe", DEF["Fe"]), ("N_lens", DEF["lens"]), ("N_T2", DEF["T2"]), ("N_3f", DEF["f3"])):
    a = run(k + "_lo", fix={k.replace("N_", "").replace("3f", "f3"): lo}); b = run(k + "_hi", fix={k.replace("N_", "").replace("3f", "f3"): hi})
    TOR[k] = {"lo_val": lo, "hi_val": hi,
              "w_AOrd_U3": [a["w_TAB"]["A_Ord"]["U3"], b["w_TAB"]["A_Ord"]["U3"]],
              "U3_TAB": [a["U3_TAB"], b["U3_TAB"]], "CASC_heavy": [a["heavy_share"]["CASC"], b["heavy_share"]["CASC"]],
              "U3_heavy": [a["heavy_share"]["U3"], b["heavy_share"]["U3"]]}
    for m in ("w_AOrd_U3", "U3_TAB", "CASC_heavy", "U3_heavy"):
        TOR[k][m + "_range"] = abs(TOR[k][m][1] - TOR[k][m][0])
R["C_tornado"] = TOR

# ================= 仮説 =================
b0 = base; e1 = V["E1"]; e2 = V["E2"]
dh = {k: e1["heavy_share"][k] - b0["heavy_share"][k] for k in ("CASC", "BIO", "U3")}
tmax = max(TOR, key=lambda k: TOR[k]["w_AOrd_U3_range"])
wE2 = e2["w_TAB"]["A_Ord"]["U3"]; wM10 = V["E2_M10"]["w_TAB"]["A_Ord"]["U3"]; wB = b0["w_TAB"]["A_Ord"]["U3"]
R["hypotheses"] = {
 "H58a": {"value": [e1["U3_TAB"], b0["U3_TAB"], e1["U3_TAB"] / b0["U3_TAB"]], "rule": "E1 U3 TAB within +-10% of 045", "supported": bool(abs(e1["U3_TAB"] / b0["U3_TAB"] - 1) <= 0.10)},
 "H58b": {"value": dh, "rule": "E1 |change in heavy share| < 0.02 for CASC, BIO, U3", "supported": bool(all(abs(v) < 0.02 for v in dh.values()))},
 "H58c": {"value": [e2["heavy_share"]["CASC"], b0["heavy_share"]["CASC"], e2["heavy_share"]["CASC"] / b0["heavy_share"]["CASC"]], "rule": "E2 CASC heavy share <= 0.8 x 045", "supported": bool(e2["heavy_share"]["CASC"] <= 0.8 * b0["heavy_share"]["CASC"])},
 "H58d": {"value": [tmax, {k: TOR[k]["w_AOrd_U3_range"] for k in TOR}], "rule": "largest tornado range (weighted A_Ord U3 TAB) is N_T2", "supported": bool(tmax == "N_T2")},
 "H58e": {"value": [abs(wM10 - wE2), abs(wE2 - wB)], "rule": "|E2_M10 - E2| > |E2 - 045| (weighted A_Ord U3 TAB)", "supported": bool(abs(wM10 - wE2) > abs(wE2 - wB))},
 "H58f": {"value": [e2["n_star"], b0["n_star"]], "rule": "E2 n* within +-20% of 045", "supported": bool(e2["n_star"] is not None and abs(e2["n_star"] / b0["n_star"] - 1) <= 0.20)},
}

# ================= 図 =================
fig, ax = plt.subplots(figsize=(11, 5))
col = {"trial": "C0", "fail": "C3", "partial": "C1", "sustained": "C2"}
stages = ["Fe", "glass", "optics"]; ypos = {}
y = 0
for s in stages:
    for c in CASES:
        if c[0] == s:
            ax.plot([c[2], c[3] * (1.08 if c[3] == c[2] else 1)], [y, y], "-", color=col[c[4]], lw=6, solid_capstyle="butt")
            ax.text(c[3] * 1.4, y, f"{c[1]} [{c[4]}]", va="center", fontsize=7); y += 1
    y += 0.6
bands = [("038 N_Fe", DEF["Fe"], "#999"), ("046 E1 N_Fe", E1["Fe"], "C2"), ("038 N_lens", DEF["lens"], "#999"), ("046 E1 N_lens", E1["lens"], "C2"),
         ("024 N_T2", DEF["T2"], "#999"), ("046 E2 N_T2", E2["T2"], "C2")]
for i, (lab, (lo, hi), c) in enumerate(bands):
    yy = y + 0.5 + i * 0.7; ax.plot([lo, hi], [yy, yy], color=c, lw=4, alpha=0.7); ax.text(hi * 1.4, yy, lab, va="center", fontsize=7)
ax.set_xscale("log"); ax.set_xlim(10, 3e9); ax.set_yticks([]); ax.invert_yaxis()
ax.set_xlabel("population of the society (log)")
ax.set_title("Cases of knowledgeable small societies making iron / glass / optics (red fail, blue trial, orange partial, green sustained)\nbottom: threshold priors (grey = 038/024 guess, green = 046 evidence-based)", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_cases_thresholds.png"), dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
labs = list(V); xs = np.arange(len(labs))
ax[0].bar(xs - 0.2, [V[k]["w_TAB"]["A_Ord"]["U3"] for k in labs], 0.4, label="weighted A_Ord, U3 (total)")
ax[0].bar(xs + 0.2, [V[k]["w_TAB_from_heavy"]["A_Ord"]["U3"] for k in labs], 0.4, label="of which from heavy collapse")
ax2 = ax[0].twinx(); ax2.plot(xs, [V[k]["heavy_share"]["CASC"] for k in labs], "ks-", label="CASC heavy share"); ax2.plot(xs, [V[k]["heavy_share"]["U3"] for k in labs], "r^-", label="U3 heavy share")
ax2.set_ylabel("heavy share"); ax2.legend(fontsize=7, loc="upper right")
ax[0].set_xticks(xs); ax[0].set_xticklabels(labs, rotation=30, fontsize=8); ax[0].set_ylabel("tablet-guide value"); ax[0].legend(fontsize=7, loc="upper left")
ax[0].set_title("threshold variants (E1: Fe+lens evidence, E2: +T2, M: closed-group multiplier)", fontsize=9)
names = list(TOR)
for i, k in enumerate(names):
    ax[1].plot(TOR[k]["w_AOrd_U3"], [i, i], "k-", lw=6, alpha=0.3); ax[1].plot(TOR[k]["w_AOrd_U3"][0], i, "bo"); ax[1].plot(TOR[k]["w_AOrd_U3"][1], i, "ro")
ax[1].axvline(wB, color="k", ls=":", label="045 main"); ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names)
ax[1].set_xlabel("weighted A_Ord U3 tablet value (blue = threshold at low end, red = high end)"); ax[1].legend(fontsize=8); ax[1].set_title("tornado over stage thresholds", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_variants_tornado.png"), dpi=120); plt.close()

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in R if k not in ("A_cases",)}, ensure_ascii=False, indent=1, default=float))
print("runtime", round(R["runtime_seconds"], 1))
