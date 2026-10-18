"""研究047：閉じた集団の倍率 M（open_problems 133）と、つながり χ（120）。事前登録：PREREGISTRATION.md（16:47Z）。
046 の sim.py を「base = run(」の手前まで exec して（中で 045・044・042・040・039・038 を exec）、閾値の差し替えの道具を借りる。
新しい乱数は seed 20261047 の K の bootstrap と標本ごとの K の割り当てだけ。numpy＋matplotlib のみ、ネット接続なし。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261047
R = {"study": "047-closed-group-multiplier", "seed": SEED}
EX = json.load(open(os.path.join(HERE, "sources", "excerpts.json"), encoding="utf8"))
assert all(e["found_in_fetched_text"] for e in EX["excerpts"])
R["n_excerpts_verified"] = len(EX["excerpts"])

# ---------- 046 の道具 ----------
P46 = os.path.abspath(os.path.join(HERE, "..", "046-tech-stage-thresholds", "sim.py"))
src = open(P46, encoding="utf8").read()
H = {"__file__": P46, "__name__": "sim046"}
exec(compile(src[:src.index("base = run(")], P46, "exec"), H)
T = H["T"]; N = H["N"]; lu = H["lu"]; U = H["U"]; DEF = H["DEF"]; ORIG = H["ORIG"]
evaluate = H["evaluate"]; weighted = H["weighted"]; sweep = H["sweep"]; G = H["G"]; SRC = G["SRC"]; POP46 = H["POP"]
r46 = json.load(open(os.path.join(HERE, "..", "046-tech-stage-thresholds", "results.json")))
E2 = {"Fe": (2500, 5e4), "lens": (1e4, 1e6), "T2": (0.5e6, float(POP46["1770"]))}
NFe0 = lu(U["Fe"], *E2["Fe"]); NL0 = lu(U["lens"], *E2["lens"]); NT20 = lu(U["T2"], *E2["T2"])

def set_thr2(Mfe=1.0, Ml=1.0, Mt2=1.0):
    """046 の set_thr の E2 の枝と同じ式で、段ごとの倍率（配列可）を掛ける。"""
    nfe = NFe0 * Mfe; nl = np.maximum(NL0 * Ml, nfe); nt2 = NT20 * Mt2
    nt2e = np.maximum(nt2, nl)
    n3f = np.maximum(lu(U["f3"], *DEF["f3"]), nt2e); n3 = np.maximum(lu(U["t3"], *DEF["t3"]), n3f)
    T["N_Fe"] = nfe; T["N_lens"] = nl; T["N_T2"] = nt2; T["N_T2e"] = nt2e; T["N_3f"] = n3f; T["N_3"] = n3
    T["pop_ok"] = T["N_min"] < nt2

SV0 = {k: SRC[k]["sv"] for k in SRC}
def set_conn(mult=None, keys=None):
    """つながり：nmax を mult 倍（ntot で頭打ち）。mult=None で元に戻す。"""
    for k in SRC:
        if mult is None or (keys is not None and k not in keys): SRC[k]["sv"] = SV0[k]; continue
        sv = dict(SV0[k]); sv["nmax"] = np.minimum(sv["nmax"] * mult, sv["ntot"]); SRC[k]["sv"] = sv

KEYS = ("U3", "CASC", "BIO", "NUC_G", "NUC_R", "NAT", "AS")
def run(Mfe=1.0, Ml=1.0, Mt2=1.0, conn=None, conn_keys=None, nstar=False):
    set_thr2(Mfe, Ml, Mt2); set_conn(conn, conn_keys)
    D, hv, ph, info = evaluate({}, keys=("U3", "CASC", "BIO"))
    W = weighted(D, hv, anchors=("A_Ord", "A_XPT_S", "A_XPT_E"))
    out = {"U3_TAB": ph["U3"]["TAB"], "CASC_TAB": ph["CASC"]["TAB"], "BIO_TAB": ph["BIO"]["TAB"],
           "heavy_share": {k: float(hv[k].mean()) for k in KEYS},
           "w_TAB": {a: {u: W[a][u]["TAB"]["total"] for u in ("U1", "U2", "U3")} for a in W},
           "w_TAB_from_heavy": {a: {u: W[a][u]["TAB"]["from_heavy"] for u in ("U1", "U2", "U3")} for a in W},
           "thr_q": {k: [float(x) for x in np.percentile(T[k], [5, 50, 95])] for k in ("N_Fe", "N_lens", "N_T2e")}}
    if nstar: out["n_star"] = sweep({})["n_star"]
    set_conn(None); set_thr2()
    return out

# ---------- 確認：M=1 で 046 の E2 を再現 ----------
e2 = run(nstar=True)
ref = r46["B_variants"]["E2"]
R["check_046_E2"] = {"U3_TAB": e2["U3_TAB"], "ref": ref["U3_TAB"], "w": e2["w_TAB"]["A_Ord"]["U3"], "w_ref": ref["w_TAB"]["A_Ord"]["U3"],
                     "n_star": e2["n_star"], "n_star_ref": ref["n_star"],
                     "reproduced": bool(e2["U3_TAB"] == ref["U3_TAB"] and e2["w_TAB"]["A_Ord"]["U3"] == ref["w_TAB"]["A_Ord"]["U3"] and e2["n_star"] == ref["n_star"])}

# ---------- A. K の分布（Kline & Boyd 表1、046 の explore.py と同じ値と式） ----------
KB = [("Malekula", 1100, "low", 13), ("Tikopia", 1500, "low", 22), ("Santa Cruz", 3600, "low", 24),
      ("Yap", 4791, "high", 43), ("Lau Fiji", 7400, "high", 33), ("Trobriand", 8000, "high", 19),
      ("Chuuk", 9200, "high", 40), ("Manus", 13000, "low", 28), ("Tonga", 17500, "high", 55), ("Hawaii", 275000, "low", 71)]
lp = np.log([k[1] for k in KB]); lt = np.log([k[3] for k in KB]); hc = np.array([k[2] == "high" for k in KB], float)
def fit(i):
    return np.linalg.lstsq(np.c_[np.ones(len(i)), lp[i], hc[i]], lt[i], rcond=None)[0]
g = np.random.default_rng(SEED); bs = []
for _ in range(20000):
    i = g.integers(0, 10, 10)
    if np.ptp(lp[i]) > 0 and 0 < hc[i].sum() < 10:
        bb = fit(i)
        if bb[1] > 0.02: bs.append(np.exp(bb[2] / bb[1]))
bs = np.array(bs)
K = g.choice(bs, N)                     # 標本ごとの K
M0 = np.maximum(1.0, K)
CAP = {"Fe": 8e4, "lens": 3e7, "T2": 3e7}  # Iceland の上端、江戸の日本 約3,000万
R["A_K"] = {"point": float(np.exp(fit(np.arange(10))[2] / fit(np.arange(10))[1])), "n_boot": int(bs.size),
            "boot_q05_25_50_75_95_99": [float(x) for x in np.percentile(bs, [5, 25, 50, 75, 95, 99])],
            "P_K_lt_1": float((bs < 1).mean()), "M0_q": [float(x) for x in np.percentile(M0, [5, 25, 50, 75, 95, 99])]}

def prior(kind):
    a = {"P0": (1, 1, 1), "P1": (1, 1, 1), "P2": (0.5, 1, 1.5), "P3": (2, 2, 2)}[kind]
    mfe, ml, mt2 = M0 ** a[0], M0 ** a[1], M0 ** a[2]
    if kind != "P0":
        mfe = np.minimum(mfe, np.maximum(1.0, CAP["Fe"] / NFe0)); ml = np.minimum(ml, np.maximum(1.0, CAP["lens"] / NL0))
        mt2 = np.minimum(mt2, np.maximum(1.0, CAP["T2"] / NT20))
    return mfe, ml, mt2
PRI = {}; PQ = {}
for p in ("P0", "P1", "P2", "P3"):
    mfe, ml, mt2 = prior(p)
    PQ[p] = {"M_Fe": [float(x) for x in np.percentile(mfe, [5, 50, 95])], "M_lens": [float(x) for x in np.percentile(ml, [5, 50, 95])],
             "M_T2": [float(x) for x in np.percentile(mt2, [5, 50, 95])],
             "share_capped": {"Fe": float((M0 ** (0.5 if p == "P2" else (2 if p == "P3" else 1)) > mfe + 1e-12).mean()),
                              "T2": float((M0 ** (1.5 if p == "P2" else (2 if p == "P3" else 1)) > mt2 + 1e-12).mean())}}
    PRI[p] = run(mfe, ml, mt2, nstar=(p == "P1"))
R["A_prior_quantiles"] = PQ
R["B_priors"] = {"E2": e2, **PRI}

# ---------- C. 固定 M の格子 ----------
GRID = [1, 2, 3, 5, 7, 10, 15, 20, 30]
GR = {}
for m in GRID:
    o = run(float(m), float(m), float(m)); GR[m] = {"heavy": o["heavy_share"], "w_AOrd_U3": o["w_TAB"]["A_Ord"]["U3"], "U3_TAB": o["U3_TAB"],
                                                     "w_AXPTS_U2": o["w_TAB"]["A_XPT_S"]["U2"]}
# T2 だけに M を掛けた格子（NUC_G の分かれ目は T2 で決まる）
GRT2 = {m: run(1.0, 1.0, float(m))["heavy_share"]["NUC_G"] for m in GRID}
def crossing(xs, ys, thr):
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 < thr <= y1: return float(np.exp(np.interp(thr, [y0, y1], np.log([x0, x1]))))
    return None
nug = [GR[m]["heavy"]["NUC_G"] for m in GRID]
R["C_grid"] = {"M": GRID, "rows": GR, "NUC_G_heavy_T2only": GRT2,
               "M_cross_NUC_G_0.05": crossing(GRID, nug, 0.05), "M_cross_NUC_G_0.20": crossing(GRID, nug, 0.20),
               "M_cross_NUC_G_0.05_T2only": crossing(GRID, [GRT2[m] for m in GRID], 0.05),
               "P1_P_M_T2_ge_cross": None}
mt2_p1 = prior("P1")[2]
if R["C_grid"]["M_cross_NUC_G_0.05"]:
    R["C_grid"]["P1_P_M_T2_ge_cross"] = float((mt2_p1 >= R["C_grid"]["M_cross_NUC_G_0.05"]).mean())
# 閉じた豪州：NUC_G の nmax（豪州か次の大国）と M·N_T2e の比
nn = SV0["NUC_G"]["nmax"]
R["C_australia"] = {"nmax_q05_50_95": [float(x) for x in np.percentile(nn, [5, 50, 95])],
                    "M_needed_q05_50_95": [float(x) for x in np.percentile(nn / np.maximum(NT20, NL0), [5, 50, 95])]}

# ---------- D. つながり χ（120） ----------
# 逸脱（事前登録からの変更）：事前登録の近似（nmax を max(1,K)^χ 倍）は、χ=1・切り詰めなしで E2 に戻るはずの確認で −8.3% ずれた
# （nmax を動かすと M を掛けていない N_3f・N_3 やペストの N_c との比較まで動くため）。そこで閾値の倍率を直接 M_k^(1−χ) にする（全シナリオ共通）。
# 近似版の結果も記録として残す。
def run_conn_approx(mfe, ml, mt2, chi):
    o = run(mfe, ml, mt2, conn=M0 ** chi); return {"heavy": o["heavy_share"], "w_AOrd_U3": o["w_TAB"]["A_Ord"]["U3"]}
mfe, ml, mt2 = prior("P1")
CH = {}
for chi in (0.0, 0.25, 0.5, 0.75, 1.0):
    CH[str(chi)] = run(mfe ** (1 - chi), ml ** (1 - chi), mt2 ** (1 - chi))
f3, l3, t3 = prior("P3")
CH["P3_chi0.5"] = run(f3 ** 0.5, l3 ** 0.5, t3 ** 0.5)
R["D_connectivity"] = {k: {"heavy": v["heavy_share"], "w_AOrd_U3": v["w_TAB"]["A_Ord"]["U3"], "U3_TAB": v["U3_TAB"],
                           "w_AXPTS_U2": v["w_TAB"]["A_XPT_S"]["U2"], "w_from_heavy": v["w_TAB_from_heavy"]["A_Ord"]["U3"]} for k, v in CH.items()}
R["D_connectivity"]["check_chi1_eq_E2"] = {"w": CH["1.0"]["w_TAB"]["A_Ord"]["U3"], "ref": e2["w_TAB"]["A_Ord"]["U3"],
                                           "exact": bool(CH["1.0"]["w_TAB"]["A_Ord"]["U3"] == e2["w_TAB"]["A_Ord"]["U3"])}
R["D_connectivity_prereg_approx"] = {"P1_chi0.5": run_conn_approx(mfe, ml, mt2, 0.5), "P0_chi1": run_conn_approx(*prior("P0"), 1.0),
                                     "note": "nmax scaling approximation; failed the chi=1 == E2 check, not used"}

# ---------- 仮説 ----------
wE2 = e2["w_TAB"]["A_Ord"]["U3"]; w = {p: PRI[p]["w_TAB"]["A_Ord"]["U3"] for p in PRI}
xc = R["C_grid"]["M_cross_NUC_G_0.05"]
casc0 = CH["0.0"]["heavy_share"]["CASC"]; casc5 = CH["0.5"]["heavy_share"]["CASC"]
R["hypotheses"] = {
 "H59a": {"value": [w["P1"], wE2, w["P1"] / wE2], "rule": "P1 weighted A_Ord U3 in [0.94, 1.00] x E2", "supported": bool(0.94 <= w["P1"] / wE2 <= 1.0)},
 "H59b": {"value": PRI["P1"]["heavy_share"]["NUC_G"], "rule": "P1 NUC_G heavy share < 0.05", "supported": bool(PRI["P1"]["heavy_share"]["NUC_G"] < 0.05)},
 "H59c": {"value": xc, "rule": "fixed M where NUC_G heavy share crosses 0.05 is >= 10", "supported": bool(xc is None or xc >= 10)},
 "H59d": {"value": [w["P1"], w["P0"], abs(w["P1"] / w["P0"] - 1)], "rule": "|P1/P0 - 1| < 0.01", "supported": bool(abs(w["P1"] / w["P0"] - 1) < 0.01)},
 "H59e": {"value": [w["P2"], w["P1"], abs(w["P2"] / w["P1"] - 1)], "rule": "|P2/P1 - 1| < 0.05", "supported": bool(abs(w["P2"] / w["P1"] - 1) < 0.05)},
 "H59f": {"value": [casc5, casc0, casc5 / casc0], "rule": "chi=0.5 CASC heavy <= 0.7 x chi=0", "supported": bool(casc5 <= 0.7 * casc0)},
 "H59g": {"value": [PRI["P1"]["n_star"], e2["n_star"]], "rule": "P1 n* within +-10% of E2", "supported": bool(PRI["P1"]["n_star"] is not None and abs(PRI["P1"]["n_star"] / e2["n_star"] - 1) <= 0.10)},
}

# ---------- 図 ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].hist(np.log10(np.clip(bs, 0.05, 1e3)), bins=80, color="C0", alpha=0.6, density=True, label="K (Kline & Boyd bootstrap)")
for p, c in (("P1", "C1"), ("P3", "C3")):
    ax[0].hist(np.log10(prior(p)[2]), bins=80, histtype="step", color=c, density=True, label=f"M_T2 under {p}")
ax[0].axvline(np.log10(2.587), color="k", ls=":", label="K point 2.6")
ax[0].set_xlabel("log10 multiplier"); ax[0].set_ylabel("density"); ax[0].legend(fontsize=7); ax[0].set_xlim(-1.3, 2.5)
ax[0].set_title("contact multiplier K and closed-group multiplier M_T2 (capped by Edo Japan 30M)", fontsize=9)
cases = [("Tasmania (closed 8,000 yr; lost bone tools? debated)", 3e3, 1.5e4, "C3"), ("Norse Greenland (Fe fail)", 2e3, 3e3, "C3"),
         ("Iceland (Fe self-sufficient)", 3e4, 8e4, "C2"), ("Confederacy (Fe kept, metal scarce)", 5e6, 9.5e6, "C1"),
         ("North Korea (factories idle, spare parts)", 2e7, 2.6e7, "C1"), ("Edo Japan (2 ships/yr; telescopes 4 gens)", 3e7, 3.2e7, "C2"),
         ("NUC_G largest group (Australia etc.) 5-95%", *R["C_australia"]["nmax_q05_50_95"][::2], "k")]
for i, (lab, lo, hi, c) in enumerate(cases):
    ax[1].plot([lo, hi * (1.05 if hi == lo else 1)], [i, i], color=c, lw=7, solid_capstyle="butt"); ax[1].text(hi * 1.3, i, lab, va="center", fontsize=7)
q = R["B_priors"]
for j, (p, col) in enumerate((("E2", "#888"), ("P1", "C1"), ("P3", "C3"))):
    lo, md, hi = q[p]["thr_q"]["N_T2e"]; yy = len(cases) + 0.5 + j * 0.8
    ax[1].plot([lo, hi], [yy, yy], color=col, lw=4); ax[1].plot(md, yy, "o", color=col); ax[1].text(hi * 1.3, yy, f"N_T2e x M ({p}) 5-50-95%", va="center", fontsize=7)
ax[1].set_xscale("log"); ax[1].set_xlim(1e3, 1e10); ax[1].set_yticks([]); ax[1].invert_yaxis(); ax[1].set_xlabel("population (log)")
ax[1].set_title("closed / semi-closed societies vs T2 threshold under closed-group multiplier", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_M_prior_cases.png"), dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for k, c in (("NUC_G", "k"), ("CASC", "C3"), ("U3", "C0"), ("BIO", "C2")):
    ax[0].plot(GRID, [GR[m]["heavy"][k] for m in GRID], "o-", color=c, label=f"{k} heavy share")
ax[0].plot(GRID, [GRT2[m] for m in GRID], "k--", label="NUC_G, M on T2 only")
ax[0].axhline(0.05, color="grey", ls=":"); ax[0].axvspan(R["A_K"]["boot_q05_25_50_75_95_99"][0], R["A_K"]["boot_q05_25_50_75_95_99"][4], color="C1", alpha=0.12, label="K bootstrap 90%")
ax[0].set_xscale("log"); ax[0].set_xlabel("fixed closed-group multiplier M (all stages)"); ax[0].set_ylabel("share restarting at T1 or below"); ax[0].legend(fontsize=7)
ax2 = ax[0].twinx(); ax2.plot(GRID, [GR[m]["w_AOrd_U3"] for m in GRID], "s:", color="C1"); ax2.set_ylabel("weighted A_Ord U3 tablet value (orange)")
ax[0].set_title("fixed-M grid on E2 thresholds", fontsize=9)
chis = [0.0, 0.25, 0.5, 0.75, 1.0]
for k, c in (("NUC_G", "k"), ("CASC", "C3"), ("U3", "C0")):
    ax[1].plot(chis, [CH[str(x)]["heavy_share"][k] for x in chis], "o-", color=c, label=f"{k} heavy share")
ax3 = ax[1].twinx(); ax3.plot(chis, [CH[str(x)]["w_TAB"]["A_Ord"]["U3"] for x in chis], "s:", color="C1"); ax3.axhline(wE2, color="C1", lw=0.8, ls="--")
ax3.set_ylabel("weighted A_Ord U3 (orange; dashed = E2 open)")
ax[1].set_xlabel("connectivity chi between surviving groups (P1 prior)"); ax[1].legend(fontsize=7); ax[1].set_title("120: connectivity offsets the closed-group penalty", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_grid_connectivity.png"), dpi=120); plt.close()

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in R if k not in ("B_priors",)}, ensure_ascii=False, indent=1, default=float))
print(json.dumps({p: {"w_AOrd_U3": v["w_TAB"]["A_Ord"]["U3"], "w_heavy": v["w_TAB_from_heavy"]["A_Ord"]["U3"], "U3_TAB": v["U3_TAB"], "heavy": v["heavy_share"],
                      "w_XPTS_U2": v["w_TAB"]["A_XPT_S"]["U2"], "n_star": v.get("n_star"), "N_T2e_q": v["thr_q"]["N_T2e"]} for p, v in R["B_priors"].items()}, indent=1))
print("runtime", round(R["runtime_seconds"], 1))
