"""研究049：孤立した生き残りでの段の喪失（open_problems 136）と h* の閾値への依存（138）。
事前登録：PREREGISTRATION.md（2026-10-18T20:46Z）。
047 の sim.py を「# ---------- C. 固定 M の格子」の手前まで exec して（中で 046・045・044・042・040・039・038 を exec）、P1 の道具を借りる（048 と同じ）。
048 の results.json（A_cases）から交易の続いた群の L_top を読む。新しい乱数は seed 20261049 の N・0.5 の振り分け・事後の標本だけ。numpy＋matplotlib のみ、ネット接続なし。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261049
R = {"study": "049-isolation-stage-loss", "seed": SEED}
EX = json.load(open(os.path.join(HERE, "sources", "excerpts.json"), encoding="utf8"))
assert all(e["found_in_fetched_text"] for e in EX["excerpts"])
R["n_excerpts_verified"] = len(EX["excerpts"])
g = np.random.default_rng(SEED)

# ================= A. 参照の集合（符号は【推測】、根拠は excerpts.json の引用番号） =================
# name, group, L, env, N(lo,hi) or None, T_iso (yr) or None, exo, after, evidence, note
# group: "iso" 孤立した生き残り / "ended" 孤立して絶えたか放棄 / "control" 孤立していない対照 / "endo" 内生の孤立 / "unk" χ不明
NA = None
CASES = [
 ("Tasmania", "iso", 0.5, 0, (3000, 15000), 8000, 1, 1, [0, 1, 2, 3, 5], "bone tools and scaled fish dropped ~4,000 yr ago; significance debated (economic alternative)"),
 ("Moriori (Chatham)", "iso", 1.0, 1, (1600, 2000), 340, 1, 1, [6, 7, 8, 9, 10], "crops unsuitable -> hunter-gatherer; no trees for ocean-going craft; invented waka korari"),
 ("Inughuit (Polar Inuit)", "iso", 1.0, 0, (100, 200), 170, 1, 1, [11, 12, 13, 14, 15], "cut off ~17th c. by cooling; lost kayak/umiak, bows, leisters; reintroduced by Baffin Inuit"),
 ("Easter Island", "iso", 0.5, 0, (2000, 3000), 570, 1, 1, [42, 43], "048 coding (L_top 0.5); colonised 12th c., 2,000-3,000 in 1722"),
 ("Hawaii (pre-1778)", "iso", 0.0, 0, (300000, 300000), 578, 1, 1, [27, 28, 29], "steady growth of population and chiefdoms; ~300,000 in 1770s"),
 ("Maori (pre-1769)", "iso", 0.0, 0, (100000, 110000), 419, 1, 1, [30, 31, 32], "several centuries in isolation; Classic period carved war canoes"),
 ("Andamanese", "iso", 0.0, 0, (7000, 7000), 3000, 1, 1, [33, 34, 35], "isolation thousands of years; no fire-making in 19th c. (not stated as lost)"),
 ("Sentinelese", "iso", 0.0, 0, (80, 100), 3000, 1, 1, [36, 37], "bows and arrows kept; censuses 82/100/98"),
 ("Guanches", "iso", 0.0, 0, NA, NA, 1, 1, [38, 39], "relative isolation, sporadic contacts; no loss stated in text"),
 ("Henderson (Polynesian)", "ended", NA, 0, NA, 600, 0.5, NA, [18, 19], "abandoned by 1600s; may relate to resource exhaustion and Pitcairn"),
 ("Pitcairn (Polynesian)", "ended", NA, 0, NA, NA, 0.5, NA, [18], "abandoned 17th or early 18th c."),
 ("Norfolk (Polynesian)", "ended", NA, 0, NA, NA, 1, NA, [24], "relative isolation and poor horticultural environment"),
 ("Raoul (Polynesian)", "ended", NA, 0, NA, NA, 1, NA, [25], "occupation relatively brief"),
 ("Nihoa", "ended", NA, 0, NA, NA, 1, NA, [26], "abandonment may be tied to deforestation"),
 ("Kangaroo Island", "ended", NA, 0, NA, NA, 1, NA, [22], "disappeared after island formation"),
 ("Flinders Island", "ended", NA, 0, NA, NA, 1, NA, [23], "until ~4,500 yr ago; thirst and hunger after El Nino shift"),
 ("King Island / Furneaux", "ended", NA, 0, NA, NA, 1, NA, [4], "stranded by rising waters and later died out"),
 ("Norse Greenland", "ended", NA, 0, (2000, 10000), 450, 1, NA, [44], "048: trade dried up, abandoned"),
 ("Tikopia", "control", 0.0, 0, (1000, 1200), NA, NA, NA, [40, 41], "frequent canoe voyaging; small but connected"),
 ("Mangareva", "endo", 0.0, 0, NA, NA, 0, NA, [20, 21], "deforestation -> economy collapse; tech loss not stated"),
 ("Dorset", "unk", 1.0, 0, NA, NA, NA, NA, [16, 17], "lost bow and drill; isolation not stated"),
]
R["A_cases"] = [{"name": c[0], "group": c[1], "L": c[2], "env": c[3], "N": c[4], "T_iso": c[5], "exo": c[6], "after": c[7],
                 "evidence": [EX["excerpts"][i]["url"] for i in c[8]], "note": c[9]} for c in CASES]
ISO = [c for c in CASES if c[1] == "iso"]; END = [c for c in CASES if c[1] == "ended"]
L_iso = np.array([c[2] for c in ISO])
R["A_counts"] = {"n_iso_surv": len(ISO), "n_ended": len(END), "n_control": sum(c[1] == "control" for c in CASES),
                 "L_iso_point": float(L_iso.mean()), "ended_share_of_isolated": len(END) / (len(END) + len(ISO))}

def jeff(v, NB=20000):
    amb = (v == 0.5); x = (v == 1).sum() + g.binomial(1, 0.5, (NB, amb.sum())).sum(1)
    p = g.beta(0.5 + x, 0.5 + v.size - x)
    return {"n": int(v.size), "point": float(v.mean()), "post_mean": float(p.mean()), "post_q05_50_95": [float(q) for q in np.percentile(p, [5, 50, 95])]}, p
A048 = json.load(open(os.path.join(HERE, "..", "048-unknown-depth-h", "results.json"), encoding="utf8"))["A_cases"]
myc = [c for c in A048 if c["name"].startswith("Mycenaean")][0]["L_top"]
conn = np.array([c["L_top"] for c in A048 if (not c["extinct"]) and c["chi"] is not None and c["chi"] >= 0.5])
P_iso, d_iso = jeff(L_iso)
P_iso_myc, _ = jeff(np.append(L_iso, myc))
P_iso_noenv, _ = jeff(np.array([0.0 if c[3] else c[2] for c in ISO]))
P_iso_and, _ = jeff(np.array([0.5 if c[0] == "Andamanese" else c[2] for c in ISO]))
P_conn, d_conn = jeff(conn)
R["A_posteriors"] = {"iso": P_iso, "iso_plus_Mycenae": P_iso_myc, "iso_env_as_0": P_iso_noenv, "iso_Andaman_0.5": P_iso_and, "conn_048_chi_ge_half": P_conn}
lossy = [c for c in ISO if c[2] > 0]
R["D_reverse_causality"] = {"n_iso_with_loss": len(lossy), "exo_and_after": int(sum(c[6] == 1 and c[7] == 1 for c in lossy)),
                            "share": float(np.mean([c[6] == 1 and c[7] == 1 for c in lossy])),
                            "env_attributed": [c[0] for c in lossy if c[3]]}

# ================= B. ロジスティック回帰（格子の事後、N と 0.5 を標本ごとに振る） =================
REG = [c for c in ISO if c[4] is not None]
def reg_posterior(cases, K=200, use_T=False, step=0.05, lim=8.0, override=None):
    ys = np.array([c[2] for c in cases]); lo = np.array([c[4][0] for c in cases], float); hi = np.array([c[4][1] for c in cases], float)
    if override: ys = np.array([override.get(c[0], c[2]) for c in cases])
    T = np.array([c[5] for c in cases], float)
    ax = np.arange(-lim, lim + 1e-9, step)
    if use_T:
        A, B, C = np.meshgrid(ax, ax, ax, indexing="ij"); A = A.ravel(); B = B.ravel(); C = C.ravel()
    else:
        A, B = np.meshgrid(ax, ax, indexing="ij"); A = A.ravel(); B = B.ravel(); C = np.zeros_like(A)
    lp0 = -0.5 * (A ** 2 + B ** 2 + C ** 2) / 2.5 ** 2
    post = np.zeros_like(A)
    for _ in range(K):
        N = np.exp(np.log(lo) + g.uniform(size=lo.size) * (np.log(hi) - np.log(lo)))
        y = np.where(ys == 0.5, g.binomial(1, 0.5, ys.size), ys)
        x = np.log10(N) - 4; z = np.log10(T) - 3
        eta = A[:, None] + B[:, None] * x[None, :] + C[:, None] * z[None, :]
        ll = (y[None, :] * eta - np.logaddexp(0, eta)).sum(1) + lp0
        w = np.exp(ll - ll.max()); post += w / w.sum()
    post /= K
    idx = g.choice(A.size, 4000, p=post)
    return A[idx], B[idx], C[idx]
def pred(a, b, n):
    return 1 / (1 + np.exp(-(a[:, None] + b[:, None] * (np.log10(np.atleast_1d(n))[None, :] - 4))))
REGS = {"main": reg_posterior(REG), "env_as_0": reg_posterior(REG, override={"Moriori (Chatham)": 0.0}),
        "Andaman_0.5": reg_posterior(REG, override={"Andamanese": 0.5}),
        "with_T": reg_posterior(REG, K=40, use_T=True, step=0.2, lim=6.0)}
NGRID = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
RB = {}
for k, (a, b, c) in REGS.items():
    pr = pred(a, b, NGRID)
    RB[k] = {"P_b_lt_0": float((b < 0).mean()), "b_q05_50_95": [float(q) for q in np.percentile(b, [5, 50, 95])],
             "pred_mean": {f"{n:.0e}": float(pr[:, i].mean()) for i, n in enumerate(NGRID)},
             "pred_q95": {f"{n:.0e}": float(np.percentile(pr[:, i], 95)) for i, n in enumerate(NGRID)}}
    if k == "with_T": RB[k]["P_c_gt_0"] = float((c > 0).mean())
R["B_regression"] = RB

# ================= 047 の P1 の道具 =================
P47 = os.path.abspath(os.path.join(HERE, "..", "047-closed-group-multiplier", "sim.py"))
src = open(P47, encoding="utf8").read()
H = {"__file__": P47, "__name__": "sim047"}
exec(compile(src[:src.index("# ---------- C. 固定 M の格子")], P47, "exec"), H)
set_thr2 = H["set_thr2"]; prior = H["prior"]; evaluate = H["evaluate"]; weighted = H["weighted"]; PRI = H["PRI"]; Nn = H["N"]
SRC = evaluate.__globals__["SRC"]
ANCH = ("A_Ord", "A_XPT_S")
def ev(mfe, ml, mt2):
    set_thr2(mfe, ml, mt2); D, hv, ph, info = evaluate({}, keys=("U3", "CASC", "BIO")); set_thr2()
    return D, hv
def at_h(W, h, a="A_Ord"):
    t = h * W[a]["U1"]["TAB"]["total"] + (1 - h) * W[a]["U2"]["TAB"]["total"]
    f = h * W[a]["U1"]["TAB"]["from_heavy"] + (1 - h) * W[a]["U2"]["TAB"]["from_heavy"]
    return t, f
def hstar(W, a="A_Ord"):
    t1, f1 = W[a]["U1"]["TAB"]["total"], W[a]["U1"]["TAB"]["from_heavy"]; t2, f2 = W[a]["U2"]["TAB"]["total"], W[a]["U2"]["TAB"]["from_heavy"]
    if f2 / t2 >= 0.5: return 0.0
    if f1 / t1 < 0.5: return None
    return float((0.5 * t2 - f2) / ((f1 - f2) - 0.5 * (t1 - t2)))
P1 = prior("P1"); D1, HV1 = ev(*P1); W1 = weighted(D1, HV1, anchors=ANCH)
HS1 = hstar(W1)
R["check_048"] = {"w_AOrd_U3": W1["A_Ord"]["U3"]["TAB"]["total"], "ref_047_P1": PRI["P1"]["w_TAB"]["A_Ord"]["U3"],
                  "h_star": HS1, "ref_048_h_star": json.load(open(os.path.join(HERE, "..", "048-unknown-depth-h", "results.json")))["B_h_star"]["A_Ord"]}
R["check_048"]["reproduced"] = bool(R["check_048"]["w_AOrd_U3"] == R["check_048"]["ref_047_P1"] and HS1 == R["check_048"]["ref_048_h_star"])

# ================= E. 孤立の罰を生き残りの大きさに写す =================
a, b, _ = REGS["main"]
sub = g.choice(Nn, 20000, replace=False)
EIS = {}
for k in ("U3", "CASC", "BIO"):
    n = np.maximum(SRC[k]["sv"]["nmax"][sub], 1.0)
    pr = pred(a, b, n)            # 4000 x 20000
    e_draw = pr.mean(1)
    EIS[k] = {"nmax_q05_50_95": [float(q) for q in np.percentile(n, [5, 50, 95])], "E_P_loss_mean": float(e_draw.mean()),
              "E_P_loss_q05_95": [float(q) for q in np.percentile(e_draw, [5, 95])], "_draws": e_draw}
for k2 in ("env_as_0", "Andaman_0.5"):
    a2, b2, _ = REGS[k2]; n = np.maximum(SRC["U3"]["sv"]["nmax"][sub], 1.0)
    EIS["U3_" + k2] = {"E_P_loss_mean": float(pred(a2, b2, n).mean())}
R["E_iso_on_model"] = {k: {kk: vv for kk, vv in v.items() if kk != "_draws"} for k, v in EIS.items()}
eU3 = EIS["U3"]["_draws"]; hc = g.choice(d_conn, eU3.size)
SV = {}
for s in (0.0, 0.25, 0.5, 1.0):
    for lab, iso_draw, iso_pt in (("size_aware", eU3, EIS["U3"]["E_P_loss_mean"]), ("naive", g.choice(d_iso, eU3.size), P_iso["post_mean"])):
        hd = s * iso_draw + (1 - s) * hc; hm = float(hd.mean()); t, f = at_h(W1, hm)
        SV[f"{lab}_s{s}"] = {"h_mean": hm, "h_q05_95": [float(q) for q in np.percentile(hd, [5, 95])], "P_h_gt_hstar": float((hd > HS1).mean()),
                            "w_AOrd": t, "share_from_heavy": f / t, "ratio_to_U3": t / W1["A_Ord"]["U3"]["TAB"]["total"],
                            "w_AXPTS": at_h(W1, hm, "A_XPT_S")[0]}
R["E_h_by_s"] = SV

# ================= F. h* の閾値の版への依存（138） =================
HSV = {"P1": HS1}
D0, HV0 = ev(1.0, 1.0, 1.0); HSV["E2 (M=1)"] = hstar(weighted(D0, HV0, anchors=ANCH))
for p in ("P0", "P2", "P3"):
    Dp, HVp = ev(*prior(p)); HSV[p] = hstar(weighted(Dp, HVp, anchors=ANCH))
for M in (3.0, 10.0, 30.0):
    Dm, HVm = ev(M, M, M); HSV[f"fixed M={M:g}"] = hstar(weighted(Dm, HVm, anchors=ANCH))
R48 = json.load(open(os.path.join(HERE, "..", "048-unknown-depth-h", "results.json")))
HSV["048 hist chi (CASC only)"] = R48["C_chi_hist"]["CASC_only"]["h_star_AOrd"]
HSV["048 hist chi (all)"] = R48["C_chi_hist"]["all"]["h_star_AOrd"]
vals = [v for v in HSV.values() if v is not None]
R["F_h_star_versions"] = {"h_star": HSV, "min": min(vals), "max": max(vals),
                          "h_readings_048": {k: R48["B_by_reading"][k]["A_Ord"]["h_mean"] for k in ("V1", "V2", "V3", "V4")}}
# 判断の分解：h の読み（V1〜V4）× 閾値の版の格子で「h > h*」かどうか
grid = {}
for vk, hv_ in R["F_h_star_versions"]["h_readings_048"].items():
    grid[vk] = {pk: (None if hs is None else bool(hv_ > hs)) for pk, hs in HSV.items()}
R["F_decision_grid"] = grid

# ================= 仮説 =================
eu3 = EIS["U3"]["E_P_loss_mean"]
R["hypotheses"] = {
 "H61a": {"value": P_iso["point"], "rule": "isolated survivors L point >= 0.40", "supported": bool(P_iso["point"] >= 0.40)},
 "H61b": {"value": [RB["main"]["P_b_lt_0"], RB["main"]["pred_mean"]["1e+05"]], "rule": "P(b<0)>=0.90 and P(L|N=1e5)<0.15",
          "supported": bool(RB["main"]["P_b_lt_0"] >= 0.90 and RB["main"]["pred_mean"]["1e+05"] < 0.15)},
 "H61c": {"value": [R["A_counts"]["ended_share_of_isolated"], P_iso["point"]], "rule": "ended share >= L mean of survivors",
          "supported": bool(R["A_counts"]["ended_share_of_isolated"] >= P_iso["point"])},
 "H61d": {"value": R["D_reverse_causality"]["share"], "rule": ">=80% of isolated losses exogenous and after isolation",
          "supported": bool(R["D_reverse_causality"]["share"] >= 0.8)},
 "H61e": {"value": [eu3, HS1], "rule": "E_U3[P(L|nmax)] < h*(P1)", "supported": bool(eu3 < HS1)},
 "H61f": {"value": [R["F_h_star_versions"]["min"], R["F_h_star_versions"]["max"]], "rule": "all h* versions in (0.028, 0.195)",
          "supported": bool(R["F_h_star_versions"]["min"] > 0.028 and R["F_h_star_versions"]["max"] < 0.195 and None not in HSV.values())},
}

# ================= 図 =================
fig, ax = plt.subplots(1, 2, figsize=(13, 5.0))
ns = np.logspace(1.5, 7.5, 200); pr = pred(a, b, ns)
ax[0].fill_between(ns, np.percentile(pr, 5, 0), np.percentile(pr, 95, 0), color="C0", alpha=0.2, label="logistic 90% band")
ax[0].plot(ns, pr.mean(0), color="C0", label="posterior mean P(loss | N)")
for c in REG:
    lo, hi = c[4]; gm = np.sqrt(lo * hi)
    ax[0].errorbar(gm, c[2], xerr=[[gm - lo], [hi - gm]], fmt="s", color="C3" if c[3] else "k", ms=6, capsize=3)
    ax[0].annotate(c[0].split(" (")[0], (gm, c[2]), textcoords="offset points", xytext=(-30 if c[0] == "Easter Island" else 4, (14 if c[0] == "Easter Island" else 6) if c[2] < 0.9 else -12), fontsize=7)
c = [x for x in CASES if x[0] == "Tikopia"][0]
ax[0].plot(np.sqrt(c[4][0] * c[4][1]), -0.06, "^", color="C2", label="Tikopia (connected control)")
ax[0].plot(np.sqrt(2000 * 10000), 1.08, "x", color="grey", label="Norse Greenland (ended)")
q = EIS["U3"]["nmax_q05_50_95"]; ax[0].axvspan(q[0], q[2], color="C1", alpha=0.1, label="model U3 survivor nmax 5-95%")
ax[0].set_xscale("log"); ax[0].set_ylim(-0.12, 1.15); ax[0].set_xlabel("population of isolated group N"); ax[0].set_ylabel("documented technology loss L")
ax[0].legend(fontsize=7, loc="center right"); ax[0].set_title("isolated survivors: loss vs size (red = loss attributed to environment)", fontsize=9)
ss = [0.0, 0.25, 0.5, 1.0]
for lab, col in (("size_aware", "C0"), ("naive", "C3")):
    ax[1].plot(ss, [SV[f"{lab}_s{s}"]["h_mean"] for s in ss], "o-", color=col, label=f"h ({lab.replace('_', '-')})")
names = list(HSV.keys()); hv_ = [HSV[k] for k in names]
for i, (k, v) in enumerate(zip(names, hv_)):
    if v is not None: ax[1].axhline(v, color="grey", lw=0.6, ls=":")
ax[1].axhline(HS1, color="k", ls="--", label=f"h* P1 = {HS1:.3f}")
ax[1].axhspan(min(vals), max(vals), color="grey", alpha=0.15, label=f"h* over {len(vals)} threshold versions {min(vals):.3f}-{max(vals):.3f}")
for vk, col in (("V2", "C1"), ("V4", "C2")):
    ax[1].axhline(R["F_h_star_versions"]["h_readings_048"][vk], color=col, lw=1, label=f"048 {vk} h = {R['F_h_star_versions']['h_readings_048'][vk]:.3f}")
ax[1].set_xlabel("s = share of survivors fully isolated"); ax[1].set_ylabel("h (unknown anthropogenic restarting at T0)")
ax[1].legend(fontsize=7); ax[1].set_title("136+138: h with isolation penalty vs h* across threshold versions", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_isolation_h.png"), dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(8, 4.2))
y = np.arange(len(names))
ax.barh(y, [0 if v is None else v for v in hv_], color="C0")
for i, v in enumerate(hv_): ax.text((v or 0) + 0.002, i, "none" if v is None else f"{v:.3f}", va="center", fontsize=8)
for vk, col in (("V1", "C3"), ("V2", "C1"), ("V3", "C4"), ("V4", "C2")):
    ax.axvline(R["F_h_star_versions"]["h_readings_048"][vk], color=col, ls="--", lw=1, label=f"048 {vk}")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.set_xlabel("h* (A_Ord: half of value from heavy collapse)"); ax.legend(fontsize=7)
ax.set_title("138: h* by threshold version vs the four h readings", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_hstar_versions.png"), dpi=120); plt.close()

R["runtime_s"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("check_048", "A_counts", "A_posteriors", "D_reverse_causality", "B_regression", "E_iso_on_model", "E_h_by_s", "F_h_star_versions", "F_decision_grid", "hypotheses")}, ensure_ascii=False, indent=1, default=float))
print("runtime", R["runtime_s"])
