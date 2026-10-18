"""研究048：未知の人為リスクの再出発の深さ h（open_problems 117）と、シナリオごとのつながり χ（135）。
事前登録：PREREGISTRATION.md（2026-10-18T18:44Z）。
047 の sim.py を「# ---------- C. 固定 M の格子」の手前まで exec して（中で 046・045・044・042・040・039・038 を exec）、P1 の閾値の道具を借りる。
新しい乱数は seed 20261048 の符号の振り分け・Beta の事後・χ の割り当てだけ。numpy＋matplotlib のみ、ネット接続なし。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261048
R = {"study": "048-unknown-depth-h", "seed": SEED}
EX = json.load(open(os.path.join(HERE, "sources", "excerpts.json"), encoding="utf8"))
assert all(e["found_in_fetched_text"] for e in EX["excerpts"])
R["n_excerpts_verified"] = len(EX["excerpts"])

# ================= A. 参照の集合（符号は【推測】、根拠は excerpts.json の引用番号） =================
# name, literate, L_top, L_multi, chi (None=不明), metal_lost, extinct, evidence idx, note
NA = None
CASES = [
 ("Mycenaean Greece (LBA)", 1, 1.0, 1.0, 0.0, 0, 0, [3, 4, 5, 6, 11], "Linear B ceased ~400 yr; monumental stone, wall painting gone; vital trade links lost"),
 ("Hittite Anatolia (LBA)", 1, 0.5, 0.0, 0.5, 0, 0, [12, 13], "cuneiform state ended; hieroglyphs sparse ~2 centuries then resume in successor states"),
 ("Levant / Ugarit (LBA)", 1, 0.0, 0.0, 1.0, 0, 0, [8, 9, 10], "literacy severely reduced; tin and Egypt trade continued"),
 ("Egypt New Kingdom -> TIP", 1, 0.0, 0.0, 0.5, 0, 0, [], "survived in weakened form"),
 ("Indus Valley (late Harappan)", 1, 1.0, 0.5, 0.5, 0, 0, [14, 15, 16], "script lost, cities died out; new faience/glass crafts; long-distance trade decreased"),
 ("Akkadian Empire", 1, 0.0, 0.0, NA, 0, 0, [1, 39], "material culture continues into Ur III"),
 ("Old Kingdom -> FIP Egypt", 1, 0.0, 0.0, 0.5, 0, 0, [38], "decline in international relations"),
 ("Classic Maya (southern lowlands)", 1, 0.5, 0.0, 0.5, 0, 0, [0, 17, 18, 19], "monumental inscriptions ceased in south; writing continued in north; coastal trade"),
 ("Western Roman Empire", 1, 0.0, 0.5, 0.5, 0, 0, [0, 22, 23, 24], "literacy decreased, moved to monasteries; papyrus import ceased"),
 ("Sub-Roman Britain", 1, 0.0, 1.0, 0.5, 0, 0, [20, 21, 25, 26, 27, 28], "Latin literacy kept; stone building, coinage ended; material below pre-Roman Iron Age; Med. pottery at Tintagel"),
 ("Byzantine Dark Ages", 1, 0.0, 0.0, 0.5, 0, 0, [29], "building ceased; education declined but persisted"),
 ("Easter Island", 0, 0.5, 0.0, 0.0, 0, 0, [0, 31, 32], "statue toppling after 1770 (entangled with contact); remote"),
 ("Olmec heartland", 0, 0.0, 0.0, NA, 0, 0, [35], "successor cultures within a few hundred years"),
 ("Tiwanaku", 0, 0.0, 0.0, NA, 0, 0, [36], "collapse ~AD 1000, cause debated"),
 ("Angkor", 1, 0.0, 0.0, NA, 0, 0, [37], "abandoned in 15th c.; Khmer state continued"),
 ("Cahokia / Mississippian", 0, 0.0, 0.0, NA, 0, 0, [2, 40], "no writing; Mississippian culture continued elsewhere"),
 ("Ancestral Puebloans", 0, 0.0, 0.0, 1.0, 0, 0, [2, 30], "pottery continued, widely traded"),
 ("Norse Greenland", 1, NA, NA, 0.0, NA, 1, [33, 34], "abandoned/extinct; trade dried up"),
]
R["A_cases"] = [{"name": c[0], "literate": c[1], "L_top": c[2], "L_multi": c[3], "chi": c[4], "metal_lost": c[5], "extinct": c[6],
                 "evidence": [EX["excerpts"][i]["url"] for i in c[7]], "note": c[8]} for c in CASES]
SURV = [c for c in CASES if not c[6]]
n = len(SURV)
Ltop = np.array([c[2] for c in SURV]); Lmul = np.array([c[3] for c in SURV])
chi = np.array([np.nan if c[4] is None else c[4] for c in SURV]); lit = np.array([c[1] for c in SURV])
brit = np.array([c[0] == "Sub-Roman Britain" for c in SURV]); metal = np.array([c[5] for c in SURV], float)
V3 = ((Lmul == 1) & (chi == 0)).astype(float)
pts = {"V1": float(Ltop.mean()), "V2": float(Lmul.mean()), "V3": float(V3.mean()), "V4": float(metal.mean())}

g = np.random.default_rng(SEED); NB = 20000
def posterior(vec, mask=None):
    v = vec if mask is None else vec[mask]; m = v.size
    amb = (v == 0.5); x = (v == 1).sum() + g.binomial(1, 0.5, (NB, amb.sum())).sum(1)
    p = g.beta(0.5 + x, 0.5 + m - x)
    return {"n": int(m), "point": float(v.mean()), "x_flip_q05_50_95": [float(q) for q in np.percentile(x / m, [5, 50, 95])],
            "post_mean": float(p.mean()), "post_q05_50_95": [float(q) for q in np.percentile(p, [5, 50, 95])], "_draws": p}
POST = {"V1": posterior(Ltop), "V2": posterior(Lmul), "V3": posterior(V3), "V4": posterior(metal),
        "V1_noBritain": posterior(Ltop, ~brit), "V2_noBritain": posterior(Lmul, ~brit),
        "V1_literate": posterior(Ltop, lit == 1), "V2_literate": posterior(Lmul, lit == 1)}
R["A_h_estimates"] = {k: {kk: vv for kk, vv in v.items() if kk != "_draws"} for k, v in POST.items()}
R["A_h_points"] = pts
R["A_counts"] = {"n_cases": len(CASES), "n_surviving": n, "n_extinct": int(sum(c[6] for c in CASES)), "n_metal_lost": int(metal.sum())}

# χ と L_top（135）
known = ~np.isnan(chi)
lo_g = known & (chi == 0); hi_g = known & (chi >= 0.5)
diff = float(Ltop[lo_g].mean() - Ltop[hi_g].mean())
perm = []
lk = Ltop[known]; nlo = int(lo_g.sum())
for _ in range(20000):
    s = g.permutation(lk); perm.append(s[:nlo].mean() - s[nlo:].mean())
perm = np.array(perm)
R["A_chi_vs_loss"] = {"n_known": int(known.sum()), "n_chi0": nlo, "n_chi_ge_half": int(hi_g.sum()),
                      "Ltop_chi0": float(Ltop[lo_g].mean()), "Ltop_chi_ge_half": float(Ltop[hi_g].mean()), "diff": diff,
                      "perm_p_one_sided": float((perm >= diff - 1e-12).mean()),
                      "Lmulti_chi0": float(Lmul[lo_g].mean()), "Lmulti_chi_ge_half": float(Lmul[hi_g].mean()),
                      "chi_values_known_incl_extinct": sorted([c[4] for c in CASES if c[4] is not None]),
                      "chi_mean_known_incl_extinct": float(np.mean([c[4] for c in CASES if c[4] is not None]))}

# ================= B. 047 の P1 の上で h を走らせる =================
P47 = os.path.abspath(os.path.join(HERE, "..", "047-closed-group-multiplier", "sim.py"))
src = open(P47, encoding="utf8").read()
H = {"__file__": P47, "__name__": "sim047"}
exec(compile(src[:src.index("# ---------- C. 固定 M の格子")], P47, "exec"), H)
set_thr2 = H["set_thr2"]; prior = H["prior"]; evaluate = H["evaluate"]; weighted = H["weighted"]; PRI = H["PRI"]; Nn = H["N"]
GW = weighted.__globals__; comp_keys = GW["comp_keys"]; anchor_weights = GW["anchor_weights"]
ANCH = ("A_Ord", "A_XPT_S", "A_XPT_E")

def ev(mfe, ml, mt2):
    set_thr2(mfe, ml, mt2); D, hv, ph, info = evaluate({}, keys=("U3", "CASC", "BIO")); set_thr2()
    return D, hv
P1 = prior("P1")
D1, HV1 = ev(*P1)
W1 = weighted(D1, HV1, anchors=ANCH)
R["check_047_P1"] = {"w": W1["A_Ord"]["U3"]["TAB"]["total"], "ref": PRI["P1"]["w_TAB"]["A_Ord"]["U3"],
                     "reproduced": bool(W1["A_Ord"]["U3"]["TAB"]["total"] == PRI["P1"]["w_TAB"]["A_Ord"]["U3"] and
                                        W1["A_XPT_S"]["U2"]["TAB"]["total"] == PRI["P1"]["w_TAB"]["A_XPT_S"]["U2"])}
WH = weighted({"HV": HV1}, HV1, anchors=ANCH)   # 重み付きの重い崩壊の確率（Dx に hv を入れる）

def at_h(W, h, a, x="TAB"):
    t = h * W[a]["U1"][x]["total"] + (1 - h) * W[a]["U2"][x]["total"]
    f = h * W[a]["U1"][x]["from_heavy"] + (1 - h) * W[a]["U2"][x]["from_heavy"]
    return t, f

def direct_mix(D, hv, h, a):
    """線形の確認：標本ごとに UNK を h·GEN ＋ (1−h)·U2 にして weighted と同じ式で直接計算。"""
    w, _ = anchor_weights(a); Dx = D["TAB"]; c, ch = comp_keys(Dx, hv); kn = [k for k in w if k != "UNK"]
    tw = sum(w[k] for k in kn); u2 = sum(w[k] * c[k] for k in kn) / tw; u2h = sum(w[k] * ch[k] for k in kn) / tw
    cu = h * Dx["GEN"] + (1 - h) * u2; chu = h * Dx["GEN"] * hv["GEN"] + (1 - h) * u2h
    return float(np.mean(sum(w[k] * c[k] for k in kn) + w["UNK"] * cu)), float(np.mean(sum(w[k] * ch[k] for k in kn) + w["UNK"] * chu))
dm = direct_mix(D1, HV1, 0.3, "A_Ord"); lm = at_h(W1, 0.3, "A_Ord")
R["check_linear_mix"] = {"direct": dm, "linear": list(lm), "max_rel_err": float(max(abs(dm[0] / lm[0] - 1), abs(dm[1] / lm[1] - 1)))}

def hstar(W, a):
    t1, f1 = W[a]["U1"]["TAB"]["total"], W[a]["U1"]["TAB"]["from_heavy"]; t2, f2 = W[a]["U2"]["TAB"]["total"], W[a]["U2"]["TAB"]["from_heavy"]
    if f2 / t2 >= 0.5: return 0.0
    if f1 / t1 < 0.5: return None
    return float((0.5 * t2 - f2) / ((f1 - f2) - 0.5 * (t1 - t2)))
HS = np.linspace(0, 1, 21)
R["B_h_scan"] = {a: {"h": HS.tolist(), "total": [at_h(W1, h, a)[0] for h in HS], "share_from_heavy": [at_h(W1, h, a)[1] / at_h(W1, h, a)[0] for h in HS],
                     "heavy_prob": [h * WH[a]["U1"]["HV"]["total"] + (1 - h) * WH[a]["U2"]["HV"]["total"] for h in HS]} for a in ANCH}
R["B_h_star"] = {a: hstar(W1, a) for a in ANCH}
R["B_U3_ref"] = {a: {"total": W1[a]["U3"]["TAB"]["total"], "from_heavy": W1[a]["U3"]["TAB"]["from_heavy"], "heavy_prob": WH[a]["U3"]["HV"]["total"]} for a in ANCH}
# 039 の h*（038 の閾値）との比較のため、E2（M=1）でも h* を出す
D0, HV0 = ev(1.0, 1.0, 1.0); W0 = weighted(D0, HV0, anchors=ANCH)
R["B_h_star_E2"] = {a: hstar(W0, a) for a in ANCH}

HV_ = {}
for k in ("V1", "V2", "V3", "V4", "V1_noBritain", "V2_noBritain"):
    p = POST[k]; out = {}
    for a in ("A_Ord", "A_XPT_S"):
        hm = p["post_mean"]; t, f = at_h(W1, hm, a)
        lo, md, hi = p["post_q05_50_95"]
        out[a] = {"h_mean": hm, "total": t, "from_heavy": f, "share_from_heavy": f / t, "ratio_to_U3": t / W1[a]["U3"]["TAB"]["total"],
                  "total_q05_95": [at_h(W1, lo, a)[0], at_h(W1, hi, a)[0]], "share_q05_95": [at_h(W1, lo, a)[1] / at_h(W1, lo, a)[0], at_h(W1, hi, a)[1] / at_h(W1, hi, a)[0]],
                  "P_h_gt_hstar": (float((p["_draws"] > R["B_h_star"][a]).mean()) if R["B_h_star"][a] is not None else 0.0),
                  "heavy_prob": hm * WH[a]["U1"]["HV"]["total"] + (1 - hm) * WH[a]["U2"]["HV"]["total"]}
    HV_[k] = out
R["B_by_reading"] = HV_

# ================= C. 歴史の χ を CASC に（135） =================
chv = np.array(R["A_chi_vs_loss"]["chi_values_known_incl_extinct"])
chi_i = g.choice(chv, Nn)
mfe, ml, mt2 = P1
Dc, HVc = ev(mfe ** (1 - chi_i), ml ** (1 - chi_i), mt2 ** (1 - chi_i))
def splice(keys):
    D = {"TAB": dict(D1["TAB"])}; hv = dict(HV1)
    for k in keys: D["TAB"][k] = Dc["TAB"][k]; hv[k] = HVc[k]
    return D, hv
CC = {}
for lab, keys in (("CASC_only", ("CASC",)), ("CASC_U3", ("CASC", "U3")), ("all", tuple(D1["TAB"].keys()))):
    D, hv = splice(keys); W = weighted(D, hv, anchors=("A_Ord", "A_XPT_S"))
    CC[lab] = {"CASC_heavy": float(hv["CASC"].mean()), "U3_heavy": float(hv["U3"].mean()), "NUC_G_heavy": float(hv["NUC_G"].mean()),
               "w_AOrd_U3": W["A_Ord"]["U3"]["TAB"]["total"], "w_AXPTS_U2": W["A_XPT_S"]["U2"]["TAB"]["total"],
               "h_star_AOrd": hstar(W, "A_Ord")}
R["C_chi_hist"] = {"chi_draw_mean": float(chi_i.mean()), "P1_CASC_heavy": float(HV1["CASC"].mean()), **CC}
# CASC だけ χ 一定（0.5）の版（047 は全シナリオ共通 χ=0.5 で 0.221）
Dh, HVh = ev(mfe ** 0.5, ml ** 0.5, mt2 ** 0.5)
R["C_chi_hist"]["CASC_chi0.5_const"] = float(HVh["CASC"].mean())

# ================= 仮説 =================
v3m = POST["V3"]["post_mean"]; t3, f3 = at_h(W1, v3m, "A_Ord"); hsO = R["B_h_star"]["A_Ord"]
R["hypotheses"] = {
 "H60a": {"value": pts["V1"], "rule": "V1 point h in [0.10, 0.35]", "supported": bool(0.10 <= pts["V1"] <= 0.35)},
 "H60b": {"value": v3m, "rule": "V3 posterior mean < 0.10", "supported": bool(v3m < 0.10)},
 "H60c": {"value": hsO, "rule": "P1 A_Ord h* in [0.15, 0.25]", "supported": bool(hsO is not None and 0.15 <= hsO <= 0.25)},
 "H60d": {"value": [f3 / t3, t3, W1["A_Ord"]["U3"]["TAB"]["total"]], "rule": "at V3 mean h: share_from_heavy < 0.5 and total < P1 U3",
          "supported": bool(f3 / t3 < 0.5 and t3 < W1["A_Ord"]["U3"]["TAB"]["total"])},
 "H60e": {"value": [R["A_chi_vs_loss"]["Ltop_chi0"], R["A_chi_vs_loss"]["Ltop_chi_ge_half"], diff], "rule": "L_top(chi=0) - L_top(chi>=0.5) >= 0.3",
          "supported": bool(diff >= 0.3)},
 "H60f": {"value": CC["CASC_only"]["CASC_heavy"], "rule": "CASC heavy share with historical chi <= 0.25", "supported": bool(CC["CASC_only"]["CASC_heavy"] <= 0.25)},
 "H60g": {"value": POST["V1"]["x_flip_q05_50_95"], "rule": "V1 flip 90% interval within [0.05, 0.40]",
          "supported": bool(POST["V1"]["x_flip_q05_50_95"][0] >= 0.05 and POST["V1"]["x_flip_q05_50_95"][2] <= 0.40)},
}

# ================= 図 =================
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
names = [c[0] for c in CASES]; y = np.arange(len(CASES))
for j, (lab, col) in enumerate((("L_top", 2), ("L_multi", 3), ("chi", 4))):
    vals = [np.nan if c[col] is None else c[col] for c in CASES]
    ax[0].scatter(np.full(len(CASES), j), y, c=vals, cmap="RdYlGn" if lab == "chi" else "Reds", vmin=0, vmax=1, s=140, edgecolors="k", marker="s")
    for yy, v in zip(y, vals):
        if np.isnan(v): ax[0].text(j, yy, "?", ha="center", va="center", fontsize=7)
ax[0].set_xticks([0, 1, 2]); ax[0].set_xticklabels(["L_top (lost top stage)", "L_multi (lost >=2 skills)", "chi (trade after)"], fontsize=8)
ax[0].set_yticks(y); ax[0].set_yticklabels(names, fontsize=7); ax[0].invert_yaxis(); ax[0].set_xlim(-0.6, 2.6)
ax[0].set_title("reference set: 18 historical collapses (Greenland = extinct)", fontsize=9)
xs = np.linspace(0, 0.6, 400)
for k, c in (("V1", "C3"), ("V2", "C1"), ("V3", "C0"), ("V4", "C2")):
    d = POST[k]["_draws"]; hgr, be = np.histogram(d, bins=120, range=(0, 0.6), density=True)
    ax[1].plot(0.5 * (be[1:] + be[:-1]), hgr, color=c, label=f"{k}: mean {POST[k]['post_mean']:.3f}")
ax[1].axvline(hsO, color="k", ls="--", label=f"h* A_Ord on P1 = {hsO:.3f}")
ax[1].axvline(0.20, color="grey", ls=":", label="h* in 039 (038 thresholds) = 0.20")
ax[1].set_xlabel("h (share of unknown anthropogenic catastrophes restarting at T0)"); ax[1].set_ylabel("posterior density"); ax[1].legend(fontsize=7)
ax[1].set_title("posterior of h under four readings of the reference set", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_cases_h_posterior.png"), dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for a, c in (("A_Ord", "C1"), ("A_XPT_S", "C0")):
    s = R["B_h_scan"][a]
    ax[0].plot(s["h"], s["share_from_heavy"], "o-", color=c, ms=3, label=f"{a}: share of value from heavy collapse")
    ax[0].axhline(R["B_U3_ref"][a]["from_heavy"] / R["B_U3_ref"][a]["total"], color=c, ls=":", lw=0.8)
ax[0].axhline(0.5, color="grey", lw=0.8)
for k, c in (("V1", "C3"), ("V3", "C0"), ("V4", "C2")):
    ax[0].axvline(POST[k]["post_mean"], color=c, ls="--", lw=0.8); ax[0].text(POST[k]["post_mean"], 0.05, k, color=c, fontsize=8)
ax[0].set_xlabel("h"); ax[0].set_ylabel("share"); ax[0].legend(fontsize=7); ax[0].set_ylim(0, 1)
ax[0].set_title("P1 thresholds: value share from heavy collapse vs h (dotted = U3 agnostic)", fontsize=9)
labs = ["P1 (chi=0)", "chi=0.5 all\n(047)", "historical chi\nCASC only", "historical chi\nall"]
vals = [R["C_chi_hist"]["P1_CASC_heavy"], R["C_chi_hist"]["CASC_chi0.5_const"], CC["CASC_only"]["CASC_heavy"], CC["all"]["CASC_heavy"]]
ax[1].bar(range(4), vals, color=["#888", "C1", "C3", "C2"]); ax[1].set_xticks(range(4)); ax[1].set_xticklabels(labs, fontsize=8)
for i, v in enumerate(vals): ax[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
ax[1].set_ylabel("CASC share restarting at T1 or below"); ax[1].set_title("135: CASC heavy share under historical connectivity", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_hscan_chi.png"), dpi=120); plt.close()

R["runtime_s"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("check_047_P1", "check_linear_mix", "A_h_points", "A_counts", "A_chi_vs_loss", "B_h_star", "B_h_star_E2", "B_U3_ref", "C_chi_hist", "hypotheses")}, ensure_ascii=False, indent=1, default=float))
for k, v in R["A_h_estimates"].items(): print(k, v)
for k, v in R["B_by_reading"].items(): print(k, json.dumps(v, default=float))
print("runtime", R["runtime_s"])
