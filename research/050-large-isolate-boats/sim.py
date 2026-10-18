"""研究050：大きな孤立集団での段の喪失（open_problems 139）と、舟の喪失が孤立を固定する自己強化（140）。
事前登録：PREREGISTRATION.md（2026-10-18T22:40Z）。
049 の sim.py を「# ================= 図」の手前まで exec して（中で 047 の前半と 046・045・044・042・040・039・038 を exec）、
049 の参照の集合・回帰の関数と 047 の P1 の道具を借りる。049 の結果を誤差0で再現してから、乱数を seed 20261050 に替える。
numpy＋matplotlib のみ、ネット接続なし。"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261050
R = {"study": "050-large-isolate-boats", "seed": SEED}
EX = json.load(open(os.path.join(HERE, "sources", "excerpts.json"), encoding="utf8"))
assert all(e["found_in_fetched_text"] for e in EX["excerpts"])
R["n_excerpts_verified"] = len(EX["excerpts"])

# ================= 049 の名前空間 =================
P49 = os.path.abspath(os.path.join(HERE, "..", "049-isolation-stage-loss", "sim.py"))
src = open(P49, encoding="utf8").read()
NS = {"__file__": P49, "__name__": "sim049"}
exec(compile(src[:src.index("# ================= 図 =================")], P49, "exec"), NS)
R49 = json.load(open(os.path.join(HERE, "..", "049-isolation-stage-loss", "results.json"), encoding="utf8"))
R["check_049"] = {"pred_1e5_main": NS["R"]["B_regression"]["main"]["pred_mean"]["1e+05"], "ref": R49["B_regression"]["main"]["pred_mean"]["1e+05"],
                  "h_star": NS["HS1"], "ref_h_star": R49["check_048"]["h_star"],
                  "E_U3": NS["R"]["E_iso_on_model"]["U3"]["E_P_loss_mean"], "ref_E_U3": R49["E_iso_on_model"]["U3"]["E_P_loss_mean"]}
R["check_049"]["reproduced"] = bool(R["check_049"]["pred_1e5_main"] == R["check_049"]["ref"] and R["check_049"]["h_star"] == R["check_049"]["ref_h_star"]
                                    and R["check_049"]["E_U3"] == R["check_049"]["ref_E_U3"])
g = np.random.default_rng(SEED); NS["g"] = g          # 以後 reg_posterior も新しい乱数を使う
reg_posterior, pred = NS["reg_posterior"], NS["pred"]
SRC, HV1, W1, HS1, ev, Nn, sub = NS["SRC"], NS["HV1"], NS["W1"], NS["HS1"], NS["ev"], NS["Nn"], NS["sub"]
weighted, P1 = NS["weighted"], NS["P1"]
ISO49 = [c for c in NS["CASES"] if c[1] == "iso"]

# ================= A. 新しい候補（符号は【推測】、根拠は excerpts.json の番号） =================
# name, group, L, env, N(lo,hi), T_iso, exo, after, evidence, note
NEW = {
 "Americas": ("Americas (pre-1492)", "iso", 0.0, 0, (3e7, 1e8), 11000, 1, 1, [0, 1, 2, 3, 5, 6], "isolated since Beringia submerged; no compelling transoceanic legacy; SA alone 30-50M (100M); no unit-wide loss stated"),
 "Australia": ("Australia mainland (pre-1788)", "iso", 0.0, 0, (3e5, 3e6), 8000, 1, 1, [8, 9, 10, 11, 12, 13], "isolated from SE Asia; but Torres Strait trade with New Guinea and Macassans from ~1700; adopted dugouts; no mainland loss stated"),
 "NewGuinea": ("New Guinea Highlands", "iso", 0.0, 0, (1e6, 2e6), 10000, 1, 1, [15, 16, 17], "relatively isolated from rest of world; independent agriculture; >1M in 1930s (upper bound assumed)"),
}
S2_SPLIT = [("Andes (Inca)", "iso", 0.0, 0, (6e6, 1.4e7), 11000, 1, 1, [7], "Inca 6-14M"),
            ("Mesoamerica", "iso", 0.5, 0, (1e6, 1e7), 11000, 1, 1, [4], "Valley of Mexico ~1M (lower); 048 Classic Maya L_top 0.5 counted for S2")]
SAHUL = ("Sahul (Australia + NG Highlands)", "iso", 0.0, 0, (1.3e6, 5e6), 8000, 1, 1, [8, 10, 11, 15, 17], "merged unit (S1)")
EXCLUDED = [{"name": "Madagascar", "reason": "regular transoceanic trade in text (Indonesian, Swahili/Arab)", "evidence": [18, 19]},
            {"name": "Iceland (Commonwealth)", "reason": "grain and cod trade with Europe in text", "evidence": [20, 21]},
            {"name": "Jomon", "reason": "isolated per text but no population figure in fetched text; Korea contact in Final Jomon", "evidence": [22, 23]}]
SETS = {"main": ISO49 + [NEW["Americas"], NEW["Australia"], NEW["NewGuinea"]],
        "S1_sahul_merged": ISO49 + [NEW["Americas"], SAHUL],
        "S2_americas_split": ISO49 + S2_SPLIT + [NEW["Australia"], NEW["NewGuinea"]],
        "S3_no_australia": ISO49 + [NEW["Americas"], NEW["NewGuinea"]]}
R["A_new_cases"] = [{"name": c[0], "L": c[2], "N": c[4], "T_iso": c[5], "evidence": [EX["excerpts"][i]["url"] for i in c[8]], "note": c[9]}
                    for c in list(NEW.values()) + S2_SPLIT + [SAHUL]]
R["A_excluded"] = EXCLUDED

def jeff(vals, NB=200000):
    v = np.array(vals, float); amb = (v == 0.5)
    x = (v == 1).sum() + (g.binomial(1, 0.5, (NB, amb.sum())).sum(1) if amb.any() else 0)
    p = g.beta(0.5 + x, 0.5 + v.size - x, NB) if np.ndim(x) == 0 else g.beta(0.5 + x, 0.5 + v.size - x)
    return {"n": int(v.size), "point": float(v.mean()), "post_mean": float(p.mean()), "q95": float(np.percentile(p, 95)), "P_gt_hstar": float((p > HS1).mean())}
BIG = {}
for k, S in SETS.items():
    big = [c for c in S if c[4] is not None and c[4][0] >= 5e4]
    BIG[k] = {"cases": [c[0] for c in big], **jeff([c[2] for c in big])}
R["A_big_isolates"] = BIG

# ================= B. 回帰を当て直す =================
NGRID = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
RB = {}; REGN = {}
for k, S in SETS.items():
    cs = [c for c in S if c[4] is not None]
    a, b, _ = reg_posterior(cs); REGN[k] = (a, b)
    pr = pred(a, b, NGRID)
    RB[k] = {"n_cases": len(cs), "P_b_lt_0": float((b < 0).mean()), "b_q05_50_95": [float(q) for q in np.percentile(b, [5, 50, 95])],
             "pred_mean": {f"{n:.0e}": float(pr[:, i].mean()) for i, n in enumerate(NGRID)},
             "pred_q95": {f"{n:.0e}": float(np.percentile(pr[:, i], 95)) for i, n in enumerate(NGRID)}}
R["B_regression"] = RB

# ================= C. 軽い崩壊の生き残りでの E[P(L|nmax)]（049 の explore.py と同じ条件、本回は登録） =================
sv = SRC["U3"]["sv"]; hvU3 = np.asarray(HV1["U3"], float)
alive = sv["ntot"] >= 5000
n_sub = np.maximum(sv["nmax"][sub], 1.0); w_light = (alive[sub] * (1 - hvU3[sub])).astype(float)
EC = {}
for k, (a, b) in list(REGN.items()) + [("049_main", NS["REGS"]["main"][:2])]:
    pr = pred(a, b, n_sub); e = (pr * w_light[None, :]).sum(1) / w_light.sum()
    EC[k] = {"mean": float(e.mean()), "q05_50_95": [float(q) for q in np.percentile(e, [5, 50, 95])], "P_gt_hstar": float((e > HS1).mean())}
R["C_E_light"] = EC
R["C_E_light"]["light_nmax_q05_50_95"] = [float(q) for q in np.percentile(n_sub[w_light > 0.5], [5, 50, 95])]

# ================= D. 140：舟の喪失の吸収的な過程 =================
BOAT = {"Inughuit (Polar Inuit)": 1.0, "Moriori (Chatham)": 1.0, "Easter Island": 1.0}
cs = [c for c in SETS["main"] if c[4] is not None]
ab, bb, _ = reg_posterior(cs, override={c[0]: BOAT.get(c[0], 0.0) for c in cs})
prB = pred(ab, bb, NGRID)
R["D_boat_regression"] = {"cases_B1": list(BOAT), "n_cases": len(cs), "P_b_lt_0": float((bb < 0).mean()),
                          "pred_mean": {f"{n:.0e}": float(prB[:, i].mean()) for i, n in enumerate(NGRID)},
                          "env_attributed_share": 2 / 3}
R48 = json.load(open(os.path.join(HERE, "..", "048-unknown-depth-h", "results.json"), encoding="utf8"))
chv = np.array(R48["A_chi_vs_loss"]["chi_values_known_incl_extinct"], float)
chi0 = g.choice(chv, Nn)
G_GEN = 12
U_loss = g.uniform(size=(Nn, G_GEN)); U_rc = g.uniform(size=(Nn, G_GEN))
idx = g.integers(0, ab.size, Nn)                      # 標本ごとに舟の回帰の事後から1組
KEYS7 = ("U3", "CASC", "BIO", "NUC_G", "NUC_R", "NAT", "AS")
def p_boat(n):
    x = np.log10(np.maximum(n, 1.0)) - 4
    return 1 / (1 + np.exp(-(ab[idx] + bb[idx] * x)))
def chi_eff(k, e=0.0, Gref=20, cconn=0.0, rho=0.0, absorb=True):
    if not absorb: return chi0.copy(), np.zeros(Nn)
    lam = 1 - (1 - p_boat(SRC[k]["sv"]["nmax"])) ** (1 / Gref)
    chi = chi0.copy(); boat = np.ones(Nn, bool); acc = np.zeros(Nn)
    for t in range(G_GEN):
        hz = lam * (1 - e) * ((1 - chi) + cconn * chi)
        lost = boat & (U_loss[:, t] < hz); boat &= ~lost; chi = np.where(lost, 0.0, chi)
        if rho > 0:
            rc = boat & (chi == 0) & (U_rc[:, t] < rho); chi = np.where(rc, 0.5, chi)
        acc += chi
    return acc / G_GEN, (~boat).astype(float)
mfe, ml, mt2 = P1
def run_cond(**kw):
    """シナリオごとに χ_eff を作って評価し、そのシナリオの結果を差し込む。KEYS7 以外は吸収なしの χ0 の評価を使う。"""
    c0, _ = chi_eff("U3", absorb=False)
    Db, hb = ev(mfe ** (1 - c0), ml ** (1 - c0), mt2 ** (1 - c0))
    D = {"TAB": dict(Db["TAB"])}; hv = dict(hb); info = {}
    for k in KEYS7:
        ce, lostb = chi_eff(k, **kw)
        if kw.get("absorb", True):
            Dk, hk = ev(mfe ** (1 - ce), ml ** (1 - ce), mt2 ** (1 - ce)); D["TAB"][k] = Dk["TAB"][k]; hv[k] = hk[k]
        alive_k = SRC[k]["sv"]["ntot"] >= 5000
        info[k] = {"chi_eff_mean_alive": float(ce[alive_k].mean()) if alive_k.any() else None,
                   "boat_lost_share_alive": float(lostb[alive_k].mean()) if alive_k.any() else None}
    W = weighted(D, hv, anchors=("A_Ord",)); WH = weighted({"HV": hv}, hv, anchors=("A_Ord",))
    h = 0.195   # 048 の V2 の事後平均
    tot = h * W["A_Ord"]["U1"]["TAB"]["total"] + (1 - h) * W["A_Ord"]["U2"]["TAB"]["total"]
    fh = h * W["A_Ord"]["U1"]["TAB"]["from_heavy"] + (1 - h) * W["A_Ord"]["U2"]["TAB"]["from_heavy"]
    hp = h * WH["A_Ord"]["U1"]["HV"]["total"] + (1 - h) * WH["A_Ord"]["U2"]["HV"]["total"]
    return {"heavy_share": {k: float(np.mean(hv[k])) for k in KEYS7}, "w_AOrd_V2": tot, "share_from_heavy_V2": fh / tot,
            "w_heavy_prob_V2": hp, "w_AOrd_U3": W["A_Ord"]["U3"]["TAB"]["total"], "per_scenario": info}
COND = {"no_absorb": dict(absorb=False), "absorb": dict(), "boat_section_e0.5": dict(e=0.5), "boat_section_e1": dict(e=1.0),
        "Gref4": dict(Gref=4), "Gref4_e1": dict(Gref=4, e=1.0),
        "cconn0.1": dict(cconn=0.1), "cconn0.1_e1": dict(cconn=0.1, e=1.0),
        "rho0.05": dict(rho=0.05), "rho0.05_e1": dict(rho=0.05, e=1.0)}
DC = {k: run_cond(**v) for k, v in COND.items()}
R["D_conditions"] = DC
R["D_chi0_mean"] = float(chi0.mean())
def eff(a, b):
    return {"d_CASC_heavy": DC[a]["heavy_share"]["CASC"] - DC[b]["heavy_share"]["CASC"],
            "d_NUC_G_heavy": DC[a]["heavy_share"]["NUC_G"] - DC[b]["heavy_share"]["NUC_G"],
            "d_w_heavy_prob_V2": DC[a]["w_heavy_prob_V2"] - DC[b]["w_heavy_prob_V2"],
            "d_w_AOrd_V2": DC[a]["w_AOrd_V2"] - DC[b]["w_AOrd_V2"]}
R["D_effects"] = {"absorption_vs_none": eff("absorb", "no_absorb"), "boat_e1_vs_absorb": eff("boat_section_e1", "absorb"),
                  "boat_e0.5_vs_absorb": eff("boat_section_e0.5", "absorb"), "Gref4_absorption": eff("Gref4", "no_absorb"),
                  "Gref4_boat_e1": eff("Gref4_e1", "Gref4"), "cconn_absorption": eff("cconn0.1", "no_absorb"),
                  "cconn_boat_e1": eff("cconn0.1_e1", "cconn0.1"), "rho_absorption": eff("rho0.05", "no_absorb"),
                  "rho_boat_e1": eff("rho0.05_e1", "rho0.05")}

# ================= 仮説 =================
E = R["D_effects"]
R["hypotheses"] = {
 "H62a": {"value": BIG["main"]["post_mean"], "rule": "main big-isolate Jeffreys mean < h*", "h_star": HS1, "supported": bool(BIG["main"]["post_mean"] < HS1)},
 "H62b": {"value": {k: BIG[k]["post_mean"] for k in BIG}, "rule": "H62a holds in S1-S3", "supported": bool(all(BIG[k]["post_mean"] < HS1 for k in BIG))},
 "H62c": {"value": RB["main"]["pred_mean"]["1e+05"], "rule": "main regression P(L|1e5) < 0.15", "supported": bool(RB["main"]["pred_mean"]["1e+05"] < 0.15)},
 "H62d": {"value": EC["main"]["mean"], "rule": "E[P(L|nmax)] over light-collapse survivors < h*", "supported": bool(EC["main"]["mean"] < HS1)},
 "H62e": {"value": E["absorption_vs_none"]["d_CASC_heavy"], "rule": "absorption raises CASC heavy share by >= 0.02", "supported": bool(E["absorption_vs_none"]["d_CASC_heavy"] >= 0.02)},
 "H62f": {"value": DC["absorb"]["heavy_share"]["NUC_G"], "rule": "NUC_G heavy share with absorption <= 0.05", "supported": bool(DC["absorb"]["heavy_share"]["NUC_G"] <= 0.05)},
 "H62g": {"value": E["boat_e1_vs_absorb"]["d_w_heavy_prob_V2"], "rule": "boat section e=1 lowers weighted heavy prob (A_Ord, h=0.195) by >= 0.001", "supported": bool(E["boat_e1_vs_absorb"]["d_w_heavy_prob_V2"] <= -0.001)},
}

# ================= 図 =================
fig, ax = plt.subplots(1, 2, figsize=(13, 5.0))
a, b = REGN["main"]; a0, b0 = NS["REGS"]["main"][:2]
ns = np.logspace(1.5, 8.2, 200)
for (aa, bbb), col, lab in (((a0, b0), "grey", "049 fit (9 cases)"), ((a, b), "C0", "050 main fit (+Americas, Australia, NG)")):
    pr = pred(aa, bbb, ns); ax[0].plot(ns, pr.mean(0), color=col, label=lab)
    ax[0].fill_between(ns, np.percentile(pr, 5, 0), np.percentile(pr, 95, 0), color=col, alpha=0.12)
for c in [c for c in SETS["main"] if c[4] is not None]:
    lo, hi = c[4]; gm = np.sqrt(lo * hi); new = c[0] in [x[0] for x in NEW.values()]
    ax[0].errorbar(gm, c[2], xerr=[[gm - lo], [hi - gm]], fmt="o" if new else "s", color="C3" if new else "k", ms=6, capsize=3)
    ax[0].annotate(c[0].split(" (")[0], (gm, c[2]), textcoords="offset points", xytext=(3, 8 if c[2] < 0.9 else -12), fontsize=7, rotation=20)
ax[0].axhline(HS1, color="k", ls="--", lw=0.8, label=f"h* (P1) = {HS1:.3f}")
q = R["C_E_light"]["light_nmax_q05_50_95"]; ax[0].axvspan(q[0], q[2], color="C1", alpha=0.1, label="light-collapse survivor nmax 5-95%")
ax[0].set_xscale("log"); ax[0].set_ylim(-0.1, 1.12); ax[0].set_xlabel("population of isolated group N"); ax[0].set_ylabel("documented technology loss L")
ax[0].legend(fontsize=7, loc="upper right"); ax[0].set_title("139: large isolates added (red = new in 050)", fontsize=9)
labs = list(BIG.keys()); y = np.arange(len(labs))
ax[1].barh(y + 0.2, [BIG[k]["post_mean"] for k in labs], height=0.4, color="C0", label="big isolates Jeffreys mean")
ax[1].barh(y - 0.2, [EC[k]["mean"] for k in labs], height=0.4, color="C2", label="E[P(L|nmax)] light-collapse survivors")
for i, k in enumerate(labs):
    ax[1].text(BIG[k]["post_mean"] + 0.003, i + 0.2, f"{BIG[k]['post_mean']:.3f} (0/{BIG[k]['n']}" + (")" if BIG[k]["point"] == 0 else f", L={BIG[k]['point']:.2f})"), va="center", fontsize=7)
    ax[1].text(EC[k]["mean"] + 0.003, i - 0.2, f"{EC[k]['mean']:.3f}", va="center", fontsize=7)
ax[1].axvline(HS1, color="k", ls="--", label="h* (P1)"); ax[1].axvline(EC["049_main"]["mean"], color="grey", ls=":", label=f"049 E = {EC['049_main']['mean']:.3f}")
ax[1].set_yticks(y); ax[1].set_yticklabels(labs, fontsize=8); ax[1].set_xlabel("probability of losing a held technology"); ax[1].legend(fontsize=7, loc="lower right")
ax[1].set_title("139: versions of the reference set vs h*", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_large_isolates.png"), dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ns2 = np.logspace(1.5, 8, 200); pb = pred(ab, bb, ns2)
ax[0].plot(ns2, pb.mean(0), color="C0", label="P(boat loss | N), isolated, ~500 yr")
ax[0].fill_between(ns2, np.percentile(pb, 5, 0), np.percentile(pb, 95, 0), color="C0", alpha=0.15)
for c in cs:
    lo, hi = c[4]; gm = np.sqrt(lo * hi); ax[0].plot(gm, BOAT.get(c[0], 0.0), "s", color="C3" if BOAT.get(c[0]) else "k")
for k, col in (("CASC", "C1"), ("NUC_G", "C4"), ("U3", "C2")):
    nm = SRC[k]["sv"]["nmax"][SRC[k]["sv"]["ntot"] >= 5000]
    if nm.size: ax[0].axvline(np.median(nm), color=col, ls=":", label=f"{k} survivor nmax median")
ax[0].set_xscale("log"); ax[0].set_xlabel("N"); ax[0].set_ylabel("boat / ocean-craft loss"); ax[0].legend(fontsize=7); ax[0].set_title("140: boat-loss fit (3 losses, 2 environmental)", fontsize=9)
cn = ["no_absorb", "absorb", "boat_section_e1", "Gref4", "Gref4_e1", "rho0.05", "rho0.05_e1"]
x = np.arange(len(cn))
ax[1].bar(x - 0.2, [DC[c]["heavy_share"]["CASC"] for c in cn], 0.4, color="C1", label="CASC heavy share")
ax[1].bar(x + 0.2, [DC[c]["heavy_share"]["NUC_G"] for c in cn], 0.4, color="C4", label="NUC_G heavy share")
ax3 = ax[1].twinx(); ax3.plot(x, [DC[c]["w_heavy_prob_V2"] for c in cn], "ks-", label="weighted heavy prob (A_Ord, h=0.195)")
ax3.set_ylabel("weighted heavy-collapse probability"); ax[1].set_xticks(x); ax[1].set_xticklabels(cn, rotation=25, fontsize=7)
ax[1].set_ylabel("heavy share of scenario"); ax[1].legend(fontsize=7, loc="upper left"); ax3.legend(fontsize=7, loc="upper right")
ax[1].set_title("140: absorbing boat loss and a boat section", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_boats_chi.png"), dpi=120); plt.close()

R["runtime_s"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: v for k, v in R.items() if k not in ("A_new_cases",)}, ensure_ascii=False, indent=1, default=float))
print("runtime", R["runtime_s"])
