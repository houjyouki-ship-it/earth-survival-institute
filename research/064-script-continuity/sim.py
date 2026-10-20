"""研究064：地球規模の重い崩壊で文字の読み書きが続く確率 p_cont（open_problems 169）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md。
063 の sim.py を行頭の「# ---------- 版 ----------」の前まで exec し 063 の主を差0で再現してから、
p_cont を「その文字を教える人が共同体の中にしかいなかった文字が局面にあった例」16件から置き直す。新しい乱数は seed 20261064。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P63 = os.path.abspath(os.path.join(HERE, "..", "063-post-collapse-reading", "sim.py"))
src = open(P63, encoding="utf8").read(); cut = src.index("\n# ---------- 版 ----------\n")
G63 = {"__file__": P63, "__name__": "sim063"}
exec(compile(src[:cut], P63, "exec"), G63)
r63 = json.load(open(os.path.join(HERE, "..", "063-post-collapse-reading", "results.json"), encoding="utf8"))
R = {"study": "064-script-continuity", "seed": 20261064}
N = G63["N"]; deltas_t = G63["deltas_t"]; ver = G63["ver"]; get = G63["get"]; rho1 = G63["rho1"]; r_reach = G63["r_reach"]
R_common = G63["R_common"]; S_ind = G63["S_ind"]; rho63 = G63["rho63"]; f_i = G63["f_i"]; taup_061 = G63["taup_061"]
C_T = G63["C_T"]; C_P = G63["C_P"]; qf = G63["qf"]; heavy_mask = G63["heavy_mask"]
KEYS = ["A_Ord/U3", "A_XPT_S/U2"]; PARTS = ("total", "heavy", "light"); M1 = "A_Ord/U3"
def ratio_all(o):
    d = deltas_t(o); return {k: {p: d[k.split("/")[0]][k.split("/")[1]]["TC"][p] / get("V0", k)["TC"][p] for p in PARTS} for k in KEYS}, d

# 検査：063 の主
chk, _ = ratio_all(ver(rho=rho63()))
dd = max(abs(chk[k][p] - r63["ratio_to_V0"]["main"][k]["TC"][p]) for k in KEYS for p in PARTS)
R["check_063_main"] = {"max_abs_diff": dd, "reproduced": dd < 1e-12}

# ---------- 資料 ----------
CJ = json.load(open(os.path.join(HERE, "sources", "cases.json"), encoding="utf8"))
CS = CJ["cases"]
R["cases"] = [{k: c[k] for k in ("id", "breadth", "depth", "y", "y_alt", "years_after", "iso_world")} for c in CS]
def tally(sel, yk="y"):
    s = [c for c in CS if sel(c)]; return len(s), float(sum(c[yk] for c in s))
deep = lambda c: c["depth"] == "deep"
R["tally"] = {nm: dict(zip(("n", "lost"), tally(f))) for nm, f in {
    "deep_narrow": lambda c: deep(c) and c["breadth"] == "narrow", "deep_broad": lambda c: deep(c) and c["breadth"] == "broad",
    "deep_unknown": lambda c: deep(c) and c["breadth"] == "unknown", "deep_all": deep,
    "nonsupp": lambda c: c["depth"] != "suppression", "all": lambda c: True, "iso_world": lambda c: c["iso_world"],
    "suppression": lambda c: c["depth"] == "suppression"}.items()}
kept = [c for c in CS if c["y"] == 0 and c["depth"] != "suppression"]
E_years = float(sum(c["years_after"] for c in kept)); R["drift"] = {"exposure_years": E_years, "n_kept": len(kept), "events": 0}

# ---------- 新しい乱数 ----------
g = np.random.default_rng(20261064)
w_mod = g.uniform(0, 1, N)
lam = g.gamma(0.5, 1 / E_years, N)
def L_class(yk="y", unk="random", depth_sel=deep):
    """deep の例から narrow と broad の喪失の Beta の標本。unknown は標本ごとに振る。"""
    nar = [c for c in CS if depth_sel(c) and c["breadth"] == "narrow"]; bro = [c for c in CS if depth_sel(c) and c["breadth"] == "broad"]
    unkc = [c for c in CS if depth_sel(c) and c["breadth"] == "unknown"]
    yn = sum(c[yk] for c in nar); nn = len(nar); yb = sum(c[yk] for c in bro); nb = len(bro)
    uy = np.array([c[yk] for c in unkc], float)
    if unk == "random": A = g.uniform(0, 1, (N, len(unkc))) < 0.5   # True → broad
    elif unk == "narrow": A = np.zeros((N, len(unkc)), bool)
    else: A = np.ones((N, len(unkc)), bool)
    yb_s = yb + (A * uy).sum(1); nb_s = nb + A.sum(1); yn_s = yn + (~A * uy).sum(1); nn_s = nn + (~A).sum(1)
    Ln = g.beta(0.5 + yn_s, 0.5 + nn_s - yn_s); Lb = g.beta(0.5 + yb_s, 0.5 + nb_s - yb_s)
    return Ln, Lb
def L_pooled(sel, yk="y"):
    n, y = tally(sel, yk); return g.beta(0.5 + y, 0.5 + n - y, N)
Ln_m, Lb_m = L_class()
Ln_un, Lb_un = L_class(unk="narrow"); Ln_ub, Lb_ub = L_class(unk="broad"); Ln_a, Lb_a = L_class(yk="y_alt")
Ln_x, Lb_x = L_class(unk="narrow")  # H76a 用：unknown を除いた数え（下で解析的に）
nN, yN = tally(lambda c: deep(c) and c["breadth"] == "narrow"); nB, yB = tally(lambda c: deep(c) and c["breadth"] == "broad")
L_pd = L_pooled(deep); L_ns = L_pooled(lambda c: c["depth"] != "suppression"); L_all = L_pooled(lambda c: True); L_iso = L_pooled(lambda c: c["iso_world"])
mix = lambda Ln, Lb, w=w_mod: w * Lb + (1 - w) * Ln
LV = {"main": mix(Ln_m, Lb_m), "narrow": Ln_m, "broad": Lb_m, "unk_narrow": mix(Ln_un, Lb_un), "unk_broad": mix(Ln_ub, Lb_ub),
      "pooled_deep": L_pd, "pooled_nonsupp": L_ns, "with_supp": L_all, "ambig_alt": mix(Ln_a, Lb_a), "nodrift": mix(Ln_m, Lb_m),
      "iso_world": L_iso, "G3": mix(Ln_m, Lb_m) ** 3}
R["L_summary"] = {nm: {"mean_keep": float(1 - v.mean()), "q05_50_95_keep": [float(x) for x in np.quantile(1 - v, [0.05, 0.5, 0.95])]} for nm, v in LV.items()}
R["L_summary"]["063_pc_main"] = {"mean_keep": float(G63["pc_main"].mean())}
R["class_means_excl_unknown"] = {"narrow": (0.5 + yN) / (1 + nN), "broad": (0.5 + yB) / (1 + nB)}
R["lambda_q"] = [float(x) for x in np.quantile(lam, [0.05, 0.5, 0.95])]

def rho64(Lv, drift=True, n_p=100, K=20):
    def f(T):
        pc = (1 - Lv) * (np.exp(-lam * T) if drift else 1.0)
        return 1 - (1 - rho1) * (1 - r_reach * R_common(T, pc) * S_ind(T, n_p, K, f_i, taup_061))
    return f
# 重い崩壊の T での p_cont の分布（U3）
B = G63["B"]; hm = np.isin(B["U3"]["st"], (4, 5)); TU = np.maximum(B["U3"]["T2s"], 0)[hm]
pcT = ((1 - LV["main"]) * np.exp(-lam * np.maximum(B["U3"]["T2s"], 0)))[hm]
R["p_cont_at_T_U3_heavy"] = {"mean": float(pcT.mean()), "q05_50_95": [float(x) for x in np.quantile(pcT, [0.05, 0.5, 0.95])],
                             "drift_factor_mean": float(np.exp(-lam * np.maximum(B["U3"]["T2s"], 0))[hm].mean())}

# ---------- 版 ----------
RT = {}; DV = {}
for nm, Lv in LV.items():
    RT[nm], DV[nm] = ratio_all(ver(rho=rho64(Lv, drift=(nm != "nodrift"))))
RT["063_main"] = {k: r63["ratio_to_V0"]["main"][k]["TC"] for k in KEYS}
R["ratio_to_V0"] = RT
R["scan_w"] = []
for wv in [0.0, 0.25, 0.5, 0.75, 1.0]:
    rr, _ = ratio_all(ver(rho=rho64(mix(Ln_m, Lb_m, np.full(N, wv)))))
    R["scan_w"].append({"w": wv, M1: rr[M1]["heavy"], "A_XPT_S/U2": rr["A_XPT_S/U2"]["heavy"]})

# ---------- 費用あたり ----------
R["CE"] = {}
for key in KEYS:
    a, u = key.split("/"); R["CE"][key] = {}
    for p in ("total", "heavy"):
        vt = get("main", key)["TAB"][p]; vtc = get("main", key)["TC"][p]; R["CE"][key][p] = {}
        for nm in LV:
            dv = DV[nm][a][u]["TC"][p] - vtc
            R["CE"][key][p][nm] = qf(dv / vt * C_T / C_P(100)) if dv > 0 else [0.0, 0.0, 0.0]
        R["CE"][key][p]["063_main"] = r63["CE"][key][p]["main"]["CE_q"]
s_ce = r63["CE_STC_060"]; R["CE_STC_060"] = s_ce

# ---------- 仮説 ----------
H = {}; h = lambda nm: RT[nm][M1]["heavy"]
cm = R["class_means_excl_unknown"]; v = cm["narrow"] - cm["broad"]
H["H76a"] = {"value": [cm["narrow"], cm["broad"], v], "rule": "deep: mean L_narrow - mean L_broad >= 0.3 (unknown excluded)", "supported": v >= 0.3}
v = 1 - R["L_summary"]["pooled_deep"]["mean_keep"]; H["H76b"] = {"value": v, "rule": "pooled deep loss mean in [0.2,0.4]", "supported": 0.2 <= v <= 0.4}
v = R["L_summary"]["main"]["mean_keep"]; H["H76c"] = {"value": v, "rule": "main mean (1-L) >= 0.75", "supported": v >= 0.75}
v = h("main"); H["H76d"] = {"value": v, "rule": "main heavy TC/V0 >= 0.82", "supported": v >= 0.82}
v = R["scan_w"][-1][M1] - R["scan_w"][0][M1]; H["H76e"] = {"value": v, "rule": "w=1 minus w=0 >= 0.05", "supported": v >= 0.05}
v = h("nodrift") - h("main"); H["H76f"] = {"value": v, "rule": "drift lowers heavy < 0.01", "supported": v < 0.01}
v = h("G3"); H["H76g"] = {"value": v, "rule": "G3 >= 0.85", "supported": v >= 0.85}
v = h("iso_world"); H["H76h"] = {"value": v, "rule": "iso_world <= 0.79", "supported": v <= 0.79}
v = [R["CE"][M1]["total"]["main"][1], s_ce]; H["H76i"] = {"value": v, "rule": "CE main > S_TC|TC", "supported": v[0] > v[1]}
for x in H.values(): x["supported"] = bool(x["supported"])
R["hypotheses"] = H

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
order = sorted(CS, key=lambda c: (c["depth"], c["breadth"], c["y"]))
col = {"narrow": "C3", "broad": "C2", "unknown": "0.6"}
for i, c in enumerate(order):
    axs[0].barh(i, 1 if c["y"] > 0 else 0.08, left=0, color=col[c["breadth"]], alpha=0.4 + 0.6 * (c["y"] if c["y"] else 1))
    axs[0].text(1.03, i, f'{c["depth"]}{" iso" if c["iso_world"] else ""}  y={c["y"]}', va="center", fontsize=7)
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([c["id"] for c in order], fontsize=7); axs[0].set_xlim(0, 1.8); axs[0].set_xticks([])
axs[0].set_title("16 crisis episodes (red narrow, green broad, grey unknown; long bar = script lost)", fontsize=8)
names = [n for n in LV] + ["063_main"]; vals = [RT[n][M1]["heavy"] for n in names]
axs[1].barh(range(len(names)), vals, color=["C0" if n != "063_main" else "k" for n in names]); axs[1].set_yticks(range(len(names))); axs[1].set_yticklabels(names, fontsize=7)
axs[1].set_xlim(0.6, 0.92); axs[1].set_xlabel("heavy-collapse TC / V0 (A_Ord/U3)")
ax2 = axs[1].inset_axes([0.62, 0.08, 0.35, 0.3]); ax2.plot([r["w"] for r in R["scan_w"]], [r[M1] for r in R["scan_w"]], "o-"); ax2.set_xlabel("w (broad weight)", fontsize=6); ax2.tick_params(labelsize=6)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_cases_versions.png"), dpi=130); plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 4))
for nm, c in (("main", "C0"), ("narrow", "C3"), ("broad", "C2"), ("iso_world", "C1")):
    ax.hist(1 - LV[nm], bins=60, range=(0, 1), histtype="step", density=True, color=c, label=f"{nm} mean {1 - LV[nm].mean():.2f}")
ax.axvline(G63["pc_main"].mean(), color="k", ls="--", label=f"063 p_cont mean {G63['pc_main'].mean():.2f}")
ax.set_xlabel("p_cont (keep writing through the shock)"); ax.set_ylabel("density"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_pcont_posteriors.png"), dpi=130); plt.close(fig)

R["runtime_s"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float, ensure_ascii=False)
print("check", R["check_063_main"]); print("tally", json.dumps(R["tally"])); print("drift", R["drift"], R["lambda_q"])
print("L_summary", json.dumps(R["L_summary"])); print("class_means", R["class_means_excl_unknown"]); print("pc_at_T", R["p_cont_at_T_U3_heavy"])
for nm in RT: print(nm, {k: {p: round(RT[nm][k][p], 4) for p in PARTS} for k in KEYS})
print("scan_w", json.dumps(R["scan_w"]))
print("CE total", json.dumps(R["CE"][M1]["total"])); print("CE heavy", json.dumps(R["CE"][M1]["heavy"]))
for k, x in H.items(): print(k, x["supported"], x["value"])
print("runtime", R["runtime_s"])
