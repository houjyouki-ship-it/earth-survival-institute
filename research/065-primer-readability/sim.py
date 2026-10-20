"""研究065：文字を知らない人が絵の読み方の手引き primer で座標の冊子を読めるようになる確率 p_primer（open_problems 170）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md。
064 の sim.py を行頭の「# ---------- 版 ----------」の前まで exec し 064 の主を差0で再現してから、
文字が切れた標本の読みに primer の経路 p_pr(T) = k_rec × k_cont × (1 − exp(−T/τ_key)) を足す。新しい乱数は seed 20261065。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P64 = os.path.abspath(os.path.join(HERE, "..", "064-script-continuity", "sim.py"))
src = open(P64, encoding="utf8").read(); cut = src.index("\n# ---------- 版 ----------\n")
G64 = {"__file__": P64, "__name__": "sim064"}
exec(compile(src[:cut], P64, "exec"), G64)
r64 = json.load(open(os.path.join(HERE, "..", "064-script-continuity", "results.json"), encoding="utf8"))
R = {"study": "065-primer-readability", "seed": 20261065}
G63 = G64["G63"]; G61 = G63["G62"]["G61"]
N = G64["N"]; ver = G64["ver"]; get = G64["get"]; ratio_all = G64["ratio_all"]; rho1 = G64["rho1"]; r_reach = G64["r_reach"]
S_ind = G64["S_ind"]; f_i = G64["f_i"]; taup_061 = G64["taup_061"]; heavy_mask = G64["heavy_mask"]; lam = G64["lam"]; LV = G64["LV"]
C_T = G64["C_T"]; C_P = G64["C_P"]; qf = G64["qf"]; tau_dec = G63["tau_dec"]; srch_lag = G63["srch_lag"]
NC = G61["NC"]; per_P = G61["per_P"]
KEYS = ["A_Ord/U3", "A_XPT_S/U2"]; PARTS = ("total", "heavy", "light"); M1 = "A_Ord/U3"

# ---------- 資料 ----------
CJ = json.load(open(os.path.join(HERE, "sources", "cases.json"), encoding="utf8"))
AD = [c for c in CJ["adoption"] if c["group"] == "main"]
R["adoption"] = {"n_main": len(AD), "y_main": float(sum(c["y"] for c in AD)), "y_alt_main": float(sum(c["y_alt"] for c in AD)),
                 "teacher_present": int(sum(c["teacher"] for c in AD)), "teacherless_nonscholar_cases": 0,
                 "with_literate_and_suppression": {"n": len(CJ["adoption"]), "y": float(sum(c["y"] for c in CJ["adoption"]))},
                 "cases": [{k: c[k] for k in ("id", "group", "y", "y_alt", "years_to_majority")} for c in CJ["adoption"]]}
def lags(keyed):
    ev, cen = [], []
    for c in CJ["decipher"]:
        if c["keyed"] != keyed or c["start"] is None: continue
        if c["end"] is not None: ev.append(c["end"] - c["start"])
        else: cen.append(2026 - c["start"])
    return ev, cen
evK, cenK = lags(True); evU, cenU = lags(False)
expK = float(sum(evK) + sum(cenK)); expU = float(sum(evU) + sum(cenU))
R["decipher"] = {"keyed": {"events": evK, "censored": cenK, "exposure": expK, "tau_mle": expK / len(evK)},
                 "unkeyed": {"events": evU, "censored": cenU, "exposure": expU, "tau_mle": expU / len(evU)}}
LG = [c["lang"] for c in CJ["decipher"] if c["keyed"] and c["lang"] is not None]
NU = [c["num"] for c in CJ["numerals"]]
R["lang_keyed"] = {"n": len(LG), "understood": float(sum(LG)), "mean_jeffreys": (0.5 + sum(LG)) / (1 + len(LG))}
R["numerals"] = {"n": len(NU), "read": float(sum(NU)), "mean_jeffreys": (0.5 + sum(NU)) / (1 + len(NU)),
                 "script_read_without_key": 0, "script_mean_jeffreys": 0.5 / (1 + len(NU))}
R["stimulus"] = [{"id": c["id"], "years": (c["end"] - c["start"]) if c["start"] else None} for c in CJ["stimulus"]]

# ---------- 新しい乱数 ----------
g = np.random.default_rng(20261065)
tau_key = 1 / g.gamma(len(evK), 1 / expK, N)
tau_unk = 1 / g.gamma(len(evU), 1 / expU, N)
k_num = g.beta(0.5 + sum(NU), 0.5 + len(NU) - sum(NU), N)
k_txt = g.beta(0.5 + sum(LG), 0.5 + len(LG) - sum(LG), N)
k_rec = g.uniform(0.5, 1.0, N)
u_spr = g.beta(0.5 + R["adoption"]["y_main"], 0.5 + len(AD) - R["adoption"]["y_main"], N)
prep_pr = np.exp(g.uniform(np.log(2e3), np.log(2e4), NC)); f_pg = g.uniform(0.05, 0.3, NC)
C_pr = prep_pr + 100 * per_P * f_pg
k_cont = 1 - (1 - k_num) * (1 - k_txt)
R["draws"] = {nm: {"mean": float(v.mean()), "q05_50_95": [float(x) for x in np.quantile(v, [0.05, 0.5, 0.95])]} for nm, v in
              {"tau_key": tau_key, "tau_unkeyed": tau_unk, "k_num": k_num, "k_txt": k_txt, "k_cont": k_cont, "k_rec": k_rec, "u_spread": u_spr}.items()}
R["costs"] = {"C_pr": qf(C_pr), "C_P_100": qf(C_P(100)), "C_pr_over_C_P": qf(C_pr / C_P(100))}

def pp_fn(krec=None, kc=None, tau=None, extra=None, srch=False, const=None):
    kr = k_rec if krec is None else krec; kcv = k_cont if kc is None else kc; tv = tau_key if tau is None else tau
    def f(T):
        if const is not None: return np.full(N, float(const))
        Ta = np.maximum(T - srch_lag, 0.0) if srch else T
        v = kr * kcv * (1 - np.exp(-Ta / tv))
        return v * extra if extra is not None else v
    return f
def R65(T, pc, ppf):
    hm = heavy_mask(T); pd = 1 - np.exp(-T / tau_dec)
    if ppf is not None: pd = 1 - (1 - pd) * (1 - ppf(T))
    read = pc + (1 - pc) * pd
    return np.where(hm, read, 1.0)
def rho65(ppf, n_p=100, K=20):
    Lv = LV["main"]
    def f(T):
        pc = (1 - Lv) * np.exp(-lam * T)
        return 1 - (1 - rho1) * (1 - r_reach * R65(T, pc, ppf) * S_ind(T, n_p, K, f_i, taup_061))
    return f
# 重い崩壊の T（U3）での p_pr
B = G63["B"]; hm = np.isin(B["U3"]["st"], (4, 5)); TU = np.maximum(B["U3"]["T2s"], 0)
def pp_at_T(ppf):
    v = ppf(TU)[hm]; return {"mean": float(v.mean()), "q05_50_95": [float(x) for x in np.quantile(v, [0.05, 0.5, 0.95])]}

# ---------- 版 ----------
PV = {"main": pp_fn(), "nokey": None, "num_only": pp_fn(kc=k_num), "txt_only": pp_fn(kc=k_txt), "rec1": pp_fn(krec=1.0),
      "rec03": pp_fn(krec=0.3), "slow10": pp_fn(tau=tau_key * 10), "pess_rec03_slow10": pp_fn(krec=0.3, tau=tau_key * 10),
      "unkeyed_tau": pp_fn(tau=tau_unk), "needs_spread": pp_fn(extra=u_spr), "srch_indep": pp_fn(srch=True), "pr_one": pp_fn(const=1.0)}
R["p_pr_at_T_U3_heavy"] = {nm: pp_at_T(f) for nm, f in PV.items() if f is not None}
RT = {}; DV = {}
for nm, f in PV.items():
    RT[nm], DV[nm] = ratio_all(ver(rho=rho65(f)))
d = max(abs(RT["nokey"][k][p] - r64["ratio_to_V0"]["main"][k][p]) for k in KEYS for p in PARTS)
R["check_064_main"] = {"max_abs_diff": d, "reproduced": d < 1e-12}
RT["064_main"] = r64["ratio_to_V0"]["main"]
R["ratio_to_V0"] = RT
R["scan_p_pr"] = []
for pv in [0.0, 0.25, 0.5, 0.75, 1.0]:
    rr, _ = ratio_all(ver(rho=rho65(pp_fn(const=pv))))
    R["scan_p_pr"].append({"p_pr": pv, M1: rr[M1]["heavy"], "A_XPT_S/U2": rr["A_XPT_S/U2"]["heavy"]})

# ---------- 費用あたり ----------
R["CE"] = {}
for key in KEYS:
    a, u = key.split("/"); R["CE"][key] = {}
    for p in ("total", "heavy"):
        vt = get("main", key)["TAB"][p]; vtc = get("main", key)["TC"][p]; v0 = DV["nokey"][a][u]["TC"][p]; R["CE"][key][p] = {}
        for nm in PV:
            if nm == "nokey": continue
            dpr = DV[nm][a][u]["TC"][p] - v0; dall = DV[nm][a][u]["TC"][p] - vtc
            R["CE"][key][p][nm] = {"primer_increment": qf(dpr / vt * C_T / C_pr) if dpr > 0 else [0.0, 0.0, 0.0],
                                   "booklet_plus_primer": qf(dall / vt * C_T / (C_P(100) + C_pr)) if dall > 0 else [0.0, 0.0, 0.0]}
        R["CE"][key][p]["064_booklet"] = r64["CE"][key][p]["main"]
s_ce = r64["CE_STC_060"]; R["CE_STC_060"] = s_ce

# ---------- 仮説 ----------
H = {}; h = lambda nm, k=M1: RT[nm][k]["heavy"]
v = [R["draws"]["tau_key"]["q05_50_95"][1], R["draws"]["tau_unkeyed"]["q05_50_95"][1]]
H["H77a"] = {"value": v + [v[0] / v[1]], "rule": "median tau_key <= median tau_unkeyed / 3", "supported": v[0] <= v[1] / 3}
v = (0.5 + R["adoption"]["y_main"]) / (1 + len(AD)); H["H77b"] = {"value": v, "rule": "adoption main Jeffreys mean >= 0.6", "supported": v >= 0.6}
v = R["numerals"]["mean_jeffreys"] - R["numerals"]["script_mean_jeffreys"]; H["H77c"] = {"value": v, "rule": "numerals - script (no key) >= 0.4", "supported": v >= 0.4}
v = R["p_pr_at_T_U3_heavy"]["main"]["mean"]; H["H77d"] = {"value": v, "rule": "main p_pr mean at T in [0.4,0.8]", "supported": 0.4 <= v <= 0.8}
v = h("main"); H["H77e"] = {"value": v, "rule": "main heavy >= 0.83", "supported": v >= 0.83}
v = [R["CE"][M1]["total"]["main"]["primer_increment"][1], r64["CE"][M1]["total"]["main"][1]]
H["H77f"] = {"value": v, "rule": "primer increment CE (median) > booklet CE 064", "supported": v[0] > v[1]}
v = h("num_only") - h("txt_only"); H["H77g"] = {"value": v, "rule": "num_only - txt_only >= 0.005", "supported": v >= 0.005}
v = [h("rec03"), h("slow10"), RT["064_main"][M1]["heavy"]]; H["H77h"] = {"value": v, "rule": "rec03 and slow10 each > 064 main", "supported": v[0] > v[2] and v[1] > v[2]}
v = h("main") - h("needs_spread"); H["H77i"] = {"value": v, "rule": "needs_spread lowers heavy < 0.01", "supported": v < 0.01}
for x in H.values(): x["supported"] = bool(x["supported"])
R["hypotheses"] = H

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
rows = [(c["id"], c["end"] - c["start"] if c["end"] else 2026 - c["start"], c["keyed"], c["end"] is not None) for c in CJ["decipher"] if c["start"] is not None]
rows.sort(key=lambda r: (not r[2], r[1]))
for i, (nm, yv, kd, ev) in enumerate(rows):
    axs[0].barh(i, yv, color=("C2" if kd else "C3") if ev else "0.75", hatch=None if ev else "//")
axs[0].set_yticks(range(len(rows))); axs[0].set_yticklabels([f'{r[0]} ({"key" if r[2] else "no key"})' for r in rows], fontsize=7)
axs[0].set_xlabel("years from good copies / key to reading (grey hatched = still unread, censored at 2026)")
off = len(rows) + 1
for j, c in enumerate(CJ["adoption"]):
    axs[0].barh(off + j, c["y"] * 100, color="C0" if c["group"] == "main" else "0.5")
axs[0].set_yticks(list(range(len(rows))) + [off + j for j in range(len(CJ["adoption"]))])
axs[0].set_yticklabels([f'{r[0]} ({"key" if r[2] else "no key"})' for r in rows] + [f'{c["id"]} [{c["group"]}] y={c["y"]}' for c in CJ["adoption"]], fontsize=6)
axs[0].set_title("decipherment lags (top) and script adoption outcome x100 (bottom, all with a teacher)", fontsize=8)
bins = np.logspace(0, 4, 60)
axs[1].hist(tau_key, bins=bins, histtype="step", density=True, label=f"tau_key (keyed) median {np.median(tau_key):.0f} y")
axs[1].hist(tau_unk, bins=bins, histtype="step", density=True, label=f"tau unkeyed median {np.median(tau_unk):.0f} y")
axs[1].hist(tau_dec, bins=bins, histtype="step", density=True, color="0.4", label=f"063 tau_dec (from loss) median {np.median(tau_dec):.0f} y")
axs[1].axvline(np.median(TU[hm]), color="k", ls="--", lw=0.8, label=f"U3 heavy T2s median {np.median(TU[hm]):.0f} y")
axs[1].set_xscale("log"); axs[1].set_xlabel("time constant (years)"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_cases_tau.png"), dpi=130); plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
names = list(PV) + ["064_main"]; vals = [RT[n][M1]["heavy"] for n in names]; vx = [RT[n]["A_XPT_S/U2"]["heavy"] for n in names]
yy = np.arange(len(names))
axs[0].barh(yy - 0.2, vals, height=0.4, label="A_Ord/U3", color=["k" if n == "064_main" else "C0" for n in names])
axs[0].barh(yy + 0.2, vx, height=0.4, label="A_XPT_S/U2", color="C1", alpha=0.7)
axs[0].set_yticks(yy); axs[0].set_yticklabels(names, fontsize=7); axs[0].set_xlim(0.7, 0.92); axs[0].set_xlabel("heavy-collapse TC / V0"); axs[0].legend(fontsize=7)
axs[0].axvline(RT["064_main"][M1]["heavy"], color="k", lw=0.7, ls=":")
sp = R["scan_p_pr"]
axs[1].plot([r["p_pr"] for r in sp], [r[M1] for r in sp], "o-", label="A_Ord/U3"); axs[1].plot([r["p_pr"] for r in sp], [r["A_XPT_S/U2"] for r in sp], "s--", label="A_XPT_S/U2")
axs[1].axvline(R["p_pr_at_T_U3_heavy"]["main"]["mean"], color="C2", ls=":", label="main p_pr mean at T")
axs[1].set_xlabel("p_pr (constant; reading via primer if script lost)"); axs[1].set_ylabel("heavy TC / V0"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_versions_scan.png"), dpi=130); plt.close(fig)

R["runtime_s"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float, ensure_ascii=False)
print("check", R["check_064_main"]); print("adoption", {k: v for k, v in R["adoption"].items() if k != "cases"})
print("decipher", json.dumps(R["decipher"])); print("lang", R["lang_keyed"], "numerals", R["numerals"])
print("draws", json.dumps(R["draws"])); print("costs", json.dumps(R["costs"]))
print("p_pr_at_T", json.dumps({k: round(v["mean"], 4) for k, v in R["p_pr_at_T_U3_heavy"].items()}))
for nm in RT: print(nm, {k: {p: round(RT[nm][k][p], 4) for p in PARTS} for k in KEYS})
print("scan_p_pr", json.dumps(R["scan_p_pr"]))
print("CE total", json.dumps(R["CE"][M1]["total"])); print("CE heavy", json.dumps(R["CE"][M1]["heavy"]))
print("CE XPT total", json.dumps({k: v for k, v in R["CE"]["A_XPT_S/U2"]["total"].items() if k in ("main", "064_booklet")}))
for k, x in H.items(): print(k, x["supported"], x["value"])
print("runtime", R["runtime_s"])
