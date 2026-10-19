"""研究062：座標の冊子の写しが重い崩壊を越えて残り、読まれる確率と写しどうしの相関（open_problems 164）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md。
061 の sim.py を行頭の「# ---------- 仮説 ----------」の前まで exec し、061 の values を差0で再現してから、
061 の P_ptr（一定の相関 c）を「地域が戦乱・放棄にあう → 施設が焼かれる → 崩壊後に朽ちる → 見つかる」の二段の模型に置き換える。新しい乱数は seed 20261062。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P61 = os.path.abspath(os.path.join(HERE, "..", "061-marker-redundancy", "sim.py"))
src = open(P61, encoding="utf8").read(); cut = src.index("\n# ---------- 仮説 ----------\n")
G61 = {"__file__": P61, "__name__": "sim061"}
exec(compile(src[:cut], P61, "exec"), G61)
r61 = json.load(open(os.path.join(HERE, "..", "061-marker-redundancy", "results.json"), encoding="utf8"))
R = {"study": "062-copy-survival-correlation", "seed": 20261062}
leaves = G61["leaves"]
a_ = dict(leaves(r61["values"])); b_ = dict(leaves(json.loads(json.dumps(G61["R"]["values"], default=float))))
R["check_061_values"] = {"n_leaves": len(a_), "max_abs_diff": max(abs(a_[k] - b_[k]) for k in a_)}
R["check_061_values"]["reproduced"] = R["check_061_values"]["max_abs_diff"] < 1e-12

N = G61["N"]; deltas_t = G61["deltas_t"]; ver = G61["ver"]; get = G61["get"]; rho1 = G61["rho1"]; tau1_ev = G61["tau1_ev"]
taup_061 = G61["taup_p"]; fp = G61["fp"]; r_reach = G61["r_reach"]; C_T = G61["C_T"]; C_P = G61["C_P"]; qf = G61["q"]
KEYS = ["A_Ord/U3", "A_XPT_S/U2"]; PARTS = ("total", "heavy", "light"); M1 = "A_Ord/U3"

# ---------- 資料：版の現存部数から τ、施設の出来事から s_b ----------
CJ = json.load(open(os.path.join(HERE, "sources", "cases.json"), encoding="utf8"))
ED = []
for e in CJ["editions"]:
    T = e["count_year"] - e["year"]; lo, hi = e["N0"]
    tau = lambda N0: T / np.log(N0 / e["n"])
    ED.append({"id": e["id"], "kind": e["kind"], "years": T, "n": e["n"], "N0": e["N0"], "surv_frac_mid": e["n"] / np.mean(e["N0"]),
               "tau_mid": float(tau(np.mean(e["N0"]))), "tau_range": [float(tau(hi)), float(tau(lo))]})
ltau = np.log([d["tau_mid"] for d in ED]); mu_t, sd_t = float(ltau.mean()), float(ltau.std(ddof=1))
R["editions"] = ED
R["census_tau"] = {"median_mid": float(np.median([d["tau_mid"] for d in ED])), "logmean": mu_t, "logsd": sd_t, "geo_mean": float(np.exp(mu_t)),
                   "tau061_median": float(np.median(taup_061)), "n": len(ED)}
EV = [e for e in CJ["events"] if e["type"] in ("library_burned", "library_dispersed")]
lg = np.log([e["s"] / (1 - e["s"]) for e in EV]); mu_b, sd_b = float(lg.mean()), float(lg.std(ddof=1))
R["events"] = {"s_burned_or_dispersed": {e["id"]: e["s"] for e in EV}, "logit_mean": mu_b, "logit_sd": sd_b, "median_s": float(np.median([e["s"] for e in EV])),
               "iraq_books_s": 0.75, "iraq_archive_s": 0.40, "warsaw_b": [0.7, 0.8]}

# ---------- 新しい乱数 ----------
g = np.random.default_rng(20261062)
h_R = g.uniform(0.2, 0.9, N); b_R = g.uniform(0.7, 1.0, N)
s_b = 1 / (1 + np.exp(-(mu_b + sd_b * np.sqrt(1 + 1 / len(EV)) * g.standard_normal(N))))
s_u = g.uniform(0.4, 0.75, N)
tau_census = np.exp(mu_t + sd_t * np.sqrt(1 + 1 / len(ED)) * g.standard_normal(N))
bp = [d for d in ED if d["id"] == "BayPsalmBook"][0]; tau_used = g.uniform(*bp["tau_range"], N)
h_high = g.uniform(0.7, 1.0, N)
q_h = (1 - b_R) * s_u + b_R * s_b

def P_new(T, n_p, K=20, taup=None, h=None):
    tp = taup_061 if taup is None else taup; hh = h_R if h is None else h
    Ke = min(n_p, K); m = n_p / Ke
    a = np.exp(-T / tp) * fp
    return 1 - ((1 - hh) * (1 - a) ** m + hh * (1 - q_h * a) ** m) ** Ke
def rho_new(n_p=100, K=20, taup=None, h=None):
    return lambda T: 1 - (1 - rho1) * (1 - r_reach * P_new(T, n_p, K, taup, h))

# ---------- 含意の c（061 の式に戻すと c はいくつか） ----------
def implied_c(T, n_p=100, K=20):
    P = P_new(np.full(N, float(T)), n_p, K)
    pbar = (1 - h_R + h_R * q_h) * np.exp(-T / taup_061) * fp
    neff = np.log(np.maximum(1 - P, 1e-300)) / np.log(1 - pbar)
    c = 1 - (neff - 1) / (n_p - 1)
    Pm = P.mean()
    def f(cs): return (1 - (1 - pbar) ** (1 + (n_p - 1) * (1 - cs))).mean() - Pm
    lo, hi = -50.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if f(mid) > 0: lo = mid
        else: hi = mid
    P61 = 1 - (1 - (1 - G61["lost0_p"]) * np.exp(-T / taup_061) * fp) ** (1 + (n_p - 1) * (1 - G61["cc"]))
    return {"T": T, "n_p": n_p, "K": K, "c_per_draw_q": [float(x) for x in np.quantile(c, [0.05, 0.5, 0.95])], "c_matching_mean": (lo + hi) / 2,
            "P_new_mean": float(Pm), "P_new_q": [float(x) for x in np.quantile(P, [0.05, 0.5, 0.95])], "P_061_mean": float(P61.mean()),
            "pbar_mean": float(pbar.mean()), "shock_survival_mean": float((1 - h_R + h_R * q_h).mean()), "lost0_061_mean": float(G61["lost0_p"].mean())}
R["implied_c"] = [implied_c(T, n, K) for T in (300, 1000) for (n, K) in ((100, 20), (10, 10), (1000, 20), (100, 1))]

# ---------- 版 ----------
def ratio(o, key, p, x="TC"):
    a, u = key.split("/"); return deltas_t(o)[a][u][x][p] / get("V0", key)[x][p]
NV = {"p62": dict(), "p62_K1": dict(K=1), "p62_K5": dict(K=5), "p62_census_tau": dict(taup=tau_census), "p62_used_tau": dict(taup=tau_used), "p62_h_high": dict(h=h_high)}
VALS = {}
for bv, t1 in (("main", None), ("tau1_ev", tau1_ev)):
    for nm, kw in NV.items():
        VALS[(bv, nm)] = deltas_t(ver(tau1=t1, rho=rho_new(**kw)))
def g2(bv, nm, key): a, u = key.split("/"); return VALS[(bv, nm)][a][u]
R["ratio_to_V0"] = {bv: {nm: {key: {x: {p: g2(bv, nm, key)[x][p] / get("V0", key)[x][p] for p in PARTS} for x in ("TAB", "TC")} for key in KEYS} for nm in NV} for bv in ("main", "tau1_ev")}
R["ratio_061"] = {v: {key: r61["ratio_to_V0"][v][key]["TC"] for key in KEYS} for v in ("main", "tau1_ev", "pointer", "tau1_ev_pointer", "marker")}

# ---------- 走査 ----------
R["scan_np"] = []; R["scan_K"] = []; R["scan_h"] = []
for n_p in [1, 3, 10, 30, 100, 300, 1000]:
    R["scan_np"].append({"n_p": n_p, **{key: {p: ratio(ver(rho=rho_new(n_p=n_p)), key, p) for p in ("heavy", "light")} for key in (M1,)}})
for K in [1, 2, 5, 10, 20, 50, 100]:
    R["scan_K"].append({"K": K, **{key: {p: ratio(ver(rho=rho_new(K=K)), key, p) for p in ("heavy", "light")} for key in (M1,)}})
for hv in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    R["scan_h"].append({"h": hv, **{key: {p: ratio(ver(rho=rho_new(h=np.full(N, hv))), key, p) for p in ("heavy", "light")} for key in (M1,)}})
m_heavy = get("main", M1)["TC"]["heavy"] / get("V0", M1)["TC"]["heavy"]
rows = {r["n_p"]: r[M1]["heavy"] for r in R["scan_np"]}
R["saturation_10_of_1000_K20"] = (rows[10] - m_heavy) / (rows[1000] - m_heavy)
rK = {r["K"]: r[M1]["heavy"] for r in R["scan_K"]}
R["K1_gain_over_K20_gain"] = (rK[1] - m_heavy) / (rK[20] - m_heavy)

# ---------- 費用あたり ----------
R["CE"] = {}
for bv in ("main", "tau1_ev"):
    R["CE"][bv] = {}
    for key in KEYS:
        R["CE"][bv][key] = {}
        for p in PARTS:
            vt = get(bv, key)["TAB"][p]; vtc = get(bv, key)["TC"][p]; d = {}
            for nm in NV:
                dv = g2(bv, nm, key)["TC"][p] - vtc
                d[nm] = {"dV_over_TAB": dv / vt, "CE_q": qf(dv / vt * C_T / C_P(100)) if dv > 0 else [0.0, 0.0, 0.0]}
            d["p61"] = {"CE_q": G61["R"]["CE"][bv][key][p]["pointer"]["CE_q"]}
            R["CE"][bv][key][p] = d
s_ce = G61["R"]["CE_STC_060_main"][M1]["total"]["S_TC_given_TC"]; R["CE_STC_060"] = s_ce

# ---------- 仮説 ----------
rt = R["ratio_to_V0"]["main"]; Hh = {}
v = R["census_tau"]["median_mid"]; Hh["H74a"] = {"value": v, "rule": "census tau median > 061 taup median", "supported": v > R["census_tau"]["tau061_median"]}
v = R["implied_c"][0]["c_per_draw_q"][1]; Hh["H74b"] = {"value": v, "rule": "implied c (median, n_p=100, K=20, T=300) in [0.5, 0.95]", "supported": 0.5 <= v <= 0.95}
v = [rt["p62"][M1]["TC"]["heavy"], R["ratio_061"]["pointer"][M1]["heavy"]]; Hh["H74c"] = {"value": v, "rule": "|heavy TC/V0 new - 061 pointer| <= 0.05", "supported": abs(v[0] - v[1]) <= 0.05}
v = R["saturation_10_of_1000_K20"]; Hh["H74d"] = {"value": v, "rule": "gain n_p=10 >= 50% of n_p=1000 (K=20, heavy)", "supported": v >= 0.5}
v = R["K1_gain_over_K20_gain"]; Hh["H74e"] = {"value": v, "rule": "gain K=1 < 50% of K=20 (n_p=100, heavy)", "supported": v < 0.5}
v = [R["CE"]["main"][M1]["total"]["p62"]["CE_q"][1], s_ce]; Hh["H74f"] = {"value": v, "rule": "CE new pointer > CE S_TC|TC", "supported": v[0] > v[1]}
v = [R["CE"]["main"][M1]["total"]["p62_K1"]["CE_q"][1], s_ce]; Hh["H74g"] = {"value": v, "rule": "CE K=1 < CE S_TC|TC", "supported": v[0] < v[1]}
v = [rt["p62_census_tau"][M1]["TC"]["heavy"], rt["p62"][M1]["TC"]["heavy"]]; Hh["H74h"] = {"value": v, "rule": "census_tau heavy TC >= p62 heavy TC", "supported": v[0] >= v[1]}
for h in Hh.values(): h["supported"] = bool(h["supported"])
R["hypotheses"] = Hh

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
ids = [d["id"] for d in ED]; ys = np.arange(len(ED))
axs[0].errorbar([d["tau_mid"] for d in ED], ys, xerr=[[d["tau_mid"] - d["tau_range"][0] for d in ED], [d["tau_range"][1] - d["tau_mid"] for d in ED]], fmt="o")
axs[0].axvspan(100, 1000, color="0.85", label="061 tau_p prior"); axs[0].set_xscale("log"); axs[0].set_yticks(ys); axs[0].set_yticklabels(ids, fontsize=7)
axs[0].set_xlabel("per-copy survival time constant tau (years)"); axs[0].legend(fontsize=7)
Ts = np.linspace(0, 2000, 201)
for (nm, n_p, K, col) in (("new n=100 K=20", 100, 20, "C2"), ("new n=10 K=10", 10, 10, "C4"), ("new n=100 K=1", 100, 1, "C3"), ("new n=1000 K=20", 1000, 20, "C1")):
    axs[1].plot(Ts, [P_new(np.full(N, t), n_p, K).mean() for t in Ts], color=col, label=nm)
P61f = lambda t, n: (1 - (1 - (1 - G61["lost0_p"]) * np.exp(-t / taup_061) * fp) ** (1 + (n - 1) * (1 - G61["cc"]))).mean()
axs[1].plot(Ts, [P61f(t, 100) for t in Ts], "k--", label="061 n=100"); axs[1].plot(Ts, [P61f(t, 10) for t in Ts], "k:", label="061 n=10")
axs[1].set_xlabel("years from collapse to reading T"); axs[1].set_ylabel("mean P(at least one pointer found)"); axs[1].legend(fontsize=7); axs[1].set_ylim(0, 1)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_census_Pptr.png"), dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].semilogx([r["n_p"] for r in R["scan_np"]], [r[M1]["heavy"] for r in R["scan_np"]], "-o", label="new, K=min(n,20)")
axs[0].semilogx([r["n_p"] for r in r61["scan_np"]], [r[M1]["main"]["heavy"] for r in r61["scan_np"]], "--s", color="k", label="061")
axs[0].axhline(m_heavy, color="0.5", lw=0.7); axs[0].set_xlabel("copies n_p"); axs[0].set_ylabel("heavy TC / V0 (A_Ord/U3)"); axs[0].legend(fontsize=7)
axs[1].semilogx([r["K"] for r in R["scan_K"]], [r[M1]["heavy"] for r in R["scan_K"]], "-o", color="C2", label="n_p=100, vary regions K")
axs[1].plot([r["h"] * 100 + 1 for r in R["scan_h"]], [r[M1]["heavy"] for r in R["scan_h"]], "-^", color="C3", label="vary hit prob h (x = 100h+1)")
axs[1].axhline(m_heavy, color="0.5", lw=0.7); axs[1].set_xlabel("K regions (or 100h+1)"); axs[1].set_ylabel("heavy TC / V0"); axs[1].legend(fontsize=7)
lab = []; pos = 0
for bv in ("main", "tau1_ev"):
    for nm in ["p61"] + list(NV):
        c = R["CE"][bv][M1]["total"][nm]["CE_q"]
        if c[1] > 0: axs[2].errorbar(pos, c[1], yerr=[[c[1] - max(c[0], 1e-3)], [c[2] - c[1]]], fmt="o", color="C0" if bv == "main" else "C3")
        lab.append(f"{bv}:{nm}"); pos += 1
axs[2].axhline(s_ce, color="C2", ls="--", label="S_TC|TC (060)"); axs[2].set_yscale("log"); axs[2].set_xticks(range(pos)); axs[2].set_xticklabels(lab, rotation=60, fontsize=6)
axs[2].set_ylabel("CE (TAB=1), median 5-95%"); axs[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_scans_ce.png"), dpi=130); plt.close(fig)

R["runtime_s"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float, ensure_ascii=False)
print("check", R["check_061_values"])
print("editions", json.dumps([(d["id"], round(d["tau_mid"]), [round(x) for x in d["tau_range"]], round(d["surv_frac_mid"], 3)) for d in ED]))
print("census_tau", json.dumps(R["census_tau"])); print("events", json.dumps(R["events"]))
print("implied_c", json.dumps(R["implied_c"]))
for bv in ("main", "tau1_ev"):
    for nm in NV: print(bv, nm, {key: {p: round(R["ratio_to_V0"][bv][nm][key]["TC"][p], 3) for p in PARTS} for key in KEYS})
print("ratio_061", json.dumps(R["ratio_061"]))
print("scan_np", json.dumps(R["scan_np"])); print("scan_K", json.dumps(R["scan_K"])); print("scan_h", json.dumps(R["scan_h"]))
print("sat", R["saturation_10_of_1000_K20"], "K1/K20", R["K1_gain_over_K20_gain"])
print("CE", json.dumps({bv: {key: {p: {nm: d["CE_q"] for nm, d in R["CE"][bv][key][p].items()} for p in PARTS} for key in KEYS} for bv in R["CE"]}))
print("S_TC CE", s_ce)
for h, d in Hh.items(): print(h, d["supported"], d["value"])
print("runtime", R["runtime_s"])
