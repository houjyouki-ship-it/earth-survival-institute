"""研究061：目印と座標の写しで、忘れられた金庫を見つける確率をどれだけ上げられるか、その費用（open_problems 162）＋遠い場所の位置の知識の時定数 τ1（161）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md。
060 の sim.py を「# ---------- 仮説 ----------」の前まで exec し、060 の values を差0で再現してから、発見の係数の ρ1 を対策ごとの ρ_eff(T) に置き換える。新しい乱数は seed 20261061。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P60 = os.path.abspath(os.path.join(HERE, "..", "060-location-memory-decay", "sim.py"))
src = open(P60, encoding="utf8").read(); cut = src.index("\n# ---------- 仮説 ----------\n")
G60 = {"__file__": P60, "__name__": "sim060"}
exec(compile(src[:cut], P60, "exec"), G60)
r60 = json.load(open(os.path.join(HERE, "..", "060-location-memory-decay", "results.json"), encoding="utf8"))
R = {"study": "061-marker-redundancy", "seed": 20261061}

def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items(): yield from leaves(v, p + "/" + k)
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from leaves(v, p + f"[{i}]")
    else: yield p, x
a_ = dict(leaves(r60["values"])); b_ = dict(leaves(json.loads(json.dumps(G60["R"]["values"], default=float))))
R["check_060_values"] = {"n_leaves": len(a_), "max_abs_diff": max(abs(a_[k] - b_[k]) for k in a_)}
R["check_060_values"]["reproduced"] = R["check_060_values"]["max_abs_diff"] < 1e-12

N = G60["N"]; VERS60 = G60["VERS"]; deltas_t = G60["deltas_t"]; G55 = G60["G55"]
KEYS = ["A_Ord/U3", "A_XPT_S/U2"]; PARTS = ("total", "heavy", "light")

# ---------- τ1 の根拠（161）：区間打ち切りの指数 ----------
S = os.path.join(HERE, "sources")
CJ = json.load(open(os.path.join(S, "cases.json"), encoding="utf8"))
CASES = [c for c in CJ["cases"] if c["class"] in ("remote", "inhabited")]
tg = np.exp(np.linspace(np.log(10), np.log(1e4), 4001))   # 事前 logU(10, 1e4)：対数の格子で一様
def post(cs):
    ll = np.zeros_like(tg)
    for c in cs:
        L, U = c["L"], c["U"]
        pL = np.exp(-L / tg)                   # L まで知られている
        pU = np.exp(-U / tg) if U is not None else 0.0
        ll += np.log(np.maximum(pL - pU, 1e-300))
    w = np.exp(ll - ll.max()); w /= w.sum(); return w
def qpost(w, qs=(0.05, 0.5, 0.95)):
    c = np.cumsum(w); return [float(tg[np.searchsorted(c, q)]) for q in qs]
W_rem = post([c for c in CASES if c["class"] == "remote"]); W_inh = post([c for c in CASES if c["class"] == "inhabited"]); W_all = post(CASES)
R["tau_posterior"] = {"remote": qpost(W_rem), "inhabited": qpost(W_inh), "all": qpost(W_all), "prior": [float(np.exp(np.log(10) + q * np.log(1000))) for q in (0.05, 0.5, 0.95)],
                      "main060_tau1_median": float(np.median(VERS60["main"]["tau1"])), "n_remote": 2, "n_inhabited": 5}

# ---------- 新しい乱数 ----------
g = np.random.default_rng(20261061)
def lu(lo, hi, n=N): return np.exp(g.uniform(np.log(lo), np.log(hi), n))
cw = np.cumsum(W_rem); tau1_ev = tg[np.minimum(np.searchsorted(cw, g.uniform(size=N)), len(tg) - 1)]
tau_M = lu(300, 5000); rho_M = g.uniform(0.5, 0.84, N)
lost0_p = g.uniform(0.3, 0.8, N); taup_p = lu(100, 1000); fp = g.uniform(0.3, 0.84, N); cc = g.uniform(0.5, 0.95, N); r_reach = g.uniform(0.5, 1.0, N)
NC = 20000
C_M = lu(4e5, 1e7, NC); prep_P = lu(1e3, 1e4, NC); per_P = lu(1e2, 1e3, NC) + g.uniform(25, 50, NC)

def rho_marker(rho1, tauM=None):
    tm = tau_M if tauM is None else tauM
    return lambda T: 1 - (1 - rho1) * (1 - np.exp(-T / tm) * rho_M)
def P_ptr(T, n_p):
    p = (1 - lost0_p) * np.exp(-T / taup_p) * fp
    neff = 1 + (n_p - 1) * (1 - cc)
    return 1 - (1 - p) ** neff
def rho_pointer(rho1, n_p=100):
    return lambda T: 1 - (1 - rho1) * (1 - r_reach * P_ptr(T, n_p))
def rho_both(rho1, n_p=100, tauM=None):
    tm = tau_M if tauM is None else tauM
    return lambda T: 1 - (1 - rho1) * (1 - np.exp(-T / tm) * rho_M) * (1 - r_reach * P_ptr(T, n_p))

# 060 の Dfac を ρ が T の関数でもよいように置き換える（ρ が配列・数のときは 060 と同じ）
_D0 = G60["Dfac"]
def Dfac(T, tau, rho):
    if tau is None: return np.ones(N)
    if callable(rho): rho = rho(T)
    m = np.exp(-T / tau); return m + (1 - m) * rho
G60["Dfac"] = Dfac

base = VERS60["main"]; rho1 = base["rho1"]
def ver(tau1=None, rho=None):
    o = dict(base)
    if tau1 is not None: o["tau1"] = tau1
    if rho is not None: o["rho1"] = rho
    return o
VERS = {"V0": {}, "main": ver(), "tau1_ev": ver(tau1=tau1_ev),
        "marker": ver(rho=rho_marker(rho1)), "pointer": ver(rho=rho_pointer(rho1)), "both": ver(rho=rho_both(rho1)),
        "tau1_ev_marker": ver(tau1=tau1_ev, rho=rho_marker(rho1)), "tau1_ev_pointer": ver(tau1=tau1_ev, rho=rho_pointer(rho1)), "tau1_ev_both": ver(tau1=tau1_ev, rho=rho_both(rho1))}
VALS = {v: deltas_t(o) for v, o in VERS.items()}
# 検査：main が 060 の main と一致
R["check_main_vs_060"] = max(abs(VALS["main"][k.split("/")[0]][k.split("/")[1]][x][p] - r60["values"]["main"][k][x][p]) for k in KEYS for x in ("TAB", "TC", "S_TC_given_TC") for p in PARTS)
def get(v, key): a, u = key.split("/"); return VALS[v][a][u]
R["values"] = {v: {key: get(v, key) for key in KEYS} for v in VALS}
R["ratio_to_V0"] = {v: {key: {x: {p: get(v, key)[x][p] / get("V0", key)[x][p] for p in PARTS} for x in ("TAB", "TC")} for key in KEYS} for v in VALS}

# ---------- 走査 ----------
NPG = [1, 3, 10, 30, 100, 300, 1000]; TMG = [30, 100, 300, 1000, 3000, 10000]
def tc_ratio(o, key, p): a, u = key.split("/"); return deltas_t(o)[a][u]["TC"][p] / get("V0", key)["TC"][p]
R["scan_np"] = []
for n_p in NPG:
    Vk = deltas_t(ver(rho=rho_pointer(rho1, n_p))); Ve = deltas_t(ver(tau1=tau1_ev, rho=rho_pointer(rho1, n_p)))
    R["scan_np"].append({"n_p": n_p, **{key: {"main": {p: Vk[key.split("/")[0]][key.split("/")[1]]["TC"][p] / get("V0", key)["TC"][p] for p in PARTS},
                                               "tau1_ev": {p: Ve[key.split("/")[0]][key.split("/")[1]]["TC"][p] / get("V0", key)["TC"][p] for p in PARTS}} for key in KEYS}})
R["scan_tauM"] = []
for tm in TMG:
    Vk = deltas_t(ver(rho=rho_marker(rho1, float(tm))))
    R["scan_tauM"].append({"tau_M": tm, **{key: {p: Vk[key.split("/")[0]][key.split("/")[1]]["TC"][p] / get("V0", key)["TC"][p] for p in PARTS} for key in KEYS}})
def gain_frac(n_lo, n_hi, key="A_Ord/U3", p="heavy", base_v="main"):
    rows = {r["n_p"]: r[key][base_v][p] for r in R["scan_np"]}; m0 = get(base_v if base_v == "main" else "tau1_ev", key)["TC"][p] / get("V0", key)["TC"][p]
    return (rows[n_lo] - m0) / (rows[n_hi] - m0)
R["pointer_saturation_10_of_1000"] = {"main_heavy": gain_frac(10, 1000), "tau1_ev_heavy": gain_frac(10, 1000, base_v="tau1_ev")}

# ---------- 費用あたり（055 の V_mix の C_T、字数 1e7〜1e8、TAB=1 の物差し） ----------
C_T = G55["cost_T"]("V_mix")[0][:NC]
def q(x): return [float(v) for v in np.quantile(x, [0.05, 0.5, 0.95])]
def C_P(n_p): return prep_P + n_p * per_P
R["costs"] = {"C_T": q(C_T), "C_M": q(C_M), "C_P_100": q(C_P(100)), "C_P_10": q(C_P(10)), "C_P_1000": q(C_P(1000))}
PCACHE = {(bv, n_p): deltas_t(ver(tau1=None if bv == "main" else tau1_ev, rho=rho_pointer(rho1, n_p))) for bv in ("main", "tau1_ev") for n_p in (10, 1000)}
R["CE"] = {}
for bv in ("main", "tau1_ev"):
    R["CE"][bv] = {}
    for key in KEYS:
        R["CE"][bv][key] = {}
        for p in PARTS:
            vt = get(bv, key)["TAB"][p]; vtc = get(bv, key)["TC"][p]; d = {}
            for x, c in (("marker", C_M), ("pointer", C_P(100)), ("both", C_M + C_P(100))):
                xv = x if bv == "main" else "tau1_ev_" + x
                dv = get(xv, key)["TC"][p] - vtc
                d[x] = {"dV_over_TAB": dv / vt, "CE_q": q(dv / vt * C_T / c) if dv > 0 else [0.0, 0.0, 0.0]}
            for n_p in (10, 1000):
                a, u = key.split("/"); dv = PCACHE[(bv, n_p)][a][u]["TC"][p] - vtc
                d[f"pointer_{n_p}"] = {"dV_over_TAB": dv / vt, "CE_q": q(dv / vt * C_T / C_P(n_p)) if dv > 0 else [0.0, 0.0, 0.0]}
            R["CE"][bv][key][p] = d
R["CE_STC_060_main"] = {key: r60["CE_medians"]["main"]["chars_1e7_1e8"][key] for key in KEYS}

# ---------- 仮説 ----------
M1 = "A_Ord/U3"; rt = R["ratio_to_V0"]; Hh = {}
v = R["tau_posterior"]["remote"][1]; Hh["H73a"] = {"value": v, "rule": "remote tau posterior median < 060 main tau1 median", "supported": v < R["tau_posterior"]["main060_tau1_median"]}
v = rt["tau1_ev"][M1]["TAB"]["heavy"]; Hh["H73b"] = {"value": v, "rule": "tau1_ev heavy TAB / V0 < main (0.63)", "supported": v < rt["main"][M1]["TAB"]["heavy"]}
v = rt["marker"][M1]["TC"]["heavy"]; Hh["H73c"] = {"value": v, "rule": "marker heavy TC / V0 >= 0.8", "supported": v >= 0.8}
v = [rt["pointer"][M1]["TC"]["heavy"], rt["marker"][M1]["TC"]["heavy"]]; Hh["H73d"] = {"value": v, "rule": "heavy TC/V0: pointer >= marker", "supported": v[0] >= v[1]}
s_ce = R["CE_STC_060_main"][M1]["total"]["S_TC_given_TC"]
v = [R["CE"]["main"][M1]["total"]["pointer"]["CE_q"][1], s_ce]; Hh["H73e"] = {"value": v, "rule": "median CE pointer > CE S_TC|TC (060 main)", "supported": v[0] > v[1]}
v = [R["CE"]["main"][M1]["total"]["marker"]["CE_q"][1], s_ce]; Hh["H73f"] = {"value": v, "rule": "median CE marker < CE S_TC|TC (060 main)", "supported": v[0] < v[1]}
v = [rt["marker"][M1]["TC"]["light"] / rt["main"][M1]["TC"]["light"] - 1, rt["pointer"][M1]["TC"]["light"] / rt["main"][M1]["TC"]["light"] - 1]
Hh["H73g"] = {"value": v, "rule": "|light TC change| < 5% for marker and pointer", "supported": all(abs(x) < 0.05 for x in v)}
v = R["pointer_saturation_10_of_1000"]["main_heavy"]; Hh["H73h"] = {"value": v, "rule": "pointer gain at n_p=10 >= 80% of n_p=1000 (heavy)", "supported": v >= 0.8}
for h in Hh.values(): h["supported"] = bool(h["supported"])
R["hypotheses"] = Hh

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for w, nm, col in ((W_rem, "remote (n=2)", "C3"), (W_inh, "inhabited (n=5)", "C0"), (W_all, "all (n=7)", "C2")):
    axs[0].semilogx(tg, w / w.max(), color=col, label=nm)
axs[0].axvspan(100, 1000, color="0.85", label="060 prior tau1"); axs[0].set_xlabel("tau (years, exact-location memory)"); axs[0].set_ylabel("posterior (scaled)"); axs[0].legend(fontsize=8)
Tg = np.linspace(0, 3000, 301)
for nm, f, col in (("forgotten find prob rho1 (median)", lambda T: np.full_like(T, np.median(rho1)), "k"),
                   ("+ marker", lambda T: 1 - (1 - np.median(rho1)) * (1 - np.exp(-T / np.median(tau_M)) * np.median(rho_M)), "C1"),
                   ("+ pointer n=100", lambda T: 1 - (1 - np.median(rho1)) * (1 - 0.75 * (1 - (1 - 0.55 * 0.57 * np.exp(-T / np.median(taup_p))) ** (1 + 99 * 0.275))), "C2"),
                   ("+ pointer n=10", lambda T: 1 - (1 - np.median(rho1)) * (1 - 0.75 * (1 - (1 - 0.55 * 0.57 * np.exp(-T / np.median(taup_p))) ** (1 + 9 * 0.275))), "C4")):
    axs[1].plot(Tg, f(Tg), color=col, label=nm)
axs[1].set_xlabel("years from collapse to reading (T2s)"); axs[1].set_ylabel("rho_eff(T) (median parameters)"); axs[1].legend(fontsize=7); axs[1].set_ylim(0, 1)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_tau_posterior_rho_eff.png"), dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
vn = [v for v in VALS if v != "V0"]; xs = np.arange(len(vn))
for j, (p, col) in enumerate((("heavy", "C3"), ("light", "C0"), ("total", "C2"))):
    axs[0].bar(xs + (j - 1) * 0.27, [rt[v][M1]["TC"][p] for v in vn], 0.27, color=col, label=p)
axs[0].axhline(1, color="k", lw=0.7); axs[0].set_xticks(xs); axs[0].set_xticklabels(vn, rotation=45, fontsize=7); axs[0].set_ylabel("TC value / V0 (A_Ord/U3)"); axs[0].legend(fontsize=7)
for bv, ls in (("main", "-o"), ("tau1_ev", "--s")):
    axs[1].semilogx(NPG, [r[M1][bv]["heavy"] for r in R["scan_np"]], ls, label=f"pointer, heavy, {bv}")
axs[1].semilogx(TMG, [r[M1]["heavy"] for r in R["scan_tauM"]], "-^", label="marker vs tau_M (x = years)")
axs[1].set_xlabel("n_p copies (or tau_M years)"); axs[1].set_ylabel("heavy TC / V0"); axs[1].legend(fontsize=7)
lab = []; pos = 0
for bv in ("main", "tau1_ev"):
    for x in ("marker", "pointer_10", "pointer", "pointer_1000", "both"):
        c = R["CE"][bv][M1]["total"][x]["CE_q"]; axs[2].errorbar(pos, max(c[1], 1e-3), yerr=[[max(c[1], 1e-3) - max(c[0], 1e-4)], [c[2] - c[1]]], fmt="o", color="C0" if bv == "main" else "C3"); lab.append(f"{bv}:{x}"); pos += 1
axs[2].axhline(s_ce, color="C2", ls="--", label="S_TC|TC (060 main)"); axs[2].axhline(1, color="k", lw=0.6)
axs[2].set_yscale("log"); axs[2].set_xticks(range(pos)); axs[2].set_xticklabels(lab, rotation=60, fontsize=6); axs[2].set_ylabel("CE (TAB=1), median 5-95%"); axs[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_versions_ce.png"), dpi=130); plt.close(fig)

R["runtime_s"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float, ensure_ascii=False)
print("checks", R["check_060_values"], R["check_main_vs_060"])
print("tau_posterior", json.dumps(R["tau_posterior"]))
for v in VALS: print(v, {key: {x: {p: round(y, 3) for p, y in d.items()} for x, d in rt[v][key].items()} for key in KEYS})
print("scan_np", json.dumps(R["scan_np"])); print("scan_tauM", json.dumps(R["scan_tauM"])); print("saturation", R["pointer_saturation_10_of_1000"])
print("costs", json.dumps(R["costs"]))
print("CE", json.dumps(R["CE"])); print("CE_STC_060", json.dumps(R["CE_STC_060_main"]))
for h, d in Hh.items(): print(h, d["supported"], d["value"])
print("runtime", R["runtime_s"])
