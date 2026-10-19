"""研究060：重い崩壊で拠点の場所の知識が忘れられる効果（open_problems 149）と、法定納本の紙の時間による喪失（152）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md。
055 の sim.py を「# ---------- 仮説 ----------」の前まで exec し、055 の main_chars を差0で再現してから、
054 の V_ext・055 の V_ld を写して発見の確率に D(T)=m(T)+(1-m(T))ρ、m=exp(-T/τ)、T=T2s を掛ける。新しい乱数は seed 20261060。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P55 = os.path.abspath(os.path.join(HERE, "..", "055-authoring-productivity", "sim.py"))
src = open(P55, encoding="utf8").read(); cut = src.index("# ---------- 仮説 ----------")
G55 = {"__file__": P55, "__name__": "sim055"}
exec(compile(src[:cut], P55, "exec"), G55)
r55 = json.load(open(os.path.join(HERE, "..", "055-authoring-productivity", "results.json")))
R = {"study": "060-location-memory-decay", "seed": 20261060}

def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items(): yield from leaves(v, p + "/" + k)
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from leaves(v, p + f"[{i}]")
    else: yield p, x
a_ = dict(leaves(r55["main_chars"])); b_ = dict(leaves(json.loads(json.dumps(G55["R"]["main_chars"], default=float))))
R["check_055_main_chars"] = {"n_leaves": len(a_), "max_abs_diff": max(abs(a_[k] - b_[k]) for k in a_)}
R["check_055_main_chars"]["reproduced"] = R["check_055_main_chars"]["max_abs_diff"] < 1e-12

G = G55["G"]; N = G55["N"]; COMPS = G55["COMPS"]; B = G55["B"]; SC = G55["SC"]; agg = G55["agg"]
ns = G["ns"]
U_lost, U_find, Z, life, H = (ns[k] for k in ("U_lost", "U_find", "Z", "life", "H"))
U_sep, p_sep, t_build, pop_ok, U_t, P_max, kappa, U_acc, p_acc = (ns[k] for k in ("U_sep", "p_sep", "t_build", "pop_ok", "U_t", "P_max", "kappa", "U_acc", "p_acc"))
U_T3, q_B, q_TC, find2 = G["U_T3"], G["q_B"], G["q_TC"], G["find2"]
U_lost2, U_find2, U_corr, lost_tc = G["U_lost2"], G["U_find2"], G["U_corr"], G["lost_tc"]
U_lost3, U_find3, U_acc3, p_accept, lost_ld_main, lost_ld_multi = (G55[k] for k in ("U_lost3", "U_find3", "U_acc3", "p_accept", "lost_ld_main", "lost_ld_multi"))
VAL54, VLD55 = G55["VAL54"], G55["VLD"]

# ---------- 資料の要約（sources/） ----------
S = os.path.join(HERE, "sources")
cases = json.load(open(os.path.join(S, "cases.json"), encoding="utf8"))
lc = json.load(open(os.path.join(S, "lost_city_counts.json"), encoding="utf8"))
nr, nu = lc["Rediscovered"]["n"], lc["Undiscovered"]["n"]; nq = lc["Uncertain or disputed"]["n"] + lc["Status Unknown"]["n"]
R["evidence"] = {"lost_city": {"rediscovered": nr, "undiscovered": nu, "uncertain_or_unknown": nq,
                               "frac_found_hi": nr / (nr + nu), "frac_found_lo": nr / (nr + nu + nq)}}
cs = {c["id"]: c for c in cases["cases"]}
R["evidence"]["copper_scroll_frac_found_max"] = cs["CopperScroll"]["n_found_max"] / cs["CopperScroll"]["n_listed"]
gaps = {c["id"]: c["found"] - c["hidden"] for c in cases["cases"] if c.get("hidden") is not None and c.get("found") is not None}
R["evidence"]["gaps_years"] = gaps
R["evidence"]["found_by_counts"] = {}
for c in cases["cases"]:
    k = str(c.get("found_by")); R["evidence"]["found_by_counts"][k] = R["evidence"]["found_by_counts"].get(k, 0) + 1
chance_caches = [c["id"] for c in cases["cases"] if c["kind"] == "cache" and c["found_by"] == "chance"]
R["evidence"]["cache_found_by_chance"] = {"n": len(chance_caches), "of_found_caches": sum(1 for c in cases["cases"] if c["kind"] == "cache" and c.get("found") and c["found_by"] != "continuous"), "ids": chance_caches,
                                          "gap_median": float(np.median([gaps[i] for i in chance_caches]))}
pp = {c["id"]: c for c in cases["paper"]}
surv = {"Aeschylus": [7 / 90, 7 / 70], "Sophocles": [7 / 130, 7 / 120], "Euripides": [0.2, 0.2], "Livy": [35 / 142, 35 / 142]}
tyrs = {"Aeschylus": (1500, 2000), "Sophocles": (1500, 2000), "Euripides": (1500, 2000), "Livy": (1300, 1500)}   # 【推測】書かれてから印刷まで
R["evidence"]["classical_survival"] = {k: {"s": v, "implied_tau_years": [tyrs[k][0] / np.log(1 / v[1]), tyrs[k][1] / np.log(1 / v[0])]} for k, v in surv.items()}
R["evidence"]["war_library_loss"] = {"Iraq2003_books": 0.25, "Iraq2003_archives": 0.60, "Zaluski": 1 - pp["Zaluski"]["survived"] / pp["Zaluski"]["items_1780s"]}

# ---------- 新しい乱数 ----------
g = np.random.default_rng(20261060)
def lu(lo, hi, n=N): return np.exp(g.uniform(np.log(lo), np.log(hi), n))
Uq1 = g.uniform(size=N); Uq2 = g.uniform(size=N)
tau1 = lu(100, 1000); rho1 = np.exp(np.log(0.02) + Uq1 * (np.log(0.8) - np.log(0.02)))
tau2 = lu(300, 3000); rho2 = 0.3 + Uq2 * (0.84 - 0.3)
taup = lu(100, 1000)
rho_city1 = g.uniform(0.77, 0.84, N); rho_city2 = g.uniform(0.77, 0.84, N)
rho_cache = g.uniform(0.0, 0.03, N)
tau_long = lu(1000, 5000); taup_short = lu(30, 300)
lost0_ev = g.uniform(0.25, 0.92, N)
rho2_corr = 0.3 + Uq1 * (0.84 - 0.3)

VERS = {
 "V0": {},
 "main": {"tau1": tau1, "rho1": rho1, "tau2": tau2, "rho2": rho2, "taup": taup},
 "city": {"tau1": tau1, "rho1": rho_city1, "tau2": tau2, "rho2": rho_city2, "taup": taup},
 "cache": {"tau1": tau1, "rho1": rho_cache, "tau2": tau2, "rho2": rho2, "taup": taup},
 "mem_long": {"tau1": tau_long, "rho1": rho1, "tau2": tau2, "rho2": rho2, "taup": taup},
 "marker": {"tau1": tau1, "rho1": 0.9, "tau2": tau2, "rho2": 0.9, "taup": taup},
 "paper_best": {"tau1": tau1, "rho1": rho1, "tau2": tau2, "rho2": rho2, "taup": 1000.0},
 "paper_short": {"tau1": tau1, "rho1": rho1, "tau2": tau2, "rho2": rho2, "taup": taup_short},
 "lost0_ev": {"tau1": tau1, "rho1": rho1, "tau2": tau2, "rho2": rho2, "taup": taup, "lost0": lost0_ev},
 "corr": {"tau1": tau1, "rho1": rho1, "tau2": tau2, "rho2": rho2_corr, "taup": taup},
}

def Dfac(T, tau, rho):
    if tau is None: return np.ones(N)
    m = np.exp(-T / tau); return m + (1 - m) * rho

def V_t(k, qv, tablet, S=None, opt=None, kind_ld=None):
    """054 の V_ext（u_on=True、S は None/'tc'）と 055 の V_ld の写しに D を掛けたもの。"""
    opt = opt or {}
    st, T2s, sc = B[k]["st"], B[k]["T2s"], SC[k]
    T = np.maximum(T2s, 0.0)
    D1 = Dfac(T, opt.get("tau1"), opt.get("rho1")); D2 = Dfac(T, opt.get("tau2"), opt.get("rho2"))
    alive = st != 0
    intact1 = (U_lost > sc["p_lost"]) & (U_find < sc["p_find"] * D1)
    T3t = T2s + 943 * sc["M"] * np.exp(0.8 * Z[2]); gen = (T3t < life) & (T3t <= H)
    cov = np.zeros(N); avail = np.zeros(N, bool)
    if tablet:
        tab_ok = U_sep > p_sep; cov = np.where(tab_ok, 1.0, cov); avail |= tab_ok
    Tt = T2s + t_build
    tgt = avail & pop_ok & (U_t < P_max * cov ** kappa) & (Tt < life) & (Tt <= H)
    frag_acc = (st == 2) & (U_acc < p_acc)
    def finish(v, c=1.0):
        v = np.where(st == 1, U_T3[k] * c, v); return np.where(frag_acc, c, v)
    v1 = finish(np.maximum(qv * tgt, gen.astype(float)))
    if kind_ld is not None:   # 055 の V_ld（紙の T だけ、T3 の社会には足さない）
        lost0 = opt.get("lost0", lost_ld_main if kind_ld == "main" else lost_ld_multi)
        lost = 1 - (1 - lost0) * (np.exp(-T / opt["taup"]) if "taup" in opt else 1.0)
        v1l = np.maximum(q_TC * tgt, gen.astype(float)); v1l = np.where(st == 1, U_T3[k], v1l); v1l = np.where(frag_acc, 1.0, v1l)
        v2 = q_B * tgt; v2 = np.where(st == 1, 0.0, v2); v2 = np.where(frag_acc, 0.0, v2)
        i3 = (U_lost3 > lost) & (U_acc3 < p_accept) & (U_find3 < find2(k, {}) * D2)
        return np.where(alive & intact1, v1l, np.where(alive & i3, v2, 0.0))
    if S is None: return np.where(alive & intact1, v1, 0.0)
    i2 = (U_lost2 > lost_tc) & (U_find2 < find2(k, {}) * D2)
    v2 = finish(np.maximum(0.76 * tgt, gen.astype(float))) if tablet else np.zeros(N)
    return np.where(alive & intact1, v1, np.where(alive & i2, v2, 0.0))

# 検査：D=1 で 054 の V_ext・055 の V_ld と一致
chk = 0.0
for k in COMPS:
    chk = max(chk, float(np.abs(V_t(k, q_TC, True) - G["V_ext"](k, q_TC, True)).max()),
              float(np.abs(V_t(k, q_TC, True, S="tc") - G["V_ext"](k, q_TC, True, S="tc")).max()),
              float(np.abs(V_t(k, q_TC, True, kind_ld="main") - G55["V_ld"](k, lost_ld_main)).max()))
R["check_V_t_equals_054_055"] = chk

def deltas_t(opt):
    D = {x: {} for x in ("TAB", "TC", "S_TC_given_TC", "S_LD_given_TC", "S_LD_multi_given_TC")}
    for k in COMPS:
        V0 = V_t(k, 0.3, False, opt=opt); VTC = V_t(k, q_TC, True, opt=opt)
        D["TAB"][k] = V_t(k, 0.3, True, opt=opt) - V0
        D["TC"][k] = VTC - V0
        D["S_TC_given_TC"][k] = V_t(k, q_TC, True, S="tc", opt=opt) - VTC
        D["S_LD_given_TC"][k] = V_t(k, q_TC, True, opt=opt, kind_ld="main") - VTC
        D["S_LD_multi_given_TC"][k] = V_t(k, q_TC, True, opt=opt, kind_ld="multi") - VTC
    return agg(D, anchors=("A_Ord", "A_XPT_S"))

KEYS = ["A_Ord/U3", "A_XPT_S/U2"]
VALS = {v: deltas_t(o) for v, o in VERS.items()}
# V0 が 054 の値と 055 の S_LD に一致するか
mx = 0.0
for key in KEYS:
    a, u = key.split("/")
    for x in ("TAB", "TC", "S_TC_given_TC"):
        for p in ("total", "heavy", "light"): mx = max(mx, abs(VALS["V0"][a][u][x][p] - VAL54[a][u][x][p]))
    for x in ("S_LD_given_TC", "S_LD_multi_given_TC"):
        for p in ("total", "heavy", "light"): mx = max(mx, abs(VALS["V0"][a][u][x][p] - VLD55[a][u][x][p]))
R["check_V0_vs_054_055_values"] = mx

def get(v, key): a, u = key.split("/"); return VALS[v][a][u]
R["values"] = {v: {key: get(v, key) for key in KEYS} for v in VALS}
R["ratio_to_V0"] = {}
for v in VALS:
    R["ratio_to_V0"][v] = {}
    for key in KEYS:
        b0, b = get("V0", key), get(v, key)
        R["ratio_to_V0"][v][key] = {x: {p: (b[x][p] / b0[x][p] if b0[x][p] != 0 else None) for p in ("total", "heavy", "light")} for x in b}
R["STC_over_TC"] = {v: {key: {p: (get(v, key)["S_TC_given_TC"][p] / get(v, key)["TC"][p] if get(v, key)["TC"][p] > 0 else None) for p in ("total", "heavy", "light")} for key in KEYS} for v in VALS}
R["heavy_share_TAB"] = {v: {key: get(v, key)["TAB"]["heavy"] / get(v, key)["TAB"]["total"] for key in KEYS} for v in VALS}

# ---------- 走査（重い崩壊の TAB の比） ----------
RG = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
TG = [30, 100, 300, 1000, 3000, 10000]
R["scan_rho1"] = []
for r_ in RG:
    o = dict(VERS["main"]); o["rho1"] = r_; Vk = deltas_t(o)
    R["scan_rho1"].append({"rho1": r_, **{key: {p: Vk[key.split("/")[0]][key.split("/")[1]]["TAB"][p] / get("V0", key)["TAB"][p] for p in ("total", "heavy")} for key in KEYS}})
R["scan_tau1"] = []
for t_ in TG:
    o = dict(VERS["main"]); o["tau1"] = float(t_); Vk = deltas_t(o)
    R["scan_tau1"].append({"tau1": t_, **{key: {p: Vk[key.split("/")[0]][key.split("/")[1]]["TAB"][p] / get("V0", key)["TAB"][p] for p in ("total", "heavy")} for key in KEYS}})
def cross(rows, xk, key, target=0.5):
    xs = [r[xk] for r in rows]; ys = [r[key]["heavy"] for r in rows]
    for i in range(1, len(xs)):
        if (ys[i - 1] - target) * (ys[i] - target) <= 0 and ys[i] != ys[i - 1]:
            return xs[i - 1] + (target - ys[i - 1]) / (ys[i] - ys[i - 1]) * (xs[i] - xs[i - 1])
    return None
R["rho1_star_heavy_half"] = {key: cross(R["scan_rho1"], "rho1", key) for key in KEYS}
R["T2s_heavy_quantiles"] = {k: [float(x) for x in np.quantile(B[k]["T2s"][np.isin(B[k]["st"], (4, 5))], [0.1, 0.5, 0.9])] for k in ("U3", "BIO", "CASC", "GEN") if np.isin(B[k]["st"], (4, 5)).any()}

# ---------- 費用あたり（055 の run を新しい価値で） ----------
def run_with(v, ch=None):
    G55["VAL54"] = VALS[v]; G55["VLD"] = VALS[v]
    out = G55["run"]("V_mix", *(ch or ()))
    G55["VAL54"] = VAL54; G55["VLD"] = VLD55
    return out
R["CE"] = {}
for v in ("V0", "main", "cache", "marker", "city"):
    R["CE"][v] = {"chars_1e7_1e8": run_with(v), "chars_reader_approx": run_with(v, (6e5, 4.17e6))}
def ce_med(v, chs, key, x, p="total"):
    d = R["CE"][v][chs]["CE"][key].get(p, {}); return d[x]["CE_q"][1] if x in d else None
R["CE_medians"] = {v: {chs: {key: {p: {x: ce_med(v, chs, key, x, p) for x in ("TC", "S_TC_given_TC", "S_LD_given_TC")} for p in ("total", "heavy", "light")} for key in KEYS} for chs in R["CE"][v]} for v in R["CE"]}

# ---------- 仮説 ----------
M1 = "A_Ord/U3"; rt = R["ratio_to_V0"]
Hh = {}
v = rt["main"][M1]["TAB"]["heavy"]; Hh["H72a"] = {"value": v, "rule": "main heavy TAB / V0 < 0.5", "supported": v < 0.5}
v = rt["main"][M1]["TAB"]["light"]; Hh["H72b"] = {"value": v, "rule": "main |light TAB / V0 - 1| < 0.05", "supported": abs(v - 1) < 0.05}
v = rt["main"][M1]["TAB"]["total"]; Hh["H72c"] = {"value": v, "rule": "main total TAB / V0 <= 0.9", "supported": v <= 0.9}
v = [R["STC_over_TC"]["main"][M1]["heavy"], R["STC_over_TC"]["V0"][M1]["heavy"]]; Hh["H72d"] = {"value": v, "rule": "heavy S_TC|TC / TC: main > V0", "supported": v[0] > v[1]}
v = rt["marker"][M1]["TAB"]["heavy"]; Hh["H72e"] = {"value": v, "rule": "marker heavy TAB / V0 >= 0.8", "supported": v >= 0.8}
v = rt["cache"][M1]["TAB"]["heavy"]; Hh["H72f"] = {"value": v, "rule": "cache heavy TAB / V0 < 0.2", "supported": v < 0.2}
v = rt["main"][M1]["S_LD_given_TC"]["heavy"]; Hh["H72g"] = {"value": v, "rule": "main heavy S_LD|TC / 055 < 0.3", "supported": (v is not None) and v < 0.3}
c = R["CE_medians"]["main"]["chars_1e7_1e8"][M1]["total"]; Hh["H72h"] = {"value": [c["S_TC_given_TC"], c["TC"]], "rule": "main V_mix 1e7-1e8: median CE S_TC|TC > TC", "supported": c["S_TC_given_TC"] > c["TC"]}
for h in Hh.values(): h["supported"] = bool(h["supported"])
R["hypotheses"] = Hh

# ---------- 図 ----------
Tg = np.linspace(0, 3000, 301)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for nm, tau, rho, col in (("vault main (median tau1, rho1)", np.median(tau1), np.median(rho1), "C0"), ("southern copy main", np.median(tau2), np.median(rho2), "C1"),
                          ("cache (rho1=0.015)", np.median(tau1), 0.015, "C3"), ("marker (rho=0.9)", np.median(tau1), 0.9, "C2"), ("city (rho=0.8)", np.median(tau1), 0.805, "C4")):
    m = np.exp(-Tg / tau); axs[0].plot(Tg, m + (1 - m) * rho, color=col, label=nm)
axs[0].set_xlabel("years from collapse to reading (T2s)"); axs[0].set_ylabel("D(T): find-prob multiplier"); axs[0].legend(fontsize=7); axs[0].set_ylim(0, 1.05)
for k, col in (("U3", "C0"), ("BIO", "C1"), ("CASC", "C2")):
    T = B[k]["T2s"][np.isin(B[k]["st"], (4, 5))]
    axs[1].hist(np.clip(T, 0, 3000), bins=60, histtype="step", density=True, color=col, label=f"{k} heavy (T1/T0)")
axs[1].set_xlabel("T2s years (heavy collapse samples)"); axs[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_D_and_T2s.png"), dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
vn = [v for v in VALS if v != "V0"]
for ax, key in zip(axs, KEYS):
    xs = np.arange(len(vn))
    for j, (x, p, col) in enumerate((("TAB", "heavy", "C0"), ("TAB", "light", "C1"), ("S_TC_given_TC", "heavy", "C2"), ("S_LD_given_TC", "heavy", "C3"))):
        ax.bar(xs + (j - 1.5) * 0.2, [rt[v][key][x][p] or 0 for v in vn], 0.2, color=col, label=f"{x} {p}")
    ax.axhline(1, color="k", lw=0.8); ax.set_xticks(xs); ax.set_xticklabels(vn, rotation=45, fontsize=8); ax.set_title(key)
axs[0].set_ylabel("value / V0 (054-055)"); axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_versions.png"), dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for key, col in zip(KEYS, ("C0", "C1")):
    axs[0].plot(RG, [r[key]["heavy"] for r in R["scan_rho1"]], "-o", color=col, label=f"{key} heavy"); axs[0].plot(RG, [r[key]["total"] for r in R["scan_rho1"]], "--", color=col, label=f"{key} total")
    axs[1].semilogx(TG, [r[key]["heavy"] for r in R["scan_tau1"]], "-o", color=col, label=f"{key} heavy"); axs[1].semilogx(TG, [r[key]["total"] for r in R["scan_tau1"]], "--", color=col)
axs[0].axhline(0.5, color="k", lw=0.6); axs[0].set_xlabel("rho1 (rediscovery of forgotten vault)"); axs[0].set_ylabel("TAB value / V0"); axs[0].legend(fontsize=7)
axs[1].set_xlabel("tau1 (years, location memory)"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_scans.png"), dpi=130); plt.close(fig)

R["runtime_s"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float, ensure_ascii=False)
print("checks", R["check_055_main_chars"], R["check_V_t_equals_054_055"], R["check_V0_vs_054_055_values"])
print("evidence", json.dumps({k: R["evidence"][k] for k in ("lost_city", "copper_scroll_frac_found_max", "cache_found_by_chance", "war_library_loss")}))
print("classical", json.dumps(R["evidence"]["classical_survival"]))
for v in VALS:
    print(v, {key: {x: {p: (round(y, 3) if y is not None else None) for p, y in d.items()} for x, d in rt[v][key].items() if x in ("TAB", "S_TC_given_TC", "S_LD_given_TC")} for key in KEYS})
print("STC/TC", json.dumps(R["STC_over_TC"]))
print("heavy_share_TAB", json.dumps(R["heavy_share_TAB"]))
print("scan_rho1", json.dumps(R["scan_rho1"])); print("scan_tau1", json.dumps(R["scan_tau1"])); print("rho1*", R["rho1_star_heavy_half"])
print("T2s heavy", R["T2s_heavy_quantiles"])
print("CE", json.dumps(R["CE_medians"]))
for h, d in Hh.items(): print(h, d["supported"], d["value"])
print("runtime", R["runtime_s"])
