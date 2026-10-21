"""研究075：到達の指標 A と重い崩壊の TC/V0 の対応、部数 N を含めて（open_problems 191＋194）。
計算のみ（ネット接続なし）。事前登録：PREREGISTRATION.md。TC/V0 も A も生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。
070 の sim.py を「版」の前まで exec して 070 の主を差0で再現し、074 の sim.py を「仮説」の前まで exec して 074 を差0で再現してから、
重い崩壊の標本の「群が届く写しの単位の数」を、074 の点の模型の「群の点から届く距離以内にある写しの数」に置き換える。新しい乱数は seed 20261075。"""
import json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, "..")
def load(d, marker, name):
    p = os.path.abspath(os.path.join(RS, d, "sim.py")); s = open(p, encoding="utf8").read(); Gd = {"__file__": p, "__name__": name}
    exec(compile(s[: s.index(marker)], p, "exec"), Gd); return Gd, json.load(open(os.path.join(RS, d, "results.json"), encoding="utf8"))
G70, r70 = load("070-group-deposit-overlap", "\n# ---------- 版 ----------\n", "sim070"); print("070 loaded", round(time.time() - T0), "s", flush=True)
G74, r74 = load("074-within-country-reach", "# ---------- 仮説 ----------", "sim074"); print("074 loaded", round(time.time() - T0), "s", flush=True)
R = {"study": "075-reach-to-tc-copies", "seed": 20261075, "note": "TC/V0 と A は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。"}
N = G70["N"]; ver = G70["ver"]; ratio_all = G70["ratio_all"]; get = G70["get"]; qf = G70["qf"]; KEYS = G70["KEYS"]; PARTS = G70["PARTS"]; M1 = G70["M1"]
C_T = G70["C_T"]; C_P = G70["C_P"]; C_pr = G70["C_pr"]; GMAX = G70["GMAX"]
# ---------- 検査 ----------
pos = G70["place"](G70["U_geo"], G70["G_main"]); D70 = G70["reach_units"](pos, G70["DEP"]["D_nat"], G70["U_reach"])
def rho75(D, mu, np_l, ku_l, spread=False):
    g = G70; Df = D.astype(float); valid = D >= 0
    def f(T):
        hm = g["heavy_mask"](T); pc = (1 - g["L66"]) * np.exp(-g["lam"] * T); Rr = g["R69"](T, pc, g["PPF"], g["PMAIN"])
        a = np.exp(-T / g["taup_061"]) * g["f_i"]
        f1 = (1 - g["h_R"]) * (1 - a) ** mu + g["h_R"] * (1 - g["q_h"] * a) ** mu
        Sg = np.where(valid, 1 - f1[:, None] ** np.maximum(Df, 0), 0.0)
        if spread: P = (1 - (1 - Rr) ** valid.sum(1)) * (1 - np.prod(1 - Sg, axis=1))
        else: P = 1 - np.prod(np.where(valid, 1 - Rr[:, None] * Sg, 1.0), axis=1)
        light = g["S_ind"](T, np_l, ku_l, g["f_i"], g["taup_061"])
        return 1 - (1 - g["rho1"]) * (1 - g["r_reach"] * np.where(hm, P, Rr * light))
    return f
RT, DV = {}, {}
RT["chk070"], DV["chk070"] = ratio_all(ver(rho=rho75(D70, G70["MU"], G70["N_P"], G70["KU"])))
d = max(abs(RT["chk070"][k][p] - r70["ratio_to_V0"]["main"][k][p]) for k in KEYS for p in PARTS)
R["check_070_main"] = {"max_abs_diff": d, "reproduced": d < 1e-12}; assert d < 1e-12, d
d74 = max(abs(G74["OUT"][v]["summary"][p][k] - r74["versions"][v]["summary"][p][k]) for v in r74["versions"] for p in r74["versions"][v]["summary"] for k in ("mean_A", "worst_A", "max_regret"))
d74m = max(abs(G74["curve"][int(t)]["A_mean12"] - r74["mesh_curve"][t]["A_mean12"]) for t in r74["mesh_curve"])
R["check_074"] = {"max_abs_diff_summary": d74, "max_abs_diff_mesh": d74m}; assert d74 == 0 and d74m == 0
print("checks ok", round(time.time() - T0), "s", flush=True)
# ---------- 群の点と届く距離（新しい乱数） ----------
plat, plon, hav, Wp, KMOV, CASES71 = G74["plat"], G74["plon"], G74["hav"], G74["W"], G74["KMOV"], np.asarray(G74["CASES71"], float)
NPT = len(plat); GEO3 = ("xia150", "xia47", "pop")
rg = np.random.default_rng(20261075)
UPT = rg.uniform(size=(N, GMAX)); IR_s = rg.integers(0, len(KMOV), N); IR_g = rg.integers(0, len(KMOV), (N, GMAX)); IC_s = rg.integers(0, len(CASES71), N)
U_geo = G70["U_geo"]; G_main = G70["G_main"]; valid = np.arange(GMAX)[None, :] < G_main[:, None]
PT = np.zeros((N, GMAX), int)
for j, gname in enumerate(GEO3):
    cdf = np.cumsum(Wp[(gname, 1.0)]); cdf /= cdf[-1]; m = U_geo == j
    PT[m] = np.minimum(np.searchsorted(cdf, UPT[m], side="right"), NPT - 1)
REACH = {"main": np.broadcast_to(KMOV[IR_s][:, None], (N, GMAX)), "per_group": KMOV[IR_g], "x2": np.broadcast_to(2 * KMOV[IR_s][:, None], (N, GMAX)),
         "ref_071": np.broadcast_to(CASES71[IC_s][:, None], (N, GMAX))}
def counts(S, reach):
    Ds = np.sort(hav(plat[:, None], plon[:, None], plat[np.array(S)][None, :], plon[np.array(S)][None, :]), axis=1).astype(np.float32)
    out = np.full((N, GMAX), -1, int); ii, jj = np.nonzero(valid); pp = PT[ii, jj]; rr = reach[ii, jj].astype(np.float32); n = np.empty(len(ii), int)
    for c in range(0, len(ii), 40000): n[c:c + 40000] = (Ds[pp[c:c + 40000]] <= rr[c:c + 40000, None]).sum(1)
    out[ii, jj] = n; return out
def run(nm, S, reach="main", mu=1, light=None, spread=False):
    D = counts(S, REACH[reach]); nl = len(S) * mu if light is None else light[0]; kl = len(S) if light is None else light[1]
    RT[nm], DV[nm] = ratio_all(ver(rho=rho75(D, mu, nl, kl, spread)))
    R.setdefault("reach_stats", {})[nm] = {"share_groups_n0": float((D[valid] == 0).mean()), "mean_n": float(D[valid].mean()), "share_samples_all_groups_n0": float(((D <= 0) | ~valid).all(1).mean())}
    print(nm, round(RT[nm][M1]["heavy"], 4), round(RT[nm][M1]["total"], 4), round(time.time() - T0), "s", flush=True)
PP = G74["PP"]; MESH = list(G74["mesh"]); PLP = list(PP) + ["mesh35"]; SITES = dict(PP); SITES["mesh35"] = MESH[:35]
for p in PLP: run(p, SITES[p])
NS = [20, 35, 50, 100, 150, 200, 300]
for n in NS:
    if n != 35: run(f"mesh{n}", MESH[:n])
for n in (35, 100, 200):
    for rv in ("per_group", "x2", "ref_071"): run(f"mesh{n}_{rv}", MESH[:n], reach=rv)
    run(f"mesh{n}_fixedlight", MESH[:n], light=(100, 20))
run("per_subregion@cty_ref_071", PP["per_subregion@cty"], reach="ref_071"); run("pop@pt_ref_071", PP["pop@pt"], reach="ref_071")
run("mesh20_mu5", MESH[:20], mu=5)   # 070 と同じ 100部を 20か所に5部ずつ
R["ratio_to_V0"] = RT; R["070_main"] = r70["ratio_to_V0"]["main"]
# ---------- 順位の対応 ----------
A74 = {p: r74["versions"]["main_kmov"]["summary"][p]["mean_A"] for p in PP}; A74["mesh35"] = r74["mesh_curve"]["35"]["A_mean12"]
def ktau(x, y):
    n = len(x); s = 0; c = 0
    for i in range(n):
        for j in range(i + 1, n): s += np.sign(x[i] - x[j]) * np.sign(y[i] - y[j]); c += 1
    return float(s / c)
xa = [A74[p] for p in PLP]; ytc = [RT[p][M1]["heavy"] for p in PLP]
R["rank"] = {"A_074": A74, "TC_heavy": {p: RT[p][M1]["heavy"] for p in PLP}, "kendall_tau_heavy": ktau(xa, ytc),
             "kendall_tau_total": ktau(xa, [RT[p][M1]["total"] for p in PLP]), "kendall_tau_heavy_XPT": ktau(xa, [RT[p]["A_XPT_S/U2"]["heavy"] for p in PLP]),
             "best_by_A": max(PLP, key=lambda p: A74[p]), "best_by_TC": max(PLP, key=lambda p: RT[p][M1]["heavy"])}
# ---------- 費用あたり ----------
def find(name):
    st = [G70]
    while st:
        g = st.pop()
        if name in g and not isinstance(g[name], dict): return g[name]
        st += [v for k, v in g.items() if isinstance(v, dict) and k.startswith("G") and "__name__" in v]
    raise KeyError(name)
per_P = find("per_P"); f_pg = find("f_pg")
def C_pr_n(n): return C_pr + (n - 100) * per_P * f_pg
R["CE"] = {}
for key in KEYS:
    a, u = key.split("/"); R["CE"][key] = {}
    for part in ("total", "heavy"):
        vt = get("main", key)["TAB"][part]; vtc = get("main", key)["TC"][part]; R["CE"][key][part] = {}
        for n in NS:
            for tag, nm in (("", f"mesh{n}"),) + ((("_fixedlight", f"mesh{n}_fixedlight"),) if n in (35, 100, 200) else ()):
                dall = DV[nm][a][u]["TC"][part] - vtc; x = dall / vt * C_T
                R["CE"][key][part][nm] = {"scaled_cost": qf(x / (C_P(n) + 2 * C_pr_n(n))) if dall > 0 else [0.0] * 3,
                                          "fixed100_cost": qf(x / (C_P(100) + 2 * C_pr)) if dall > 0 else [0.0] * 3}
        for p in ("per_subregion@cty", "chk070"):
            dall = DV[p][a][u]["TC"][part] - vtc; nn = 35 if p != "chk070" else 100
            R["CE"][key][part][p] = {"scaled_cost": qf(dall / vt * C_T / (C_P(nn) + 2 * C_pr_n(nn))) if dall > 0 else [0.0] * 3}
ce = {n: R["CE"][M1]["total"][f"mesh{n}"]["scaled_cost"][1] for n in NS}; ceh = {n: R["CE"][M1]["heavy"][f"mesh{n}"]["scaled_cost"][1] for n in NS}
R["CE_summary"] = {"total_median_by_N": ce, "heavy_median_by_N": ceh, "argmax_total": max(ce, key=ce.get), "argmax_heavy": max(ceh, key=ceh.get),
                   "ratio_200_35_total": ce[200] / ce[35], "ratio_200_35_heavy": ceh[200] / ceh[35]}
# ---------- 仮説 ----------
h = lambda nm, k=M1: RT[nm][k]["heavy"]; h70 = r70["ratio_to_V0"]["main"][M1]["heavy"]; H = {}
v = R["rank"]["kendall_tau_heavy"]; H["H87a"] = {"class": "予測", "value": v, "rule": "Kendall tau(A_074, TC heavy) over 10 plans >= 0.7", "supported": v >= 0.7}
v = R["CE_summary"]["ratio_200_35_total"]; H["H87b"] = {"class": "予測", "value": v, "rule": "CE(mesh200)/CE(mesh35), total median, scaled cost < 0.5", "supported": v < 0.5}
v = h("mesh200") - h("mesh35"); H["H87c"] = {"class": "予測", "value": v, "rule": "heavy TC mesh200 - mesh35 >= 0.03", "supported": v >= 0.03}
v = h70 - h("mesh35"); H["H87d"] = {"class": "予測", "value": [h("mesh35"), h70, v], "rule": "070 main heavy - mesh35 heavy >= 0.05", "supported": v >= 0.05}
for x in H.values(): x["supported"] = bool(x["supported"])
R["hypotheses"] = H
# ---------- 図 ----------
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
axs[0].scatter(xa, ytc)
for p, x, y in zip(PLP, xa, ytc): axs[0].annotate(p, (x, y), fontsize=6)
axs[0].set_xlabel("A (074, mean of 12 conditions)"); axs[0].set_ylabel("heavy-collapse TC / V0 (A_Ord/U3)"); axs[0].set_title(f"N=35 plans, Kendall tau={R['rank']['kendall_tau_heavy']:.2f}")
axs[1].plot(NS, [h(f"mesh{n}") for n in NS], "o-", label="heavy TC/V0 (main reach)")
axs[1].plot(NS, [h(f"mesh{n}", ) if False else RT[f"mesh{n}"][M1]["total"] for n in NS], "s-", label="total TC/V0")
axs[1].axhline(h70, color="k", ls=":", lw=.8, label="070 main heavy (region model)")
ax2 = axs[1].twinx(); ax2.plot(NS, [ce[n] for n in NS], "^--", color="C3", label="cost-effectiveness (total, median)"); ax2.set_ylabel("cost-effectiveness (TAB=1 scale)")
axs[1].set_xscale("log"); axs[1].set_xlabel("copies N (mesh)"); axs[1].legend(fontsize=7, loc="lower right"); ax2.legend(fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_rank_and_N.png"), dpi=120); plt.close()
fig, ax = plt.subplots(figsize=(8, 4.5)); nm = [k for k in RT if k != "chk070"]; yy = np.arange(len(nm))
ax.barh(yy, [h(k) for k in nm]); ax.set_yticks(yy); ax.set_yticklabels(nm, fontsize=6); ax.axvline(h70, color="k", ls=":"); ax.set_xlabel("heavy-collapse TC / V0 (not survival probability)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_versions.png"), dpi=120); plt.close()
R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("check_070_main", "check_074", "rank", "CE_summary", "hypotheses")}, ensure_ascii=False, indent=1, default=float))
print("runtime", R["runtime_s"])
