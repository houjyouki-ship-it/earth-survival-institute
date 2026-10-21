"""研究076：写しの近くに生き残りが寄る効果（196）と部数の限界の費用あたり（197）。
計算のみ（ネット接続なし）。事前登録：PREREGISTRATION.md。TC/V0 は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。
075 の sim.py を「PP = G74["PP"]」の前まで exec し（070・074 の差0の検査を含む）、075 の run/counts を使う。新しい乱数は seed 20261076。"""
import json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, "..")
p75 = os.path.abspath(os.path.join(RS, "075-reach-to-tc-copies", "sim.py")); s = open(p75, encoding="utf8").read()
G = {"__file__": p75, "__name__": "sim075"}; exec(compile(s[: s.index('PP = G74["PP"]')], p75, "exec"), G)
r75 = json.load(open(os.path.join(RS, "075-reach-to-tc-copies", "results.json"), encoding="utf8"))
print("075 loaded", round(time.time() - T0), "s", flush=True)
R = {"study": "076-copy-attraction-marginal", "seed": 20261076, "note": "TC/V0 は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。"}
RT, DV, run, M1, N, GMAX = G["RT"], G["DV"], G["run"], G["M1"], G["N"], G["GMAX"]
G74 = G["G74"]; PP = G74["PP"]; MESH = list(G74["mesh"]); SITES = dict(PP); PLP = list(PP) + ["mesh35"]; SITES["mesh35"] = MESH[:35]
hav, plat, plon, Wp, KMOV = G["hav"], G["plat"], G["plon"], G["Wp"], G["KMOV"]; GEO3 = G["GEO3"]; U_geo = G["U_geo"]; NPT = G["NPT"]
PT0 = G["PT"].copy()
h = lambda nm, k=M1: RT[nm][k]["heavy"]
# ---------- 検査：075 を差0で再現 ----------
chk = {}
for nm, S in (("mesh35", MESH[:35]), ("mesh200", MESH[:200]), ("per_subregion@cty", PP["per_subregion@cty"])):
    run("chk_" + nm, S); chk[nm] = max(abs(RT["chk_" + nm][k][p] - r75["ratio_to_V0"][nm][k][p]) for k in RT["chk_" + nm] for p in RT["chk_" + nm][k])
R["check_075"] = {"max_abs_diff": chk, "reproduced": all(v < 1e-12 for v in chk.values())}; assert R["check_075"]["reproduced"], chk
print("check ok", chk, round(time.time() - T0), "s", flush=True)
# ---------- 寄せ ----------
L0 = float(np.median(KMOV)); R["L_km"] = L0
rg = np.random.default_rng(20261076); UA = rg.uniform(size=(N, GMAX)); UB = rg.uniform(size=(N, GMAX))
def attract_PT(S, alpha, L):
    dmin = hav(plat[:, None], plon[:, None], plat[np.array(S)][None, :], plon[np.array(S)][None, :]).min(1)
    PT = PT0.copy(); m_att = UA < alpha
    for j, gname in enumerate(GEO3):
        w = Wp[(gname, 1.0)] * np.exp(-dmin / L); cdf = np.cumsum(w); cdf /= cdf[-1]; m = (U_geo[:, None] == j) & m_att
        PT[m] = np.minimum(np.searchsorted(cdf, UB[m], side="right"), NPT - 1)
    return PT
def run_att(nm, S, alpha, L=L0):
    G["PT"] = attract_PT(S, alpha, L); run(nm, S); G["PT"] = PT0
ALPHAS = (0.0, 0.1, 0.25, 0.5, 1.0); NS = (35, 100, 200)
for a in ALPHAS:
    for n in NS: run_att(f"mesh{n}_a{a}", MESH[:n], a)
d0 = max(abs(h(f"mesh{n}_a0.0") - r75["ratio_to_V0"][f"mesh{n}"][M1]["heavy"]) for n in (35, 200)); R["check_alpha0"] = d0; assert d0 < 1e-12
for p in PP: run_att(f"{p}_a0.5", SITES[p], 0.5)
for tag, L in (("Lx2", 2 * L0), ("Lhalf", L0 / 2)):
    for n in (35, 200): run_att(f"mesh{n}_a0.5_{tag}", MESH[:n], 0.5, L)
R["reach_stats"] = G["R"]["reach_stats"]
R["heavy_TC"] = {k: h(k) for k in RT if k != "chk070"}; R["total_TC"] = {k: RT[k][M1]["total"] for k in RT if k != "chk070"}
R["alpha_scan"] = {str(a): {str(n): h(f"mesh{n}_a{a}") for n in NS} for a in ALPHAS}
R["gain_35_200"] = {str(a): h(f"mesh200_a{a}") - h(f"mesh35_a{a}") for a in ALPHAS}
fl = r75["posthoc_floor_ceiling"]["floor_heavy"]; ce_ = r75["posthoc_floor_ceiling"]["ceiling35_heavy"]
R["share_gap_closed_mesh35"] = {str(a): (h(f"mesh35_a{a}") - fl) / (ce_ - fl) for a in ALPHAS}
# ---------- 限界の費用あたり ----------
def find(name):
    st = [G["G70"]]
    while st:
        g = st.pop()
        if name in g and not isinstance(g[name], dict): return g[name]
        st += [v for k, v in g.items() if isinstance(v, dict) and k.startswith("G") and "__name__" in v]
    raise KeyError(name)
per_P = find("per_P"); f_pg = find("f_pg"); C_P = G["C_P"]; C_pr = G["C_pr"]; C_T = G["C_T"]; get = G["get"]; qf = G["qf"]
def Cost(n, scale=1.0): return C_P(n) + 2 * (C_pr + (n - 100) * per_P * f_pg) - (1 - scale) * n * per_P * (1 + 2 * f_pg)
a_, u_ = M1.split("/"); CE = {}
for part in ("total", "heavy"):
    vt = get("main", M1)["TAB"][part]; vtc = get("main", M1)["TC"][part]; CE[part] = {}
    for a in ALPHAS:
        t35 = DV[f"mesh35_a{a}"][a_][u_]["TC"][part]; t200 = DV[f"mesh200_a{a}"][a_][u_]["TC"][part]; t100 = DV[f"mesh100_a{a}"][a_][u_]["TC"][part]
        for sc in (1.0, 0.1):
            k = f"a{a}" + ("" if sc == 1.0 else "_perPx0.1")
            CE[part][k] = {"avg35": qf((t35 - vtc) / vt * C_T / Cost(35, sc)), "avg200": qf((t200 - vtc) / vt * C_T / Cost(200, sc)),
                           "marg35_200": qf((t200 - t35) / vt * C_T / (Cost(200, sc) - Cost(35, sc))),
                           "marg35_100": qf((t100 - t35) / vt * C_T / (Cost(100, sc) - Cost(35, sc))), "marg100_200": qf((t200 - t100) / vt * C_T / (Cost(200, sc) - Cost(100, sc))),
                           "share_samples_marg_lt_avg35": float(np.mean((t200 - t35) / (Cost(200, sc) - Cost(35, sc)) < (t35 - vtc) / Cost(35, sc)))}
R["CE"] = CE; R["CE_check_075_avg35_total"] = [CE["total"]["a0.0"]["avg35"][1], r75["CE_summary"]["total_median_by_N"]["35"]]
# ---------- 順位 ----------
def ktau(x, y):
    s_ = 0; c = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)): s_ += np.sign(x[i] - x[j]) * np.sign(y[i] - y[j]); c += 1
    return float(s_ / c)
y0 = [r75["ratio_to_V0"][p][M1]["heavy"] for p in PLP]; y5 = [h(f"{p}_a0.5") if p != "mesh35" else h("mesh35_a0.5") for p in PLP]
R["rank"] = {"TC_a0": dict(zip(PLP, y0)), "TC_a0.5": dict(zip(PLP, y5)), "kendall_tau": ktau(y0, y5), "best_a0": PLP[int(np.argmax(y0))], "best_a0.5": PLP[int(np.argmax(y5))],
             "spread_a0": max(y0) - min(y0), "spread_a0.5": max(y5) - min(y5)}
# ---------- 仮説 ----------
H = {}; m35_75 = r75["ratio_to_V0"]["mesh35"][M1]["heavy"]
v = h("mesh35_a0.5") - m35_75; H["H88a"] = {"class": "予測", "value": v, "rule": "heavy TC mesh35 alpha=0.5 - 075 mesh35 >= 0.02", "supported": v >= 0.02}
v = CE["total"]["a0.0"]["marg35_200"][1] / CE["total"]["a0.0"]["avg35"][1]; H["H88b"] = {"class": "予測", "value": v, "rule": "marginal CE 35->200 / avg CE 35 (total, median, alpha=0) < 0.3", "supported": v < 0.3}
v = [R["gain_35_200"]["0.5"], R["gain_35_200"]["0.0"]]; H["H88c"] = {"class": "予測", "value": v, "rule": "gain 35->200 at alpha=0.5 < gain at alpha=0", "supported": v[0] < v[1]}
v = R["rank"]["kendall_tau"]; H["H88d"] = {"class": "予測", "value": v, "rule": "Kendall tau(rank alpha=0, rank alpha=0.5) over 10 plans >= 0.6", "supported": v >= 0.6}
for x in H.values(): x["supported"] = bool(x["supported"])
R["hypotheses"] = H
# ---------- 図 ----------
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for n in NS: axs[0].plot(ALPHAS, [h(f"mesh{n}_a{a}") for a in ALPHAS], "o-", label=f"mesh {n} copies")
axs[0].axhline(fl, color="k", ls=":", lw=.8, label="075 floor (no group reaches)"); axs[0].axhline(ce_, color="gray", ls="--", lw=.8, label="075 ceiling (all reach 35)")
axs[0].set_xlabel("alpha (share of groups pulled toward copies, L=144.8 km)"); axs[0].set_ylabel("heavy-collapse TC / V0 (not survival probability)"); axs[0].legend(fontsize=7)
for part, mk in (("total", "o"), ("heavy", "s")):
    axs[1].plot(ALPHAS, [CE[part][f"a{a}"]["avg35"][1] for a in ALPHAS], mk + "-", label=f"{part}: average CE at 35")
    axs[1].plot(ALPHAS, [CE[part][f"a{a}"]["marg35_200"][1] for a in ALPHAS], mk + "--", label=f"{part}: marginal CE 35->200")
axs[1].set_xlabel("alpha"); axs[1].set_ylabel("cost-effectiveness (median, TAB=1 scale)"); axs[1].set_yscale("log"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_alpha_scan.png"), dpi=120); plt.close()
fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(y0, y5)
for p, x, y in zip(PLP, y0, y5): ax.annotate(p, (x, y), fontsize=6)
lo, hi = min(y0 + y5) - .005, max(y0 + y5) + .005; ax.plot([lo, hi], [lo, hi], "k:", lw=.8)
ax.set_xlabel("heavy TC/V0, alpha=0 (075)"); ax.set_ylabel("heavy TC/V0, alpha=0.5"); ax.set_title(f"10 plans, Kendall tau={R['rank']['kendall_tau']:.2f}")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_plans_alpha.png"), dpi=120); plt.close()
R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("check_075", "alpha_scan", "gain_35_200", "share_gap_closed_mesh35", "rank", "hypotheses")}, ensure_ascii=False, indent=1, default=float))
print("CE total", json.dumps(CE["total"], default=float)); print("CE heavy", json.dumps(CE["heavy"], default=float)); print("runtime", R["runtime_s"])
