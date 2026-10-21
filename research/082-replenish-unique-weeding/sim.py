"""研究082：除籍の内訳（open_problems 208）と所蔵の確認・補充（209）。
計算のみ（資料は sources/ に保存済み、本スクリプトはネットに接続しない）。事前登録：PREREGISTRATION.md（2026-10-21T14:44Z）。
TC/V0 と費用あたりは生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。
081 の sim.py を「# ---------- 仮説 ----------」の前まで exec し（rindex）、081 の CE 全値を差0で再現してから、
k 年ごとの確認と補充（079 の p_acc で収蔵）を入れた在否の確率を f_i に掛けて 076 の run_att で TC を計算し直す。seed 20261082。"""
import json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, "..")
p81 = os.path.abspath(os.path.join(RS, "081-deaccession-rate", "sim.py")); s = open(p81, encoding="utf8").read()
G = {"__file__": p81, "__name__": "sim081"}; exec(compile(s[: s.rindex("# ---------- 仮説 ----------")], p81, "exec"), G)
r81 = json.load(open(os.path.join(RS, "081-deaccession-rate", "results.json"), encoding="utf8"))
print("081 loaded", round(time.time() - T0), "s", flush=True)
R = {"study": "082-replenish-unique-weeding", "seed": 20261082, "note": "TC/V0 と費用あたりは生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。"}
# ---------- 検査：081 の CE を差0で再現 ----------
dd = 0.0
for vn, V in r81["CE"].items():
    for part, P_ in V.items():
        for a, X in P_.items():
            for kk in ("avg35", "marg35_200"):
                A = np.asarray(G["CE"][vn][part][a][kk]); B = np.asarray(X[kk]); dd = max(dd, float(np.max(np.abs(A - B) / np.abs(B))))
R["check_081"] = {"CE_max_rel_diff": dd, "reproduced": dd < 1e-12}; assert R["check_081"]["reproduced"], R["check_081"]
run_att, MESH, RT, DV, M1, G70 = G["run_att"], G["MESH"], G["RT"], G["DV"], G["M1"], G["G70"]
Cost, get, C_T, qf, ce2, pp_main = G["Cost"], G["get"], G["C_T"], G["qf"], G["ce2"], G["pp_main"]
G80 = G["G"]; base = G80["base"]; pacc = G80["pacc79"]
assert np.allclose(pp_main, base / pacc, rtol=1e-12, atol=0); R["check_pp_main_equals_base_over_pacc"] = True
f_i0 = G["f_i0"]; N = f_i0.size; NC = pp_main.size; assert N == NC, (N, NC)
dj = np.array([c["d"] for c in G["cases"]]); q25, q75 = float(np.quantile(dj, 0.25)), float(np.quantile(dj, 0.75))
# ---------- 除籍の内訳（sources/weeding_reasons.json） ----------
WR = json.load(open(os.path.join(HERE, "sources", "weeding_reasons.json"), encoding="utf8"))
rdup = np.array([c["dup"] / c["total"] for c in WR["qualifying"]])
kochi = [c for c in WR["qualifying"] if "高知" in c["inst"]][0]
R["weeding_reasons"] = {"n": int(rdup.size), "r_dup": dict(zip([c["inst"] for c in WR["qualifying"]], rdup.tolist())), "median": float(np.median(rdup)),
                        "kochi_incl_serials_desc": kochi["dup"] / kochi["alt_total_incl_serials_approx"],
                        "kochi_not_held_anywhere_local_share": 2193 / 25432, "n_not_qualifying": len(WR["not_qualifying"])}
uniq_ok = rdup.size >= 2
# ---------- 新しい乱数 ----------
rg = np.random.default_rng(20261082)
u_d, u_D, u_c, u_dm = rg.uniform(size=(4, N))
d_main = q25 + (q75 - q25) * u_d
d_minmax = dj.min() + (dj.max() - dj.min()) * u_dm
Delta = np.exp(np.log(10) + (np.log(200) - np.log(10)) * u_D)
c_chk = np.exp(np.log(2) + (np.log(20) - np.log(2)) * u_c)
def presence(d, k, p, life=np.inf, disc=0.0):
    """在否の確率と、1館あたりの将来の費用（確認＋補充の送付）。k=inf は補充なし。"""
    if not np.isfinite(k):
        return np.exp(-d * Delta), np.zeros_like(Delta)
    n = np.floor(Delta / k); n = np.minimum(n, np.floor(life / k)) if np.isfinite(life) else n
    sv = np.exp(-d * k); pi = np.ones_like(Delta); cost = np.zeros_like(Delta)
    for j in range(1, int(n.max()) + 1):
        act = j <= n; miss = 1 - pi * sv; w = (1 + disc) ** (-j * k)
        cost = cost + act * w * (c_chk + miss * base)
        pi = np.where(act, pi * sv + miss * p, pi)
    return pi * np.exp(-d * (Delta - n * k)), cost
x0, c0 = presence(d_main, np.inf, pacc); assert np.array_equal(x0, np.exp(-d_main * Delta)) and not c0.any()
x1, _ = presence(d_main, 1e9, pacc); assert np.allclose(x1, np.exp(-d_main * Delta), rtol=1e-12); R["check_k_inf_equals_exp"] = True
def run_k(tag, kfac, n, a):
    G70["f_i"] = f_i0 * kfac
    try: run_att(f"mesh{n}_a{a}_{tag}", MESH[:n], float(a))
    finally: G70["f_i"] = f_i0
h = lambda nm: RT[nm][M1]["heavy"]; t = lambda nm: RT[nm][M1]["total"]
a_, u_ = M1.split("/")
def TCs(nm, part): return DV[nm][a_][u_]["TC"][part]
def inc_ce(nm_from, nm_to, cost_from, cost_to, part):
    vt = get("main", M1)["TAB"][part]; g = (TCs(nm_to, part) - TCs(nm_from, part)) / vt * C_T / (cost_to - cost_from); return g
VERS = {}  # tag -> (d, k, life, disc)
for k in (5, 10, 25, 50): VERS[f"k{k}"] = (d_main, k, np.inf, 0.0)
VERS["k10_disc3_desc"] = (d_main, 10, np.inf, 0.03)
VERS["k10_life30_desc"] = (d_main, 10, 30.0, 0.0); VERS["k10_life100_desc"] = (d_main, 10, 100.0, 0.0)
VERS["k10_dminmax_desc"] = (d_minmax, 10, np.inf, 0.0)
if uniq_ok: VERS["k10_unique_desc"] = (d_main * (1 - np.median(rdup)), 10, np.inf, 0.0)
NONE = {"none": d_main, "none_dminmax_desc": d_minmax}
if uniq_ok: NONE["none_unique_desc"] = d_main * (1 - np.median(rdup))
for tag, dv in NONE.items():
    for a in ("0.0", "0.5"):
        for n in (35, 200): run_k(tag, np.exp(-dv * Delta), n, a)
    print("none", tag, round(time.time() - T0), "s", flush=True)
# 検査：主の d を 081 と同じ U(min,max) と標本にすれば 081 の TC になる（d_minmax は新しい標本なので別。ここは 081 の d_main で1本だけ再計算して一致を確かめる）
run_k("chk081", np.exp(-G["d_main"] * G["Delta"]), 35, "0.5"); assert abs(h("mesh35_a0.5_chk081") - r81["TC"]["main"]["heavy_a0.5_mesh35"]) < 1e-12; R["check_081_main_TC_rerun"] = True
FUT = {}
for tag, (dv, k, life, disc) in VERS.items():
    x, c = presence(dv, k, pacc, life, disc); FUT[tag] = {"x": x, "c": c}
    for a in (("0.0", "0.5") if tag == "k10" else ("0.5",)): run_k(tag, x, 35, a)
    print("ver", tag, round(time.time() - T0), "s", flush=True)
x, c = presence(d_main, 10, pacc); run_k("k10R200", x, 200, "0.5"); FUT["k10R200_desc"] = {"x": x, "c": c}
# ---------- 集計 ----------
C35, C200 = Cost(35, pp_main), Cost(200, pp_main)
def none_of(tag): return "none_dminmax_desc" if "dminmax" in tag else ("none_unique_desc" if "unique" in tag else "none")
OUT = {}
for tag in VERS:
    nn = none_of(tag); o = {}
    for a in (("0.0", "0.5") if tag == "k10" else ("0.5",)):
        for part in ("heavy", "total"):
            C35R = C35 + 35 * FUT[tag]["c"]
            ceR = inc_ce(f"mesh35_a{a}_{nn}", f"mesh35_a{a}_{tag}", C35, C35R, part)
            ce200 = inc_ce(f"mesh35_a{a}_{nn}", f"mesh200_a{a}_{nn}", C35, C200, part)
            o[f"{part}_a{a}"] = {"TC_none35": RT[f"mesh35_a{a}_{nn}"][M1][part], "TC_R35": RT[f"mesh35_a{a}_{tag}"][M1][part], "TC_none200": RT[f"mesh200_a{a}_{nn}"][M1][part],
                                 "ce_inc_R35": qf(ceR), "ce_inc_none200": qf(ce200), "ratio_median": float(np.median(ceR) / np.median(ce200)), "share_R_gt_200": float(np.mean(ceR > ce200)),
                                 "extra_cost_R35_median_usd": float(np.median(35 * FUT[tag]["c"])), "extra_cost_200_median_usd": float(np.median(C200 - C35))}
    o["presence_quantiles"] = qf(FUT[tag]["x"]); OUT[tag] = o
R["presence_none_quantiles"] = qf(np.exp(-d_main * Delta))
R["versions"] = OUT
R["R200_desc"] = {"TC_heavy_a0.5": h("mesh200_a0.5_k10R200"), "TC_heavy_a0.5_none200": h("mesh200_a0.5_none")}
R["d_main_range"] = [q25, q75]
# ---------- 仮説 ----------
H = {}
H["H94a"] = {"class": "確認的", "value": {"n": int(rdup.size), "r_dup": rdup.tolist()}, "rule": "n>=2 and count(r_dup>0.5)>=2", "supported": bool(rdup.size >= 2 and (rdup > 0.5).sum() >= 2)}
H["H94b"] = {"class": "確認的", "value": {"n": int(rdup.size), "median": float(np.median(rdup)) if rdup.size else None}, "rule": "n>=2 and median r_dup >= 0.33", "supported": bool(rdup.size >= 2 and np.median(rdup) >= 0.33)}
m = OUT["k10"]["heavy_a0.5"]
H["H94c"] = {"class": "予測", "value": {"TC_R35": m["TC_R35"], "TC_none200": m["TC_none200"]}, "rule": "heavy a0.5 TC(R35,k=10) < TC(none200)", "supported": bool(m["TC_R35"] < m["TC_none200"])}
H["H94d"] = {"class": "予測", "value": m["ratio_median"], "rule": "median CE(35->R35) / median CE(35->200) > 1 (heavy a0.5, k=10)", "supported": bool(m["ratio_median"] > 1)}
R["hypotheses"] = H
# ---------- 図 ----------
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
fig, axs = plt.subplots(1, 3, figsize=(17, 5))
ks = [5, 10, 25, 50]
axs[0].plot(ks, [OUT[f"k{k}"]["heavy_a0.5"]["TC_R35"] for k in ks], "-o", label="35 copies + check/replenish every k yr")
axs[0].axhline(m["TC_none35"], color="C1", ls="--", label="35 copies, no replenish"); axs[0].axhline(m["TC_none200"], color="C2", ls="--", label="200 copies, no replenish")
axs[0].axhline(r81["TC_base_080"]["heavy_a0.5_mesh35"], color="k", ls=":", label="35 copies, no weeding (080)")
axs[0].set_xlabel("check interval k (yr)"); axs[0].set_ylabel("heavy a=0.5 TC/V0 (not a survival probability)"); axs[0].legend(fontsize=7)
axs[1].semilogy(ks, [np.median(OUT[f"k{k}"]["heavy_a0.5"]["ce_inc_R35"][1]) if isinstance(OUT[f"k{k}"]["heavy_a0.5"]["ce_inc_R35"], list) else OUT[f"k{k}"]["heavy_a0.5"]["ce_inc_R35"][1] for k in ks], "-o", label="35 -> 35+replenish")
axs[1].axhline(m["ce_inc_none200"][1], color="C2", ls="--", label="35 -> 200 (no replenish)")
axs[1].set_xlabel("check interval k (yr)"); axs[1].set_ylabel("incremental cost-effectiveness (median, heavy a0.5)"); axs[1].legend(fontsize=7)
names = list(OUT); xx = np.arange(len(names))
axs[2].bar(xx, [OUT[n]["heavy_a0.5"]["ratio_median"] for n in names]); axs[2].axhline(1, color="k", ls=":")
axs[2].set_xticks(xx); axs[2].set_xticklabels([n.replace("_desc", "") for n in names], fontsize=7, rotation=30); axs[2].set_yscale("log")
axs[2].set_ylabel("CE(35->replenish) / CE(35->200), heavy a0.5"); 
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_replenish.png"), dpi=120); plt.close()
R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("check_081", "weeding_reasons", "presence_none_quantiles", "R200_desc", "hypotheses")}, ensure_ascii=False, indent=1, default=float))
for tag, o in OUT.items():
    v = o["heavy_a0.5"]; print(tag, "TC none35 %.4f R35 %.4f none200 %.4f ratio %.2f share %.2f extraR %.0f extra200 %.0f" % (v["TC_none35"], v["TC_R35"], v["TC_none200"], v["ratio_median"], v["share_R_gt_200"], v["extra_cost_R35_median_usd"], v["extra_cost_200_median_usd"]))
print("runtime", R["runtime_s"])
