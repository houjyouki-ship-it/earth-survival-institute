"""研究074：国の中の地理（Natural Earth の点の人口）で到達の指標 A を計算し直し、写しの置き方と必要な写しの数を比べる。
計算のみ（ネット接続なし）。A は「生存者（地理の重み）のうち届く範囲に生き残った写しが1つ以上ある割合」の厳密な期待値で、
生存確率でも復興確率でもない。提言は模型の中の順位（外部の検証はまだ）。乱数は 073 から引き継ぐものだけ（本回の新しい乱数なし）。"""
import json, math, os, time
import numpy as np
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__))
P73 = os.path.join(HERE, "..", "073-forager-mobility-reach", "sim.py")
src = open(P73, encoding="utf8").read(); G = {"__file__": P73, "__name__": "x73"}
exec(src[: src.index("# ---------- 図 ----------")], G)
R73 = json.load(open(os.path.join(os.path.dirname(P73), "results.json"), encoding="utf8"))
R = {"note": "A は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。"}
# 検査：073 の主の summary を差0で再現
for p, v in R73["versions"]["main_kmov"]["summary"].items():
    for k, x in v.items(): assert x == G["OUT"]["main_kmov"]["summary"][p][k], (p, k)
R["check_073_main_identical"] = True
G72 = G["G"]; K, A2, SUB, LAT, LON, GEO, PLANS, hav = G72["K"], G72["A2"], G72["SUB"], G72["LAT"], G72["LON"], G72["GEO"], G72["PLANS"], G72["hav"]
ISO, ALIAS = G72["ISO"], G72["ALIAS"]; NK = len(K); PL = list(PLANS)
KMOV = G["km"].kmov.values.astype(float); CASES71 = G["CASES71"]; QS, GEOS = (0.0, 0.5, 0.8), ("xia150", "xia47", "pop", "refuge")
# ---------- 点 ----------
NE = json.load(open(os.path.join(HERE, "sources", "ne_10m_populated_places_simple.geojson"), encoding="utf8"))["features"]
a2i = {a: i for i, a in enumerate(A2)}; a3i = {ISO[ALIAS.get(k, k)]["alpha-3"]: i for i, k in enumerate(K)}
plat, plon, ppop, pc = [], [], [], []; unmatched = 0
for f in NE:
    p = f["properties"]; i = a2i.get(p["iso_a2"], a3i.get(p["adm0_a3"]))
    if i is None: unmatched += 1; continue
    if not p["pop_max"] or p["pop_max"] <= 0: continue
    plat.append(p["latitude"]); plon.append(p["longitude"]); ppop.append(float(p["pop_max"])); pc.append(i)
nopt = sorted(set(range(NK)) - set(pc))
for i in nopt: plat.append(LAT[i]); plon.append(LON[i]); ppop.append(1.0); pc.append(i)  # 点のない国は国の中心1点
plat, plon, ppop, pc = map(np.array, (plat, plon, ppop, pc)); NP = len(plat)
R["points"] = {"n_features": len(NE), "n_points_used": int(NP), "n_unmatched_country": unmatched, "countries_without_points": [K[i] for i in nopt],
               "points_per_country_median": float(np.median(np.bincount(pc, minlength=NK)))}
def wpts(g, expo=1.0):
    s = ppop ** expo; tot = np.bincount(pc, weights=s, minlength=NK); return GEO[g][pc] * s / tot[pc]
W = {(g, e): wpts(g, e) for g in GEOS for e in (1.0, 0.5)}
for g in GEOS: assert abs(W[(g, 1.0)].sum() - 1) < 1e-9
big = {}
for j in np.argsort(-ppop, kind="stable"): big.setdefault(int(pc[j]), int(j))
BIGPT = np.array([big[i] for i in range(NK)])  # 国ごとの pop_max 最大の点
def Dto(S): return hav(plat[:, None], plon[:, None], plat[np.array(S)][None, :], plon[np.array(S)][None, :]) if len(S) else np.full((NP, 0), np.inf)
def A_pts(S, w, reach, q):
    D = np.sort(Dto(S), axis=1); r = np.sort(np.asarray(reach, float))
    n = np.stack([np.searchsorted(D[i], r, side="right") for i in range(NP)])  # 各点・各届く距離での写しの数
    return float(w @ (1 - q ** n).mean(1))
# ---------- 置き方（N=35） ----------
N = len(PLANS["per_subregion"])
PP = {f"{p}@cty": [int(BIGPT[i]) for i in P] for p, P in PLANS.items()}
wp = W[("pop", 1.0)]; subs = sorted(set(SUB)); ps = []
for s in subs:
    m = np.where(np.array([SUB[c] == s for c in pc]))[0]
    if wp[m].sum() <= 0: continue
    Dm = hav(plat[m][:, None], plon[m][:, None], plat[m][None, :], plon[m][None, :]); ps.append(int(m[np.argmin(wp[m] @ Dm)]))
CANDB = sorted(set(int(BIGPT[i]) for i in G72["CAND"]))
while len(ps) < N:
    dm = Dto(ps)[CANDB].min(1); ps.append(CANDB[int(np.argmax(dm))])
PP["per_subregion@pt"] = ps[:N]; PP["pop@pt"] = list(np.argsort(-ppop, kind="stable")[:N].astype(int))
PLP = list(PP)
def compare(reach, expo=1.0):
    T = {p: np.array([A_pts(S, W[(g, expo)], reach, q) for q in QS for g in GEOS]) for p, S in PP.items()}
    M = np.vstack([T[p] for p in PLP]); best = M.max(0); reg = best - M
    S = {p: {"mean_A": float(T[p].mean()), "worst_A": float(T[p].min()), "max_regret": float(reg[i].max())} for i, p in enumerate(PLP)}
    return T, S, min(PLP, key=lambda p: S[p]["max_regret"]), max(PLP, key=lambda p: S[p]["worst_A"])
VER = {"main_kmov": (KMOV, 1.0), "kmov_sqrtpop": (KMOV, 0.5), "kmov_x2": (2 * KMOV, 1.0), "ref_071": (CASES71, 1.0)}
OUT = {}
for v, (reach, e) in VER.items():
    T, S, a, b = compare(reach, e)
    OUT[v] = {"summary": S, "minimax_regret_plan": a, "maximin_plan": b, "mean_A_cty7": float(np.mean([S[p + "@cty"]["mean_A"] for p in PL])),
              "table": {p: [round(x, 6) for x in t] for p, t in T.items()}}
    print(v, a, b, round(time.time() - T0), "s", flush=True)
R["versions"] = OUT
# 073 の国の中心の模型での同じ7つ・同じ12条件（gc）の平均 A
def A_country(reach):
    return float(np.mean([G["A_w"](P, G72["DM"]["gc"], GEO[g], reach, np.full(len(reach), 1 / len(reach)), q) for P in PLANS.values() for q in QS for g in GEOS]))
R["country_model_mean_A"] = {"kmov": A_country(KMOV), "ref_071": A_country(CASES71)}
# ---------- mesh：主の届く距離での到達の期待値（q=0、4地理の平均の重み）を貪欲に最大化 ----------
wbar = np.mean([W[(g, 1.0)] for g in GEOS], axis=0); ks = np.sort(KMOV)
def cov(d): return 1 - np.searchsorted(ks, d, side="left") / len(ks)  # P(届く距離 >= d)
cur = np.zeros(NP); mesh = []; CH = [35, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000]; curve = {}
for t in range(1, CH[-1] + 1):
    best_g, best_j = -1.0, -1
    for c0 in range(0, NP, 1500):
        C = cov(hav(plat[:, None], plon[:, None], plat[None, c0:c0 + 1500], plon[None, c0:c0 + 1500]))
        g = wbar @ np.maximum(C - cur[:, None], 0); j = int(np.argmax(g))
        if g[j] > best_g: best_g, best_j = float(g[j]), c0 + j
    mesh.append(best_j); cur = np.maximum(cur, cov(hav(plat, plon, plat[best_j], plon[best_j])))
    if t in CH:
        curve[t] = {"A_mean12": float(np.mean([A_pts(mesh, W[(g, 1.0)], KMOV, q) for q in QS for g in GEOS])),
                    "A_q0_wbar": float(wbar @ cur), "by_geo_q0.5": {g: A_pts(mesh, W[(g, 1.0)], KMOV, 0.5) for g in GEOS}}
        print("mesh", t, round(curve[t]["A_mean12"], 4), round(time.time() - T0), "s", flush=True)
    if t >= 35 and t in CH and curve[t]["A_mean12"] >= 0.5 and t >= 300: break
R["mesh_curve"] = {str(k): v for k, v in curve.items()}
reach05 = [t for t in curve if curve[t]["A_mean12"] >= 0.5]
R["mesh_N_for_A05"] = {"first_checkpoint_ge_0.5": min(reach05) if reach05 else None,
                       "last_checkpoint_below": max([t for t in curve if curve[t]["A_mean12"] < 0.5], default=None)}
R["mesh35_vs_plans_main"] = curve[35]["A_mean12"]
R["mesh_countries_first100"] = len(set(int(pc[j]) for j in mesh[:100]))
# ---------- 仮説 ----------
H = {}; m = OUT["main_kmov"]
H["H86a"] = {"class": "予測", "value": m["minimax_regret_plan"], "rule": "主の minimax regret の最良が per_subregion@cty か per_subregion@pt", "supported": m["minimax_regret_plan"] in ("per_subregion@cty", "per_subregion@pt")}
n05 = R["mesh_N_for_A05"]["first_checkpoint_ge_0.5"]
H["H86b"] = {"class": "予測", "value": R["mesh_N_for_A05"], "rule": "mesh の12条件の平均 A が 0.5 に達する写しの数が 100〜1,000", "supported": (n05 is not None and 100 <= n05 <= 1000 and (R["mesh_N_for_A05"]["last_checkpoint_below"] or 0) >= 70)}
rc = m["mean_A_cty7"] / R["country_model_mean_A"]["kmov"]
H["H86c"] = {"class": "予測", "value": rc, "rule": "点の模型/国の模型 の7つの平均 A（kmov）< 0.5", "supported": rc < 0.5}
r71 = OUT["ref_071"]["mean_A_cty7"] / R["country_model_mean_A"]["ref_071"]
H["H86d"] = {"class": "予測", "value": r71, "rule": "点の模型/国の模型（071 の19件）>= 0.8", "supported": r71 >= 0.8}
R["hypotheses"] = H
# ---------- 図 ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
fig, ax = plt.subplots(figsize=(7, 4)); xs = sorted(curve)
ax.plot(xs, [curve[t]["A_mean12"] for t in xs], "o-", label="mesh (greedy, mean of 12 q×geo)")
for g in GEOS: ax.plot(xs, [curve[t]["by_geo_q0.5"][g] for t in xs], ".--", lw=.8, label=f"mesh {g}, q=0.5")
ax.axhline(.5, color="k", lw=.6); ax.axhline(m["summary"][m["minimax_regret_plan"]]["mean_A"], color="r", lw=.6, label=f"best N=35 plan ({m['minimax_regret_plan']})")
ax.set_xscale("log"); ax.set_xlabel("number of copies N"); ax.set_ylabel("A (not survival probability)"); ax.legend(fontsize=7); ax.set_title("Reach at forager mobility (Binford kmov), point populations")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_mesh_curve.png"), dpi=120); plt.close()
fig, ax = plt.subplots(figsize=(9, 4)); x = np.arange(len(PLP))
for j, v in enumerate(VER): ax.bar(x + j * 0.2, [OUT[v]["summary"][p]["mean_A"] for p in PLP], 0.2, label=v)
ax.set_xticks(x + 0.3); ax.set_xticklabels(PLP, rotation=25, fontsize=7); ax.set_ylabel("mean A over 12 conditions"); ax.legend(fontsize=7)
ax.set_title("N=35 placements at point resolution"); fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_plans_points.png"), dpi=120); plt.close()
fig, ax = plt.subplots(figsize=(11, 5.5)); ax.scatter(plon, plat, s=1, c="0.8")
ax.scatter(plon[mesh[:300]], plat[mesh[:300]], s=6, c="C3", label="mesh first 300"); ax.scatter(plon[PP["per_subregion@pt"]], plat[PP["per_subregion@pt"]], marker="^", facecolors="none", edgecolors="C2", label="per_subregion@pt")
ax.legend(); ax.set_title("Sites"); fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_sites.png"), dpi=110); plt.close()
R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("points", "country_model_mean_A", "mesh_N_for_A05", "mesh_countries_first100", "hypotheses")}, ensure_ascii=False, indent=1, default=float))
for v in OUT: print(v, OUT[v]["minimax_regret_plan"], OUT[v]["maximin_plan"], round(OUT[v]["mean_A_cty7"], 4), {p: (round(OUT[v]["summary"][p]["mean_A"], 4), round(OUT[v]["summary"][p]["max_regret"], 4)) for p in PLP})
print("runtime", R["runtime_s"])
