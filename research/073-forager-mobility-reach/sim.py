"""研究073：狩猟採集民の年間の移動距離（Binford 2001、CRAN binford 0.1.0）で届く距離を置き直し、072 の置き方の順位を確かめる。
計算のみ（ネット接続なし）。指標 A は「生存者（地理の重み）のうち届く範囲に生き残った写しが1つ以上ある割合」の厳密な期待値で、
生存確率でも復興確率でもない。乱数は 072 の random の置き方の抽出（seed 20261072）と本回の bootstrap（seed 20261073）だけ。"""
import json, math, os, time
import numpy as np, pandas as pd
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__))
P72 = os.path.join(HERE, "..", "072-barrier-placement-robustness", "sim.py")
src = open(P72, encoding="utf8").read(); G = {"__file__": P72, "__name__": "x72"}
exec(src[: src.index("# ---------- 図 ----------")], G)
R72 = json.load(open(os.path.join(os.path.dirname(P72), "results.json"), encoding="utf8"))
R = {"note": "A は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。"}
# 検査1：072 の summary を差0で再現
diff = [(p, k) for p in R72["summary"] for k in R72["summary"][p] if R72["summary"][p][k] != G["R"]["summary"][p][k]]
assert not diff, diff; R["check_072_summary_identical"] = True
PLANS, DM, GEO, NK, CASES71, TA, SC = G["PLANS"], G["DM"], G["GEO"], G["NK"], G["CASES"], G["TA"], G["SC"]
PL = list(PLANS); QS, GEOS, DISTS, KS = G["QS"], G["GEOS"], G["DISTS"], G["KS"]
SC60 = [(q, g, d) for q in QS for g in GEOS for d in DISTS]
def A_w(P, D, w, reach, wt, q):
    """届く距離 reach（km の配列）を重み wt で平均した A。072 の A_of（hazard なし）と同じ式。"""
    X = np.sort(D[:, P], axis=1); out = np.zeros(NK)
    for r, a in zip(reach, wt):
        out += a * (1 - q ** (X <= r).sum(1))
    return float(w @ out)
# 検査2：A_w が 072 の A_of と一致
for (k, q, g, d) in SC[:12]:
    a = A_w(PLANS["per_subregion"], DM[d], GEO[g], k * CASES71, np.full(len(CASES71), 1 / len(CASES71)), q)
    assert abs(a - G["A_of"](PLANS["per_subregion"], DM[d], GEO[g], k, q)) < 1e-12
R["check_A_w_equals_A_of"] = True
# ---------- 資料 ----------
L = pd.read_csv(os.path.join(HERE, "sources", "lrb_mobility.csv"))
km = L.dropna(subset=["kmov"]); km = km[km.kmov > 0]
diam = 2 * np.sqrt(L.area.values * 100 / math.pi)
reg_w = np.array(1 / km["wc.area"].map(km["wc.area"].value_counts()), dtype=float); reg_w = reg_w / reg_w.sum()
VER = {"main_kmov": (km.kmov.values, None), "diam_area": (diam, None), "kmov_x2": (2 * km.kmov.values, None),
       "per_move": (L.kspmov.dropna().values, None), "region_balanced": (km.kmov.values, reg_w),
       "kmov_max_by_region": (km.groupby("wc.area").kmov.max().values, None), "ref_071": (CASES71, None)}
R["data"] = {"n_societies": int(len(L)), "n_kmov": int(len(km)), "kmov_median": float(np.median(km.kmov)), "kmov_p90": float(np.percentile(km.kmov, 90)),
             "kmov_max": float(km.kmov.max()), "radius_median_km": float(np.median(diam / 2)), "diam_median_km": float(np.median(diam)),
             "kspmov_median": float(np.median(L.kspmov.dropna())), "regions": km["wc.area"].value_counts().to_dict(),
             "kmov_median_by_region": km.groupby("wc.area").kmov.median().round(1).to_dict(),
             "share_kmov_ge_1606": float(np.mean(km.kmov >= 1606)), "share_kmov_ge_803": float(np.mean(km.kmov >= 803)),
             "spearman_kmov_abslat": float(pd.Series(km.kmov.values).rank().corr(pd.Series(np.abs(km.latitude.values)).rank())),
             "spearman_kmov_area": float(km.kmov.rank().corr(km.area.rank()))}
rng = np.random.default_rng(20261073)
bs = [np.median(rng.choice(km.kmov.values, len(km))) for _ in range(2000)]; R["data"]["kmov_median_boot90"] = [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]
# ---------- 置き方の比較 ----------
def run(reach, wt):
    wt = np.full(len(reach), 1 / len(reach)) if wt is None else wt
    T = {p: np.array([A_w(P, DM[d], GEO[g], reach, wt, q) for (q, g, d) in SC60]) for p, P in PLANS.items()}
    M = np.vstack([T[p] for p in PL]); best = M.max(0); reg = best - M
    S = {p: {"mean_A": float(T[p].mean()), "worst_A": float(T[p].min()), "max_regret": float(reg[i].max())} for i, p in enumerate(PL)}
    return T, S, min(PL, key=lambda p: S[p]["max_regret"]), max(PL, key=lambda p: S[p]["worst_A"])
OUT = {}; TV = {}
for v, (reach, wt) in VER.items():
    T, S, a, b = run(reach, wt); TV[v] = T
    OUT[v] = {"summary": S, "minimax_regret_plan": a, "maximin_plan": b, "mean_A_all_plans": float(np.mean([S[p]["mean_A"] for p in PL]))}
R["versions"] = OUT
# 072 の格子の k ごとの同じ60条件の平均 A
idx = {k: [i for i, s in enumerate(SC) if s[0] == k] for k in KS}
R["grid072_mean_A"] = {str(k): float(np.mean([TA[p][idx[k]].mean() for p in PL])) for k in KS}
assert abs(R["grid072_mean_A"]["1.0"] - OUT["ref_071"]["mean_A_all_plans"]) < 1e-12
# 主の平均 A に等しい k（071 の19件×k）を二分法で
def meanA_k(k): return float(np.mean([A_w(P, DM[d], GEO[g], k * CASES71, np.full(19, 1 / 19), q) for P in PLANS.values() for (q, g, d) in SC60]))
def k_equiv(target):
    lo, hi = 0.001, 1.0
    for _ in range(30):
        m = (lo + hi) / 2; lo, hi = (m, hi) if meanA_k(m) < target else (lo, m)
    return (lo + hi) / 2
R["k_equiv"] = {v: k_equiv(OUT[v]["mean_A_all_plans"]) for v in ("main_kmov", "diam_area", "kmov_x2", "region_balanced")}
R["k_median_ratio"] = R["data"]["kmov_median"] / float(np.median(CASES71))
# 主で地理・距離ごとの最良
R["main_best_by_geo"] = {g: PL[int(np.argmax([np.mean([TV["main_kmov"][p][i] for i, s in enumerate(SC60) if s[1] == g]) for p in PL]))] for g in GEOS}
# 小さい届く距離で写しが本当に要る数：主の届く距離で、各国に1つの写し（N=国の数）のときの A（上限の目安）
CAND = G["CAND"]
R["main_all_countries_A_q0"] = {g: A_w(CAND, DM["gc"], GEO[g], km.kmov.values, np.full(len(km), 1 / len(km)), 0.0) for g in GEOS}
# ---------- 仮説 ----------
H = {}
H["H85a"] = {"class": "確認的", "value": R["data"]["kmov_median"], "rule": "kmov の中央値 < 401 km", "supported": R["data"]["kmov_median"] < 401}
H["H85b"] = {"class": "確認的", "value": R["data"]["radius_median_km"], "rule": "sqrt(AREA/π) の中央値 < 100 km", "supported": R["data"]["radius_median_km"] < 100}
H["H85c"] = {"class": "確認的（弱い：一般論を既知）", "value": R["data"]["spearman_kmov_abslat"], "rule": "Spearman(kmov, |lat|) >= 0.3", "supported": R["data"]["spearman_kmov_abslat"] >= 0.3}
H["H85d"] = {"class": "予測", "value": OUT["main_kmov"]["minimax_regret_plan"], "rule": "主の minimax regret の最良 = per_subregion", "supported": OUT["main_kmov"]["minimax_regret_plan"] == "per_subregion"}
ratio = OUT["main_kmov"]["mean_A_all_plans"] / OUT["ref_071"]["mean_A_all_plans"]
H["H85e"] = {"class": "予測", "value": ratio, "rule": "主の平均 A / 071(k=1) の平均 A < 0.5", "supported": ratio < 0.5}
near = min(KS, key=lambda k: abs(R["grid072_mean_A"][str(k)] - OUT["main_kmov"]["mean_A_all_plans"]))
H["H85f"] = {"class": "予測", "value": near, "rule": "072 の格子で主に最も近い k = 0.25", "supported": near == 0.25}
R["hypotheses"] = H
# ---------- 図 ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.logspace(0, 4, 41)
ax.hist(km.kmov, bins=bins, alpha=.6, label=f"Binford kmov (n={len(km)})"); ax.hist(diam, bins=bins, alpha=.4, label=f"territory diameter (n={len(diam)})")
ax.hist(CASES71, bins=bins, alpha=.7, label="071 text-seeking trips (n=19)"); ax.set_xscale("log"); ax.set_xlabel("km"); ax.set_ylabel("count"); ax.legend(fontsize=8)
ax.set_title("Reach distances: foragers' annual residential movement vs 071 trips"); fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_reach_hist.png"), dpi=120)
fig, ax = plt.subplots(figsize=(8, 4)); vs = ["main_kmov", "region_balanced", "diam_area", "kmov_x2", "per_move", "ref_071"]
x = np.arange(len(PL))
for j, v in enumerate(vs): ax.bar(x + j * 0.13, [OUT[v]["summary"][p]["mean_A"] for p in PL], 0.13, label=v)
ax.set_xticks(x + 0.33); ax.set_xticklabels(PL, rotation=20, fontsize=8); ax.set_ylabel("mean A over 60 conditions (not survival prob.)"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_A_by_plan.png"), dpi=120)
R["runtime_s"] = time.time() - T0
json.dump(R, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("data", "k_equiv", "k_median_ratio", "grid072_mean_A", "main_best_by_geo", "main_all_countries_A_q0", "hypotheses")}, ensure_ascii=False, indent=1, default=float))
for v in OUT: print(v, OUT[v]["minimax_regret_plan"], OUT[v]["maximin_plan"], round(OUT[v]["mean_A_all_plans"], 4), {p: (round(OUT[v]["summary"][p]["mean_A"], 4), round(OUT[v]["summary"][p]["max_regret"], 4)) for p in PL})
print("runtime", round(R["runtime_s"], 1))
