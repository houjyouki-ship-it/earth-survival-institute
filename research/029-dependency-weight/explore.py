"""研究029 探索（事前登録外）：(a) 依存重みの生態系の内訳、(b) 生態系ごとに総量をそろえた重み、
(c) 依存0の repo が持つ星重み、(d) 生態系正規化重みでの予算つき選択の交差評価。出力 explore_results.json。"""
import os, json, runpy, contextlib, io, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["OUT029"] = "_explore_tmp.json"
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(f"{HERE}/sim.py", run_name="sim")
os.remove(f"{HERE}/_explore_tmp.json")
items, raw, ok, D1, w, sw, SETS, cen, WEIGHTS = (G[k] for k in "items raw ok D1 w sw SETS cen WEIGHTS".split())
greedy_set, q_frac, q_vec, need0, dV = G["greedy_set"], G["q_frac"], G["q_vec"], G["need0"], G["dV"]
eco = np.array([(raw[i["name"]]["top"][0]["ecosystem"] if o and raw[i["name"]]["top"] and D1[k] > 0 else "none")
                for k, (i, o) in enumerate(zip(items, ok))], dtype=object)
dw = WEIGHTS["dep_D1"]
OUT = {"note": "事前登録外の探索"}
OUT["dep_share_by_top_ecosystem"] = dict(sorted({e: float(dw[eco == e].sum() / dw.sum()) for e in set(eco)}.items(), key=lambda x: -x[1]))
OUT["star_share_by_top_ecosystem"] = dict(sorted({e: float(sw[eco == e].sum() / sw.sum()) for e in set(eco)}.items(), key=lambda x: -x[1])[:12])
OUT["star_share_with_D1_zero"] = float(sw[D1 == 0].sum() / sw.sum())
# (b) 生態系ごとの総量をそろえる（各生態系の依存重みの合計を 1 に）
en = np.zeros(len(items))
for e in set(eco):
    if e == "none": continue
    m = eco == e; en[m] = dw[m] / dw[m].sum()
OUT["eco_norm_n_ecosystems"] = int(len(set(eco)) - 1)
OUT["eco_norm_share"] = {s: float(en[m].sum() / en.sum()) for s, m in SETS.items()}
base = {lab: float(q_vec(need0, v).mean()) for lab, v in (("stars", sw), ("dep_D1", dw), ("eco_norm", en))}
OUT["budget_c4_0.1_cross"] = {}
for sl, sv in (("stars", sw), ("dep_D1", dw), ("eco_norm", en), ("mix3", sw / sw.sum() + dw / dw.sum() + en / en.sum())):
    fr, nc = greedy_set("c4_4KiB_lc0", 0.1, sv)
    OUT["budget_c4_0.1_cross"][sl] = {el: float(q_frac(fr, nc, ev).mean() - base[el]) for el, ev in (("stars", sw), ("dep_D1", dw), ("eco_norm", en))}
# (c) Featured15 の依存
OUT["featured_D1_zero"] = [i["name"] for k, i in enumerate(items) if i["featured"] and D1[k] == 0]
json.dump(OUT, open(f"{HERE}/explore_results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=1))
# (e) 全数の ≥20,000 星だけを母集団にした交差評価（標本の重い裾の影響を除く）
cost_bytes, REEL, pM, needc = G["cost_bytes"], G["REEL"], G["pM"], G["needc"]
g20 = SETS["all_ge20k"]
def sub(v): return np.where(g20, v, 0.0)
U = {"stars": sub(sw), "dep_D1": sub(dw), "dep_log1p_D1": sub(WEIGHTS["dep_log1p_D1"]), "eco_norm": sub(en)}
U["mix50"] = U["stars"] / U["stars"].sum() + U["dep_D1"] / U["dep_D1"].sum()
base = {el: float(q_vec(need0, v).mean()) for el, v in U.items() if el != "mix50"}
tot_c4 = float((w * cost_bytes("c4_4KiB_lc0"))[g20].sum() / REEL)
OUT["census_only"] = {"cost_all_ge20k_c4_reels": tot_c4, "spearman_note": "sim.py ge20k を参照",
                      "share_featured_in_ge20k": {el: float(v[SETS["featured15"]].sum() / v.sum()) for el, v in U.items() if el != "mix50"},
                      "budgets": {}}
for B in (0.003, 0.01, 0.03):
    row = {}; FRs = {}
    for sl, sv in U.items():
        fr, nc = greedy_set("c4_4KiB_lc0", B, sv); fr = fr * g20; FRs[sl] = fr
        row[sl] = {"n": int((fr > 0).sum()), **{el: float(q_frac(fr, nc, ev).mean() - base[el]) for el, ev in U.items() if el != "mix50"}}
    a = set(np.where(FRs["stars"] > 0)[0]); b = set(np.where(FRs["dep_D1"] > 0)[0])
    row["jaccard_stars_dep_D1"] = len(a & b) / max(1, len(a | b))
    row["ratio_star_sel_eval_dep"] = row["stars"]["dep_D1"] / row["dep_D1"]["dep_D1"]
    row["ratio_dep_sel_eval_star"] = row["dep_D1"]["stars"] / row["stars"]["stars"]
    row["mix_vs_opt"] = {"stars": row["mix50"]["stars"] / row["stars"]["stars"], "dep_D1": row["mix50"]["dep_D1"] / row["dep_D1"]["dep_D1"]}
    OUT["census_only"]["budgets"][str(B)] = row
json.dump(OUT, open(f"{HERE}/explore_results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(OUT["census_only"], ensure_ascii=False, indent=1))
