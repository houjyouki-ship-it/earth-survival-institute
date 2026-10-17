"""研究028 探索（事前登録外）：sim.py の関数を再利用して、(a) 予算つき欲張り選択の ΔV、(b) 価値の重みを星から変えた場合、
(c) chromium の TAR 上限、(d) 025 の圧縮率での費用、を調べる。出力 explore_results.json。"""
import os, json, runpy, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["OUT_JSON"] = "_explore_tmp.json"
G = runpy.run_path(f"{HERE}/sim.py", run_name="sim")
os.remove(f"{HERE}/_explore_tmp.json")
items, w, st, tar, sw, W, M, need0, needc, cost_bytes, dV, REEL, SETS, KiB = (G[k] for k in
    "items w st tar sw W M need0 needc cost_bytes dV REEL SETS KiB".split())
N = len(M); Ms = np.sort(M); pM = lambda x: np.searchsorted(Ms, x, side="right") / N
def q_vec(e, swv):
    o = np.argsort(e); cw = np.cumsum(swv[o]) / swv.sum(); k = np.searchsorted(e[o], M, side="right")
    return np.where(k > 0, cw[np.maximum(k - 1, 0)], 0.0)
def greedy_set(c, budget, swv):
    nc = needc(c); cb = w * cost_bytes(c)
    gain = swv * np.clip(pM(need0 - 1) - pM(nc - 1), 0, None)
    od = np.argsort(-(gain / np.maximum(cb, 1))); cc = np.cumsum(cb[od])
    frac = np.zeros(len(items)); k = np.searchsorted(cc, budget * REEL); frac[od[:k]] = 1.0
    if k < len(od) and items[od[k]]["src"] == "sample025": frac[od[k]] = (budget * REEL - (cc[k - 1] if k else 0)) / cb[od[k]]
    return frac, nc
def q_frac(frac, nc, swv):
    """frac：各項目の二重書きされた割合（標本の束は部分的に）。開ける＝元の need か、二重書き分は nc。"""
    q1 = q_vec(np.minimum(need0, nc), swv * frac); q0 = q_vec(need0, swv * (1 - frac))
    a = (swv * frac).sum() / swv.sum(); return a * q1 + (1 - a) * q0
OUT = {"note": "事前登録外の探索"}
# (a) 欲張り選択の ΔV（星重み）
OUT["greedy_dV_star"] = {}
for c in ("c3_16KiB_lc0", "c4_4KiB_lc0", "u_uncompressed"):
    for B in (0.01, 0.1, 1.0):
        fr, nc = greedy_set(c, B, sw); qv = q_frac(fr, nc, sw)
        ci = [i for i in np.argsort(-fr * sw) if fr[i] > 0 and items[i]["src"] == "census"]; top = [items[i]["name"] for i in ci[:8]]
        OUT["greedy_dV_star"][f"{c}@{B}"] = {"mean_q": float(qv.mean()), "dV_P2": dV(qv), "n_repos": float((w * fr).sum()),
                                            "featured_included": int(sum(fr[i] > 0 for i in range(len(items)) if items[i]["featured"])),
                                            "example_census_repos": top}
# (b) 価値の重みを変える：repo 数重み（w）と非圧縮バイト重み（w·tar）
OUT["alt_weights"] = {}
for lab, swv in (("repo_count", w.copy()), ("bytes", w * tar)):
    d = {"as_archived_q": float(q_vec(need0, swv).mean()), "share_featured15": float(swv[SETS["featured15"]].sum() / swv.sum()),
         "share_ge20k": float(swv[SETS["all_ge20k"]].sum() / swv.sum())}
    for s in ("featured15", "all_ge20k", "all_high"):
        e = np.where(SETS[s], np.minimum(need0, needc("c4_4KiB_lc0")), need0); qv = q_vec(e, swv)
        d[f"c4_{s}"] = {"mean_q": float(qv.mean()), "dV_P2": dV(qv)}
    fr, nc = greedy_set("c4_4KiB_lc0", 0.1, swv); qv = q_frac(fr, nc, swv)
    d["c4_greedy@0.1"] = {"mean_q": float(qv.mean()), "dV_P2": dV(qv)}
    OUT["alt_weights"][lab] = d
# (b2) 星重みで選んだ欲張り集合（c4、0.1 リール）を、バイト重みで評価すると
fr, nc = greedy_set("c4_4KiB_lc0", 0.1, sw); qv = q_frac(fr, nc, w * tar)
OUT["star_greedy_evaluated_by_bytes"] = {"mean_q": float(qv.mean()), "dV_P2": dV(qv)}
# (c) chromium の TAR を API の大きさ（全履歴の pack、上限）にした場合の ≥20k 全数の費用
ic = [i for i, it in enumerate(items) if it["name"] == "chromium/chromium"][0]
OUT["chromium_upper"] = {}
for c in ("c2_64KiB_lc0", "c4_4KiB_lc0"):
    cb = w * cost_bytes(c); base = cb[SETS["all_ge20k"]].sum()
    up = base - cb[ic] + 67517178 * 1024 * G["rho"][c]
    OUT["chromium_upper"][c] = {"ge20k_reels_main": float(base / REEL), "ge20k_reels_chromium_upper": float(up / REEL)}
json.dump(OUT, open(f"{HERE}/explore_results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=1))
