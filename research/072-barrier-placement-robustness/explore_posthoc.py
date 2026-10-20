"""事後の探索（事前登録の外）。sim.py を「# ---------- 仮説」の前まで exec して、
(1) 国を等しい重みにした地理（査読への返答の粗い模型と同じ重み）で cover750 と cover1500 を比べる、
(2) 各距離の定義で、コストが届く上限（2×19件の最大）以内の国の組の割合を出す。"""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); src = open(os.path.join(HERE, "sim.py"), encoding="utf8").read()
G = {"__file__": os.path.join(HERE, "sim.py"), "__name__": "x"}; exec(src[: src.index("# ---------- 仮説")], G)
NK, CAND, PLANS, DM, A_of, KS, QS, DISTS = (G[x] for x in ("NK", "CAND", "PLANS", "DM", "A_of", "KS", "QS", "DISTS"))
w = np.zeros(NK); w[CAND] = 1.0; w /= w.sum(); out = {}
for p in PLANS: out[p] = [A_of(PLANS[p], DM[d], w, k, q) for k in KS for q in QS for d in DISTS]
c = float(np.mean(np.array(out["cover750"]) >= np.array(out["cover1500"]) - 1e-12))
E = {"equal_weight_mean_A": {p: float(np.mean(v)) for p, v in out.items()}, "equal_weight_share_cover750_ge_cover1500": c,
     "share_pairs_within_cut": {d: float(np.isfinite(DM[d]).mean()) if d != "gc" else float((DM[d] <= G["CUT"]).mean()) for d in DISTS}}
json.dump(E, open(os.path.join(HERE, "explore_posthoc.json"), "w"), ensure_ascii=False, indent=1); print(json.dumps(E, ensure_ascii=False, indent=1))
