"""事後の探索（事前登録の外）：主の届く距離の A のうち、写しが自国にあるだけで届く分（距離0）の割合。国の中心の解像度の限界を見る。"""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); src = open(os.path.join(HERE, "sim.py"), encoding="utf8").read()
G = {"__file__": os.path.join(HERE, "sim.py"), "__name__": "x73"}; exec(src[: src.index("# ---------- 仮説 ----------")], G)
PLANS, DM, GEO, SC60, A_w, km = G["PLANS"], G["DM"], G["GEO"], G["SC60"], G["A_w"], G["km"]
out = {}
for p, P in PLANS.items():
    own = np.mean([A_w(P, DM[d], GEO[g], np.array([1e-6]), np.array([1.0]), q) for (q, g, d) in SC60])
    main = G["OUT"]["main_kmov"]["summary"][p]["mean_A"]; out[p] = {"own_country_only_A": own, "main_A": main, "own_share": own / main}
json.dump(out, open(os.path.join(HERE, "explore_posthoc.json"), "w"), indent=1)
for p, v in out.items(): print(p, {k: round(x, 4) for k, x in v.items()})
