"""探索（事前登録の外）：遠い場所の τ の事後が事前の下端にどれだけ依るか。sim.py と同じ区間打ち切りの指数、計算のみ。"""
import json, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
C = [c for c in json.load(open(os.path.join(H, "sources", "cases.json"), encoding="utf8"))["cases"] if c["class"] == "remote"]
out = {}
for lo in (10, 30, 100):
    tg = np.exp(np.linspace(np.log(lo), np.log(1e4), 4001)); ll = np.zeros_like(tg)
    for c in C: ll += np.log(np.maximum(np.exp(-c["L"] / tg) - np.exp(-c["U"] / tg), 1e-300))
    w = np.exp(ll - ll.max()); w /= w.sum(); cw = np.cumsum(w)
    out[f"prior_lo_{lo}"] = [float(tg[np.searchsorted(cw, q)]) for q in (0.05, 0.5, 0.95)]
# 尤度だけ：P(2件とも U より前に失われる)=0.5 となる τ
tg = np.exp(np.linspace(np.log(1), np.log(1e5), 20001)); L = np.prod([1 - np.exp(-c["U"] / tg) for c in C], axis=0)
out["tau_where_lik_0.5"] = float(tg[np.argmin(abs(L - 0.5))]); out["tau_where_lik_0.1"] = float(tg[np.argmin(abs(L - 0.1))])
json.dump(out, open(os.path.join(H, "explore_results.json"), "w"), indent=1); print(out)
