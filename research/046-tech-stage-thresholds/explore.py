"""探索（事前登録の外）：Kline & Boyd 2010 表1（038 の sources で確認済みの値）で、接触の多さを人口の倍率に換算する。
log(道具の数) = a + b·log(人口) + c·[接触 high]。接触 high は人口を exp(c/b) 倍にしたのと同じ。seed 20261046 の bootstrap。"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
KB = [("Malekula", 1100, "low", 13), ("Tikopia", 1500, "low", 22), ("Santa Cruz", 3600, "low", 24),
      ("Yap", 4791, "high", 43), ("Lau Fiji", 7400, "high", 33), ("Trobriand", 8000, "high", 19),
      ("Chuuk", 9200, "high", 40), ("Manus", 13000, "low", 28), ("Tonga", 17500, "high", 55), ("Hawaii", 275000, "low", 71)]
lp = np.log([k[1] for k in KB]); lt = np.log([k[3] for k in KB]); hc = np.array([k[2] == "high" for k in KB], float)
def fit(i):
    X = np.c_[np.ones(len(i)), lp[i], hc[i]]
    return np.linalg.lstsq(X, lt[i], rcond=None)[0]
a, b, c = fit(np.arange(10))
g = np.random.default_rng(20261046); bs = []
for _ in range(20000):
    i = g.integers(0, 10, 10)
    if np.ptp(lp[i]) > 0 and 0 < hc[i].sum() < 10:
        bb = fit(i)
        if bb[1] > 0.02: bs.append(np.exp(bb[2] / bb[1]))
bs = np.array(bs)
noH = np.arange(9)
a2, b2, c2 = fit(noH)
out = {"a": a, "b": b, "c": c, "contact_pop_multiplier": float(np.exp(c / b)),
       "bootstrap_multiplier_q05_50_95": [float(x) for x in np.percentile(bs, [5, 50, 95])], "n_boot_used": int(bs.size),
       "without_Hawaii": {"b": b2, "c": c2, "multiplier": float(np.exp(c2 / b2)) if b2 > 0 else None}}
json.dump(out, open(os.path.join(HERE, "explore_results.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
