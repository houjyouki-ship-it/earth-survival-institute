"""研究057 の探索（事前登録の外、初回実行の後に追加）：重い崩壊で T2 到達を縛っているのは段の年数か人口の待ちか。
sim.py を「FS = [」の前まで exec する（同じ乱数）。"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "sim.py"), encoding="utf8").read(); cut = src.index("FS = [")
G = {"__file__": os.path.join(HERE, "sim.py"), "__name__": "sim057"}
exec(compile(src[:cut], os.path.join(HERE, "sim.py"), "exec"), G)
B, T2s_m, HEAVY, COMPS, life = G["B"], G["T2s_m"], G["G54"]["HEAVY"], G["COMPS"], G["life"]
out = {}
for k in COMPS:
    hv = np.isin(B[k]["st"], HEAVY)
    if hv.sum() < 100: continue
    b = B[k]["T2s"][hv]; f0 = T2s_m(k, 0.0, 0.0, 0.0, 1.0)[hv]; t3 = T2s_m(k, 1, 1, 1, 0.1)[hv]
    lf = life[hv]
    out[k] = {"n_heavy": int(hv.sum()), "T2s_base_q": [float(np.quantile(b, p)) for p in (0.05, 0.5, 0.95)],
              "T2s_f0_q": [float(np.quantile(f0, p)) for p in (0.05, 0.5, 0.95)],
              "share_knowledge_binding": float(np.mean(f0 < 0.9 * b)),
              "share_base_T2s_lt_life": float(np.mean(b < lf)), "share_f0_T2s_lt_life": float(np.mean(f0 < lf)),
              "share_thr0.1_T2s_lt_life": float(np.mean(t3 < lf))}
json.dump(out, open(os.path.join(HERE, "explore_results.json"), "w"), indent=1)
for k, v in out.items(): print(k, json.dumps(v))
