"""研究049 の探索（事前登録の外）。sim.py の E は U3 の nmax の全標本（絶えた標本と、閾値の模型ですでに重い崩壊の標本を含む）で平均したので、
小さな nmax への外挿が二重計上になる。ここでは「生き残り（ntot>=5000）で、P1 の閾値の模型では軽い崩壊（T2 以上を保つ）の標本」だけで
E[P(L|nmax)] を出す。孤立の罰が閾値の模型に上乗せする重い崩壊の上限にあたる。sim.py の results.json と同じ乱数の流れを使うため、
sim.py を丸ごと exec してからその名前空間で計算する（seed 20261049、追加の乱数なし）。"""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NS = {"__file__": os.path.join(HERE, "sim.py"), "__name__": "sim049"}
_src = open(os.path.join(HERE, "sim.py"), encoding="utf8").read()
_src = _src[:_src.index("# ================= 図 =================")]   # 図と results.json の書き出しは省く
exec(compile(_src, os.path.join(HERE, "sim.py"), "exec"), NS)
SRC, HV1, pred, REGS, sub, HS1, CASES = NS["SRC"], NS["HV1"], NS["pred"], NS["REGS"], NS["sub"], NS["HS1"], NS["CASES"]
OUT = {"note": "exploratory, not preregistered"}
sv = SRC["U3"]["sv"]; hv = np.asarray(HV1["U3"], float)
OUT["U3_hv_shape"] = list(np.shape(hv)); OUT["U3_hv_mean"] = float(hv.mean())
alive = sv["ntot"] >= 5000
OUT["U3_share_alive"] = float(alive.mean())
for lab in ("main", "env_as_0", "Andaman_0.5", "with_T"):
    a, b, c = REGS[lab]
    n = np.maximum(sv["nmax"][sub], 1.0); w = (alive[sub] * (1 - hv[sub])).astype(float)
    pr = pred(a, b, n)
    e = (pr * w[None, :]).sum(1) / w.sum()
    OUT[f"E_cond_light_{lab}"] = {"mean": float(e.mean()), "q05_95": [float(q) for q in np.percentile(e, [5, 95])], "P_gt_hstar": float((e > HS1).mean())}
    if lab == "main":
        OUT["light_nmax_q05_50_95"] = [float(q) for q in np.percentile(n[w > 0.5], [5, 50, 95])]
        wa = alive[sub].astype(float); ea = (pr * wa[None, :]).sum(1) / wa.sum()
        OUT["E_cond_alive_main"] = {"mean": float(ea.mean()), "q05_95": [float(q) for q in np.percentile(ea, [5, 95])]}
# 大きな孤立集団（N>=5e4）だけの Jeffreys：0/2
g2 = np.random.default_rng(20261149)
big = [c for c in CASES if c[1] == "iso" and c[4] is not None and c[4][0] >= 5e4]
x = sum(c[2] == 1 for c in big); p = g2.beta(0.5 + x, 0.5 + len(big) - x, 200000)
OUT["big_isolated_jeffreys"] = {"cases": [c[0] for c in big], "x": int(x), "n": len(big), "post_mean": float(p.mean()),
                                "q95": float(np.percentile(p, 95)), "P_gt_hstar": float((p > HS1).mean())}
# h* 以下に P(L) を抑えるのに要る「喪失なしの大きな孤立集団の数」k（Jeffreys の事後平均 0.5/(k+1) < h*）
OUT["k_needed_post_mean_below_hstar"] = int(np.ceil(0.5 / HS1 - 1))
json.dump(OUT, open(os.path.join(HERE, "explore_results.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=1))
