"""登録外（支持に数えない）：谷からの回復の遅れ（(t_rec − t_min) − 038 の予測）で phi を出し直す記述の版。
注：F_heavy・V は生存確率や復興確率ではない。"""
import json, math, os
HERE = os.path.dirname(os.path.abspath(__file__))
CJ = json.load(open(os.path.join(HERE, "cases.json"))); R = json.load(open(os.path.join(HERE, "results.json")))
r = CJ["R_MED"]; C = CJ["cases"]
def lag(P0, Pmin, t_min, t_rec): return (t_rec - t_min) - math.log(P0 / Pmin) / r
L = {"Britain": lag(3e6, 1e6, 600, 1184), "Italy": lag(8.5e6, 4e6, 800, C["Italy|main"]["t_rec"]),
     "France": lag(7.2e6, 5e6, 800, C["France|Pmin5.0(Carolingian)"]["t_rec"]), "Iberia": lag(7.5e6, 4e6, 800, C["Iberia|Frier7.5"]["t_rec"])}
phi = {k: max(v, 0) / L["Britain"] for k, v in L.items() if k != "Britain"}
g = R["grid"]
def F_at(p, vn="D2rz|none"):
    ys = [R["OUT"][f"{x:.6f}|{vn}"]["F_heavy"] for x in g]
    for i in range(len(g) - 1):
        if g[i] <= p <= g[i + 1]: return ys[i] + (ys[i + 1] - ys[i]) * (p - g[i]) / (g[i + 1] - g[i])
med = sorted(phi.values())[1]
out = {"lag_years": L, "phi": phi, "median_all": med, "F_heavy_D2rz_at_median": F_at(med), "F_heavy_D2rz_at_Italy": F_at(phi["Italy"]),
       "note": "谷の年 800 は Charlemagne の頃の推定の年（Medieval_demography）。Britain の谷 600 は 125 の推測"}
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open(os.path.join(HERE, "explore_posttrough.json"), "w"), indent=1, ensure_ascii=False)
