"""登録外（支持に数えない）：Italy の谷の高さを Malanima（itwiki）の 4〜6M で動かしたときの lag と phi。注：F_heavy は生存確率ではない。"""
import json, math, os
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
CJ = json.load(open(os.path.join(RES, "131-phi-same-collapse", "cases.json"))); EX = json.load(open(os.path.join(RES, "131-phi-same-collapse", "explore_posttrough.json")))
r = CJ["R_MED"]; t_rec = CJ["cases"]["Italy|main"]["t_rec"]; LB = EX["lag_years"]["Britain"]
out = {}
for Pmin in (4e6, 5e6, 6e6):
    lag = (t_rec - 800) - math.log(8.5e6 / Pmin) / r; out[f"{Pmin/1e6:.0f}M"] = {"lag": lag, "phi_Italy": max(lag, 0) / LB,
        "phi_mean3": (max(lag, 0) / LB) / 3}
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, "explore_italy_trough.json"), "w"), indent=1)
