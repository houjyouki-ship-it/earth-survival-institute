"""探索（初回結果を見た後に追加、事前登録外）：重い崩壊の価値がどのシナリオから来るか。results.json だけを読む。"""
import json
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
r = json.load(open(f"{HERE}/results.json"))
r32 = json.load(open(f"{HERE}/../032-collapse-scenarios/results.json"))
out = {}
for a in r["weighted_038"]:
    mw = r32["weighted"][a]["mean_weights"]
    heavy = {s: mw[s] * r["scenarios_038"][s]["R_tab_from_heavy"] for s in mw}
    tot = sum(heavy.values()); tab = {s: mw[s] * r["scenarios_038"][s]["R_tab"] for s in mw}; tt = sum(tab.values())
    hs = {s: mw[s] * r["scenarios_038"][s]["heavy_share"] for s in mw}; hst = sum(hs.values())
    out[a] = {"heavy_value_share_by_scen": {s: (v / tot if tot else None) for s, v in heavy.items()},
              "heavy_value_total_approx": tot, "R_tab_total_approx": tt, "heavy_value_frac_of_R_tab": tot / tt if tt else None,
              "heavy_prob_share_by_scen": {s: (v / hst if hst else None) for s, v in hs.items()},
              "heavy_prob_weighted_excl_GEN": hst - hs["GEN"], "heavy_prob_weighted": hst}
json.dump(out, open(f"{HERE}/explore_results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
