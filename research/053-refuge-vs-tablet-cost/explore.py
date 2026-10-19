# 研究053 探索（事前登録の外）：重い崩壊と軽い崩壊に分けた費用あたりの比、最も避難所に有利な組み合わせ、金庫の物差しの外で避難所が要る価値。
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
W = json.load(open(os.path.join(HERE, "..", "040-cohesion-vs-knowledge", "results.json")))
V = W["weighted"]; S = W["sensitivity"]["p_ref_1"]["weighted"]
cTAB = R["C_TAB_q"]; tot = R["refuge_total_cost_q"]; cap = R["refuge_capital_cost_q"]
out = {}
for key in ("A_Ord/U3", "A_XPT_S/U2"):
    a, u = key.split("/"); w = V[a][u]; out[key] = {}
    th, tt = w["TAB"]["from_heavy"], w["TAB"]["total"]; tl = tt - th
    for iv, n in (("REF_1e5", 1e5), ("REF_1e6", 1e6)):
        rh = w[iv]["from_heavy"]; rt = w[iv]["total"]; rl = rt - rh
        e = {"value_ratio_heavy_only": rh / th, "value_ratio_light_only": rl / tl if tl > 0 else None,
             "CE_median_heavy_only": rh / th * cTAB[1] / tot[iv][1], "CE_median_light_only": (rl / tl) * cTAB[1] / tot[iv][1] if tl > 0 else None}
        # 避難所に最も有利な組み合わせ：p_ref=1 の価値、TAB 費用の95%点、避難所の資本費の5%点（維持費なし）
        rv_best = S[a][u][iv] / S[a][u]["TAB"] if a in S and u in S[a] else None
        if rv_best:
            e["CE_best_case"] = rv_best * cTAB[2] / cap[iv][0]
            e["value_ratio_pref1"] = rv_best
        # 金庫の物差しの外で避難所が持つべき価値（金庫の価値の単位、TAB と費用あたり同じになる量）
        e["required_extra_value_vault_units_median"] = tot[iv][1] / cTAB[1] * tt - rt
        e["required_extra_over_TAB_value"] = e["required_extra_value_vault_units_median"] / tt
        out[key][iv] = e
json.dump(out, open(os.path.join(HERE, "explore_results.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, indent=1))
