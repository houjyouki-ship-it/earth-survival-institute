"""研究039 探索（初回結果の後に追加、判定には使わない）：XPT 表2・表3 の中央値の引き算による「説明されない残り」。
中央値は足せない（README の方法論規則）ので参考値。ネット接続なし。"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
cat = {"S": {"AI": 2.13, "eng": 0.8, "nat": 1.0, "nuc": 4.0, "non": 0.05, "tot": 9.05},
       "E": {"AI": 12, "eng": 3, "nat": 0.85, "nuc": 8, "non": 0.09, "tot": 20}}
ext = {"S": {"AI": 0.38, "eng": 0.01, "nat": 0.0018, "nuc": 0.074, "non": 0.0043, "tot": 1},
       "E": {"AI": 3, "eng": 1, "nat": 0.01, "nuc": 0.55, "non": 0.004, "tot": 6}}
out = {}
for g in ("S", "E"):
    rc = cat[g]["tot"] - sum(v for k, v in cat[g].items() if k != "tot")
    re = ext[g]["tot"] - sum(v for k, v in ext[g].items() if k != "tot")
    out[g] = {"residual_catastrophe_pct": rc, "residual_extinction_pct": re, "ratio_ext_over_cat": (re / rc) if rc > 0 else None,
              "named_ext_over_cat": {k: ext[g][k] / cat[g][k] for k in ("AI", "eng", "nat", "nuc", "non")}}
r = json.load(open(os.path.join(HERE, "results.json")))
out["abs_heavy_value_range"] = {a: [min(r["weighted_039"][a][u]["R_tab_from_heavy"] for u in ("U1", "U2", "U3")),
                                    max(r["weighted_039"][a][u]["R_tab_from_heavy"] for u in ("U1", "U2", "U3"))] for a in r["weighted_039"]}
json.dump(out, open(os.path.join(HERE, "explore_results.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
