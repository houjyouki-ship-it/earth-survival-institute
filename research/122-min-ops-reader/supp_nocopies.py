"""補足：5 重の写しを使わない（1 点＝1 ビット）D3 の p_relay を、120 の prelay と同じ式で計算（results.json の値から）。"""
import json
R = json.load(open("results.json", encoding="utf8"))
px, YR = 5.28e11, 365.25 * 86400
B = {"p_build": 0.8, "M50": 1e5, "L_arm": 1e6, "L_reed": 1e9}
pr = R["p_relay_versions"]["D3_realistic"]; p_reed_full = pr["p_reed"] * (1 + pr["M_reed"] / B["M50"])  # p_build*p_seal*p_glass*p_NiFe
out = {}
for key in ("D3_nocopies|rand", "D3_realistic|rand", "D4_stepper|rand"):
    d = R["designs"][key]; C, k = d["relays"], d["ops_per_point"]; N0 = px / (10 * 20 * YR) * C
    Ma, Mr = N0 + px * k / B["L_arm"], N0 + px * k / B["L_reed"]
    pa = B["p_build"] / (1 + Ma / B["M50"]); prd = p_reed_full / (1 + Mr / B["M50"])
    out[key] = {"ops_per_point": k, "p_arm": pa, "p_reed": prd, "p_relay": 1 - (1 - pa) * (1 - prd)}
assert abs(out["D3_realistic|rand"]["p_relay"] - pr["p_relay"]) < 1e-9
json.dump(out, open("supp_nocopies.json", "w"), indent=1); print(out)
