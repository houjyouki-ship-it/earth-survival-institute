"""研究107：制裁・封鎖・戦時の共食い整備の事例から、廃墟の道具が T2e の段の閾値を下げる倍率 m を置く（open_problems 256）。
あわせて工作機械の削り直し 1 回の厚さ δ と削れる総量（255）を公開資料で置き直す。計算のみ、ネット接続なし。
事前登録：PREREGISTRATION.md（2026-10-24 03:44 JST）。106 の sim.py を「# ---------- 仮説」の前まで exec（106 の乱数をそのまま再現）。
新しい乱数は seed 20261107。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P106 = os.path.abspath(os.path.join(HERE, "..", "106-rework-lifetime", "sim.py"))
src = open(P106, encoding="utf8").read(); cut = src.index("\n# ---------- 仮説")
S = {"__file__": P106, "__name__": "sim106"}
exec(compile(src[:cut], P106, "exec"), S)
R106 = json.load(open(os.path.join(os.path.dirname(P106), "results.json"), encoding="utf8"))
d106 = max(abs(S["runs"][k] - R106["runs_F_heavy"][k]) for k in R106["runs_F_heavy"])
assert d106 < 1e-12
G = S["G"]; G["T2s_mod"] = S["T2s_bridge"]; FB = S["FB"]
N_lens, N_T2e, L_lens, L_prec, L_bulk, ok_n, Nn = G["N_lens"], G["N_T2e"], G["L_lens"], G["L_tool_prec"], G["L_tool_bulk"], G["ok_n"], G["N"]
L_rw_full = S["L_rw_full"]
R = {"study": "107-cannibalization-mprod", "seed": 20261107, "check_106_runs_reproduced_maxdiff": d106}
g = np.random.default_rng(20261107)
def lu(lo, hi, n=Nn): return np.exp(g.uniform(np.log(lo), np.log(hi), n))

# ---------- 事例（sources/cases.json、抜粋は各 wikitext） ----------
CASES = json.load(open(os.path.join(HERE, "sources", "cases.json"), encoding="utf8"))["cases"]
q3 = lambda v: [float(np.quantile(v, p)) for p in (0.05, 0.5, 0.95)]
ratio = {}; out = {}
for c in CASES:
    if not c.get("N"): continue
    lo, hi = c["N"]; Nc = g.uniform(lo, hi, Nn) if hi > lo else np.full(Nn, float(lo))
    ratio[c["id"]] = {"N_T2e": Nc / N_T2e, "N_lens": Nc / N_lens}
    out[c["id"]] = {"qualifies_10y": c["qualifies_10y"], "resupply": c["resupply"], "N": c["N"],
                    "ratio_N_T2e_q": q3(Nc / N_T2e), "ratio_N_lens_q": q3(Nc / N_lens),
                    "share_ratio_T2e_lt1": float(np.mean(Nc / N_T2e < 1))}
R["cases"] = out
qual_main = [c["id"] for c in CASES if c.get("N") and c["qualifies_10y"] and not c.get("variant")]
qual_var = [c["id"] for c in CASES if c.get("N") and c["qualifies_10y"]]
minr = lambda ids, tier: np.min(np.vstack([ratio[i][tier] for i in ids]), axis=0)
mt_main = np.minimum(1.0, minr(qual_main, "N_T2e")); mt_var = np.minimum(1.0, minr(qual_var, "N_T2e"))
ml_var = np.minimum(1.0, minr(qual_var, "N_lens"))
R["m_case"] = {"main_T2e_q": q3(mt_main), "main_share_lt1": float(np.mean(mt_main < 1)),
               "whites_T2e_q": q3(mt_var), "whites_lens_q": q3(ml_var), "qual_main": qual_main, "qual_var": qual_var}

# ---------- 255：削り直し 1 回の厚さと総量（Myford の再研削、Practical Machinist） ----------
# 【事実（Myford）】再研削 1 回で 0.005〜0.010 in（0.127〜0.254 mm）、整列の限界まで最大 0.025 in（0.635 mm）、理論上 2〜5 回。
# 【事実（Practical Machinist）】手のスクレーパー 1 押しの平均の深さ 0.0003 in（7.6 µm）、例：0.003 in の摩耗の窪み。
IN = 25.4
d_myf = lu(0.005 * IN, 0.010 * IN)
n_myf = 0.025 * IN / d_myf                    # 2.5〜5
n_myf_5mm = 5.0 / d_myf                       # 106 の削れる総量 5 mm のまま
L_myf = np.minimum(L_bulk, n_myf * L_prec)
L_myf5 = np.minimum(L_bulk, n_myf_5mm * L_prec)
L_pm = np.minimum(L_bulk, (5.0 / (0.003 * IN)) * L_prec)   # 1 回で窪み 0.076 mm を削る版
R["rework_inputs"] = {"delta_mm_myford": [0.127, 0.254], "total_mm_myford": 0.635, "n_myford_q": q3(n_myf),
                      "n_myford_5mm_q": q3(n_myf_5mm), "L_myford_q": q3(L_myf), "L_myford_5mm_q": q3(L_myf5),
                      "L_pm_q": q3(L_pm), "L_rw106_q": q3(L_rw_full), "stroke_depth_mm": 0.0003 * IN}

m_main_l, m_main_t = S["m_main_l"], S["m_main_t"]; m_prod = S["m_prod"]
runs = {
 # 256：事例の m
 "case_main_rw106": FB("c_main_rw106", m_t=mt_main, L_t=L_rw_full, ok=ok_n),
 "case_whites_T2e_rw106": FB("c_wh_rw106", m_t=mt_var, L_t=L_rw_full, ok=ok_n),
 "case_whites_both_rw106": FB("c_wh_both_rw106", m_l=ml_var, L_l=L_lens, m_t=mt_var, L_t=L_rw_full, ok=ok_n),
 "case_whites_both_bulk": FB("c_wh_both_bulk", m_l=ml_var, L_l=L_lens, m_t=mt_var, L_t=L_bulk, ok=ok_n),
 "case_whites_both_myford": FB("c_wh_both_myf", m_l=ml_var, L_l=L_lens, m_t=mt_var, L_t=L_myf, ok=ok_n),
 # 255：δ と総量を置き直した L
 "m0.3_myford": FB("m03_myf", m_l=0.3, L_l=L_lens, m_t=0.3, L_t=L_myf, ok=ok_n),
 "m0.3_myford_5mm": FB("m03_myf5", m_l=0.3, L_l=L_lens, m_t=0.3, L_t=L_myf5, ok=ok_n),
 "m0.3_pm_0.076mm": FB("m03_pm", m_l=0.3, L_l=L_lens, m_t=0.3, L_t=L_pm, ok=ok_n),
 "mmain_myford": FB("mmain_myf", m_l=m_main_l, L_l=L_lens, m_t=m_main_t, L_t=L_myf, ok=ok_n),
 "mprod_myford": FB("mprod_myf", m_l=m_prod, L_l=L_lens, m_t=m_prod, L_t=L_myf, ok=ok_n),
 "m0.1_myford": FB("m01_myf", m_l=0.1, L_l=L_lens, m_t=0.1, L_t=L_myf, ok=ok_n),
}
R["runs_F_heavy"] = runs
R["ref_106"] = {k: R106["runs_F_heavy"][k] for k in ("RW_full_m0.3", "RW_full_mmain", "RW_full_mprod", "RW_full_m0.1")}

# ---------- 仮説 ----------
H = {}
H["H119a"] = {"class": "confirmatory", "rule": ">=1 opened source: T2e machines kept running >=10 y by cannibalization/improvised parts under sanctions",
              "supported": True, "note": "Yank_tank: ~60,000 pre-1962 US cars still driving in Cuba, kept by adapting household products and Soviet technology, parting out cars"}
small_nores = [i for i in qual_var if out[i]["ratio_N_T2e_q"][1] < 1 and out[i]["resupply"] is False]
H["H119b"] = {"class": "confirmatory", "rule": "0 cases (>=10 y) with N < median N_T2e and no external resupply",
              "value": small_nores, "supported": len(small_nores) == 0,
              "note": "rhodesia_whites (0.27-0.31M) is < N_T2e but sanctions were evaded via South Africa/Portugal"}
mn = min(out[i]["ratio_N_T2e_q"][1] for i in qual_main)
H["H119c"] = {"class": "prediction", "rule": "min over >=10y cases (country population) of median N_case/N_T2e >= 1",
              "value": mn, "supported": bool(mn >= 1)}
H["H119d"] = {"class": "prediction", "rule": "m_case versions with rework L (106): F_heavy < 1.10",
              "value": {k: runs[k] for k in ("case_main_rw106", "case_whites_T2e_rw106", "case_whites_both_rw106")},
              "supported": bool(max(runs[k] for k in ("case_main_rw106", "case_whites_T2e_rw106", "case_whites_both_rw106")) < 1.10)}
H["H119e"] = {"class": "prediction", "rule": "public per-rebuild removal depth lies within 0.02-0.2 mm",
              "value": {"myford_mm": [0.127, 0.254], "pm_worn_spot_mm": 0.0762},
              "supported": False, "note": "Myford range straddles the upper bound (0.254 > 0.2); total removable 0.635 mm vs 5 mm assumed in 106 (8x smaller)"}
R["hypotheses"] = H

# ---------- 図 ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ids = list(out)
med = [out[i]["ratio_N_T2e_q"][1] for i in ids]
lo = [med[j] - out[i]["ratio_N_T2e_q"][0] for j, i in enumerate(ids)]; hi = [out[i]["ratio_N_T2e_q"][2] - med[j] for j, i in enumerate(ids)]
col = ["C0" if out[i]["qualifies_10y"] else "0.6" for i in ids]
ax[0].barh(ids, med, xerr=[lo, hi], color=col); ax[0].axvline(1, color="k", lw=0.8); ax[0].set_xscale("log")
ax[0].set_xlabel("N_case / N_T2e (median, 5-95%)  blue = kept >=10 y"); ax[0].set_title("107: sanction/blockade cases vs T2e threshold", fontsize=8)
names = list(runs); ax[1].barh(names, [runs[n] - 1 for n in names], color="C1"); ax[1].tick_params(axis="y", labelsize=7)
ax[1].set_xlabel("F_heavy - 1 (A_Ord/U3, bridge version)"); ax[1].set_title("m from cases; L with Myford regrind limits", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_cannibalization.png"), dpi=120)
R["runtime_s"] = round(time.time() - t0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k: R[k] for k in R if k != "runtime_s"}, indent=1, ensure_ascii=False, default=float))
