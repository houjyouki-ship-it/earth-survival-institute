"""研究105：外から道具・部品が入った小さな孤立集団の事例で、廃墟の品が段の閾値を下げる倍率 m と寿命 L を絞る（open_problems 252）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md（23:43 JST）。
104 の sim.py を「# ---------- 仮説」の前まで exec（057→…→038 の乱数と 104 の入力をそのまま使う）。新しい乱数は seed 20261105。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P104 = os.path.abspath(os.path.join(HERE, "..", "104-ruin-glass-thresholds", "sim.py"))
src = open(P104, encoding="utf8").read(); cut = src.index("# ---------- 仮説")
G = {"__file__": P104, "__name__": "sim104"}
exec(compile(src[:cut], P104, "exec"), G)
R104 = json.load(open(os.path.join(os.path.dirname(P104), "results.json"), encoding="utf8"))
K = "A_Ord/U3"
# 再現：104 の runs と差 0
diff = max(abs(G["R"]["runs"][t][K]["F_heavy"] - R104["runs"][t][K]["F_heavy"]) for t in R104["runs"])
R = {"study": "105-imported-tools-m", "seed": 20261105, "check_104_runs_reproduced_maxdiff": diff}
assert diff < 1e-12
evaluate, N_Fe, N_lens, N_T2e, L_tool_bulk, L_lens, ok_n, Nn = (G[k] for k in ("evaluate", "N_Fe", "N_lens", "N_T2e", "L_tool_bulk", "L_lens", "ok_n", "N"))
g = np.random.default_rng(20261105)
def lu(lo, hi, n=Nn): return np.exp(g.uniform(np.log(lo), np.log(hi), n))

# ---------- 事例：人数と段の閾値の比 ----------
CASES = json.load(open(os.path.join(HERE, "sources", "cases.json"), encoding="utf8"))["cases"]
TH = {"N_Fe": N_Fe, "N_lens": N_lens, "N_T2e": N_T2e}
cases_out = {}
for c in CASES:
    lo = c.get("N", c.get("N_lo")); hi = c.get("N", c.get("N_hi"))
    Nc = np.exp(g.uniform(np.log(lo), np.log(hi), Nn)) if lo != hi else np.full(Nn, float(lo))
    cases_out[c["id"]] = {k: [float(np.quantile(Nc / v, p)) for p in (0.05, 0.5, 0.95)] for k, v in TH.items()}
    cases_out[c["id"]].update({"N": [lo, hi], "resupply": c["resupply"], "tier_made": c["tier_made"],
                               "years": c.get("years_no_resupply", c.get("years")), "censored": c["censored"]})
R["cases_ratio_N_over_threshold_q05_50_95"] = cases_out
R["max_tier_with_local_production"] = "iron (re-forging scrap: grafton N=5; smelting imported bog iron: greenland N=2-3k). No case at lens or T2e tier."
L_case = [c.get("years_no_resupply") for c in CASES if not c["resupply"]]
R["no_resupply_years"] = L_case

# ---------- 走査：m と寿命 L（一定）の効き ----------
Ls = [10, 18, 50, 100, 200, 500, 1000, 2000, 5000, np.inf]
sweep = {}
for m in (0.1, 0.3, 0.7):
    sweep[str(m)] = []
    for L in Ls:
        o = evaluate(f"sweep_m{m}_L{L}", m_l=m, L_l=L, m_t=m, L_t=L)
        sweep[str(m)].append(o[K]["F_heavy"])
R["sweep_F_heavy_vs_L"] = {"L": [float(x) for x in Ls], "F": sweep}

# ---------- 事例に合わせた版 ----------
# S1：レンズ・T2e の段に m_prod の証拠なし → logU(0.7,1)、寿命は 104 と同じ（H117d）
m_l1 = lu(0.7, 1.0); m_t1 = lu(0.7, 1.0)
s1 = evaluate("S1_mprod_0.7to1_bulk", m_l=m_l1, L_l=L_lens, m_t=m_t1, L_t=L_tool_bulk, ok=ok_n)
s1i = evaluate("S1_mprod_0.7to1_inf", m_l=m_l1, m_t=m_t1)
# S2：使うだけ（m_use はごく小さい）を、補給なしの事例の年数 L（1.5〜18 年、logU）だけ効かせる（H117e）
L_use = lu(1.5, 18.0)
m_use = np.minimum(27.0 / N_T2e, 1.0)      # Pitcairn の人数 / N_T2e（使うだけなら人数の制約はほぼない、という寛大な置き方）
s2 = evaluate("S2_use_only_caseL", m_l=np.minimum(27.0 / N_lens, 1.0), L_l=L_use, m_t=m_use, L_t=L_use)
s2b = evaluate("S2b_use_only_L100", m_l=np.minimum(27.0 / N_lens, 1.0), L_l=100.0, m_t=m_use, L_t=100.0)   # 登録外の感度
# S3：事例の上限を T2e の段まで外挿（Greenland の比を全段に、登録外の楽観の上限）
mg = np.minimum(np.exp(g.uniform(np.log(2000), np.log(3000), Nn)) / N_Fe, 1.0)
s3 = evaluate("S3_greenland_ratio_all_tiers_bulk", m_l=mg, L_l=L_lens, m_t=mg, L_t=L_tool_bulk, ok=ok_n)
R["greenland_ratio_q"] = [float(np.quantile(mg, p)) for p in (0.05, 0.5, 0.95)]


# ---------- 登録外：橋渡しの版（品は寿命 L の間だけ段を与え、L までに自前の閾値 N に届かなければ効かない） ----------
# 104 の式は wait(m·N) <= L なら段を恒久に得たとみなす。品が尽きた後も段が残るには、L までに人口が元の閾値に届く必要がある。
_orig = G["T2s_mod"]; FIXED, B, parts, Wc = G["FIXED"], G["B"], G["parts"], G["W"]
def T2s_bridge(k, m_l=None, L_l=np.inf, m_t=None, L_t=np.inf, ok=True):
    if k in FIXED: return B[k]["T2s"]
    st, n, wait, d01e, tFe = parts(k)
    wl = wait(N_lens); wt = wait(N_T2e)
    if m_l is not None:
        wl2 = wait(np.maximum(m_l * N_lens, N_Fe)); wl = np.where(ok & (wl <= L_l), wl2, wl)
    if m_t is not None:
        wt2 = wait(m_t * N_T2e); wt = np.where(ok & (wt <= L_t), wt2, wt)
    t1 = np.where(st >= 5, np.maximum(tFe + d01e, wl), 0.0)
    t2 = np.where(st >= 4, np.maximum(t1 + d12b, wt), 0.0)
    return np.select([st == 2, st == 3, st >= 4], [Wc, np.zeros(Nn), t2], default=np.zeros(Nn))
d12b = G["d12"]
G["T2s_mod"] = T2s_bridge
bridge = {}
for tag, kw in {"B_m0.3_inf": dict(m_l=0.3, m_t=0.3), "B_m0.3_bulk": dict(m_l=0.3, L_l=L_lens, m_t=0.3, L_t=L_tool_bulk, ok=ok_n),
                "B_m0.1_bulk": dict(m_l=0.1, L_l=L_lens, m_t=0.1, L_t=L_tool_bulk, ok=ok_n),
                "B_use_caseL": dict(m_l=np.minimum(27.0 / N_lens, 1.0), L_l=L_use, m_t=m_use, L_t=L_use),
                "B_use_L100": dict(m_l=np.minimum(27.0 / N_lens, 1.0), L_l=100.0, m_t=m_use, L_t=100.0),
                "B_use_bulk": dict(m_l=np.minimum(27.0 / N_lens, 1.0), L_l=L_lens, m_t=m_use, L_t=L_tool_bulk, ok=ok_n),
                "B_greenland_bulk": dict(m_l=mg, L_l=L_lens, m_t=mg, L_t=L_tool_bulk, ok=ok_n)}.items():
    bridge[tag] = evaluate(tag, **kw)[K]["F_heavy"]
G["T2s_mod"] = _orig
R["bridge_F_heavy"] = bridge
# 104 の式の S2 の増分のうち、段を品の寿命の後まで持ち越した標本の割合（重い崩壊 st>=4）
share = []
for k in G["COMPS"]:
    if k in FIXED: continue
    st, n, wait, *_ = parts(k); m = st >= 4
    if not m.any(): continue
    share.append(((wait(m_use * N_T2e) <= L_use) & (wait(N_T2e) > L_use))[m])
R["S2_share_samples_kept_stage_after_tools_died"] = float(np.concatenate(share).mean())

RU = G["R"]["runs"]
R["runs"] = {t: RU[t] for t in RU if t.startswith(("S", "sweep", "B_"))}
# ---------- 仮説 ----------
H = {}
H["H117a"] = {"class": "confirmatory", "rule": "a no-resupply case (pitcairn/grafton) with documented metal-tool use >= 10 years",
              "value": {"grafton_years": 1.5, "pitcairn_years": 18.1, "pitcairn_tool_use_documented": False}, "supported": None,
              "note": "Grafton は道具の記述ありだが 1.5 年、Pitcairn は 18 年だが開いた頁に道具の使用の記述なし（Bible で読み書きを教えた記述のみ）→検定できず"}
H["H117b"] = {"class": "confirmatory", "rule": "no source says the group came to make the import-dependent tier itself", "supported": False,
              "note": "Grafton の 5 人が難破船の金属でふいごを作り道具を鍛え直した記述がある（鉄の段の再加工）。レンズ・T2e の段の自前の生産の記述はどの事例にもない"}
mx = max(cases_out[c]["N_T2e"][1] for c in cases_out)
H["H117c"] = {"class": "prediction", "rule": "all case m_use medians (N/N_tier) < 0.05 (tier=N_T2e and N_lens)",
              "value": {c: (cases_out[c]["N_lens"][1], cases_out[c]["N_T2e"][1]) for c in cases_out},
              "supported": bool(all(max(cases_out[c]["N_lens"][1], cases_out[c]["N_T2e"][1]) < 0.05 for c in cases_out))}
H["H117d"] = {"class": "prediction", "rule": "S1 (m_prod logU 0.7-1, 104 lifetimes) F_heavy-1 < 0.10", "value": s1[K]["F_heavy"],
              "supported": bool(s1[K]["F_heavy"] - 1 < 0.10)}
H["H117e"] = {"class": "prediction", "rule": "S2 use-only with case L (1.5-18 y) F_heavy-1 < 0.02", "value": s2[K]["F_heavy"],
              "supported": bool(s2[K]["F_heavy"] - 1 < 0.02)}
R["hypotheses"] = H

# ---------- 図 ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
xL = [1e5 if not np.isfinite(L) else L for L in Ls]
for m in sweep: ax[0].plot(xL, sweep[m], "o-", label=f"m={m} (both thresholds)")
for L in L_case: ax[0].axvline(L, color="grey", ls=":", lw=1)
ax[0].axvspan(400, 450, color="C3", alpha=0.2, label="Greenland (with resupply)")
ax[0].set_xscale("log"); ax[0].set_xlabel("usable life L of ruin tools (years; 1e5 = infinite)"); ax[0].set_ylabel("F_heavy (A_Ord/U3)")
ax[0].legend(fontsize=7); ax[0].set_title("105: effect needs L of centuries; observed no-resupply L <= 18 y (dotted)", fontsize=8)
ids = list(cases_out); 
for j, tier in enumerate(("N_Fe", "N_lens", "N_T2e")):
    ax[1].scatter([i + 0.2 * (j - 1) for i in range(len(ids))], [cases_out[c][tier][1] for c in ids], label=tier)
ax[1].set_yscale("log"); ax[1].set_xticks(range(len(ids))); ax[1].set_xticklabels(ids, fontsize=7, rotation=15)
ax[1].axhline(0.3, color="k", ls="--", lw=0.8); ax[1].axhline(0.1, color="k", ls=":", lw=0.8)
ax[1].set_ylabel("median N_case / N_tier"); ax[1].legend(fontsize=7); ax[1].set_title("case population vs stage thresholds (local production only at iron tier)", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_imported_tools.png"), dpi=120)

R["runtime_s"] = round(time.time() - t0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k: R[k] for k in R if k != "runs"}, indent=1, ensure_ascii=False, default=float))
for t, r in R["runs"].items():
    print(f"{t:36s} F_heavy {r[K]['F_heavy']:.4f} F_total {r[K]['F_total']:.4f} T2med {r['heavy_st4to6']['T2_median_rel_change']:+.3f}")
