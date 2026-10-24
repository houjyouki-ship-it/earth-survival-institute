"""研究111：T_loss を引き金から数える規則と、技能と資本を分けた模型（open_problems 263＋261）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md（2026-10-24 11:36 JST）。
109 の sim.py を「# ---------- 仮説」の前まで exec（108→104→…の乱数と 109 の全版を再現、assert）。新しい乱数は seed 20261111。
段 k を保つ ⇔ z_min ≥ ln N_k（108 の技能）かつ t_below,k = ln(N_k/n)/r ≤ L（資本の寿命）。κ→∞ か L→0 で二値の模型。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P109 = os.path.abspath(os.path.join(HERE, "..", "109-kappa-historical", "sim.py"))
src = open(P109, encoding="utf8").read(); cut = src.index("# ---------- 仮説")
G9 = {"__file__": P109, "__name__": "sim109"}
exec(compile(src[:cut], P109, "exec"), G9)
R109 = json.load(open(os.path.join(os.path.dirname(P109), "results.json"), encoding="utf8"))
d109 = max(abs(G9["res"][t]["F_heavy"] - R109["runs"][t]["F_heavy"]) for t in R109["runs"])
assert d109 < 1e-12, d109
G = G9["G"]   # 108 の名前空間
N, B, SRC, COMPS, FIXED, V_m, agg, q_TC, v0 = (G[k] for k in ("N", "B", "SRC", "COMPS", "FIXED", "V_m", "agg", "q_TC", "v0"))
N_Fe, N_lens, N_T2e, r_pop, CONS, zmin, T2_from = (G[k] for k in ("N_Fe", "N_lens", "N_T2e", "r_pop", "CONS", "zmin", "T2_from"))
K = "A_Ord/U3"; k_hist = G["k_hist"]; kf = G9["kf"]; CASES9 = G9["CASES"]
g = np.random.default_rng(20261111)
q3 = lambda v: [float(np.quantile(v, p)) for p in (0.05, 0.5, 0.95)]
OUT = {"study": "111-trigger-skill-capital", "seed": 20261111, "check_109_runs_reproduced_maxdiff": d109}

THR = [(3, N_T2e), (4, N_lens), (5, N_Fe)]
def new_st(k, kap, L):
    st = B[k]["st"]; n = np.maximum(SRC[k]["sv"]["nmax"], 1.0)
    zm, _ = zmin(n, r_pop, kap, G["POP_W"])
    lev = np.full(N, 6)
    for code, Nk in reversed(THR):          # 低い段から：保てば上書き
        tb = np.maximum(0.0, np.log(Nk / n) / r_pop)
        keep = (zm >= np.log(Nk)) & (tb <= L)
        lev = np.where(keep, code, lev)
    # 段は入れ子：上の段を保つなら下も保つ（N_T2e ≥ N_lens ≥ N_Fe なので tb も z の条件も単調）
    return np.where(CONS[k], np.minimum(st, lev), st)

def run(tag, kap, L):
    D = {}; sh = {}
    for k in COMPS:
        if k in FIXED: T2n = B[k]["T2s"]
        else:
            s2 = new_st(k, kap, L); T2n = T2_from(k, s2)
            sh[k] = (float(np.isin(B[k]["st"], (4, 5, 6)).mean()), float(np.isin(s2, (4, 5, 6)).mean()))
        D[k] = V_m(k, q_TC, True, T2n) - V_m(k, 0.3, False, T2n)
    v1 = agg({"x": D}, anchors=("A_Ord", "A_XPT_S"))
    b_ = v0["A_Ord"]["U3"]["x"]; n_ = v1["A_Ord"]["U3"]["x"]
    o = {"F_heavy": n_["heavy"] / b_["heavy"], "F_total": n_["total"] / b_["total"], "U3_heavy_rel_drop": 1 - sh["U3"][1] / sh["U3"][0]}
    OUT.setdefault("runs", {})[tag] = o
    return o

# ---------- 検算 ----------
INF = np.full(N, 1e12)
c1 = run("check_L_inf_hist", k_hist, INF)["F_heavy"]
c108 = G["R"]["runs"]["hist"][K]["F_heavy"]
OUT["check_L_inf_equals_108_hist"] = abs(c1 - c108); assert abs(c1 - c108) < 1e-12, (c1, c108)
OUT["check_L0_binary"] = abs(run("check_L0_hist", k_hist, np.zeros(N))["F_heavy"] - 1.0); assert OUT["check_L0_binary"] < 1e-9

# ---------- 資本の寿命 L の版（分位から対数正規で近似【推測】） ----------
def lnorm(q05, q50, q95):
    s = (np.log(q95) - np.log(q05)) / (2 * 1.6449); return np.exp(g.normal(np.log(q50), s, N))
LV = {"precision_104": lnorm(1.97, 19.7, 411.5), "myford_107": lnorm(6.92, 69.5, 1450.3),
      "mixed_106": lnorm(2.32, 116.1, 10695), "rework_106": lnorm(113.2, 1373.9, 19393), "inf": INF}
OUT["L_versions_q"] = {k: q3(v) for k, v in LV.items() if k != "inf"}

# ---------- 引き金の規則での事例（登録の規則 1） ----------
SLOW = 0.0063
TRIG = {"Inughuit_boats": "i", "Tasmania_108": "i", "Britain_coin": "i", "Britain_wheel_pottery": "i", "Greece_LinearB": "i",
        "Greece_monumental_stone": "i", "Tonga_pottery": "ii_undated", "EasterIsland_seaworthy_vessels": "iii_capital",
        "England_mortared_stone_111": "ii_or_capital", "Britain_primary_glass_111": "never_local"}
# 110 の κ の中央値（results.json から）
R110 = json.load(open(os.path.join(HERE, "..", "110-isolate-skill-w", "results.json"), encoding="utf8"))
kap_med = {c: R110["cases"][c]["kappa_q"][1] for c in R110["cases"]}
counted = [c for c, t in TRIG.items() if t == "i"]
slow = sum(kap_med[c] <= SLOW for c in counted); fast = len(counted) - slow
wpost = g.beta(0.5 + slow, 0.5 + fast, N)
OUT["trigger_rule"] = {"types": TRIG, "counted_for_kappa": counted, "slow": int(slow), "fast": int(fast), "w_post_q": q3(wpost)}
mixk = np.where(g.random(N) < wpost, k_hist, kf)

# ---------- 主な表：κ × L ----------
KAPS = {"slow_Tasmania": k_hist, "trigger_mix": mixk, "fast_state": kf}
TAB = {}
for kn, kv in KAPS.items():
    for ln, lv in LV.items():
        TAB[f"{kn}|{ln}"] = run(f"{kn}|{ln}", kv, lv)
OUT["table"] = TAB
# L 一定の走査（遅い κ と混合）
LS = np.logspace(0, 4, 21); SW = {}
for kn in ("slow_Tasmania", "trigger_mix"):
    SW[kn] = [{"L": float(l), **run(f"sweep_{kn}_{l:.0f}", KAPS[kn], np.full(N, l))} for l in LS]
OUT["sweep"] = SW
def crit(kn, thr=1.1):
    fh = np.array([s["F_heavy"] for s in SW[kn]]); i = np.where(fh >= thr)[0]
    return float(LS[i.min()]) if i.size else None
OUT["L_crit_F1.1"] = {kn: crit(kn) for kn in SW}
# t_below（重い崩壊の標本、U3、T2e の段）の分布：資本が縛る理由の記述
st = B["U3"]["st"]; n = np.maximum(SRC["U3"]["sv"]["nmax"], 1.0); h = np.isin(st, (4, 5, 6)) & CONS["U3"]
OUT["t_below_U3_heavy_q"] = {nm: q3(np.maximum(0, np.log(Nk / n) / r_pop)[h]) for nm, Nk in (("T2e", N_T2e), ("lens", N_lens), ("Fe", N_Fe))}

# ---------- 仮説 ----------
H = {}
H["H123a"] = {"class": "confirmatory", "supported": True, "value": "Anglo-Saxon glass: 'very little evidence for glass making from the raw materials in Roman Britain and even less evidence in Anglo-Saxon Britain'; cullet recycling; re-use of Roman opaque glass",
              "note": "primary glass making was never local even in Roman Britain, so this is not a post-collapse loss of the stage"}
H["H123b"] = {"class": "confirmatory", "supported": True, "value": "masonry churches from 597 (Augustine 'probably imported workmen from Frankish Gaul'); Wearmouth glaziers from Gaul 675; gap c.410-597 = ~187 y",
              "note": "'probably'; Shapland: return to timber was a cultural choice, i.e. demand (ii), not skill decay"}
v = TAB["slow_Tasmania|myford_107"]["F_heavy"]; H["H123c"] = {"class": "prediction", "value": v, "rule": "<=1.2", "supported": bool(v <= 1.2)}
v = TAB["trigger_mix|myford_107"]["F_heavy"]; H["H123d"] = {"class": "prediction", "value": v, "rule": "<1.3", "supported": bool(v < 1.3)}
v = OUT["L_crit_F1.1"]["slow_Tasmania"]; H["H123e"] = {"class": "prediction", "value": v, "rule": ">200 y", "supported": bool(v is not None and v > 200)}
H["H123f"] = {"class": "descriptive", "value": TRIG, "supported": None}
OUT["hypotheses"] = H; OUT["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
for kn, c in (("slow_Tasmania", "C3"), ("trigger_mix", "C0")):
    axs[0].semilogx(LS, [s["F_heavy"] for s in SW[kn]], "o-", color=c, label=f"{kn}, constant L")
    axs[0].axhline(TAB[f"{kn}|inf"]["F_heavy"], color=c, ls=":", lw=0.8)
for ln, lv in LV.items():
    if ln != "inf": axs[0].axvline(np.median(lv), color="gray", lw=0.5); axs[0].text(np.median(lv), 1.02, ln, rotation=90, fontsize=6)
axs[0].axhline(1.1, color="k", lw=0.5); axs[0].set_xlabel("capital lifetime L (years)"); axs[0].set_ylabel("F_heavy (A_Ord/U3)")
axs[0].set_title("111: stage kept only if skill AND capital survive the trough", fontsize=9); axs[0].legend(fontsize=7)
lns = list(LV); x = np.arange(len(lns))
for j, kn in enumerate(KAPS):
    axs[1].bar(x + (j - 1) * 0.27, [TAB[f"{kn}|{l}"]["F_heavy"] for l in lns], 0.27, label=kn)
axs[1].set_xticks(x); axs[1].set_xticklabels(lns, fontsize=7); axs[1].axhline(1.1, color="k", lw=0.5); axs[1].legend(fontsize=7); axs[1].set_ylabel("F_heavy")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_skill_capital.png"), dpi=120); plt.close(fig)
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("check_109", d109, "check_Linf", OUT["check_L_inf_equals_108_hist"], "check_L0", OUT["check_L0_binary"])
print("trigger", OUT["trigger_rule"]["slow"], OUT["trigger_rule"]["fast"], [round(x, 3) for x in OUT["trigger_rule"]["w_post_q"]])
print("L_q", {k: [round(x, 1) for x in v] for k, v in OUT["L_versions_q"].items()})
print("t_below", {k: [round(x, 0) for x in v] for k, v in OUT["t_below_U3_heavy_q"].items()})
for t, o in TAB.items(): print(t, "F_heavy", round(o["F_heavy"], 4), "drop", round(o["U3_heavy_rel_drop"], 3))
for kn in SW: print("sweep", kn, [(round(s["L"]), round(s["F_heavy"], 3)) for s in SW[kn]])
print("L_crit", OUT["L_crit_F1.1"])
for k, h in H.items(): print(k, h["class"], h["supported"], str(h["value"])[:120])
print("runtime", round(OUT["runtime_s"], 1))
