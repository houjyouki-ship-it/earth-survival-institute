"""研究113：閾値を下回った小さな群が T2e の段の精密な道具を作り直せるか（open_problems 267）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md（2026-10-24 15:37 JST）。
112 の sim.py を「# ---------- 仮説」の前まで exec（112 の表を差 0 で再現、assert）。新しい乱数は seed 20261113。
T2e の精密な道具の条件：群の人口が N_tool に達した時点で作り直せる（R）。それまでは廃墟の道具（Myford の L）でつなぐ。
  t_tool = max(0, t_below − ln(N_T2e/N_tool)/r_pop)。条件 min(t_below, t_tool) ≤ L_myf（N_tool ≥ N_T2e なら 112 の X と同じ）。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P112 = os.path.abspath(os.path.join(HERE, "..", "112-nonrebuildable-capital", "sim.py"))
src = open(P112, encoding="utf8").read(); cut = src.index("\n# ---------- 仮説")
G2 = {"__file__": P112, "__name__": "sim112"}
exec(compile(src[:cut], P112, "exec"), G2)
R112 = json.load(open(os.path.join(os.path.dirname(P112), "results.json"), encoding="utf8"))
d112 = max(abs(G2["TAB"][t]["F_heavy"] - R112["table"][t]["F_heavy"]) for t in R112["table"])
assert d112 < 1e-12, d112
run, KAPS, V, N, N_T2e, r_pop, L_myf, L_rew, L_bulk = (G2[k] for k in ("run", "KAPS", "V", "N", "N_T2e", "r_pop", "L_myf", "L_rew", "L_bulk"))
G2["OUT"].pop("runs", None)
g = np.random.default_rng(20261113)
q3 = lambda v: [float(np.quantile(v, p)) for p in (0.05, 0.5, 0.95)]
OUT = {"study": "113-precision-tool-rebuild", "seed": 20261113, "check_112_table_reproduced_maxdiff": d112,
       "N_T2e_q": q3(N_T2e), "r_pop_q": q3(np.broadcast_to(r_pop, (N,)))}

# ---------- 資料の事例（sources/ の wikitext から抜粋。社会の人口は同じく sources/ の人口の頁） ----------
CASES = [
 {"case": "Hindley の目盛り機・ねじ切り盤", "year": 1739, "place": "England (York)", "society_pop": 5.2e6, "pop_note": "England 1700 5.2M (Demographics of England)", "prereq": "時計の歯切り盤から（Dividing engine）"},
 {"case": "Wilkinson の中ぐり盤", "year": 1774, "place": "England", "society_pop": 5.2e6, "pop_note": "下限として 1700 の値", "prereq": "Watt は数年失敗（Machine tool）"},
 {"case": "Ramsden のねじ切り盤→目盛り機", "year": 1775, "place": "London", "society_pop": 5.2e6, "pop_note": "下限として 1700 の値", "prereq": "最初の盤でより正確な盤を作る（Screw-cutting lathe）。Dollond の家の工房と徒弟"},
 {"case": "Senot のねじ切り盤", "year": 1795, "place": "France", "society_pop": 24.6e6, "pop_note": "France 1740 24.6M（下限）", "prereq": "不明"},
 {"case": "D. Wilkinson の往復台", "year": 1798, "place": "Rhode Island, US", "society_pop": 5.308e6, "pop_note": "1800 US census 5,308,483", "prereq": "Britain の機械と交易あり"},
 {"case": "Maudslay のねじ切り盤・平面の三枚・Lord Chancellor", "year": 1800, "place": "London", "society_pop": 8.29e6, "pop_note": "England 1801 census 8,287,907", "prereq": "Bramah の工房で 8 年、1810 年に 80 人"},
 {"case": "Whitworth の三面の削り（1830年代）・1840 年の報告", "year": 1840, "place": "Manchester", "society_pop": 10.5e6, "pop_note": "GB 1801 10.5M（下限）", "prereq": "Maudslay の工房で徒弟（1825〜）"},
]
OUT["cases"] = CASES
soc = np.array([c["society_pop"] for c in CASES]); OUT["case_society_pop_min_median"] = [float(soc.min()), float(np.median(soc))]

# ---------- T2e の条件 ----------
s_metal = g.random(N) < 0.3     # 112 と同じ p_metal（新しい乱数で引き直す）
def t2e_tool(N_tool, L_tool=None):
    L_tool = L_myf if L_tool is None else L_tool
    gap = np.log(N_T2e / N_tool) / r_pop                     # N_tool に達してから N_T2e までの年数（負なら N_tool>N_T2e）
    def f(tb):
        t_tool = np.maximum(0.0, tb - gap)
        ok = np.where(s_metal, True, tb <= L_bulk)
        return ok & (np.minimum(tb, t_tool) <= L_tool)
    return f
fe_d, le_d = V["data_main"]["Fe"], V["data_main"]["lens"]
VER = {}
def ver(nm, N_tool, L_tool=None): VER[nm] = {"Fe": fe_d, "lens": le_d, "T2e": t2e_tool(N_tool, L_tool)}
ver("base_X_inf", np.inf)                     # 112 の data_main と同じ構造（金属の乱数だけ違う）
ver("hist_min_5.2e6", 5.2e6)                  # 事例の社会の人口の最小（England 1700）
ver("hist_median_5.3e6", float(np.median(soc)))
ver("hist_England1801_8.3e6", 8.29e6)
ver("hist_div10_5.2e5", 5.2e5)                # 手引きがあれば 1 桁小さくてよいとする楽観【推測】
ver("workshop_logu_1e3_1e5", np.exp(g.uniform(np.log(1e3), np.log(1e5), N)))   # 工房＋鋳物＋鍛冶の規模【推測】
ver("workshop_1e4", 1e4)
ver("tool_R", 1.0)                            # 112 の T2e の道具 R に相当（N_tool=1）
TAB = {}
for vn, Lk in VER.items():
    for kn in KAPS:
        TAB[f"{kn}|{vn}"] = run(f"{kn}|{vn}", KAPS[kn], Lk, store=False)
OUT["table"] = TAB
OUT["check_base_vs_112_data_main"] = {kn: TAB[f"{kn}|base_X_inf"]["F_heavy"] - R112["table"][f"{kn}|data_main"]["F_heavy"] for kn in KAPS}
OUT["check_toolR_vs_112_T2e_tool_R"] = {kn: TAB[f"{kn}|tool_R"]["F_heavy"] - R112["table"][f"{kn}|data_T2e_tool_R"]["F_heavy"] for kn in KAPS}
# 群のうち事例の最小の社会の大きさを越える N_T2e の割合
OUT["share_N_T2e_below_hist_min"] = float((N_T2e < 5.2e6).mean())

# ---------- 比 c = N_tool/N_T2e の走査（二重の数えの判定） ----------
CS = np.logspace(-4, 0, 17)
OUT["sweep_ratio"] = {kn: [{"c": float(c), **run(f"c_{kn}_{c:.4g}", KAPS[kn], {"Fe": fe_d, "lens": le_d, "T2e": t2e_tool(c * N_T2e)}, store=False)} for c in CS]
                      for kn in ("slow_Tasmania", "trigger_mix")}
def crit(kn, thr):
    fh = np.array([s["F_heavy"] for s in OUT["sweep_ratio"][kn]]); i = np.where(fh >= thr)[0]
    return float(CS[i.max()]) if i.size else None
OUT["ratio_crit"] = {kn: {"F>=1.1": crit(kn, 1.1), "F>=1.5": crit(kn, 1.5)} for kn in OUT["sweep_ratio"]}

# ---------- 仮説 ----------
H = {}
H["H125a"] = {"class": "confirmatory", "supported": bool(soc.min() >= 1e6), "value": f"7 件すべて人口 {soc.min():.2g} 以上の社会（England/France/US）。小さな孤立した群の例 0 件"}
H["H125b"] = {"class": "confirmatory", "supported": True, "value": "Surface plate: 'surface plates do not derive their precision from more-precise standards ... three approximately flat surfaces are progressively refined'"}
H["H125c"] = {"class": "confirmatory", "supported": None, "value": "Ramsden/Dividing engine の頁に製作の年数の記述なし（Chaulnes の 5 年後、1777 年に記述を出版のみ）→検定できず"}
H["H125d"] = {"class": "confirmatory", "supported": True, "value": "Maudslay: Bramah の工房で 8 年（管理者）、1810 年に 80 人を雇用。Ramsden は最初の盤でより正確な盤を作った"}
v = TAB["slow_Tasmania|hist_min_5.2e6"]["F_heavy"]; b = TAB["slow_Tasmania|base_X_inf"]["F_heavy"]
H["H125e"] = {"class": "prediction", "value": v, "base": b, "rule": "F - base < 0.1", "supported": bool(v - b < 0.1)}
v = TAB["slow_Tasmania|hist_div10_5.2e5"]["F_heavy"]
H["H125f"] = {"class": "prediction", "value": v, "rule": "< 1.73", "supported": bool(v < 1.732)}
H["H125g"] = {"class": "descriptive", "supported": None, "value": "cases 表"}
OUT["hypotheses"] = H; OUT["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
vns = list(VER); x = np.arange(len(vns))
for j, kn in enumerate(KAPS):
    axs[0].bar(x + (j - 1) * 0.27, [TAB[f"{kn}|{v}"]["F_heavy"] for v in vns], 0.27, label=kn)
axs[0].set_xticks(x); axs[0].set_xticklabels(vns, rotation=45, ha="right", fontsize=7); axs[0].axhline(1.1, color="k", lw=0.5)
axs[0].set_ylabel("F_heavy (A_Ord/U3)"); axs[0].set_title("113: T2e tools rebuildable once group reaches N_tool", fontsize=9); axs[0].legend(fontsize=7)
for kn, c in (("slow_Tasmania", "C3"), ("trigger_mix", "C0")):
    axs[1].semilogx(CS, [s["F_heavy"] for s in OUT["sweep_ratio"][kn]], "o-", color=c, label=kn)
axs[1].axhline(1.1, color="k", lw=0.5); axs[1].set_xlabel("c = N_tool / N_T2e"); axs[1].set_ylabel("F_heavy"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_tool_rebuild.png"), dpi=120); plt.close(fig)
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("check_112", d112, "base-112", OUT["check_base_vs_112_data_main"], "toolR-112", OUT["check_toolR_vs_112_T2e_tool_R"])
print("N_T2e_q", [f"{v:.3g}" for v in OUT["N_T2e_q"]], "r_pop_q", OUT["r_pop_q"], "share N_T2e<5.2e6", OUT["share_N_T2e_below_hist_min"])
for t, o in TAB.items(): print(t, "F_heavy", round(o["F_heavy"], 4), "drop", round(o["U3_heavy_rel_drop"], 3))
for kn, s in OUT["sweep_ratio"].items(): print("ratio", kn, [(f"{a['c']:.3g}", round(a["F_heavy"], 3)) for a in s])
print("ratio_crit", OUT["ratio_crit"])
for k, hh in H.items(): print(k, hh["class"], hh["supported"], str(hh["value"])[:110])
print("runtime", round(OUT["runtime_s"], 1))
