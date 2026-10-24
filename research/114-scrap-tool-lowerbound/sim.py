"""研究114：手順の知識を持つ小さな集団が廃墟の材料で精密な道具を作り直せる場合の N_tool の下限（open_problems 270）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md（2026-10-24 17:35 JST）。
113 の sim.py を「# ---------- 仮説」の前まで exec（113 の表を差 0 で再現、assert）。新しい乱数は seed 20261114。
T2e の道具の条件（2 経路）【独自仮説】：
  経路2（113 の X）：t_below ≤ L_myf。
  経路1（廃墟の材料）：t_small = max(0, t_below − ln(N_T2e/N_small)/r_pop) ≤ max(L_scrap, L_myf) かつ 精度が足りる（確率 p_prec）。
  L_scrap=0 なら 113 の t2e_tool(N_small) と同じ（assert）。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P113 = os.path.abspath(os.path.join(HERE, "..", "113-precision-tool-rebuild", "sim.py"))
src = open(P113, encoding="utf8").read(); cut = src.index("\n# ---------- 仮説")
G3 = {"__file__": P113, "__name__": "sim113"}
exec(compile(src[:cut], P113, "exec"), G3)
R113 = json.load(open(os.path.join(os.path.dirname(P113), "results.json"), encoding="utf8"))
d113 = max(abs(G3["TAB"][t]["F_heavy"] - R113["table"][t]["F_heavy"]) for t in R113["table"])
assert d113 < 1e-12, d113
run, KAPS, N, N_T2e, r_pop, L_myf, L_bulk, s_metal, fe_d, le_d, t2e_tool = (G3[k] for k in
    ("run", "KAPS", "N", "N_T2e", "r_pop", "L_myf", "L_bulk", "s_metal", "fe_d", "le_d", "t2e_tool"))
g = np.random.default_rng(20261114)
OUT = {"study": "114-scrap-tool-lowerbound", "seed": 20261114, "check_113_table_reproduced_maxdiff": d113}

# ---------- 資料の事例（sources/ の wikitext） ----------
CASES = [
 {"case": "Gingery の Build Your Own Metal Working Shop From Scrap", "people": 1, "people_note": "'the hobbyist'（人数の数値なし、1 人の趣味の人を想定）", "years": None,
  "ruin_material": True, "materials": "自動車の廃部品の Al・Zn 合金、木炭の炉、生砂の鋳型。旋盤→形削り盤→フライス盤→ボール盤→割り出し台", "precision": "数値なし"},
 {"case": "Johansson のブロックゲージ（1896）", "people": 2, "people_note": "本人と妻 Margareta（試作）", "years": None,
  "ruin_material": True, "materials": "Singer のミシンを研削・ラップ盤に改造、鋼。のちに勤め先の工場が時間と資源を提供", "precision": "超精密のブロック（数値は頁の別の節）"},
 {"case": "Open Source Ecology（GVCS）", "people": None, "people_note": "中心の人数の記述なし。家は 50 人で 5 日", "years": 10,
  "ruin_material": True, "materials": "Kickstarter 63,573 ドルなどの資金。50 機械のうち 2014 年に 12 を試作、2018 年に 33%。Multimachine（旋盤・フライス・ボール盤）を含む", "precision": "記述なし"},
 {"case": "Colditz Cock（捕虜の滑空機）", "people": 14, "people_note": "Goldfinch・Best と 12 人の助手", "years": None,
  "ruin_material": True, "materials": "寝台の板・床板・電線・寝袋。道具も寝台の板などから手作り。金属の精密な道具ではない", "precision": "該当なし（木工）"},
 {"case": "三面の擦り合わせ（Hand scraper / Surface plate）", "people": None, "people_note": "人数の記述なし", "years": None,
  "ruin_material": None, "materials": "三枚の生の鋳物の平板（応力を抜いたもの）、平らなきさげ、光明丹（engineer's blue）", "precision": "より精密な基準なしに平面を生む"},
]
OUT["cases"] = CASES

# ---------- T2e の 2 経路 ----------
def t2e_two(N_small, L_scrap, p_prec=1.0):
    prec = g.random(N) < p_prec
    gap = np.log(N_T2e / N_small) / r_pop
    def f(tb):
        ok = np.where(s_metal, True, tb <= L_bulk)
        path2 = tb <= L_myf
        t_small = np.maximum(0.0, tb - gap)
        path1 = prec & (t_small <= np.maximum(L_scrap, L_myf))
        return ok & (path2 | path1)
    return f
VER = {
 "check_X": (np.inf, 0.0, 1.0),                 # 113 の base_X_inf と同じ
 "check_1e4_Lscrap0": (1e4, 0.0, 1.0),          # 113 の workshop_1e4 と同じ
 "n10_Lbulk": (10.0, L_bulk, 1.0),              # 事例の人数の桁、廃墟の鉄の部材の寿命
 "n100_Lbulk": (100.0, L_bulk, 1.0),            # 事例の人数＋食料を作る人【推測】
 "n1000_Lbulk": (1000.0, L_bulk, 1.0),          # 鋳物・木炭・鍛冶を含む村【推測】
 "n100_Lbulk_prec0.5": (100.0, L_bulk, 0.5),    # 趣味の水準の盤で T2e の精度に届く群が半分【推測】
 "n100_Lbulk_prec0.2": (100.0, L_bulk, 0.2),
 "n100_L200": (100.0, 200.0, 1.0),              # 使える廃材（砥粒・鋼の棒・Al）が 200 年で尽きる【推測】
}
TAB = {}
for vn, (ns, ls, pp) in VER.items():
    fT = t2e_two(ns, ls, pp)
    for kn in KAPS:
        TAB[f"{kn}|{vn}"] = run(f"{kn}|{vn}", KAPS[kn], {"Fe": fe_d, "lens": le_d, "T2e": fT}, store=False)
OUT["table"] = TAB
OUT["check_X_vs_113_base"] = {kn: TAB[f"{kn}|check_X"]["F_heavy"] - R113["table"][f"{kn}|base_X_inf"]["F_heavy"] for kn in KAPS}
OUT["check_1e4_vs_113_workshop"] = {kn: TAB[f"{kn}|check_1e4_Lscrap0"]["F_heavy"] - R113["table"][f"{kn}|workshop_1e4"]["F_heavy"] for kn in KAPS}
for d in (OUT["check_X_vs_113_base"], OUT["check_1e4_vs_113_workshop"]):
    assert max(abs(v) for v in d.values()) < 1e-12, d
OUT["L_bulk_q"] = [float(np.quantile(L_bulk, p)) for p in (0.05, 0.5, 0.95)]
OUT["L_myf_q"] = [float(np.quantile(L_myf, p)) for p in (0.05, 0.5, 0.95)]

# ---------- 走査：L_scrap（一定）と p_prec ----------
LS = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 5000]
OUT["sweep_Lscrap_n100"] = {kn: [{"L_scrap": l, **run(f"L_{kn}_{l}", KAPS[kn], {"Fe": fe_d, "lens": le_d, "T2e": t2e_two(100.0, float(l))}, store=False)} for l in LS]
                            for kn in ("slow_Tasmania", "trigger_mix")}
PS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
OUT["sweep_pprec_n100_Lbulk"] = {kn: [{"p_prec": p, **run(f"P_{kn}_{p}", KAPS[kn], {"Fe": fe_d, "lens": le_d, "T2e": t2e_two(100.0, L_bulk, p)}, store=False)} for p in PS]
                                 for kn in ("slow_Tasmania", "trigger_mix")}
def crit(seq, key, thr, smallest=True):
    xs = [s[key] for s in seq if s["F_heavy"] >= thr]
    return (min(xs) if smallest else max(xs)) if xs else None
OUT["crit_Lscrap"] = {kn: {"F>=1.1": crit(s, "L_scrap", 1.1), "F>=1.5": crit(s, "L_scrap", 1.5)} for kn, s in OUT["sweep_Lscrap_n100"].items()}
OUT["crit_pprec"] = {kn: {"F>=1.1": crit(s, "p_prec", 1.1), "F>=1.5": crit(s, "p_prec", 1.5)} for kn, s in OUT["sweep_pprec_n100_Lbulk"].items()}

# ---------- 仮説 ----------
H = {}
H["H126a"] = {"class": "confirmatory", "supported": True, "value": "Gingery: 'The hobbyist starts by constructing a small foundry capable of melting silicon-aluminum and zinc alloys from recycled automotive parts ... The lathe is the first machine built'。書名 The Charcoal Foundry。人数は数値なし（'the hobbyist' 単数）"}
H["H126b"] = {"class": "confirmatory", "supported": True, "value": "4/4 の小さな集団（Gingery・Johansson・OSE・Colditz）が工業の社会の材料に頼る。鉱石から外の供給なしの例 0 件"}
H["H126c"] = {"class": "confirmatory", "supported": True, "value": "OSE の頁に中心の人数の記述なし（家は 50 人で 5 日）。多数の工場の分業の記述なし。ただし資金・購入の材料に頼る"}
v = TAB["slow_Tasmania|n10_Lbulk"]["F_heavy"]
H["H126d"] = {"class": "prediction", "value": v, "rule": ">= 1.5", "supported": bool(v >= 1.5)}
c = OUT["crit_Lscrap"]["slow_Tasmania"]["F>=1.1"]
H["H126e"] = {"class": "prediction", "value": c, "rule": "crit L_scrap (F>=1.1) < 500", "supported": bool(c is not None and c < 500)}
H["H126f"] = {"class": "descriptive", "supported": None, "value": "cases 表"}
OUT["hypotheses"] = H; OUT["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
vns = [v for v in VER if not v.startswith("check_1e4")]; x = np.arange(len(vns))
for j, kn in enumerate(KAPS):
    axs[0].bar(x + (j - 1) * 0.27, [TAB[f"{kn}|{v}"]["F_heavy"] for v in vns], 0.27, label=kn)
axs[0].set_xticks(x); axs[0].set_xticklabels(vns, rotation=45, ha="right", fontsize=7); axs[0].axhline(1.1, color="k", lw=0.5)
axs[0].set_ylabel("F_heavy (A_Ord/U3)"); axs[0].set_title("114: T2e tools from ruin scrap by small group", fontsize=9); axs[0].legend(fontsize=7)
for kn, col in (("slow_Tasmania", "C3"), ("trigger_mix", "C0")):
    axs[1].plot([s["L_scrap"] for s in OUT["sweep_Lscrap_n100"][kn]], [s["F_heavy"] for s in OUT["sweep_Lscrap_n100"][kn]], "o-", color=col, label=kn)
    axs[2].plot(PS, [s["F_heavy"] for s in OUT["sweep_pprec_n100_Lbulk"][kn]], "o-", color=col, label=kn)
axs[1].set_xscale("symlog", linthresh=25); axs[1].set_xlabel("L_scrap (years, N_small=100)"); axs[1].axhline(1.1, color="k", lw=0.5); axs[1].legend(fontsize=7)
axs[2].set_xlabel("p_prec (share of groups whose scrap-built tools reach T2e precision)"); axs[2].axhline(1.1, color="k", lw=0.5); axs[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_scrap_tool.png"), dpi=120); plt.close(fig)
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("check_113", d113, "X-113", OUT["check_X_vs_113_base"], "1e4-113", OUT["check_1e4_vs_113_workshop"])
print("L_bulk_q", OUT["L_bulk_q"], "L_myf_q", OUT["L_myf_q"])
for t, o in TAB.items(): print(t, "F_heavy", round(o["F_heavy"], 4))
for kn, s in OUT["sweep_Lscrap_n100"].items(): print("Lscrap", kn, [(a["L_scrap"], round(a["F_heavy"], 3)) for a in s])
for kn, s in OUT["sweep_pprec_n100_Lbulk"].items(): print("pprec", kn, [(a["p_prec"], round(a["F_heavy"], 3)) for a in s])
print("crit_Lscrap", OUT["crit_Lscrap"], "crit_pprec", OUT["crit_pprec"])
for k, hh in H.items(): print(k, hh["class"], hh["supported"], str(hh["value"])[:110])
print("runtime", round(OUT["runtime_s"], 1))
