"""研究112：重い崩壊の段の入力を「地元で建て直せる R／代わりがありうる S／代わりなし X」に分け、
建て直せない入力だけに在庫の寿命 L を掛ける版の F_heavy（open_problems 265）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md（2026-10-24 13:36 JST）。
111 の sim.py を「# ---------- 仮説」の前まで exec（109→108→… の乱数と 111 の全表を再現、assert）。新しい乱数は seed 20261112。
段 k を保つ ⇔ 技能 z_min ≥ ln N_k かつ 段 k の各入力 i の在庫の条件（R：なし、S：代わりあり→min(t_below,t_sub)≤L_i／代わりなし→t_below≤L_i、X：t_below≤L_i）
かつ 下の段を保つ（入れ子）。R の入力の人手は N_k に入っているので L を掛けない（二重に数えない）。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P111 = os.path.abspath(os.path.join(HERE, "..", "111-trigger-skill-capital", "sim.py"))
src = open(P111, encoding="utf8").read(); cut = src.index("\n# ---------- 仮説")
G1 = {"__file__": P111, "__name__": "sim111"}
exec(compile(src[:cut], P111, "exec"), G1)
R111 = json.load(open(os.path.join(os.path.dirname(P111), "results.json"), encoding="utf8"))
d111 = max(abs(G1["TAB"][t]["F_heavy"] - R111["table"][t]["F_heavy"]) for t in R111["table"])
d111s = max(abs(a["F_heavy"] - b["F_heavy"]) for kn in R111["sweep"] for a, b in zip(G1["SW"][kn], R111["sweep"][kn]))
assert d111 < 1e-12 and d111s < 1e-12, (d111, d111s)
G = G1["G"]
N, B, SRC, COMPS, FIXED, V_m, agg, q_TC, v0 = (G[k] for k in ("N", "B", "SRC", "COMPS", "FIXED", "V_m", "agg", "q_TC", "v0"))
N_Fe, N_lens, N_T2e, r_pop, CONS, zmin, T2_from = (G[k] for k in ("N_Fe", "N_lens", "N_T2e", "r_pop", "CONS", "zmin", "T2_from"))
KAPS, LV, INF = G1["KAPS"], G1["LV"], G1["INF"]
g = np.random.default_rng(20261112)
q3 = lambda v: [float(np.quantile(v, p)) for p in (0.05, 0.5, 0.95)]
OUT = {"study": "112-nonrebuildable-capital", "seed": 20261112,
       "check_111_table_reproduced_maxdiff": d111, "check_111_sweep_reproduced_maxdiff": d111s}

STAGES = [("Fe", 5, N_Fe), ("lens", 4, N_lens), ("T2e", 3, N_T2e)]   # 低い段から

def new_st(k, kap, Lk):
    """Lk: {"Fe": array, "lens": array, "T2e": array}＝段ごと・群ごとの有効な寿命（∞ なら資本の条件なし）"""
    st = B[k]["st"]; n = np.maximum(SRC[k]["sv"]["nmax"], 1.0)
    zm, _ = zmin(n, r_pop, kap, G["POP_W"])
    lev = np.full(N, 6); prev = np.ones(N, bool); keep_share = {}
    for nm, code, Nk in STAGES:
        tb = np.maximum(0.0, np.log(Nk / n) / r_pop)
        keep = (zm >= np.log(Nk)) & Lk[nm](tb) & prev   # Lk[nm] は t_below を受けて条件を返す関数
        lev = np.where(keep, code, lev); prev = keep
    return np.where(CONS[k], np.minimum(st, lev), st)

def run(tag, kap, Lk, store=True):
    D = {}; sh = {}
    for k in COMPS:
        if k in FIXED: T2n = B[k]["T2s"]
        else:
            s2 = new_st(k, kap, Lk); T2n = T2_from(k, s2)
            sh[k] = (float(np.isin(B[k]["st"], (4, 5, 6)).mean()), float(np.isin(s2, (4, 5, 6)).mean()))
        D[k] = V_m(k, q_TC, True, T2n) - V_m(k, 0.3, False, T2n)
    v1 = agg({"x": D}, anchors=("A_Ord", "A_XPT_S"))
    b_ = v0["A_Ord"]["U3"]["x"]; n_ = v1["A_Ord"]["U3"]["x"]
    o = {"F_heavy": n_["heavy"] / b_["heavy"], "F_total": n_["total"] / b_["total"], "U3_heavy_rel_drop": 1 - sh["U3"][1] / sh["U3"][0]}
    if store: OUT.setdefault("runs", {})[tag] = o
    return o

# ---- 条件の部品 ----
def cond_const(L):            # 111 と同じ：t_below ≤ L
    return lambda tb: tb <= L
ALWAYS = lambda tb: np.ones(N, bool)
def cond_inputs(inputs):
    """inputs: list of (cls, p_sub, t_sub, L)。R は条件なし。S は群ごとに確率 p_sub で代わりあり。"""
    def f(tb):
        ok = np.ones(N, bool)
        for cls, p, ts, L in inputs:
            if cls == "R": continue
            if cls == "X": ok &= tb <= L; continue
            ok &= np.where(SUBS[(id(inputs), cls, p, id(L))], np.minimum(tb, ts) <= L, tb <= L)
        return ok
    return f
SUBS = {}
def mk(inputs):
    for cls, p, ts, L in inputs:
        if cls == "S": SUBS[(id(inputs), cls, p, id(L))] = g.random(N) < p
    return cond_inputs(inputs)

# ---------- 検算：全段に同じ L・代わりなし（p_sub=0）で 111 と一致 ----------
CHK = {}
for kn in ("slow_Tasmania", "trigger_mix"):
    for ln in ("myford_107", "rework_106"):
        L = LV[ln]; c = cond_const(L)
        a = run(f"check_sameL|{kn}|{ln}", KAPS[kn], {"Fe": c, "lens": c, "T2e": c}, store=False)["F_heavy"]
        b = R111["table"][f"{kn}|{ln}"]["F_heavy"]; CHK[f"{kn}|{ln}"] = abs(a - b)
        inp = [("S", 0.0, 0.0, L)]; cc = mk(inp)
        a2 = run("chk2", KAPS[kn], {"Fe": cc, "lens": cc, "T2e": cc}, store=False)["F_heavy"]
        CHK[f"{kn}|{ln}|S_p0"] = abs(a2 - b)
OUT["check_sameL_equals_111"] = CHK; assert max(CHK.values()) < 1e-12, CHK
a = run("chk_R", KAPS["slow_Tasmania"], {"Fe": ALWAYS, "lens": ALWAYS, "T2e": ALWAYS}, store=False)["F_heavy"]
OUT["check_allR_equals_111_inf"] = abs(a - R111["table"]["slow_Tasmania|inf"]["F_heavy"]); assert OUT["check_allR_equals_111_inf"] < 1e-12

# ---------- 在庫の寿命の分布 ----------
def lnorm(q05, q50, q95):
    s = (np.log(q95) - np.log(q05)) / (2 * 1.6449); return np.exp(g.normal(np.log(q50), s, N))
L_bulk = lnorm(551.4, 1878.4, 21467.8)              # 104 の 20 mm の部材（廃墟の鉄・金属の在庫）
L_glass104 = np.exp(g.uniform(np.log(500), np.log(5000), N))   # 104 のガラスの寿命の置き値【推測】
L_cullet_lb = np.full(N, 740.0)                      # Freestone 2015：Roman glass を 450→約1190 年まで再溶解（下限）【計算】
L_myf, L_rew = LV["myford_107"], LV["rework_106"]
OUT["L_q"] = {"bulk_104": q3(L_bulk), "glass_104": q3(L_glass104), "myford_107": q3(L_myf), "rework_106": q3(L_rew)}

# ---------- 入力の表と段ごとの条件 ----------
# 透明な素材：新しいガラスは融剤・耐火土の代わりが両方揃う群だけ作れ、揃うまで（max(t_flux,t_refr)）は廃墟のガラスの在庫でつなぐ。揃わない群は在庫だけ（t_below ≤ L_glass）。
def lens_cond(p_flux, t_flux, p_refr, t_refr, L_glass, crystal=0.0):
    s1 = g.random(N) < p_flux; s2 = g.random(N) < p_refr; cr = g.random(N) < crystal
    def f(tb):
        new_glass = s1 & s2
        ok = np.where(new_glass, np.minimum(tb, max(t_flux, t_refr)) <= L_glass, tb <= L_glass)
        return ok | cr     # 岩の水晶のレンズの経路（Visby）がある群
    return f
def fe_cond(p_ore, p_fuel, L=None):
    L = L_bulk if L is None else L
    s = (g.random(N) < p_ore) & (g.random(N) < p_fuel)
    return lambda tb: np.where(s, True, tb <= L)
def t2e_cond(tool="X", L_tool=None, p_metal=0.3, L_metal=None):
    L_tool = L_myf if L_tool is None else L_tool; L_metal = L_bulk if L_metal is None else L_metal
    s = g.random(N) < p_metal
    def f(tb):
        ok = np.where(s, True, tb <= L_metal)
        if tool == "X": ok &= tb <= L_tool
        return ok
    return f

FRAC = {"Fe": {"inputs": ["furnace_clay R", "bellows R", "hammer R", "anvil R", "ore S", "fuel_charcoal S"], "nonR_share": 2 / 6},
        "lens": {"inputs": ["sand R", "lime R", "fuel R", "abrasive_sand R", "polish_ironoxide R", "flux S", "refractory_crucible S", "clear_glass_stock S"], "nonR_share": 3 / 8},
        "T2e": {"inputs": ["iron_from_Fe R", "glass_from_lens R", "fuel R", "precision_machine_tools X", "nonferrous_metals S"], "nonR_share": 2 / 5}}
OUT["input_table"] = FRAC

# ---------- 版 ----------
V = {}
def ver(name, fe, le, t2):
    V[name] = {"Fe": fe, "lens": le, "T2e": t2}
# 登録の主（prior）：代わりは即時（t_sub=0）、ガラスの在庫は 104 の置き値
ver("prior_registered", fe_cond(0.8, 1.0), lens_cond(0.9, 0.0, 0.7, 0.0, L_glass104), t2e_cond("X", L_myf))
# 資料の版（data）：燃料も S（Norse Greenland は木炭の不足で輸入頼み）p 0.9【推測】、融剤の代わりの遅れ 550 年（Forest glass 1000−450）、
# 耐火土の遅れも同じ（Forest glass：potash は +200 °C で高温の陶器が要る）、ガラスの在庫は Freestone の下限 740 年
ver("data_main", fe_cond(0.8, 0.9), lens_cond(0.9, 550.0, 0.7, 550.0, L_cullet_lb), t2e_cond("X", L_myf))
ver("data_rework", fe_cond(0.8, 0.9), lens_cond(0.9, 550.0, 0.7, 550.0, L_cullet_lb), t2e_cond("X", L_rew))
ver("data_crystal0.1", fe_cond(0.8, 0.9), lens_cond(0.9, 550.0, 0.7, 550.0, L_cullet_lb, crystal=0.1), t2e_cond("X", L_myf))
# H124g：T2e だけが建て直せない
ver("only_T2e_X", ALWAYS, ALWAYS, t2e_cond("X", L_myf, p_metal=1.0))
# H124h：T2e を建て直せる、製鉄・レンズだけに 111 の Myford の L
cM = cond_const(L_myf)
ver("only_Fe_lens_myford", cM, cM, ALWAYS)
# 段ごとの寄与：一つの段にだけ 111 の Myford の L
ver("myford_Fe_only", cM, ALWAYS, ALWAYS); ver("myford_lens_only", ALWAYS, cM, ALWAYS); ver("myford_T2e_only", ALWAYS, ALWAYS, cM)
# T2e の道具を R とする版（作り直しの知識が確実、106 の三面の擦り合わせ）
ver("data_T2e_tool_R", fe_cond(0.8, 0.9), lens_cond(0.9, 550.0, 0.7, 550.0, L_cullet_lb), t2e_cond("R"))
# 悲観：鉱石・燃料が地元にない群が多い、耐火土も乏しい
ver("data_pessimistic", fe_cond(0.5, 0.7), lens_cond(0.7, 1000.0, 0.4, 1000.0, L_cullet_lb), t2e_cond("X", L_myf, p_metal=0.1))

TAB = {}
for vn, Lk in V.items():
    for kn in KAPS:
        TAB[f"{kn}|{vn}"] = run(f"{kn}|{vn}", KAPS[kn], Lk)
OUT["table"] = TAB

# 融剤の代わりの遅れ t_sub の走査（遅い κ、data_main の他は固定）と、T2e の L 一定の走査（製鉄・レンズは data_main）
TS = [0, 100, 200, 400, 550, 740, 800, 1000, 1500, 2000]
OUT["sweep_t_sub"] = [{"t_sub": t, **run(f"tsub_{t}", KAPS["slow_Tasmania"],
                       {"Fe": V["data_main"]["Fe"], "lens": lens_cond(0.9, t, 0.7, t, L_cullet_lb), "T2e": V["data_main"]["T2e"]})} for t in TS]
LS = np.logspace(1, 4, 13)
fe_d, le_d = V["data_main"]["Fe"], V["data_main"]["lens"]
OUT["sweep_T2e_L"] = {kn: [{"L": float(l), **run(f"t2eL_{kn}_{l:.0f}", KAPS[kn], {"Fe": fe_d, "lens": le_d, "T2e": cond_const(np.full(N, l))})} for l in LS]
                      for kn in ("slow_Tasmania", "trigger_mix")}
def crit(kn, thr=1.1):
    fh = np.array([s["F_heavy"] for s in OUT["sweep_T2e_L"][kn]]); i = np.where(fh >= thr)[0]
    return float(LS[i.min()]) if i.size else None
OUT["T2e_L_crit_F1.1"] = {kn: crit(kn) for kn in OUT["sweep_T2e_L"]}

# 縛りの診断：U3 の重い崩壊の群のうち、段の技能の条件を満たすのに資本で落ちる割合（遅い κ、data_main）
st = B["U3"]["st"]; n = np.maximum(SRC["U3"]["sv"]["nmax"], 1.0); h = np.isin(st, (4, 5, 6)) & CONS["U3"]
zm, _ = zmin(n, r_pop, KAPS["slow_Tasmania"], G["POP_W"]); DG = {}
for nm, code, Nk in STAGES:
    tb = np.maximum(0.0, np.log(Nk / n) / r_pop); sk = (zm >= np.log(Nk)) & h
    DG[nm] = {"skill_ok_share": float(sk.mean() / h.mean()),
              "capital_fail_given_skill_data": float((~V["data_main"][nm](tb) & sk).sum() / max(sk.sum(), 1)),
              "capital_fail_given_skill_111myford": float((~(tb <= L_myf) & sk).sum() / max(sk.sum(), 1))}
OUT["diag_binding_U3_slow"] = DG

# ---------- 仮説 ----------
H = {}
H["H124a"] = {"class": "confirmatory", "supported": True, "value": "Cape York meteorite: 'known to the Inughuit ... for centuries, who used it as a source of meteoritic iron for tools'; found 'about 1,000 years ago'; removed 1894-97"}
H["H124b"] = {"class": "confirmatory", "supported": True, "value": "Forest glass: 'approximately 1000-1700 AD using wood ash and sand'; 'For several centuries after the fall ... around 450 AD, recycling of Roman glass formed the major part of the local industry'"}
H["H124c"] = {"class": "confirmatory", "supported": True, "value": "Natron: used 'in ceramic and glass making by the Romans and others until at least AD 640'; 'Soda ash also replaced natron in the ancient glass trade' (Egypt/Wadi El Natrun named)",
              "note": "replacement named is soda ash, not plant ash; the prediction said 'plant ash etc.'"}
H["H124d"] = {"class": "confirmatory", "supported": True, "value": "Visby lenses: rock crystal (quartz), 11th or 12th century, local manufacture at Frojel"}
H["H124e"] = {"class": "confirmatory", "supported": True, "value": "Freestone 2015: recycling of Roman glass 'made a significant and sometimes very substantial contribution to the northern European glass industry from the eighth to 12th centuries'; Limoges enamels natron (Roman) glass up to about 1190",
              "note": "not an isolated group: fresh eastern glass also reached the west in 5th-7th c; the Peake 2013 thesis (5th-6th c recycling) could not be opened (bot check) and is not counted"}
v = TAB["slow_Tasmania|data_main"]["F_heavy"]; H["H124f"] = {"class": "prediction", "value": v, "rule": "|F-1.091|<=0.05", "supported": bool(abs(v - 1.0911) <= 0.05)}
v2 = TAB["slow_Tasmania|prior_registered"]["F_heavy"]; H["H124f"]["value_prior_registered"] = v2
v = TAB["slow_Tasmania|only_T2e_X"]["F_heavy"]; H["H124g"] = {"class": "prediction", "value": v, "rule": "<1.2", "supported": bool(v < 1.2)}
v = TAB["slow_Tasmania|only_Fe_lens_myford"]["F_heavy"]; H["H124h"] = {"class": "prediction", "value": v, "rule": ">=2.0", "supported": bool(v >= 2.0)}
H["H124i"] = {"class": "descriptive", "value": FRAC, "supported": None}
OUT["hypotheses"] = H; OUT["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
vn_show = ["prior_registered", "data_main", "data_pessimistic", "data_rework", "data_T2e_tool_R", "only_T2e_X", "only_Fe_lens_myford",
           "myford_Fe_only", "myford_lens_only", "myford_T2e_only"]
x = np.arange(len(vn_show))
for j, kn in enumerate(KAPS):
    axs[0].bar(x + (j - 1) * 0.27, [TAB[f"{kn}|{v}"]["F_heavy"] for v in vn_show], 0.27, label=kn)
axs[0].set_xticks(x); axs[0].set_xticklabels(vn_show, rotation=55, ha="right", fontsize=7); axs[0].axhline(1.1, color="k", lw=0.5)
axs[0].set_ylabel("F_heavy (A_Ord/U3)"); axs[0].set_title("112: L only on non-rebuildable inputs", fontsize=9); axs[0].legend(fontsize=7)
for kn, c in (("slow_Tasmania", "C3"), ("trigger_mix", "C0")):
    axs[1].semilogx(LS, [s["F_heavy"] for s in OUT["sweep_T2e_L"][kn]], "o-", color=c, label=f"{kn}: Fe/lens data_main, T2e constant L")
    axs[1].semilogx(R111["sweep"][kn][0]["L"] * 0 + np.array([s["L"] for s in R111["sweep"][kn]]), [s["F_heavy"] for s in R111["sweep"][kn]], ":", color=c, label=f"{kn}: 111 (same L on all stages)")
axs[1].axhline(1.1, color="k", lw=0.5); axs[1].set_xlabel("L of T2e precision tools (years)"); axs[1].set_ylabel("F_heavy"); axs[1].legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_nonrebuildable.png"), dpi=120); plt.close(fig)
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("check_111", d111, d111s, "check_sameL", max(CHK.values()), "check_allR", OUT["check_allR_equals_111_inf"])
print("L_q", {k: [round(x, 0) for x in v] for k, v in OUT["L_q"].items()})
for t, o in TAB.items(): print(t, "F_heavy", round(o["F_heavy"], 4), "drop", round(o["U3_heavy_rel_drop"], 3))
print("sweep_t_sub", [(s["t_sub"], round(s["F_heavy"], 3)) for s in OUT["sweep_t_sub"]])
for kn, s in OUT["sweep_T2e_L"].items(): print("sweep_T2e_L", kn, [(round(a["L"]), round(a["F_heavy"], 3)) for a in s])
print("T2e_L_crit", OUT["T2e_L_crit_F1.1"])
print("diag", json.dumps(DG))
for k, hh in H.items(): print(k, hh["class"], hh["supported"], str(hh["value"])[:110])
print("runtime", round(OUT["runtime_s"], 1))
