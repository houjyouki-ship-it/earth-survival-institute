"""研究120：継電器の並列の読み取り回路の部品数 C と接点の寿命（open_problems 282）。
計算のみ、ネット接続なし（資料は sources/ に保存済み）。事前登録：PREREGISTRATION.md（2026-10-25 05:36 JST）。
119 の sim.py を「# ---------- 仮説」の前まで exec（119 の p_vac と表を差 0 で再現、assert）。
接点の寿命は公開の値（汎用継電器：機械 10^7 回・定格負荷 10^5 回、Omron MK・Deltrol 260・P&B KH。リード継電器：10^9 回まで、Wikipedia）。
部品数 C、動作の割合 a、作り続ける能力 M50、部品の確率はすべて【推測】。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P119 = os.path.abspath(os.path.join(HERE, "..", "119-t2e-reader-vacuum", "sim.py"))
src = open(P119, encoding="utf8").read(); cut = src.index("\n# ---------- 仮説")
G = {"__file__": P119, "__name__": "sim119"}
exec(compile(src[:cut], P119, "exec"), G)
R119 = json.load(open(os.path.join(os.path.dirname(P119), "results.json"), encoding="utf8"))
dP = max(abs(G["PV"][k] - R119["p_vac_versions"][k]) for k in R119["p_vac_versions"])
assert dP < 1e-12, dP
dT = max(abs(G["TAB"][t]["F_heavy"] - R119["table"][t]["F_heavy"]) for t in R119["table"])
run, KAPS, fe_d, le_d, t2e_two, L_bulk, PPREC, TB, ptube, PX_REEL, B_FRAME, PX_FRAME, YR = (G[k] for k in (
    "run", "KAPS", "fe_d", "le_d", "t2e_two", "L_bulk", "PPREC", "TB", "ptube", "PX_REEL", "B_FRAME", "PX_FRAME", "YR"))
OUT = {"study": "120-relay-contact-life", "check_119_pvac_maxdiff": dP, "check_119_table_F_maxdiff_newrng": dT}

# ---------- 1. 接点の消耗（総動作数＝点数×C×a。速さ・チャンネル数に依らない）----------
LIFE = {"armature_rated_load": 1e5, "armature_mid": 1e6, "armature_mech_dry": 1e7, "reed_dry": 1e9}  # 事実：1e5・1e7（データシート）、1e9（Reed relay）。1e6 は【推測】の中間
DATA = {"reel_120GB": PX_REEL, "q0.3_36GB": 0.3 * PX_REEL, "prio_10MB": 10e6 / B_FRAME * PX_FRAME}
T_READ, RATE = 20.0, 10.0                                   # 119 と同じ 20 年・10 Hz
def demand(px, C, a, L):
    ops = px * C * a; ch = px / (RATE * T_READ * YR); N0 = ch * C
    return {"ops": ops, "N0": N0, "replacements": ops / L, "ratio_to_N0": ops / L / max(N0, 1e-9), "M_total": N0 + ops / L,
            "days_to_wear_one_relay": L / (RATE * a) / 86400}
DEM = {}
for dn, px in DATA.items():
    for ln, L in LIFE.items():
        for C in (20, 50, 100):
            DEM[f"{dn}|{ln}|C{C}"] = demand(px, C, 0.5, L)
OUT["demand"] = DEM
# 比較：Z4 約 2,500 継電器（Wikipedia）、Mark I 765,000 部品・1944–1959 運用（Wikipedia）
OUT["reference_counts"] = {"Z4_relays": 2500, "MarkI_components": 765000, "MarkI_years": 1959 - 1944}

# ---------- 2. p_relay の置き直し（すべて【推測】）----------
B = {"p_build": 0.8,      # 119 の p_relay：最初の一揃いを組めて動く確率
     "M50": 1e5,          # 20 年で作り続けられる継電器（または接点の組）の数の中央：1 日 15 個×20 年 ≈ 1.1e5
     "C": 50, "a": 0.5, "L_arm": 1e6, "L_reed": 1e9,
     "p_NiFe": 0.6,       # リードの磁性合金（Ni-Fe）を作れる確率
     "refurb": 1.0,       # 接点の磨き直し・付け替えで新品の何分の一の手間で済むか（1＝磨き直しなし）
     "data": "reel_120GB"}
def S(M, M50): return 1.0 / (1.0 + M / M50)
def prelay(d):
    px = DATA[d["data"]]
    Ma = demand(px, d["C"], d["a"], d["L_arm"]); Mr = demand(px, d["C"], d["a"], d["L_reed"])
    Ma_eff = Ma["N0"] + Ma["replacements"] / d["refurb"]; Mr_eff = Mr["N0"] + Mr["replacements"] / d["refurb"]
    p_arm = d["p_build"] * S(Ma_eff, d["M50"])
    p_reed = d["p_build"] * TB["p_seal"] * TB["p_glass"] * d["p_NiFe"] * S(Mr_eff, d["M50"])   # ガラスと金属の封じ（真空は要らない）
    return {"p_arm": p_arm, "p_reed": p_reed, "p_relay": 1 - (1 - p_arm) * (1 - p_reed), "M_arm": Ma_eff, "M_reed": Mr_eff}
def mod(**kw): d = dict(B); d.update(kw); return d
VERS = {
 "main": B,
 "L_arm_1e5": mod(L_arm=1e5), "L_arm_1e7": mod(L_arm=1e7),
 "C20": mod(C=20), "C100": mod(C=100),
 "M50_3e4": mod(M50=3e4), "M50_1e6": mod(M50=1e6),
 "refurb10": mod(refurb=10.0),
 "q0.3": mod(data="q0.3_36GB"), "prio10MB": mod(data="prio_10MB"),
 "noreed": mod(p_NiFe=0.0),
 "optimistic": mod(L_arm=1e7, C=20, M50=1e6, refurb=10.0, p_NiFe=0.9),
 "pessimistic": mod(L_arm=1e5, C=100, M50=3e4, p_NiFe=0.3),
}
PR = {k: prelay(v) for k, v in VERS.items()}
P_TUBE = ptube(TB)
PV = {k: 1 - (1 - v["p_relay"]) * (1 - P_TUBE) for k, v in PR.items()}
PV["either_119_relay0.8"] = 1 - (1 - 0.8) * (1 - P_TUBE)
OUT["base"] = B; OUT["p_relay_versions"] = PR; OUT["p_tube"] = P_TUBE; OUT["p_vac_versions"] = PV

TAB = {}
for vn, pv in PV.items():
    fT = t2e_two(100.0, L_bulk, PPREC * pv)
    for kn in KAPS:
        TAB[f"{kn}|{vn}"] = run(f"{kn}|{vn}", KAPS[kn], {"Fe": fe_d, "lens": le_d, "T2e": fT}, store=False)
OUT["table"] = TAB
assert abs(TAB["slow_Tasmania|either_119_relay0.8"]["F_heavy"] - R119["table"]["slow_Tasmania|either_main"]["F_heavy"]) < 0.02

# ---------- 仮説 ----------
H = {}
H["H132a"] = {"class": "confirmatory", "supported": False,
  "value": "Relay の記事は寿命を『機械的寿命と接点の寿命があり負荷で変わる』と述べるだけで、回数の数値を挙げない（数値は Reed relay の 10^9 と Reed switch の『数千〜数十億』、データシートの 10^7／10^5）"}
H["H132b"] = {"class": "confirmatory", "supported": True, "value": "Mark I 1944-08-07 運用開始、1959 年に解体：15 年"}
H["H132c"] = {"class": "confirmatory", "supported": True, "value": "ガラスの管に封じ（hermetic seal）、寿命 10^9 回まで。ただし記事は封じの中身（不活性ガス・真空）を書かない"}
d = DEM["reel_120GB|armature_mid|C50"]; dlo = DEM["reel_120GB|armature_mech_dry|C20"]
H["H132d"] = {"class": "prediction", "value": {"ratio_L1e6": d["ratio_to_N0"], "ratio_L1e7": dlo["ratio_to_N0"]},
  "rule": "寿命 1e6〜1e7 回で 交換数/初めの個数 > 100", "supported": bool(min(d["ratio_to_N0"], dlo["ratio_to_N0"]) > 100),
  "note": "比は rate×a×T/L で C・データの量に依らず、ほぼ算術で決まる（自明に近い予測）"}
fh = TAB["slow_Tasmania|main"]["F_heavy"]
H["H132e"] = {"class": "prediction", "value": {"p_relay": PR["main"]["p_relay"], "F_heavy_slow": fh},
  "rule": "0.4<=p_relay<=0.7 かつ 1.35<=F_heavy<=1.55", "supported": bool(0.4 <= PR["main"]["p_relay"] <= 0.7 and 1.35 <= fh <= 1.55)}
H["H132f"] = {"class": "descriptive", "supported": None, "value": "寿命と負荷・材料の表"}
OUT["hypotheses"] = H; OUT["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
for ln, L in LIFE.items():
    xs = [20, 50, 100]; axs[0].plot(xs, [DEM[f"reel_120GB|{ln}|C{c}"]["M_total"] for c in xs], "o-", label=f"{ln} (L={L:.0e})")
axs[0].axhline(2500, color="gray", ls=":", lw=0.8); axs[0].text(21, 2700, "Z4 relays", fontsize=7)
axs[0].axhline(765000, color="gray", ls="--", lw=0.8); axs[0].text(21, 8e5, "Mark I components", fontsize=7)
axs[0].axhline(B["M50"], color="k", lw=0.5); axs[0].set_yscale("log"); axs[0].set_xlabel("relays per channel C"); axs[0].set_ylabel("relays to make (initial + wear), one reel")
axs[0].legend(fontsize=7); axs[0].set_title("120: contact wear for one 120 GB reel (a=0.5)", fontsize=9)
vn = list(PV); x = np.arange(len(vn))
for j, kn in enumerate(KAPS):
    axs[1].bar(x + (j-1)*0.27, [TAB[f"{kn}|{w}"]["F_heavy"] for w in vn], 0.27, label=kn)
axs[1].set_xticks(x); axs[1].set_xticklabels([f"{w}\n{PV[w]:.2f}" for w in vn], fontsize=6, rotation=35)
axs[1].axhline(1.1, color="k", lw=0.5); axs[1].axhline(1.5, color="k", lw=0.5, ls=":"); axs[1].legend(fontsize=7); axs[1].set_ylabel("F_heavy (A_Ord/U3)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_relay_wear.png"), dpi=120); plt.close(fig)
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("check_119", dP, "F newrng diff", dT)
for k, v in DEM.items():
    if "C50" in k or "C20" in k: print("demand", k, {kk: f"{vv:.3g}" for kk, vv in v.items()})
for k, v in PR.items(): print("p_relay", k, {kk: round(vv, 4) if vv < 10 else f"{vv:.3g}" for kk, vv in v.items()}, "p_vac", round(PV[k], 4))
print("p_tube", round(P_TUBE, 4))
for t, o in TAB.items(): print(t, "F_heavy", round(o["F_heavy"], 4))
for kk, hh in H.items(): print(kk, hh["class"], hh["supported"], str(hh.get("value"))[:160])
print("runtime", round(OUT["runtime_s"], 1))
