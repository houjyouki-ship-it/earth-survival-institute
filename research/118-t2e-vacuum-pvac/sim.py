"""研究118：T2e の機械以外の部品（電球の真空・封じ・フィラメント）の因子 p_vac（open_problems 276）。
計算のみ、ネット接続なし（資料は sources/ に保存済み）。事前登録：PREREGISTRATION.md（2026-10-25 01:43 JST、追記 01:47 JST）。
117 の sim.py を「# ---------- 仮説」の前まで exec（117 の p_prec と表を差 0 で再現、assert）。新しい乱数は seed 20261118。
p_prec(117) × p_vac を t2e_two の第 3 引数に入れ、F_heavy を再計算する。
p_vac = p_pump × p_seal × p_fil × p_glass、p_pump = 1 − (1 − p_Hg)(1 − p_mech1mm × p_P)。値はすべて【推測】。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P117 = os.path.abspath(os.path.join(HERE, "..", "117-scrap-builder-pexec", "sim.py"))
src = open(P117, encoding="utf8").read(); cut = src.index("\n# ---------- 仮説")
G = {"__file__": P117, "__name__": "sim117"}
exec(compile(src[:cut], P117, "exec"), G)
R117 = json.load(open(os.path.join(os.path.dirname(P117), "results.json"), encoding="utf8"))
dP = max(abs(G["PP"][k] - R117["p_prec_versions"][k]) for k in R117["p_prec_versions"])
dT = max(abs(G["TAB"][t]["F_heavy"] - R117["table"][t]["F_heavy"]) for t in R117["table"])
assert dP < 1e-12 and dT < 1e-12, (dP, dT)
run, KAPS, fe_d, le_d, t2e_two, L_bulk = (G[k] for k in ("run", "KAPS", "fe_d", "le_d", "t2e_two", "L_bulk"))
PPREC = G["PP"]["main_single"]                    # 117 主の p_prec = 0.912 × 0.5
OUT = {"study": "118-t2e-vacuum-pvac", "seed": 20261118, "check_117_pprec_maxdiff": dP,
       "check_117_table_maxdiff": dT, "p_prec_117main": PPREC}

# ---------- 資料から読んだ要件（H130f、記述）----------
PARTS = [
 {"part": "排気（水銀ポンプ）", "need": "Sprengel 式・Geissler 式。ガラス管・水銀・ゴム管・ゴム栓、前段に機械のポンプ（Geissler 系、特別な作りは要らない）。Sprengel は前段なしでも動く",
  "fact": "Ram 1894 ch.XII：機械のポンプでは足りず水銀のポンプが要る。Thompson 1888：Müller の小型ポンプは水銀 300 cc、Edison 社（1885）は鉄管に fall-tube を差すだけ",
  "numbers": {"Hg_cc_small_pump": 300, "Hg_kg_small_pump": 300 * 13.546 / 1000, "exhaust_h_1881": 5.0, "exhaust_min_improved": 30}},
 {"part": "排気（化学の代わり）", "need": "機械のピストンのポンプで約 1 mm Hg＋赤燐の蒸気（Malignani 1896）。水銀なし",
  "fact": "Howell & Schroeder 1927 ch.4：Malignani は水銀のポンプを持たず機械のポンプで約 1 mm が限界、燐の蒸気で良い真空、排気は 30 分→2 分未満",
  "numbers": {"mech_limit_mmHg": 1.0, "exhaust_min": 2}},
 {"part": "乾燥", "need": "五酸化燐（phosphoric anhydride）を排気の管に。燐は骨灰・尿から（1669 Brand、1769 骨灰）",
  "fact": "Howell & Schroeder 1927 ch.4・Wikipedia Phosphorus", "numbers": {}},
 {"part": "封じの導入線", "need": "白金線（ガラスと熱膨張が近く、吹管の炎で溶けず酸化しない）。代わりの合金は長い封じが要り漏れやすい",
  "fact": "Ram 1894 ch.IV：'Platinum wire is always used'、直径 0.014 in（1 A まで）。中は銅線を継げる",
  "numbers": {"Pt_wire_in": 0.014}},
 {"part": "フィラメント", "need": "炭にした木綿糸（Edison 1879-10-21）、竹、Swan の羊皮紙化した木綿糸（硫酸）。炉で 8 時間以上かけて炭化、12 時間以上冷ます",
  "fact": "Dyer & Martin 1910（Gutenberg 820）、Ram 1894 ch.II–III", "numbers": {"carbonise_heat_h_min": 8, "cool_h_min": 12}},
 {"part": "フィラメントの計測", "need": "径の測定は 1/5 mil（0.0002 in）まで読めるねじのマイクロメータ、またはランプとレンズで像を幕に映す方法。抵抗で揃える（flashing to resistance）",
  "fact": "Ram 1894 ch.VIII・V", "numbers": {"micrometer_read_in": 0.0002}},
 {"part": "ガラス細工", "need": "吹管でのランプの封じ・ポンプの製作と修理。'easy work'、ポンプは 'always breaking, and had to be repaired on the spot'",
  "fact": "Thompson 1888 の討論（Swinburne の発言）", "numbers": {"pump_price_before_gbp": [10, 15], "pump_price_after_s": 10}},
]
OUT["parts"] = PARTS
# 白金の量（1 灯あたり）：長さは【推測】
L_cm = 1.0
Pt_mg = 2 * np.pi * (0.014 * 2.54 / 2) ** 2 * L_cm * 21.452 * 1000
OUT["Pt_mg_per_lamp_if_2x1cm"] = float(Pt_mg)

# ---------- p_vac の版（すべて【推測】）----------
BASE = {"p_Hg": 0.5, "p_mech1mm": 0.5, "p_P": 0.9, "p_seal": 0.9, "p_fil": 0.97, "p_glass": 0.9}
def pvac(d):
    p_pump = 1 - (1 - d["p_Hg"]) * (1 - d["p_mech1mm"] * d["p_P"])
    return float(p_pump * d["p_seal"] * d["p_fil"] * d["p_glass"])
def mod(**kw): d = dict(BASE); d.update(kw); return d
VERS = {
 "vac1_117main":       None,                                 # p_vac=1（117 の主の再現）
 "main":               BASE,
 "Hg0.2":              mod(p_Hg=0.2),
 "Hg0.8":              mod(p_Hg=0.8),
 "Hg0_mech_only":      mod(p_Hg=0.0),                        # 水銀がまったくない
 "mech0.8_cond":       mod(p_mech1mm=0.8),                   # 盤が成功した群の条件つき
 "glass0.5":           mod(p_glass=0.5),                     # 117 の p_exec と同じ悲観
 "all_low":            mod(p_Hg=0.2, p_mech1mm=0.3, p_P=0.7, p_seal=0.7, p_fil=0.9, p_glass=0.5),
 "all_high":           mod(p_Hg=0.8, p_mech1mm=0.8, p_P=0.97, p_seal=0.99, p_fil=0.99, p_glass=0.97),
}
PV = {k: (1.0 if v is None else pvac(v)) for k, v in VERS.items()}
OUT["p_vac_inputs"] = {k: v for k, v in VERS.items()}; OUT["p_vac_versions"] = PV

# 部品の不確かさ：各値を平均＝主の Beta（強さ 10）で引く【推測】
g = np.random.default_rng(20261118); M = 200_000
def bdraw(m, k=10.0): return g.beta(m * k, (1 - m) * k, M)
D = {kk: bdraw(v) for kk, v in BASE.items()}
pv_mc = (1 - (1 - D["p_Hg"]) * (1 - D["p_mech1mm"] * D["p_P"])) * D["p_seal"] * D["p_fil"] * D["p_glass"]
q10, q50, q90 = (float(np.quantile(pv_mc, q)) for q in (0.1, 0.5, 0.9))
OUT["p_vac_mc"] = {"q10": q10, "q50": q50, "q90": q90, "mean": float(pv_mc.mean()), "beta_strength": 10}
PV["mc_q10"] = q10; PV["mc_q90"] = q90

TAB = {}
for vn, pv in PV.items():
    fT = t2e_two(100.0, L_bulk, PPREC * pv)
    for kn in KAPS:
        TAB[f"{kn}|{vn}"] = run(f"{kn}|{vn}", KAPS[kn], {"Fe": fe_d, "lens": le_d, "T2e": fT}, store=False)
OUT["table"] = TAB
assert abs(TAB["slow_Tasmania|vac1_117main"]["F_heavy"] - R117["table"]["slow_Tasmania|main_single"]["F_heavy"]) < 0.02
# 走査：p_vac 0.1〜1（遅い κ・混合）
SW = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SWEEP = {}
for pv in SW:
    fT = t2e_two(100.0, L_bulk, PPREC * pv)
    for kn in ("slow_Tasmania", "trigger_mix"):
        SWEEP[f"{kn}|{pv}"] = run(f"{kn}|sw{pv}", KAPS[kn], {"Fe": fe_d, "lens": le_d, "T2e": fT}, store=False)["F_heavy"]
OUT["sweep"] = SWEEP
def cross(kn, thr):
    xs = np.array(SW); ys = np.array([SWEEP[f"{kn}|{p}"] for p in SW])
    if ys.min() >= thr: return None
    i = int(np.where(ys >= thr)[0][0]); return float(np.interp(thr, ys[i-1:i+1], xs[i-1:i+1])) if i > 0 else float(xs[0])
OUT["p_vac_crossing"] = {"slow_1.5": cross("slow_Tasmania", 1.5), "slow_1.1": cross("slow_Tasmania", 1.1), "mix_1.1": cross("trigger_mix", 1.1)}

# ---------- 仮説 ----------
H = {}
H["H130a"] = {"class": "confirmatory", "supported": True, "value": "Sprengel/Geissler：ガラス管・水銀・ゴム。前段の機械のポンプは 'do not require to be of any very special construction'（Ram 1894 p.177）",
  "note": "精密に削った部品を必須とする記述なし。Geissler 系の前段の弁は機械で開け閉めするとあり、ピストンのポンプ自体は要る（Sprengel は不要）"}
H["H130b"] = {"class": "confirmatory", "supported": True, "value": "1881 年 6 月に 1 灯 5 時間（Howell & Schroeder 1927）、改良後 30 分", "note": "30 分以上"}
H["H130c"] = {"class": "confirmatory", "supported": True, "value": "'Platinum wire is always used ... coefficient of expansion nearly the same as that of the glass'（Ram 1894 p.33）",
  "note": "理由に加え吹管の炎で溶けず酸化しないこと"}
H["H130d"] = {"class": "confirmatory", "supported": True, "value": "Malignani 1896：機械のポンプ（約 1 mm）＋燐の蒸気、水銀のポンプなし。Novak 灯の臭素 1.5 mm（1894 以前）",
  "note": "事前の知識と重なり独立性は弱い"}
fh = TAB["slow_Tasmania|main"]["F_heavy"]
H["H130e"] = {"class": "prediction", "value": {"p_vac": PV["main"], "F_heavy_slow": fh},
  "rule": "0.5<=p_vac<=0.95 かつ 1.30<=F_heavy(遅いκ)<=1.65", "supported": bool(0.5 <= PV["main"] <= 0.95 and 1.30 <= fh <= 1.65),
  "note": "p_vac の主は登録の追記で置いた推測の積なので範囲に入るのは作りから。非自明なのは F_heavy の範囲だけ"}
H["H130f"] = {"class": "descriptive", "supported": None, "value": "部品の要件の表"}
OUT["hypotheses"] = H; OUT["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
for kn, c in (("slow_Tasmania", "C0"), ("trigger_mix", "C1")):
    axs[0].plot(SW, [SWEEP[f"{kn}|{p}"] for p in SW], "o-", color=c, label=kn)
axs[0].axhline(1.1, color="k", lw=0.5); axs[0].axhline(1.5, color="k", lw=0.5, ls=":")
axs[0].axvline(PV["main"], color="C3", ls="--", lw=0.8, label=f"main p_vac={PV['main']:.2f}")
axs[0].axvspan(q10, q90, color="C3", alpha=0.1)
axs[0].set_xlabel("p_vac (vacuum lamp factor)"); axs[0].set_ylabel("F_heavy (A_Ord/U3)"); axs[0].legend(fontsize=7)
axs[0].set_title(f"118: F_heavy vs p_vac (p_prec 117 main={PPREC:.3f})", fontsize=9)
vn = list(PV); x = np.arange(len(vn))
for j, kn in enumerate(KAPS):
    axs[1].bar(x + (j-1)*0.27, [TAB[f"{kn}|{w}"]["F_heavy"] for w in vn], 0.27, label=kn)
axs[1].set_xticks(x); axs[1].set_xticklabels([f"{w}\n{PV[w]:.2f}" for w in vn], fontsize=6, rotation=25)
axs[1].axhline(1.1, color="k", lw=0.5); axs[1].axhline(1.5, color="k", lw=0.5, ls=":"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_pvac.png"), dpi=120); plt.close(fig)
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("check_117", dP, dT, "p_prec_117main", round(PPREC, 4))
print("Pt_mg_per_lamp(2x1cm)", round(float(Pt_mg), 1), "Hg_kg_small_pump", round(300 * 13.546 / 1000, 2))
for kk, v in PV.items(): print("p_vac", kk, round(v, 4), "p_eff", round(PPREC * v, 4))
print("mc", OUT["p_vac_mc"])
for t, o in TAB.items(): print(t, "F_heavy", round(o["F_heavy"], 4))
for kk, v in SWEEP.items(): print("sweep", kk, round(v, 4))
print("crossing", OUT["p_vac_crossing"])
for kk, hh in H.items(): print(kk, hh["class"], hh["supported"], str(hh.get("value"))[:100])
print("runtime", round(OUT["runtime_s"], 1))
