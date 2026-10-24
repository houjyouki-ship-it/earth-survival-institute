"""研究119：T2e と読み取り装置に封じた真空の器具が要るか（open_problems 280）。
計算のみ、ネット接続なし（資料は sources/ に保存済み）。事前登録：PREREGISTRATION.md（2026-10-25 03:38 JST）。
118 の sim.py を「# ---------- 仮説」の前まで exec（118 の p_vac と表を差 0 で再現、assert）。
読み取りの年数は公開の速度（Z3 5–10 Hz、Mark I 3 加算/秒、Colossus 5,000 字/秒×5 ビット、光導電セルの遅れ約 10 ms）と
018 の点の数（1 コマ 8.8M 点・2 MB、1 リール 120 GB）から計算する。部品の確率はすべて【推測】。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P118 = os.path.abspath(os.path.join(HERE, "..", "118-t2e-vacuum-pvac", "sim.py"))
src = open(P118, encoding="utf8").read(); cut = src.index("\n# ---------- 仮説")
G = {"__file__": P118, "__name__": "sim118"}
exec(compile(src[:cut], P118, "exec"), G)
R118 = json.load(open(os.path.join(os.path.dirname(P118), "results.json"), encoding="utf8"))
dP = max(abs(G["PV"][k] - R118["p_vac_versions"][k]) for k in R118["p_vac_versions"])
dT = max(abs(G["TAB"][t]["F_heavy"] - R118["table"][t]["F_heavy"]) for t in R118["table"])
assert dP < 1e-12 and dT < 1e-12, (dP, dT)
run, KAPS, fe_d, le_d, t2e_two, L_bulk, PPREC, pvac118, BASE118 = (G[k] for k in ("run", "KAPS", "fe_d", "le_d", "t2e_two", "L_bulk", "PPREC", "pvac", "BASE"))
OUT = {"study": "119-t2e-reader-vacuum", "check_118_pvac_maxdiff": dP, "check_118_table_maxdiff": dT, "p_prec_117main": PPREC}

# ---------- 1. 読み取りの量と速度（事実は sources/、並列の幅と部品数は【推測】）----------
PX_FRAME, B_FRAME, REEL_B = 8.8e6, 2e6, 120e9            # 018（Piql）
PX_REEL = REEL_B / B_FRAME * PX_FRAME                    # 5.28e11 点
YR = 365.25 * 86400
RATE = {  # 1 チャンネルが 1 秒に判定できる点の数
 "relay_Z3_5Hz": 5.0, "relay_Z3_10Hz": 10.0, "relay_MarkI_3add": 3.0,
 "photoresistor_10ms": 100.0,                            # 遅れ約 10 ms → 上限 100/秒（暗への戻りは最大 1 秒）
 "tube_Colossus_char": 5000.0, "tube_Colossus_bit": 25000.0}
single = {k: PX_REEL / r / YR for k, r in RATE.items()}  # 1 チャンネル 1 台での年数
OUT["reel_pixels"] = PX_REEL; OUT["single_channel_years"] = single
# 並列にして T 年で 1 リールを読むのに要るチャンネル数と部品数（1 チャンネルあたり C 個の継電器・管、【推測】20〜100）
def need(rate, T, C): ch = PX_REEL / (rate * T * YR); return ch, ch * C
PAR = {}
for k in ("relay_Z3_10Hz", "relay_Z3_5Hz", "tube_Colossus_char"):
    for T in (1, 5, 20, 100):
        for C in (20, 100):
            ch, n = need(RATE[k], T, C); PAR[f"{k}|T{T}|C{C}"] = {"channels": ch, "components": n}
OUT["parallel"] = PAR
# 優先データ 10 MB（018）と、q=0.3（024 の非圧縮の割合）の 36 GB
for lab, b in (("prio10MB", 10e6), ("q0.3_36GB", 0.3 * REEL_B)):
    px = b / B_FRAME * PX_FRAME
    OUT[f"years_{lab}"] = {k: px / r / YR for k, r in RATE.items()}

# ---------- 2. 部品ごとの真空の要否（H131g、記述）----------
PARTS = [
 {"part": "光源", "no_vac": "炭素のアーク灯（棒は空気中）、日光", "vac": "白熱電球（118）", "fact": "Wikipedia Arc lamp：'carbon rods in free air'"},
 {"part": "光電素子", "no_vac": "光導電セル（遅れ約 10 ms、暗への戻り最大 1 秒）", "vac": "光電管（Colossus の 8 つの photocell の読み取り）", "fact": "Wikipedia Photoresistor・Colossus computer"},
 {"part": "論理", "no_vac": "継電器（Z3 2,600 個・5–10 Hz、Mark I 3 加算/秒）", "vac": "高真空の管（Colossus 1,600〜2,400 本、5,000 字/秒）", "fact": "Wikipedia Z3・Harvard Mark I・Colossus"},
 {"part": "高真空の作り", "no_vac": "—", "vac": "Geissler の水銀ポンプでは部分の真空（de Forest、青い光）。高真空は Arnold（AT&T）と Langmuir（GE、Gaede の拡散ポンプの改良）、GE は 1915 年に hard vacuum の三極管。今は barium の getter", "fact": "Wikipedia Vacuum tube・Getter"},
]
OUT["parts"] = PARTS

# ---------- 3. p_vac の版（部品の確率はすべて【推測】）----------
# 真空管の経路：118 の排気（水銀 or 機械＋燐）× 封じ × ガラス × 高真空 × 陰極 × getter（電球のフィラメントの代わりに陰極）
TB = {"p_Hg": BASE118["p_Hg"], "p_mech1mm": BASE118["p_mech1mm"], "p_P": BASE118["p_P"], "p_seal": BASE118["p_seal"], "p_glass": BASE118["p_glass"],
      "p_hard_Hg": 0.8,    # Sprengel は 1 mPa 未満（118 の資料）だが、管の量産の高真空は拡散ポンプで実現
      "p_hard_mech": 0.2,  # 機械 1 mm＋燐で管の高真空に届くかは資料なし
      "p_cath": 0.8, "p_getter": 0.8}
def ptube(d):
    pump = 1 - (1 - d["p_Hg"] * d["p_hard_Hg"]) * (1 - d["p_mech1mm"] * d["p_P"] * d["p_hard_mech"])
    return float(pump * d["p_seal"] * d["p_glass"] * d["p_cath"] * d["p_getter"])
def tmod(**kw): d = dict(TB); d.update(kw); return d
P_RELAY = 0.8   # 継電器＋光導電セル＋アーク灯の経路が組めて動く確率【推測】（銅線・ばね・接点、電源は p_prec 側）
PV = {
 "pvac118_bulb": float(R118["p_vac_versions"]["main"]),          # 118 の主（再現）
 "novac_upper": 1.0,                                               # 真空の器具なし、確率 1
 "relay_only": P_RELAY,
 "tube_only_main": ptube(TB),
 "tube_Hg0.2": ptube(tmod(p_Hg=0.2)), "tube_Hg0.8": ptube(tmod(p_Hg=0.8)),
 "tube_low": ptube(tmod(p_Hg=0.2, p_hard_Hg=0.5, p_hard_mech=0.0, p_cath=0.5, p_getter=0.5, p_seal=0.7, p_glass=0.5)),
 "tube_high": ptube(tmod(p_Hg=0.8, p_hard_Hg=0.95, p_hard_mech=0.5, p_cath=0.95, p_getter=0.95, p_seal=0.99, p_glass=0.97)),
}
PV["either_main"] = 1 - (1 - P_RELAY) * (1 - PV["tube_only_main"])
PV["either_relay0.5"] = 1 - (1 - 0.5) * (1 - PV["tube_only_main"])
PV["either_relay0.2"] = 1 - (1 - 0.2) * (1 - PV["tube_only_main"])
OUT["tube_inputs"] = TB; OUT["p_relay_main"] = P_RELAY; OUT["p_vac_versions"] = PV

TAB = {}
for vn, pv in PV.items():
    fT = t2e_two(100.0, L_bulk, PPREC * pv)
    for kn in KAPS:
        TAB[f"{kn}|{vn}"] = run(f"{kn}|{vn}", KAPS[kn], {"Fe": fe_d, "lens": le_d, "T2e": fT}, store=False)
OUT["table"] = TAB
assert abs(TAB["slow_Tasmania|pvac118_bulb"]["F_heavy"] - R118["table"]["slow_Tasmania|main"]["F_heavy"]) < 0.02

# ---------- 仮説 ----------
H = {}
H["H131a"] = {"class": "confirmatory", "supported": True, "value": "Z3 5–10 Hz（2,600 継電器）、Mark I 3 加算/秒"}
H["H131b"] = {"class": "confirmatory", "supported": True, "value": "Colossus 5,000 字/秒（試験で 9,700、5 並列で実効 25,000）"}
H["H131c"] = {"class": "confirmatory", "supported": True, "value": "de Forest は Geissler の水銀ポンプで hard vacuum でない。Arnold・Langmuir（Gaede の拡散ポンプ 1915 の改良）、GE の hard vacuum 三極管 1915"}
H["H131d"] = {"class": "confirmatory", "supported": True, "value": "光導電セルの遅れ約 10 ms、暗へは最大 1 秒", "note": "記事は光導電セル一般でセレンを名指ししない"}
s = single
H["H131e"] = {"class": "prediction", "value": {k: round(v, 2) for k, v in s.items()},
  "rule": "tube_Colossus_char < 20 年 かつ relay_Z3_10Hz > 500 年",
  "supported": bool(s["tube_Colossus_char"] < 20 and s["relay_Z3_10Hz"] > 500),
  "note": "1 チャンネルでの話。並列にすると継電器でも 20 年で読める（parallel の表）ので、『真空のない経路は足りない』という結論は支持されない"}
fh = TAB["slow_Tasmania|tube_only_main"]["F_heavy"]
H["H131f"] = {"class": "prediction", "value": {"p_vac_tube": PV["tube_only_main"], "F_heavy_slow": fh},
  "rule": "0.25<=p_vac_tube<=0.50 かつ 1.20<=F_heavy<=1.40",
  "supported": bool(0.25 <= PV["tube_only_main"] <= 0.50 and 1.20 <= fh <= 1.40),
  "note": "p_vac の値は推測の積で範囲に入るのは作りから。非自明なのは F_heavy の範囲だけ"}
H["H131g"] = {"class": "descriptive", "supported": None, "value": "部品の表"}
OUT["hypotheses"] = H; OUT["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
ks = list(RATE); axs[0].barh(ks, [single[k] for k in ks], color=["C3" if "relay" in k else ("C2" if "photo" in k else "C0") for k in ks])
axs[0].set_xscale("log"); axs[0].axvline(20, color="k", lw=0.6, ls=":"); axs[0].axvline(500, color="k", lw=0.6)
axs[0].set_xlabel("years to scan one 120 GB reel, single channel"); axs[0].set_title("119: reader throughput (no vacuum vs tube)", fontsize=9)
vn = list(PV); x = np.arange(len(vn))
for j, kn in enumerate(KAPS):
    axs[1].bar(x + (j-1)*0.27, [TAB[f"{kn}|{w}"]["F_heavy"] for w in vn], 0.27, label=kn)
axs[1].set_xticks(x); axs[1].set_xticklabels([f"{w}\n{PV[w]:.2f}" for w in vn], fontsize=6, rotation=25)
axs[1].axhline(1.1, color="k", lw=0.5); axs[1].axhline(1.5, color="k", lw=0.5, ls=":"); axs[1].legend(fontsize=7)
axs[1].set_ylabel("F_heavy (A_Ord/U3)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_reader_vac.png"), dpi=120); plt.close(fig)
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("check_118", dP, dT)
print("reel_pixels", PX_REEL)
for k, v in single.items(): print("single_years", k, round(v, 2))
for k, v in PAR.items(): print("parallel", k, round(v["channels"], 1), round(v["components"]))
print("q0.3", {k: round(v, 1) for k, v in OUT["years_q0.3_36GB"].items()})
print("prio10MB", {k: round(v, 4) for k, v in OUT["years_prio10MB"].items()})
for k, v in PV.items(): print("p_vac", k, round(v, 4))
for t, o in TAB.items(): print(t, "F_heavy", round(o["F_heavy"], 4))
for kk, hh in H.items(): print(kk, hh["class"], hh["supported"], str(hh.get("value"))[:120])
print("runtime", round(OUT["runtime_s"], 1))
