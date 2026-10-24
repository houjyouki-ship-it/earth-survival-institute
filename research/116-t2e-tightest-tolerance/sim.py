"""研究116：T2e の段に 0.0001 in 級の部品が要るか（open_problems 273）。
計算のみ、ネット接続なし（資料は sources/ に保存済み）。事前登録：PREREGISTRATION.md（2026-10-24 21:42 JST）。
115 の sim.py を「# ---------- 仮説」の前まで exec（115 の p_prec と表を差 0 で再現、assert）。新しい乱数は seed 20261116。
ε_r の下端 lo_r を、資料で見つかった最も厳しい「代わりのきかない」部品の公差に置き直す【独自仮説】。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P115 = os.path.abspath(os.path.join(HERE, "..", "115-scrap-precision-pprec", "sim.py"))
src = open(P115, encoding="utf8").read(); cut = src.index("\n# ---------- 仮説")
G = {"__file__": P115, "__name__": "sim115"}
exec(compile(src[:cut], P115, "exec"), G)
R115 = json.load(open(os.path.join(os.path.dirname(P115), "results.json"), encoding="utf8"))
dP = max(abs(G["PP"][k] - R115["pprec_versions"][k]["p_prec"]) for k in R115["pprec_versions"])
dT = max(abs(G["TAB"][t]["F_heavy"] - R115["table"][t]["F_heavy"]) for t in R115["table"])
assert dP < 1e-12 and dT < 1e-12, (dP, dT)
run, KAPS, fe_d, le_d, t2e_two, L_bulk = (G[k] for k in ("run", "KAPS", "fe_d", "le_d", "t2e_two", "L_bulk"))
IN = 25.4e-3
OUT = {"study": "116-t2e-tightest-tolerance", "seed": 20261116, "check_115_pprec_maxdiff": dP, "check_115_table_maxdiff": dT}

# ---------- 部品ごとの公差の表（H128e、記述） ----------
U = "Unwin 1909 The Elements of Machine Design（https://archive.org/details/elementsofmachin00unwirich）"
PARTS = [
 {"part": "発電機の空隙（半径の機械的な隙間、最小：歯つき・穴あき電機子）", "year": 1901, "in": 1/32, "substitutable": False,
  "src": "Wiener 1901 Practical calculation of dynamo-electric machines §58 'ranges between 1/32 and 7/16 inch'（https://archive.org/details/practicalcalcula00wieniala）"},
 {"part": "発電機の空隙（同、最大）", "year": 1901, "in": 7/16, "substitutable": False, "src": "Wiener 1901 同上"},
 {"part": "発電機の機械的な隙間（設計例）", "year": 1891, "in": 0.14, "substitutable": False,
  "src": "Urquhart 1891 Dynamo construction（https://archive.org/details/dynamoconstructi00urqurich）'allowing ·14 inch mechanical clearance'"},
 {"part": "電機子軸の端の遊び（最小）", "year": 1891, "in": 0.25, "substitutable": False, "src": "Urquhart 1891 'end-play of at least one-fourth of an inch'"},
 {"part": "平軸受の走りばめ・最上の等級 I・軸径 3 in の軸の公差", "year": 1909, "in": 0.0018, "substitutable": False, "src": U + " §§ Running fits（British Standards Committee の表の抜粋）"},
 {"part": "同・軸径 9 in", "year": 1909, "in": 0.0030, "substitutable": False, "src": U},
 {"part": "圧入の締め代（直径 1 in あたり）", "year": 1909, "in": 0.001, "substitutable": False,
  "src": "Kimball & Barr 1909 Elements of machine design（https://archive.org/details/elementsofmachin01kimb）'.001\" per inch of diameter is often allowed'"},
 {"part": "玉軸受の玉の寸法の揃い", "year": 1909, "in": 1e-4, "substitutable": True,
  "src": U + " 'an accuracy to 1/10,000th inch is usual'。発電機の設計書 6 冊（1891〜1910）の本文に ball bearing の語は 0 回、軸受は babbitt・brass・ring-oil の平軸受"},
]
OUT["parts"] = PARTS
nonsub = [p for p in PARTS if not p["substitutable"]]
tight = min(nonsub, key=lambda p: p["in"])
# 等級 I の公差を軸径で冪に外挿して 1 in の軸に（2 点の当てはめ）【推測】
b = np.log(0.0030/0.0018)/np.log(9/3); tol1in = 0.0018*(1/3)**b
OUT["gradeI_exponent"] = float(b); OUT["gradeI_tol_1in_in"] = float(tol1in)
OUT["tightest_nonsub"] = {"part": tight["part"], "in": tight["in"]}

# ---------- p_prec の版（lo_r を置き直す） ----------
g = np.random.default_rng(20261116); M = 400_000
def pprec(med_a, sig_a, lo_r, hi_r, p_exec=1.0, w_ball=0.0):
    ea = med_a * np.exp(sig_a * g.standard_normal(M))
    lo = np.where(g.random(M) < w_ball, 1e-4*IN, lo_r)   # 確率 w_ball で玉軸受級の部品が要る
    er = np.exp(np.log(lo) + g.random(M) * (np.log(hi_r) - np.log(lo)))
    return float(np.mean(ea <= er) * p_exec)
PV = {
 "main_gradeI_1in":   (0.001*IN, 1.0, tol1in*IN, IN/16, 1.0, 0.0),  # 主：最も厳しい代わりのきかない部品＝等級 I の 1 in の軸（外挿）
 "gradeI_3in":        (0.001*IN, 1.0, 0.0018*IN, IN/16, 1.0, 0.0),  # 外挿なし（資料の最小の軸径 3 in）
 "half_gradeI":       (0.001*IN, 1.0, 0.5*tol1in*IN, IN/16, 1.0, 0.0),  # 小型の機械で公差を半分に【推測】
 "airgap_only":       (0.001*IN, 1.0, IN/32, 7/16*IN, 1.0, 0.0),   # 発電機の空隙だけが縛り
 "ball_w0.1":         (0.001*IN, 1.0, tol1in*IN, IN/16, 1.0, 0.1), # 1 割の群で玉軸受が要る【推測】
 "pess_main":         (0.004*IN, 1.5, tol1in*IN, IN/16, 1.0, 0.0), # 115 の悲観の届く精度
 "pess_exec0.5":      (0.004*IN, 1.5, tol1in*IN, IN/16, 0.5, 0.0),
}
PP = {k: pprec(*v) for k, v in PV.items()}
OUT["pprec_versions"] = {k: {"params_mm": [float(x) for x in PV[k]], "p_prec": PP[k]} for k in PV}

TAB = {}
for vn, p in PP.items():
    fT = t2e_two(100.0, L_bulk, p)
    for kn in KAPS:
        TAB[f"{kn}|{vn}"] = run(f"{kn}|{vn}", KAPS[kn], {"Fe": fe_d, "lens": le_d, "T2e": fT}, store=False)
OUT["table"] = TAB

# ---------- 仮説 ----------
H = {}
gap_min = min(p["in"] for p in PARTS if "空隙" in p["part"] or "機械的な隙間" in p["part"])
H["H128a"] = {"class": "confirmatory", "value_in": gap_min, "rule": ">=1/32 in", "supported": bool(gap_min >= 1/32 - 1e-12),
  "note": "Wiener・Urquhart の 2 冊。Atwood 1893 の値は OCR が読めず数えない"}
brg = min(p["in"] for p in PARTS if "平軸受" in p["part"])
H["H128b"] = {"class": "confirmatory", "value_in": brg, "extrap_1in": float(tol1in), "rule": ">=0.0005 in", "supported": bool(brg >= 0.0005),
  "note": "資料の値は等級 I の公差（3 in 軸）。1 in への外挿でも 0.0011 in"}
H["H128c"] = {"class": "confirmatory", "value_in": tight["in"], "part": tight["part"], "rule": ">=0.0005 in（0.0002 in 以下の必須部品なし）",
  "supported": bool(tight["in"] >= 0.0005), "note": "0.0001 in 級は玉軸受の玉だけで、発電機の設計書は平軸受を使う"}
v = PP["main_gradeI_1in"]
H["H128d"] = {"class": "prediction", "value": v, "rule": "|p-0.903| < |p-0.642|", "supported": bool(abs(v-0.903) < abs(v-0.642)),
  "note": "lo_r が 0.001 in 以上になったので登録の自明さの点検により自明（構造から決まる）。支持に数えない"}
H["H128e"] = {"class": "descriptive", "supported": None, "value": "部品の公差の表"}
OUT["hypotheses"] = H; OUT["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
for i, p in enumerate(sorted(PARTS, key=lambda p: p["in"])):
    axs[0].barh(i, p["in"]*IN, color="C1" if p["substitutable"] else "C0")
axs[0].set_xscale("log"); axs[0].set_yticks(range(len(PARTS)))
axs[0].set_yticklabels([f"{p['year']} {p['in']:.4g} in" for p in sorted(PARTS, key=lambda p: p["in"])], fontsize=7)
axs[0].axvline(0.0005*IN, color="k", lw=0.6, ls="--"); axs[0].axvline(0.001*IN, color="g", lw=0.6)
axs[0].set_xlabel("mm (orange = substitutable: ball bearing)"); axs[0].set_title("116: first-generation T2e part tolerances", fontsize=9)
vn = list(PP); x = np.arange(len(vn))
for j, kn in enumerate(KAPS):
    axs[1].bar(x + (j-1)*0.27, [TAB[f"{kn}|{w}"]["F_heavy"] for w in vn], 0.27, label=kn)
axs[1].set_xticks(x); axs[1].set_xticklabels([f"{w}\np={PP[w]:.2f}" for w in vn], fontsize=6.5); axs[1].axhline(1.1, color="k", lw=0.5)
axs[1].set_ylabel("F_heavy (A_Ord/U3)"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_tolerance.png"), dpi=120); plt.close(fig)
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("check_115", dP, dT)
print("tightest_nonsub", tight["part"], tight["in"], "gradeI 1in", round(tol1in, 5), "b", round(b, 3))
for k, w in PP.items(): print("p_prec", k, round(w, 4))
for t, o in TAB.items(): print(t, "F_heavy", round(o["F_heavy"], 4))
for k, hh in H.items(): print(k, hh["class"], hh["supported"], str(hh.get("value", hh.get("value_in")))[:100])
print("runtime", round(OUT["runtime_s"], 1))
