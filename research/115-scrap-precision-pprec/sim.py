"""研究115：廃材から作った盤・平面が T2e の精度に届く確率 p_prec（open_problems 272）。
計算のみ、ネット接続なし（資料は sources/ に保存済み）。事前登録：PREREGISTRATION.md（2026-10-24 19:36 JST）。
114 の sim.py を「# ---------- 仮説」の前まで exec（114 の表を差 0 で再現、assert）。新しい乱数は seed 20261115。
p_prec = P(ε_a ≤ ε_r) × p_exec【独自仮説】。ε_a：廃材・手作業の方法で届く精度、ε_r：T2e の縛りの部品に要る精度。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P114 = os.path.abspath(os.path.join(HERE, "..", "114-scrap-tool-lowerbound", "sim.py"))
src = open(P114, encoding="utf8").read(); cut = src.index("\n# ---------- 仮説")
G = {"__file__": P114, "__name__": "sim114"}
exec(compile(src[:cut], P114, "exec"), G)
R114 = json.load(open(os.path.join(os.path.dirname(P114), "results.json"), encoding="utf8"))
d114 = max(abs(G["TAB"][t]["F_heavy"] - R114["table"][t]["F_heavy"]) for t in R114["table"])
assert d114 < 1e-12, d114
run, KAPS, fe_d, le_d, t2e_two, L_bulk = (G[k] for k in ("run", "KAPS", "fe_d", "le_d", "t2e_two", "L_bulk"))
OUT = {"study": "115-scrap-precision-pprec", "seed": 20261115, "check_114_table_reproduced_maxdiff": d114}
IN = 25.4e-3  # mm per inch

# ---------- 資料の精度の表（H127f、記述） ----------
ACH = [  # 届く精度（手作業・廃材の方法）
 {"what": "Gingery 式の盤（廃材の Al の鋳物、きさげ）で試し削り 12 in の先細り", "year": 2013, "mm": 0.0005*IN, "over_mm": 12*IN,
  "src": "https://www.madmodder.net/index.php?topic=8191.25（vtsteam、dial caliper で測定、'less than 1/2 a thou'。電動機・pillow bearings は市販）"},
 {"what": "Maudslay の卓上の micrometer の読み", "year": 1800, "mm": 0.0001*IN, "over_mm": None, "src": "https://en.wikipedia.org/wiki/Henry_Maudslay"},
 {"what": "Nasmyth：Maudslay の測り方は千分の一インチを測れ、それは機械の要求を越えていた", "year": 1830, "mm": 0.001*IN, "over_mm": None, "src": "Roe 1916（https://www.gutenberg.org/cache/epub/72046/pg72046.txt）"},
 {"what": "Whitworth の end measurement の絶対精度（分解能は百万分の一インチ）", "year": 1840, "mm": 500e-6, "over_mm": None, "src": "https://en.wikipedia.org/wiki/Joseph_Whitworth"},
 {"what": "三面の擦り合わせの平面（'millionths of an inch'）", "year": 1830, "mm": 1e-6*IN, "over_mm": None, "src": "https://en.wikipedia.org/wiki/Machine_tool"},
]
REQ = [  # 要る精度（T2e の最初の世代）
 {"what": "Watt の 18 in のシリンダー（失敗：長径と短径の差）", "year": 1769, "mm": 0.375*IN, "ok": False, "src": "Roe 1916"},
 {"what": "Wilkinson の 50 in のシリンダー（商業の成功：'thickness of an old shilling'）", "year": 1776, "mm": IN/16, "ok": True,
  "src": "Roe 1916（Boulton の言葉）＋ 1 シリングの厚さ約 1/16 in（https://chestofbooks.com/reference/Wonder-Book-Of-Knowledge/The-Beginning-Of-Precision-In-Mechanics.html）"},
 {"what": "Nasmyth：千分の一インチは 'quite beyond all the requirements of engineering mechanism'（1830〜40 年代の見方）", "year": 1840, "mm": 0.001*IN, "ok": True, "src": "Roe 1916"},
 {"what": "現代の旋盤の中ぐりで容易に保てる公差（参考）", "year": 2020, "mm": 0.25, "ok": True, "src": "https://en.wikipedia.org/wiki/Boring_(manufacturing)"},
]
OUT["achievable"] = ACH; OUT["required"] = REQ

# ---------- p_prec の版 ----------
g = np.random.default_rng(20261115); M = 400_000
def pprec(med_a, sig_a, lo_r, hi_r, p_exec=1.0):
    ea = med_a * np.exp(sig_a * g.standard_normal(M))
    er = np.exp(g.uniform(np.log(lo_r), np.log(hi_r), M))
    return float(np.mean(ea <= er) * p_exec)
PV = {
 # 主：届く精度の中央値 0.001 in（Gingery の報告 0.0005 in の 2 倍、選ばれた成功例の補正）【推測】、σ 1.0。
 #     要る精度は Nasmyth の 0.001 in（要求の下限）〜 Wilkinson の 1/16 in の対数一様【推測】。
 "main":            (0.001*IN, 1.0, 0.001*IN, IN/16, 1.0),
 "report_median":   (0.0005*IN, 1.0, 0.001*IN, IN/16, 1.0),   # 報告そのまま
 "pess_4x":         (0.004*IN, 1.5, 0.001*IN, IN/16, 1.0),    # 成功例の偏りを 8 倍に、幅も広く
 "rolling_bearing": (0.001*IN, 1.0, 0.0001*IN, IN/16, 1.0),   # T2e に転がり軸受級（十分の一 thou）を要する部品がある【推測、資料なし】
 "main_exec0.5":    (0.001*IN, 1.0, 0.001*IN, IN/16, 0.5),    # 手順を実行しきる群が半分【推測】
 "worst":           (0.004*IN, 1.5, 0.0001*IN, IN/16, 0.5),
}
PP = {k: pprec(*v) for k, v in PV.items()}
OUT["pprec_versions"] = {k: {"params_mm": list(PV[k]), "p_prec": PP[k]} for k in PV}

# ---------- 114 の模型に当てる（N_small=100, L_scrap=L_bulk） ----------
TAB = {}
for vn, p in PP.items():
    fT = t2e_two(100.0, L_bulk, p)
    for kn in KAPS:
        TAB[f"{kn}|{vn}"] = run(f"{kn}|{vn}", KAPS[kn], {"Fe": fe_d, "lens": le_d, "T2e": fT}, store=False)
OUT["table"] = TAB

# ---------- 仮説 ----------
H = {}
H["H127a"] = {"class": "confirmatory", "supported": True,
  "value": "≤25 µm の記述 4 件（Maudslay の micrometer 0.0001 in・1800 年ごろ、Nasmyth の千分の一 in、三面の平面 'millionths'、Whitworth 1840 の絶対 約 500 nm）。ただし測る精度で、作る精度の数値ではない"}
H["H127b"] = {"class": "confirmatory", "supported": True,
  "value": "蒸気機関の 2 件：Wilkinson 1/16 in=1.59 mm（成功）、Watt 3/8 in=9.5 mm（失敗）はどちらも ≥0.1 mm。発電機の空隙は開いて確かめた資料がなく検定できず"}
H["H127c"] = {"class": "confirmatory", "supported": True,
  "value": "Wikipedia に数値なし。検索で開いて確かめた実測は 1 件（madmodder の vtsteam、<0.0005 in/12 in）。Gingery 本人の 0.001 in は無許可の複製の PDF の検索の抜粋でしか見えず、開いていないので数えない"}
v = PP["main"]; f = TAB["slow_Tasmania|main"]["F_heavy"]
H["H127d"] = {"class": "prediction", "value": {"p_prec": v, "F_slow": f}, "rule": "p_prec>=0.5 and F_slow>=1.5", "supported": bool(v >= 0.5 and f >= 1.5)}
f2 = TAB["slow_Tasmania|main_exec0.5"]["F_heavy"]
H["H127e"] = {"class": "prediction", "value": f2, "rule": ">=1.1", "supported": bool(f2 >= 1.1), "note": "p_prec の主が 0.2 以上なので p_prec≥0.1 の構造から自明に近い"}
H["H127f"] = {"class": "descriptive", "supported": None, "value": "精度の表"}
OUT["hypotheses"] = H; OUT["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
for a in ACH: axs[0].scatter(a["year"], a["mm"], marker="o", color="C0")
for r in REQ: axs[0].scatter(r["year"], r["mm"], marker="s" if r["ok"] else "x", color="C3")
axs[0].set_yscale("log"); axs[0].set_ylabel("mm"); axs[0].set_title("115: achievable (blue) vs required (red; x=failed)", fontsize=9)
vn = list(PP); x = np.arange(len(vn))
for j, kn in enumerate(KAPS):
    axs[1].bar(x + (j - 1) * 0.27, [TAB[f"{kn}|{v}"]["F_heavy"] for v in vn], 0.27, label=kn)
axs[1].set_xticks(x); axs[1].set_xticklabels([f"{v}\np={PP[v]:.2f}" for v in vn], fontsize=7); axs[1].axhline(1.1, color="k", lw=0.5)
axs[1].set_ylabel("F_heavy (A_Ord/U3)"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_pprec.png"), dpi=120); plt.close(fig)
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("check_114", d114)
for k, v in PP.items(): print("p_prec", k, round(v, 4))
for t, o in TAB.items(): print(t, "F_heavy", round(o["F_heavy"], 4))
for k, hh in H.items(): print(k, hh["class"], hh["supported"], str(hh["value"])[:120])
print("runtime", round(OUT["runtime_s"], 1))
