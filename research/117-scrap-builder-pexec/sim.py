"""研究117：廃材の盤の作り手の実行しきる確率 p_exec（open_problems 274）。
計算のみ、ネット接続なし（資料は sources/ に保存済み）。事前登録：PREREGISTRATION.md（2026-10-24 23:37 JST）。
116 の sim.py を「# ---------- 仮説」の前まで exec（116 の p_prec と表を差 0 で再現、assert）。新しい乱数は seed 20261117。
p_prec = P(ε_a ≤ ε_r) × p_exec（115 の式）の p_exec を、2006 年の Gingery の wiki が挙げた製作記録 9 件の結末で置く。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P116 = os.path.abspath(os.path.join(HERE, "..", "116-t2e-tightest-tolerance", "sim.py"))
src = open(P116, encoding="utf8").read(); cut = src.index("\n# ---------- 仮説")
G = {"__file__": P116, "__name__": "sim116"}
exec(compile(src[:cut], P116, "exec"), G)
R116 = json.load(open(os.path.join(os.path.dirname(P116), "results.json"), encoding="utf8"))
dP = max(abs(G["PP"][k] - R116["pprec_versions"][k]["p_prec"]) for k in R116["pprec_versions"])
dT = max(abs(G["TAB"][t]["F_heavy"] - R116["table"][t]["F_heavy"]) for t in R116["table"])
assert dP < 1e-12 and dT < 1e-12, (dP, dT)
run, KAPS, fe_d, le_d, t2e_two, L_bulk = (G[k] for k in ("run", "KAPS", "fe_d", "le_d", "t2e_two", "L_bulk"))
P_ACC = G["PP"]["main_gradeI_1in"] / 1.0      # 116 主の P(ε_a≤ε_r)（p_exec=1）＝0.912
OUT = {"study": "117-scrap-builder-pexec", "seed": 20261117, "check_116_pprec_maxdiff": dP,
       "check_116_table_maxdiff": dT, "P_acc_116main": P_ACC}

WB = "https://web.archive.org/web/"
# ---------- 製作記録の表（H129e、記述）：wiki の lathe の頁（最終更新 2006-12-14）が挙げた 9 件 ----------
LOGS = [
 {"id": 1, "who": "Lionel Oliver II", "url": WB+"20140208043442/http://backyardmetalcasting.com/lathe7.html", "status": "complete_used",
  "evidence": "Part 7 で盤そのもので面板の木型を削った。'I've kinda lost interest in the lathe'、Part 8 は 'coming as soon as I get to it! 2002' のまま", "accuracy_number": None},
 {"id": 2, "who": "Barry Workman（友人 M. Horgan の頁）", "url": WB+"20111210175317/http://www.dm.net/~lughaid/barry.htm", "status": "complete",
  "evidence": "盤と shaper を完成、盤で機関 2 台と礼砲を作った（1998）", "accuracy_number": None},
 {"id": 3, "who": "W. E. Johns", "url": WB+"20121230212354/http://www.gizmology.net/gingerylathe.htm", "status": "not_started",
  "evidence": "'I hope to build one too'、AutoCAD で描いただけ（2003）", "accuracy_number": None},
 {"id": 4, "who": "John C. Schwytzer", "url": WB+"20120711143953/http://home.earthlink.net/~jschwytzer/lathe.html", "status": "complete",
  "evidence": "'The lathe took me nine months to build'、約 $230、四つ爪チャックなど付属品も（2002）", "accuracy_number": None},
 {"id": 5, "who": "Jayson Craig", "url": WB+"20041210014820/http://foundry.psibersport.com/lathe.html", "status": "stalled",
  "evidence": "2004-10-18 の更新で往復台の段階。途中で市販の 1.5hp の立てフライス盤を買う。2006・2008 の写しは空。往復台の寸法の狂い 0.0005〜0.0015 in を caliper で", "accuracy_number": "0.0005-0.0015 in（未完成の部品）"},
 {"id": 6, "who": "goesworkshop.com", "url": WB+"20050210065346/http://www.goesworkshop.com/lathe.html", "status": "complete",
  "evidence": "'This is my lathe i built around 2 years ago ... i have run the crap out of this thing'（2005）", "accuracy_number": None},
 {"id": 7, "who": "T. Janstrom", "url": WB+"20061128212929/http://www.tjanstrom.com/latheparttwo.html", "status": "stalled",
  "evidence": "2000 年 10 月に滑り面の擦り合わせの段階。2006 年の写しも同じ頁まで", "accuracy_number": None},
 {"id": 8, "who": "daedalus.csgeeks.org/~stock", "url": WB+"20060211003711/http://daedalus.csgeeks.org/~stock/foundry/gingery/", "status": "stalled",
  "evidence": "wiki で '(in progress)'。ベッドと往復台まで、Last Modified June 13, 2002（2006 年の写しも同じ）", "accuracy_number": None},
 {"id": 9, "who": "Donald Qualls", "url": WB+"20130312045009/http://silent1.home.netcom.com/HomebuiltLathe.htm", "status": "unknown",
  "evidence": "Gingery でない自前の設計（友人と 2 台）。目次は主軸・心押し台・ねじ切りの歯車まであるが完成・使用の明記なし", "accuracy_number": None},
]
OUT["logs"] = LOGS
st = [l["status"] for l in LOGS]
n_c = sum(s.startswith("complete") for s in st); n_s = st.count("stalled"); n_ns = st.count("not_started"); n_u = st.count("unknown")
COUNTS = {"complete": n_c, "stalled": n_s, "not_started": n_ns, "unknown": n_u, "N": len(LOGS)}
OUT["counts"] = COUNTS
# 完成の割合の版（途中で止まった＝断念に数える：登録の主）
FR = {
 "main_excl_unknown":     n_c / (len(LOGS) - n_u),                 # 4/8
 "unknown_as_fail":       n_c / len(LOGS),                         # 4/9
 "unknown_as_complete":  (n_c + n_u) / len(LOGS),                  # 5/9
 "excl_nonstarter":       n_c / (len(LOGS) - n_u - n_ns),          # 4/7（始めた者だけ）
 "strict_no_oliver":     (n_c - 1) / (len(LOGS) - n_u),            # 3/8（Oliver を未完成に）
 "stalled_excluded":      n_c / (n_c),                             # 登録の「途中で止まったものを除く版」＝自明に 1
}
OUT["completion_fraction_versions"] = FR

# ---------- p_exec の版：愛好家の 1 回の試み → 崩壊後の群【推測】 ----------
g = np.random.default_rng(20261117)
p1 = FR["main_excl_unknown"]
k, n = n_c, len(LOGS) - n_u
beta = g.beta(k + 0.5, n - k + 0.5, 200_000)                       # Jeffreys の事後
q10, q50, q90 = (float(np.quantile(beta, q)) for q in (0.1, 0.5, 0.9))
OUT["beta_posterior"] = {"k": k, "n": n, "q10": q10, "q50": q50, "q90": q90}
def attempts(p, m, rho=0.0):
    """m 回の試み（世代・家族）。rho は試みどうしの相関（群に共通の失敗の原因の割合）【推測】"""
    return float(rho * p + (1 - rho) * (1 - (1 - p) ** m))
PE = {
 "exec1_116main":     1.0,                       # 116 の主（再現）
 "main_single":       p1,                        # 主：愛好家 1 回の試みの完成の割合 0.5
 "beta_q10":          q10,
 "beta_q90":          q90,
 "strict_3of8":       FR["strict_no_oliver"],
 "excl_nonstarter":   FR["excl_nonstarter"],
 "attempts3_indep":   attempts(p1, 3),           # 3 回の独立な試み【推測】
 "attempts3_rho0.5":  attempts(p1, 3, 0.5),      # 失敗の原因の半分が群に共通【推測】
}
OUT["p_exec_versions"] = PE
PP = {kk: P_ACC * v for kk, v in PE.items()}
OUT["p_prec_versions"] = PP
TAB = {}
for vn, p in PP.items():
    fT = t2e_two(100.0, L_bulk, p)
    for kn in KAPS:
        TAB[f"{kn}|{vn}"] = run(f"{kn}|{vn}", KAPS[kn], {"Fe": fe_d, "lens": le_d, "T2e": fT}, store=False)
OUT["table"] = TAB
assert abs(TAB["slow_Tasmania|exec1_116main"]["F_heavy"] - R116["table"]["slow_Tasmania|main_gradeI_1in"]["F_heavy"]) < 0.02

# ---------- 仮説 ----------
H = {}
fa = FR["main_excl_unknown"]
H["H129a"] = {"class": "confirmatory", "value": fa, "rule": "0.3<=f<=0.9（途中で止まった＝断念）", "supported": bool(0.3 <= fa <= 0.9),
  "range_versions": [min(v for kk, v in FR.items() if kk != "stalled_excluded"), max(v for kk, v in FR.items() if kk != "stalled_excluded")],
  "note": "n=8（不明 1 を除く）。Oliver の使ったが仕上げを断念した例を完成に数えた。版 0.375〜0.571 すべて範囲内"}
fb = (n_s + n_ns) / len(LOGS)
H["H129b"] = {"class": "confirmatory", "value": fb, "rule": ">=0.2", "supported": bool(fb >= 0.2),
  "note": "途中で止まった 4＋始めていない 1。始めていない者を除いても 4/9=0.44"}
acc_c = sum(1 for l in LOGS if l["status"].startswith("complete") and l["accuracy_number"])
H["H129c"] = {"class": "confirmatory", "value": acc_c / n_c, "rule": "< 0.5", "supported": bool(acc_c / n_c < 0.5),
  "note": "完成 4 件で精度の数値 0。数値は未完成の Craig の 1 件だけ"}
fh = TAB["slow_Tasmania|main_single"]["F_heavy"]
H["H129d"] = {"class": "prediction", "value": fh, "rule": "1.5<=F_heavy(遅いκ)<=2.2", "supported": bool(1.5 <= fh <= 2.2),
  "note": "下がるのは自明、範囲に入るかだけが非自明"}
H["H129e"] = {"class": "descriptive", "supported": None, "value": "製作記録の表"}
OUT["hypotheses"] = H; OUT["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
cols = {"complete": "C2", "complete_used": "C2", "stalled": "C3", "not_started": "C7", "unknown": "C0"}
for l in LOGS:
    axs[0].barh(l["id"], 1, color=cols[l["status"]]); axs[0].text(0.02, l["id"], f'{l["id"]} {l["status"]}', va="center", fontsize=8, color="w")
axs[0].set_xticks([]); axs[0].invert_yaxis(); axs[0].set_title(f"117: 9 Gingery lathe build logs (wiki list 2006): complete {n_c}/{len(LOGS)-n_u}", fontsize=9)
vn = list(PP); x = np.arange(len(vn))
for j, kn in enumerate(KAPS):
    axs[1].bar(x + (j-1)*0.27, [TAB[f"{kn}|{w}"]["F_heavy"] for w in vn], 0.27, label=kn)
axs[1].set_xticks(x); axs[1].set_xticklabels([f"{w}\np_exec={PE[w]:.2f}" for w in vn], fontsize=6, rotation=20); axs[1].axhline(1.1, color="k", lw=0.5); axs[1].axhline(1.5, color="k", lw=0.5, ls=":")
axs[1].set_ylabel("F_heavy (A_Ord/U3)"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_pexec.png"), dpi=120); plt.close(fig)
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("check_116", dP, dT, "P_acc", round(P_ACC, 4))
print("counts", COUNTS)
for kk, v in FR.items(): print("frac", kk, round(v, 4))
print("beta", OUT["beta_posterior"])
for kk in PE: print("p_exec", kk, round(PE[kk], 4), "p_prec", round(PP[kk], 4))
for t, o in TAB.items(): print(t, "F_heavy", round(o["F_heavy"], 4))
for kk, hh in H.items(): print(kk, hh["class"], hh["supported"], str(hh.get("value"))[:80])
print("runtime", round(OUT["runtime_s"], 1))
