"""研究124：時刻と送りを担うカム・歩進の切替器の摩耗（open_problems 290）。系列のまとめの回。
計算のみ、ネット接続なし（資料は sources/ に保存済み）。事前登録：PREREGISTRATION.md（2026-10-25 13:43 JST）。
123 の sim.py を「# ---------- 仮説」の前まで exec し、123 の E[p_relay] と F_heavy を差 0 で再現（assert）。
事実（sources/）：Finder 26 系の歩進の継電器（照明用の現行品）は機械的寿命 300·10³ 回、定格負荷の電気的寿命 100·10³ 回（finder26.pdf）。
Telephone exchange：Strowger の step-by-step の局は掃除などの continuous maintenance を要した。Crossbar switch：設計の基準は 40 年で保守の停止 2 時間で、
earlier electromechanical systems より大きく改善。点火の接点（カムで開閉し負荷を切る）の交換までの開閉 3.0×10⁷〜1.2×10⁸ 回（123 の sources）。
カム・歩進の寿命の分布、手作りの倍率、部品の重み w（継電器何個分の手間か）、1 点あたりの機械の動作数は【推測】。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P123 = os.path.abspath(os.path.join(HERE, "..", "123-dry-contact-life", "sim.py"))
src = open(P123, encoding="utf8").read(); cut = src.index("\n# ---------- 仮説")
G = {"__file__": P123, "__name__": "sim123"}
exec(compile(src[:cut], P123, "exec"), G)
R123 = json.load(open(os.path.join(os.path.dirname(P123), "results.json"), encoding="utf8"))
dE = max(abs(G["EXP"][k]["mean"] - R123["expected_prelay"][k]["mean"]) for k in R123["expected_prelay"])
dF = max(abs(G["TAB"][t]["F_heavy"] - R123["table"][t]["F_heavy"]) for t in R123["table"])
assert dE < 1e-12, dE
G120 = G["G"]["G120"]
from_design, DES, P_TUBE, DV = G["from_design"], G["DES"], G["P_TUBE"], G["DV"]
run, KAPS, fe_d, le_d, t2e_two, L_bulk, PPREC = (G[k] for k in ("run", "KAPS", "fe_d", "le_d", "t2e_two", "L_bulk", "PPREC"))
prelay, demand, S, TB, DATA = G120["prelay"], G120["demand"], G120["S"], G120["TB"], G120["DATA"]
OUT = {"study": "124-cam-stepper-wear", "check_123_Eprelay_maxdiff": dE, "check_123_table_F_maxdiff": dF}
PX = DATA["reel_120GB"]; OUT["points_per_reel"] = PX

# ---------- 1. 機械の部品の摩耗を同じ物差しで ----------
def prelay_mech(d, mech_ops_pt=0.0, L_mech=1e9, w=1.0, refurb_m=1.0):
    """120 の prelay に機械の部品の作り直し（継電器 w 個分の手間）を、腕の経路とリードの経路の両方に足す。"""
    Ma = demand(PX, d["C"], d["a"], d["L_arm"]); Mr = demand(PX, d["C"], d["a"], d["L_reed"])
    Mm = w * PX * mech_ops_pt / L_mech / refurb_m
    Ma_eff = Ma["N0"] + Ma["replacements"] / d["refurb"] + Mm; Mr_eff = Mr["N0"] + Mr["replacements"] / d["refurb"] + Mm
    p_arm = d["p_build"] * S(Ma_eff, d["M50"])
    p_reed = d["p_build"] * TB["p_seal"] * TB["p_glass"] * d["p_NiFe"] * S(Mr_eff, d["M50"])
    return {"p_arm": p_arm, "p_reed": p_reed, "p_relay": 1 - (1 - p_arm) * (1 - p_reed), "M_mech": Mm}
for dk in ("D1_min|rand", "D3_realistic|rand", "D4_stepper|rand"):          # 機械 0 で 120 の式と一致
    for L in (1e5, 1e6, 1e7):
        a_ = prelay(from_design(dk, L_arm=L))["p_relay"]; b_ = prelay_mech(from_design(dk, L_arm=L))["p_relay"]
        assert abs(a_ - b_) < 1e-12

# 1 点あたりの機械の動作（122 の模擬の数え）
nd1 = DES["D1_min|rand"]; nd4 = DES["D4_stepper|rand"]
MECH_OPS = {"D1_cam_every_point": 1.0,                                  # カムは点ごとに従節を持ち上げる【推測：主】
            "D1_cam_load_only": nd1["cam_ops"] / nd1["points"],          # 1 の点だけ数える（楽観）
            "D4_stepper_steps": nd4["breakdown"]["bit_counter"] / nd4["points"]}  # 1 ビット 1 歩
OUT["mech_ops_per_point"] = MECH_OPS
ign_lo, ign_hi = G["OUT"]["ignition_ops_range"]
CAM_ANCH = 10 ** ((np.log10(ign_lo) + np.log10(ign_hi)) / 2)            # 点火の接点そのもの（カム＋従節＋負荷を切る接点）
STEP_ANCH = 10 ** ((np.log10(3e5) + np.log10(1e7)) / 2)                 # Finder 26 の機械 3e5 とデータシートの継電器の機械 1e7 の間【推測】
OUT["anchors"] = {"cam_factory_geo_mid": CAM_ANCH, "stepper_factory_geo_mid": STEP_ANCH, "finder26_mech": 3e5, "finder26_elec": 1e5}
# 機械の寿命の分布（log10 正規）：中央＝錨×手作りの倍率 h、幅＝hypot(h の幅, 錨の幅)【推測】
MV = {
 "D1|main":      dict(d="D1_min|rand", ops=MECH_OPS["D1_cam_every_point"], anch=CAM_ANCH, h=0.1, hs=0.5, as_=0.3, w=1.0, rm=1.0),
 "D1|loadonly":  dict(d="D1_min|rand", ops=MECH_OPS["D1_cam_load_only"], anch=CAM_ANCH, h=0.1, hs=0.5, as_=0.3, w=1.0, rm=1.0),
 "D1|h0.3":      dict(d="D1_min|rand", ops=1.0, anch=CAM_ANCH, h=0.3, hs=0.5, as_=0.3, w=1.0, rm=1.0),
 "D1|file3":     dict(d="D1_min|rand", ops=1.0, anch=CAM_ANCH, h=0.1, hs=0.5, as_=0.3, w=1.0, rm=3.0),   # やすりで面を戻す（一時しのぎ）
 "D1|w0.3":      dict(d="D1_min|rand", ops=1.0, anch=CAM_ANCH, h=0.1, hs=0.5, as_=0.3, w=0.3, rm=1.0),   # 接点の板だけ替える
 "D1|pess":      dict(d="D1_min|rand", ops=1.0, anch=ign_lo, h=0.03, hs=1.0, as_=0.3, w=1.0, rm=1.0),
 "D4|main":      dict(d="D4_stepper|rand", ops=MECH_OPS["D4_stepper_steps"], anch=STEP_ANCH, h=0.1, hs=0.5, as_=0.75, w=3.0, rm=1.0),
 "D4|tel1e7":    dict(d="D4_stepper|rand", ops=MECH_OPS["D4_stepper_steps"], anch=1e7, h=0.1, hs=0.5, as_=0.3, w=3.0, rm=1.0),
 "D4|w1":        dict(d="D4_stepper|rand", ops=MECH_OPS["D4_stepper_steps"], anch=STEP_ANCH, h=0.1, hs=0.5, as_=0.75, w=1.0, rm=1.0),
 "D4|finder3e5": dict(d="D4_stepper|rand", ops=MECH_OPS["D4_stepper_steps"], anch=3e5, h=0.1, hs=0.5, as_=0.3, w=3.0, rm=1.0),
 "D4|pess":      dict(d="D4_stepper|rand", ops=MECH_OPS["D4_stepper_steps"], anch=3e5, h=0.03, hs=1.0, as_=0.3, w=3.0, rm=1.0),
 "D3|none":      dict(d="D3_realistic|rand", ops=0.0, anch=1e9, h=1.0, hs=0.0, as_=0.0, w=0.0, rm=1.0),
 "D2rz|none":    dict(d="D2_rz|rand", ops=0.0, anch=1e9, h=1.0, hs=0.0, as_=0.0, w=0.0, rm=1.0),     # カムなし：感知の継電器の接点が直接打つ
}
xs, ws = np.polynomial.hermite_e.hermegauss(31); ws = ws / ws.sum()
mu_c, sd_c = DV["main"]                                                   # 123 の主：継電器の乾いた接点の寿命
def expect2(v):
    mu_m = np.log10(v["anch"] * v["h"]); sd_m = float(np.hypot(v["hs"], v["as_"]))
    vals, wts, mm = [], [], []
    for xc, wc in zip(xs, ws):
        d = from_design(v["d"], L_arm=10 ** (mu_c + sd_c * xc))
        for xm, wm in zip(xs, ws):
            r = prelay_mech(d, v["ops"], 10 ** (mu_m + sd_m * xm), v["w"], v["rm"])
            vals.append(r["p_relay"]); wts.append(wc * wm); mm.append(r["M_mech"])
    vals, wts, mm = map(np.array, (vals, wts, mm)); o = np.argsort(vals); cdf = np.cumsum(wts[o])
    q = lambda p: float(vals[o][np.searchsorted(cdf, p)])
    om = np.argsort(mm); cm = np.cumsum(wts[om])
    return {"mean": float((wts * vals).sum()), "p10": q(0.1), "p50": q(0.5), "p90": q(0.9),
            "L_mech_median": 10 ** mu_m, "sigma_mech": sd_m, "M_mech_median": float(mm[om][np.searchsorted(cm, 0.5)]),
            "M_mech_at_median_L": v["w"] * PX * v["ops"] / (10 ** mu_m) / v["rm"] if v["ops"] else 0.0}
EXP = {k: expect2(v) for k, v in MV.items()}
OUT["mech_versions"] = MV; OUT["expected_prelay"] = EXP
# 123 との一致：D3 は機械なしで 123 の E と同じ（積分の格子が違うので許容 1e-3）
OUT["check_D3_vs_123"] = EXP["D3|none"]["mean"] - R123["expected_prelay"]["D3|main|refurb1"]["mean"]
# 走査：機械の寿命の中央と E[p_relay]（D1・D4、主の他の値）
Ls = np.logspace(5, 10, 26); SC = {}
for nm in ("D1|main", "D4|main"):
    SC[nm] = []
    for L in Ls:
        v = dict(MV[nm]); v["anch"] = L; v["h"] = 1.0; SC[nm].append(expect2(v)["mean"])
OUT["scan_Lmech"] = list(map(float, Ls)); OUT["scan_E"] = SC

# ---------- 2. p_vac と F_heavy ----------
PVN = {k: 1 - (1 - EXP[k]["mean"]) * (1 - P_TUBE) for k in ("D1|main", "D1|file3", "D1|pess", "D4|main", "D4|tel1e7", "D3|none", "D2rz|none")}
OUT["p_vac"] = PVN; OUT["p_tube"] = P_TUBE
best = max(PVN, key=PVN.get); OUT["best_design"] = best
TAB = {}
for vn in dict.fromkeys(["D1|main", "D4|main", best]):
    fT = t2e_two(100.0, L_bulk, PPREC * PVN[vn])
    for kn in KAPS:
        TAB[f"{kn}|{vn}"] = run(f"{kn}|{vn}", KAPS[kn], {"Fe": fe_d, "lens": le_d, "T2e": fT}, store=False)
OUT["table"] = TAB
F123_D3 = R123["table"]["slow_Tasmania|D3|main|refurb1"]["F_heavy"]; F123_D1 = R123["table"]["slow_Tasmania|D1|main|refurb1"]["F_heavy"]

# ---------- 仮説 ----------
H = {}
H["H136a"] = {"class": "confirmatory", "supported": False, "value": "Strowger switch の記事に保守・摩耗の記述なし。Telephone exchange の記事に『Strowger-based, step-by-step offices ... required continuous maintenance, such as cleaning』（登録の記事でないので数えない）"}
H["H136b"] = {"class": "confirmatory", "supported": False, "value": "Crossbar switch の記事は『two hours of downtime for service every forty years ... a large improvement over earlier electromechanical systems』。step-by-step と名指しの摩耗・保守の比較はない（規則どおり棄却、内容は近い）"}
H["H136c"] = {"class": "confirmatory", "supported": False, "value": "Cam (mechanism) の記事に wear の語なし"}
H["H136d"] = {"class": "confirmatory", "supported": False, "value": "見つかった数値は Finder 26 の歩進の継電器の機械 3×10⁵・電気 1×10⁵（10⁶ 未満）。電話の歩進の切替器の寿命の数値は見つからず（PO の uniselector の冊子は保守中で開けず）"}
d1 = EXP["D1|main"]["mean"]; d1_123 = R123["expected_prelay"]["D1|main|refurb1"]["mean"]; d4 = EXP["D4|main"]["mean"]
H["H136e"] = {"class": "prediction", "value": {"D1_124": d1, "D1_123": d1_123, "drop": d1_123 - d1}, "rule": "drop>=0.12", "supported": bool(d1_123 - d1 >= 0.12)}
H["H136f"] = {"class": "prediction", "value": {"D4": d4, "D1": d1}, "rule": "D4>D1", "supported": bool(d4 > d1)}
fbest = TAB[f"slow_Tasmania|{best}"]["F_heavy"]
H["H136g"] = {"class": "prediction", "value": {"best": best, "F_best": fbest, "F_123_D3": F123_D3, "F_123_D1": F123_D1, "move_vs_123_D3": fbest - F123_D3},
  "rule": "|F_best - 1.480| < 0.03", "supported": bool(abs(fbest - F123_D3) < 0.03)}
H["H136h"] = {"class": "descriptive", "supported": None, "value": "機械の作り直しの数と p_relay・p_vac・F_heavy の版"}
OUT["hypotheses"] = H; OUT["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
axs[0].plot(Ls, SC["D1|main"], "g", label="D1 (cam contact, 1 op/pt)"); axs[0].plot(Ls, SC["D4|main"], "m", label="D4 (stepper, 0.2 step/pt, w=3)")
axs[0].axhline(EXP["D3|none"]["mean"], color="b", ls="--", label="D3 (relays only)"); axs[0].axhline(EXP["D2rz|none"]["mean"], color="c", ls="--", label="D2 RZ (no cam)")
axs[0].axhline(d1_123, color="g", ls=":", lw=0.8, label="D1 in 123 (no cam wear)")
axs[0].axvspan(ign_lo, ign_hi, color="r", alpha=0.1); axs[0].axvline(3e5, color="k", ls=":", lw=0.8); axs[0].text(3.2e5, 0.05, "Finder 26 mech 3e5", fontsize=7)
axs[0].text(ign_lo, 0.05, "ignition points\n(factory)", fontsize=7, color="r")
axs[0].set_xscale("log"); axs[0].set_xlabel("median life of the mechanical part (operations, already handmade)"); axs[0].set_ylabel("E[p_relay]"); axs[0].legend(fontsize=7)
axs[0].set_title("124: cam / stepper wear on the reader (model quantity, not survival prob.)", fontsize=9)
ks = list(EXP.keys()); xx = np.arange(len(ks)); mean = [EXP[k]["mean"] for k in ks]
err = np.clip([[mean[i] - EXP[k]["p10"] for i, k in enumerate(ks)], [EXP[k]["p90"] - mean[i] for i, k in enumerate(ks)]], 0, None)
cols = ["g" if k.startswith("D1") else "m" if k.startswith("D4") else "b" if k.startswith("D3") else "c" for k in ks]
axs[1].bar(xx, mean, color=cols, yerr=err, capsize=2); axs[1].set_xticks(xx); axs[1].set_xticklabels(ks, rotation=40, fontsize=7, ha="right")
axs[1].set_ylabel("E[p_relay] (bars p10-p90)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_cam_stepper.png"), dpi=120); plt.close(fig)
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("check_123", dE, dF, "D3vs123", OUT["check_D3_vs_123"])
print("anchors", OUT["anchors"], "mech_ops", MECH_OPS)
for k, v in EXP.items(): print("E", k, {kk: (round(vv, 4) if vv < 10 else f"{vv:.3g}") for kk, vv in v.items()})
for k, v in PVN.items(): print("p_vac", k, round(v, 4))
for t, o in TAB.items(): print(t, "F_heavy", round(o["F_heavy"], 4))
for kk, hh in H.items(): print(kk, hh["class"], hh["supported"], str(hh.get("value"))[:220])
print("runtime", round(OUT["runtime_s"], 1))
