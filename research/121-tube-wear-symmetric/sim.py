"""研究121：真空管の経路のすり減り（open_problems 285）。継電器（120）と同じ物差しで管の作り直しを数え、p_vac を両経路で対称に置き直す。
計算のみ、ネット接続なし（資料は sources/ に保存済み）。事前登録：PREREGISTRATION.md（2026-10-25 07:39 JST）。
120 の sim.py を「# ---------- 仮説」の前まで exec（120 の p_vac と表を差 0 で再現、assert）。
事実（sources/）：ENIAC 18,000 本（1955 年の終わり、Vacuum tube の記事は『over 17,000』）、初めは毎日数本が切れ、後に 2 日に 1 本、故障の多くは暖機と冷却のとき（ENIAC）。
Colossus：電源を切らないことで故障を非常に低くした（数値なし）。特別品質の管は 1950 年代後半に数十万時間（Vacuum tube）。
手作りの管の寿命の倍率、M50_tube、入り切りの倍率はすべて【推測】。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P120 = os.path.abspath(os.path.join(HERE, "..", "120-relay-contact-life", "sim.py"))
src = open(P120, encoding="utf8").read(); cut = src.index("\n# ---------- 仮説")
G = {"__file__": P120, "__name__": "sim120"}
exec(compile(src[:cut], P120, "exec"), G)
R120 = json.load(open(os.path.join(os.path.dirname(P120), "results.json"), encoding="utf8"))
dP = max(abs(G["PV"][k] - R120["p_vac_versions"][k]) for k in R120["p_vac_versions"])
assert dP < 1e-12, dP
dT = max(abs(G["TAB"][t]["F_heavy"] - R120["table"][t]["F_heavy"]) for t in R120["table"])
run, KAPS, fe_d, le_d, t2e_two, L_bulk, PPREC, TB, ptube, PX_REEL, YR, PR, DATA, S = (G[k] for k in (
    "run", "KAPS", "fe_d", "le_d", "t2e_two", "L_bulk", "PPREC", "TB", "ptube", "PX_REEL", "YR", "PR", "DATA", "S"))
OUT = {"study": "121-tube-wear-symmetric", "check_120_pvac_maxdiff": dP, "check_120_table_F_maxdiff": dT}

# ---------- 1. 資料から管 1 本の平均の寿命（一定の故障率と置く）----------
H_DAY = 24.0
ENIAC_N = 18000
LIFE_FACT = {
 "ENIAC_late_1per2days": ENIAC_N * 2 * H_DAY,      # 8.64e5 時間・本
 "ENIAC_early_3perday": ENIAC_N * H_DAY / 3.0,     # 『several tubes almost every day』を 3 本/日と読む【推測の読み】
 "ENIAC_early_10perday": ENIAC_N * H_DAY / 10.0,   # 『several』の上側の読み【推測の読み】
}
# 注：ENIAC は常時通電でなく、故障の多くは暖機・冷却のとき。上の値は『運用の時間あたり』の平均の寿命で、通電の時間あたりより短めに出る。
OUT["life_from_sources_h"] = LIFE_FACT

# ---------- 2. 管の経路の作り直しの数（通電の時間で消耗）----------
RATE_TUBE, T_MAX = 5000.0, 20.0                     # 119 の Colossus 級の 1 チャンネル、20 年の上限
def tube_demand(px, C, L_h, cyc=1.0):
    t_single = px / RATE_TUBE / YR                   # 1 チャンネルでの年数
    ch = max(1.0, np.ceil(t_single / T_MAX))         # 20 年に収めるチャンネル数（主は 1）
    years = px / (RATE_TUBE * ch) / YR
    tube_h = ch * C * years * 365.25 * H_DAY         # 通電の延べ時間（全部の管が常に点灯）
    N0 = ch * C; repl = tube_h * cyc / L_h
    return {"channels": ch, "years": years, "tube_hours": tube_h, "N0": N0, "replacements": repl,
            "ratio_to_N0": repl / N0, "M_total": N0 + repl, "failures_per_day": N0 * cyc / L_h * H_DAY}
LIFE_GRID = {"1e2h": 1e2, "1e3h": 1e3, "1e4h": 1e4, "1.44e5h": LIFE_FACT["ENIAC_early_3perday"], "8.64e5h": LIFE_FACT["ENIAC_late_1per2days"]}
TD = {}
for dn, px in DATA.items():
    for ln, L in LIFE_GRID.items():
        for C in (20, 50, 100):
            TD[f"{dn}|{ln}|C{C}"] = tube_demand(px, C, L)
OUT["tube_demand"] = TD
# 1 点あたりの作り直し（物差しをそろえる）：継電器 C a / L_ops、管 C /(rate*3600*L_h)
OUT["per_point_replacement"] = {"relay_arm_L1e6_C50_a0.5": 50 * 0.5 / 1e6, "relay_reed_L1e9": 50 * 0.5 / 1e9,
                                "tube_L1e4h_C50": 50 / (RATE_TUBE * 3600 * 1e4), "tube_L1e2h_C50": 50 / (RATE_TUBE * 3600 * 1e2)}

# ---------- 3. p_tube の置き直し（すべて【推測】）----------
BT = {"C": 50, "L_h": 1e4,      # 手作りの管：ENIAC 初期の 1.44e5 時間の約 1/14【推測】
      "M50": 1e4,                # 管は継電器（M50 1e5）より 10 倍作りにくい【推測】
      "cyc": 1.0,                # 入り切りの倍率（毎日の停電で暖機・冷却が増えると >1）
      "data": "reel_120GB"}
def ptube_w(d):
    td = tube_demand(DATA[d["data"]], d["C"], d["L_h"], d["cyc"])
    return float(ptube(TB) * S(td["M_total"], d["M50"])), td["M_total"]
def mod(**kw): d = dict(BT); d.update(kw); return d
TV = {
 "main": BT,
 "L1e2": mod(L_h=1e2), "L1e3": mod(L_h=1e3), "L1.44e5": mod(L_h=LIFE_FACT["ENIAC_early_3perday"]),
 "M50_1e3": mod(M50=1e3), "M50_1e5": mod(M50=1e5),
 "C100": mod(C=100), "cyc10": mod(cyc=10.0),
 "q0.3": mod(data="q0.3_36GB"),
 "pessimistic": mod(L_h=1e2, M50=1e3, C=100, cyc=10.0),
}
PT = {k: ptube_w(v) for k, v in TV.items()}
P_TUBE0 = ptube(TB)
# 対称の p_vac：120 の主の p_relay と、すり減りを入れた p_tube
PRm = PR["main"]["p_relay"]
PV = {k: 1 - (1 - PRm) * (1 - v[0]) for k, v in PT.items()}
PV["tube_pess_relay_pess"] = 1 - (1 - PR["pessimistic"]["p_relay"]) * (1 - PT["pessimistic"][0])
PV["tube_main_relay_opt"] = 1 - (1 - PR["optimistic"]["p_relay"]) * (1 - PT["main"][0])
PV["120_main_nowear"] = 1 - (1 - PRm) * (1 - P_TUBE0)
OUT["base"] = BT; OUT["p_tube_nowear"] = P_TUBE0
OUT["p_tube_versions"] = {k: {"p_tube": v[0], "M_total": v[1]} for k, v in PT.items()}; OUT["p_vac_versions"] = PV
assert abs(PV["120_main_nowear"] - R120["p_vac_versions"]["main"]) < 1e-12

# 走査：寿命 L_h と M50 で p_tube が無すり減りの半分になる境目
Ls = np.logspace(1, 6, 51); Ms = (1e3, 1e4, 1e5)
SCAN = {f"M50_{m:.0e}": [float(ptube_w(mod(L_h=l, M50=m))[0]) for l in Ls] for m in Ms}
half = {}
for m in Ms:
    arr = np.array(SCAN[f"M50_{m:.0e}"]); idx = np.where(arr >= P_TUBE0 / 2)[0]
    half[f"M50_{m:.0e}"] = float(Ls[idx[0]]) if len(idx) else None
OUT["scan_L"] = list(map(float, Ls)); OUT["scan_ptube"] = SCAN; OUT["L_half_ptube"] = half

TAB = {}
for vn in ("main", "L1e2", "cyc10", "pessimistic", "tube_pess_relay_pess", "120_main_nowear"):
    fT = t2e_two(100.0, L_bulk, PPREC * PV[vn])
    for kn in KAPS:
        TAB[f"{kn}|{vn}"] = run(f"{kn}|{vn}", KAPS[kn], {"Fe": fe_d, "lens": le_d, "T2e": fT}, store=False)
OUT["table"] = TAB
assert abs(TAB["slow_Tasmania|120_main_nowear"]["F_heavy"] - R120["table"]["slow_Tasmania|main"]["F_heavy"]) < 0.02

# ---------- 仮説 ----------
H = {}
H["H133a"] = {"class": "confirmatory", "supported": True,
  "value": "ENIAC：1955 年の終わりに 18,000 本（Vacuum tube の記事は over 17,000）。『one tube every two days』（Eckert 1989『about every two days』）"}
H["H133b"] = {"class": "confirmatory", "supported": True,
  "value": "Colossus：『not powering a machine down reduced failure rates to very low levels』（数値なし）"}
H["H133c"] = {"class": "confirmatory", "supported": True,
  "value": "Vacuum tube：特別品質の小信号管が 1950 年代後半に『hundreds of thousands of hours』、トリウム陰極『tens of thousands of hours』"}
d = TD["reel_120GB|1e4h|C50"]
H["H133d"] = {"class": "prediction", "value": {"ratio_to_N0_L1e4": d["ratio_to_N0"], "relay_arm_ratio_120": R120["demand"]["reel_120GB|armature_mid|C50"]["ratio_to_N0"]},
  "rule": "主（L 1e4 h）の作り直し/初めの本数 < 10 かつ 継電器の汎用（120）の比 > 1000", "supported": bool(d["ratio_to_N0"] < 10 and R120["demand"]["reel_120GB|armature_mid|C50"]["ratio_to_N0"] > 1000)}
fh = TAB["slow_Tasmania|main"]["F_heavy"]
H["H133e"] = {"class": "prediction", "value": {"p_tube": PT["main"][0], "p_vac": PV["main"], "F_heavy_slow": fh},
  "rule": "0.20<=p_tube<=0.235 かつ 0.45<=p_vac<=0.52 かつ 1.34<=F<=1.40",
  "supported": bool(0.20 <= PT["main"][0] <= 0.235 and 0.45 <= PV["main"] <= 0.52 and 1.34 <= fh <= 1.40)}
H["H133f"] = {"class": "descriptive", "supported": None, "value": "管の数・寿命・故障の頻度の表"}
OUT["hypotheses"] = H; OUT["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
for k, v in SCAN.items(): axs[0].plot(Ls, v, label=f"tube path, {k}")
axs[0].axhline(P_TUBE0, color="gray", ls=":"); axs[0].axhline(PRm, color="r", ls="--", lw=0.8); axs[0].text(12, PRm + 0.005, "relay path p_relay (120 main)", fontsize=7, color="r")
for ln in ("ENIAC_early_10perday", "ENIAC_early_3perday", "ENIAC_late_1per2days"): axs[0].axvline(LIFE_FACT[ln], color="k", lw=0.5)
axs[0].set_xscale("log"); axs[0].set_xlabel("mean tube life L_h (hours); vertical lines = ENIAC-derived"); axs[0].set_ylabel("p_tube with wear")
axs[0].legend(fontsize=7); axs[0].set_title("121: tube-path wear for one 120 GB reel (C=50, 1 channel)", fontsize=9)
vn = [t.split("|")[1] for t in TAB if t.startswith("slow_Tasmania")]; x = np.arange(len(vn))
for j, kn in enumerate(KAPS):
    axs[1].bar(x + (j-1)*0.27, [TAB[f"{kn}|{w}"]["F_heavy"] for w in vn], 0.27, label=kn)
axs[1].set_xticks(x); axs[1].set_xticklabels([f"{w}\n{PV[w]:.3f}" for w in vn], fontsize=6, rotation=30)
axs[1].axhline(1.1, color="k", lw=0.5); axs[1].axhline(1.5, color="k", lw=0.5, ls=":"); axs[1].legend(fontsize=7); axs[1].set_ylabel("F_heavy (A_Ord/U3)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_tube_wear.png"), dpi=120); plt.close(fig)
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("check_120", dP, "F diff", dT)
print("life_from_sources", LIFE_FACT)
for k, v in TD.items():
    if "C50" in k: print("tube_demand", k, {kk: f"{vv:.3g}" for kk, vv in v.items()})
print("per_point", OUT["per_point_replacement"])
for k, v in PT.items(): print("p_tube", k, round(v[0], 4), "M", f"{v[1]:.3g}", "p_vac", round(PV[k], 4))
for k in ("tube_pess_relay_pess", "tube_main_relay_opt", "120_main_nowear"): print("p_vac", k, round(PV[k], 4))
print("L_half", half)
for t, o in TAB.items(): print(t, "F_heavy", round(o["F_heavy"], 4))
for kk, hh in H.items(): print(kk, hh["class"], hh["supported"], str(hh.get("value"))[:200])
print("runtime", round(OUT["runtime_s"], 1))
