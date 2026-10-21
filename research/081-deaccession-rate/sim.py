"""研究081：収蔵された冊子の除籍（weeding）の率（open_problems 204）。
計算のみ（資料は sources/ に保存済み、本スクリプトはネットに接続しない）。事前登録：PREREGISTRATION.md。
TC/V0 と費用あたりは生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。
080 の sim.py を「# ---------- 仮説 ----------」の前まで exec し（rindex）、080 の CE を差0で再現してから、
1部の生き残りの係数 f_i（063）に崩壊の前の除籍 exp(−m·d·Δ) を掛けて 076 の run_att で TC を計算し直す。新しい乱数は seed 20261081。"""
import json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, "..")
p80 = os.path.abspath(os.path.join(RS, "080-offer-acceptance-rate", "sim.py")); s = open(p80, encoding="utf8").read()
G = {"__file__": p80, "__name__": "sim080"}; exec(compile(s[: s.rindex("# ---------- 仮説 ----------")], p80, "exec"), G)
r80 = json.load(open(os.path.join(RS, "080-offer-acceptance-rate", "results.json"), encoding="utf8"))
print("080 loaded", round(time.time() - T0), "s", flush=True)
R = {"study": "081-deaccession-rate", "seed": 20261081, "note": "TC/V0 と費用あたりは生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。"}
# ---------- 検査：080 の CE を差0で再現 ----------
d = 0.0
for vn, V in r80["CE"].items():
    for part, P_ in V.items():
        for a, X in P_.items():
            for kk in ("avg35", "marg35_200"):
                A = np.asarray(G["CE"][vn][part][a][kk]); B = np.asarray(X[kk]); d = max(d, float(np.max(np.abs(A - B) / np.abs(B))))
R["check_080"] = {"CE_max_rel_diff": d, "reproduced": d < 1e-12}; assert R["check_080"]["reproduced"], R["check_080"]
def find(key):
    st, seen = [G], set()
    while st:
        g = st.pop()
        if id(g) in seen: continue
        seen.add(id(g))
        if key in g: return g
        st += [v for k, v in g.items() if isinstance(v, dict) and "__name__" in v]
    raise KeyError(key)
G76 = find("run_att"); G79 = find("pacc"); G70 = G76["G"]["G70"]
run_att, MESH, RT, DV, M1 = G76["run_att"], G76["MESH"], G76["RT"], G76["DV"], G76["M1"]
Cost, get, C_T, qf = G79["Cost"], G79["get"], G79["C_T"], G79["qf"]
pp_main = G["VERS"]["main"]; NC = pp_main.size; N = G70["f_i"].size
a_, u_ = M1.split("/")
def ce2(dv35, dv200, pp, part):
    vt = get("main", M1)["TAB"][part]; vtc = get("main", M1)["TC"][part]
    t35 = dv35[a_][u_]["TC"][part]; t200 = dv200[a_][u_]["TC"][part]
    avg = (t35 - vtc) / vt * C_T / Cost(35, pp); mg = (t200 - t35) / vt * C_T / (Cost(200, pp) - Cost(35, pp))
    return {"avg35": qf(avg), "marg35_200": qf(mg), "ratio_median": float(np.median(mg) / np.median(avg)), "share_marg_gt_avg": float(np.mean(mg > avg))}
for part in ("total", "heavy"):
    for a in ("0.0", "0.5"):
        x = ce2(DV[f"mesh35_a{a}"], DV[f"mesh200_a{a}"], pp_main, part)["ratio_median"]; y = r80["CE"]["main"][part][f"a{a}"]["ratio_median"]
        assert abs(x - y) < 1e-12 * abs(y), (part, a, x, y)
R["check_ce2_equals_080_main"] = True
print("checks ok", round(time.time() - T0), "s", flush=True)
# ---------- 除籍の率 d_j（sources/deaccession.json） ----------
DJ = json.load(open(os.path.join(HERE, "sources", "deaccession.json"), encoding="utf8"))
cases = []
for c in DJ["qualifying"]:
    if c.get("numeric", True) is False: continue
    rates = [w / n for (w, n) in c["years"].values()]
    cases.append({"inst": c["inst"], "type": c["type"], "n_years": len(rates), "d": float(np.mean(rates)), "d_years": rates})
dj = np.array([c["d"] for c in cases]); typ = np.array([c["type"] for c in cases])
pooled = sum(w for c in DJ["qualifying"] if c.get("numeric", True) is not False for (w, n) in c["years"].values()) / sum(n for c in DJ["qualifying"] if c.get("numeric", True) is not False for (w, n) in c["years"].values())
R["deaccession"] = {"cases": cases, "n": int(dj.size), "median": float(np.median(dj)), "q25": float(np.quantile(dj, 0.25)), "q75": float(np.quantile(dj, 0.75)),
                    "min": float(dj.min()), "max": float(dj.max()), "mean": float(dj.mean()), "pooled_volume_weighted": float(pooled),
                    "median_university": float(np.median(dj[typ == "university"])), "n_university": int((typ == "university").sum()),
                    "national_numeric": {c["inst"]: c["d"] for c in cases if c["type"] == "national"},
                    "national_non_numeric": [c["inst"] for c in DJ["qualifying"] if c.get("numeric", True) is False],
                    "tau_weed_median_years": float(1 / np.median(dj)), "tau_weed_pooled_years": float(1 / pooled), "tau_copy_062_median": 522.0,
                    "descriptive": {c["inst"]: c["w"] / c["N"] for c in DJ["descriptive"]},
                    "share_cases_d_gt_1_over_522": float(np.mean(dj > 1 / 522))}
main_ok = dj.size >= 2
# ---------- 新しい乱数 ----------
rg = np.random.default_rng(20261081)
u_d = rg.uniform(size=N); u_D = rg.uniform(size=N)
d_main = dj.min() + (dj.max() - dj.min()) * u_d
Delta = np.exp(np.log(10) + (np.log(200) - np.log(10)) * u_D)
f_i0 = G70["f_i"].copy()
def run_ver(tag, k, alphas=("0.0", "0.5")):
    G70["f_i"] = f_i0 * k
    try:
        for a in alphas:
            for n in (35, 200): run_att(f"mesh{n}_a{a}_{tag}", MESH[:n], float(a))
    finally:
        G70["f_i"] = f_i0
VK = {"main": np.exp(-1 * d_main * Delta),
      "m3_desc": np.exp(-3 * d_main * Delta), "m10_desc": np.exp(-10 * d_main * Delta),
      "Delta30_desc": np.exp(-d_main * 30), "Delta100_desc": np.exp(-d_main * 100), "Delta300_desc": np.exp(-d_main * 300),
      "d_median_desc": np.exp(-np.median(dj) * Delta), "d_pooled_desc": np.exp(-pooled * Delta),
      "d_median_m10_desc": np.exp(-10 * np.median(dj) * Delta)}
R["survival_factor_quantiles"] = {k: qf(v) for k, v in VK.items()}
for tag, k in VK.items():
    run_ver(tag, k); print("ver", tag, round(time.time() - T0), "s", flush=True)
# 走査（記述）：d を固定、Δ は主の標本
DS = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
for dv in DS: run_ver(f"dscan{dv}", np.exp(-dv * Delta), alphas=("0.5",))
# ---------- 集計 ----------
h = lambda nm: RT[nm][M1]["heavy"]; t = lambda nm: RT[nm][M1]["total"]
base_TC = {"heavy_a0.5_mesh35": h("mesh35_a0.5"), "heavy_a0.5_mesh200": h("mesh200_a0.5"), "total_a0.5_mesh35": t("mesh35_a0.5")}
R["TC_base_080"] = base_TC
TC, CE = {}, {}
for tag in VK:
    TC[tag] = {f"{p}_a{a}_mesh{n}": RT[f"mesh{n}_a{a}_{tag}"][M1][p] for p in ("heavy", "total") for a in ("0.0", "0.5") for n in (35, 200)}
    CE[tag] = {p: {f"a{a}": ce2(DV[f"mesh35_a{a}_{tag}"], DV[f"mesh200_a{a}_{tag}"], pp_main, p) for a in ("0.0", "0.5")} for p in ("total", "heavy")}
R["TC"] = TC; R["CE"] = CE
R["dscan"] = {str(dv): {"heavy_mesh35": h(f"mesh35_a0.5_dscan{dv}"), "heavy_mesh200": h(f"mesh200_a0.5_dscan{dv}"),
                        "ratio_heavy": ce2(DV[f"mesh35_a0.5_dscan{dv}"], DV[f"mesh200_a0.5_dscan{dv}"], pp_main, "heavy")["ratio_median"],
                        "ratio_total": ce2(DV[f"mesh35_a0.5_dscan{dv}"], DV[f"mesh200_a0.5_dscan{dv}"], pp_main, "total")["ratio_median"]} for dv in DS}
assert abs(R["dscan"]["0.0"]["heavy_mesh35"] - base_TC["heavy_a0.5_mesh35"]) < 1e-12
R["check_dscan0_equals_080"] = True
R["drop_heavy_a0.5_mesh35"] = {tag: base_TC["heavy_a0.5_mesh35"] - TC[tag]["heavy_a0.5_mesh35"] for tag in VK}
r0 = r80["CE"]["main"]["heavy"]["a0.5"]["ratio_median"]
R["ratio_rel_change_heavy_a0.5"] = {tag: CE[tag]["heavy"]["a0.5"]["ratio_median"] / r0 - 1 for tag in VK}
# ---------- 仮説 ----------
H = {}
v = float(np.median(dj)) if main_ok else None
H["H93a"] = {"class": "確認的", "value": {"n": int(dj.size), "median": v}, "rule": "n>=2 and median d_j < 0.002/yr", "supported": bool(main_ok and v < 0.002)}
nat = [c["d"] for c in cases if c["type"] == "national"]; uni = [c["d"] for c in cases if c["type"] == "university"]
H["H93b"] = {"class": "確認的", "value": {"n_national_numeric": len(nat), "national_median": float(np.median(nat)) if nat else None, "university_median": float(np.median(uni)) if uni else None},
             "rule": "n_national>=1 and n_university>=1 and median(university) > median(national)", "supported": bool(nat and uni and np.median(uni) > np.median(nat)),
             "caveat": "national = CISTI（NRC の National Science Library、1966 年に公式化）。LAC は CARL 表で除籍 N/A（数値なし）。NLA は分母が別頁の概数のため記述。"}
v = R["drop_heavy_a0.5_mesh35"]["main"]; H["H93c"] = {"class": "予測", "value": v, "rule": "heavy a=0.5 mesh35 TC/V0 drop < 0.01", "supported": bool(v < 0.01)}
v = R["ratio_rel_change_heavy_a0.5"]["main"]; H["H93d"] = {"class": "予測", "value": v, "rule": "|heavy a=0.5 marginal/average rel. change vs 080 main| < 5%", "supported": bool(abs(v) < 0.05)}
R["hypotheses"] = H
# ---------- 図 ----------
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
fig, axs = plt.subplots(1, 3, figsize=(17, 5))
o = np.argsort(dj); cols = {"university": "C0", "national": "C3", "state": "C2"}
axs[0].barh(range(dj.size), dj[o] * 100, color=[cols[typ[i]] for i in o]); axs[0].set_yticks(range(dj.size)); axs[0].set_yticklabels([cases[i]["inst"][:26] for i in o], fontsize=5.5)
axs[0].set_xscale("symlog", linthresh=0.01); axs[0].axvline(100 / 522, color="k", ls=":", label="1/522 yr (062 copy tau)"); axs[0].axvline(0.2, color="r", ls="--", label="H93a 0.2%")
axs[0].set_xlabel("annual withdrawal rate d (%/yr)  blue=univ, red=national, green=state"); axs[0].legend(fontsize=7)
xs = [max(x, 1e-4) for x in DS]
axs[1].semilogx(xs, [R["dscan"][str(x)]["heavy_mesh35"] for x in DS], "-o", label="heavy a=0.5 mesh35"); axs[1].semilogx(xs, [R["dscan"][str(x)]["heavy_mesh200"] for x in DS], "-s", label="heavy a=0.5 mesh200")
axs[1].axvline(np.median(dj), color="C2", ls=":", label=f"median d {np.median(dj)*100:.2f}%"); axs[1].axvline(pooled, color="C4", ls=":", label=f"pooled d {pooled*100:.2f}%")
axs[1].set_xlabel("d (/yr), Delta~logU(10,200) yr; x=1e-4 means d=0"); axs[1].set_ylabel("TC/V0 (not a survival probability)"); axs[1].legend(fontsize=7)
names = list(VK); x = np.arange(len(names))
axs[2].bar(x - 0.2, [CE[n]["heavy"]["a0.5"]["ratio_median"] for n in names], 0.4, label="heavy a0.5"); axs[2].bar(x + 0.2, [CE[n]["total"]["a0.5"]["ratio_median"] for n in names], 0.4, label="total a0.5")
axs[2].axhline(1, color="k", ls=":"); axs[2].axhline(r0, color="C0", ls="--", lw=0.8, label=f"080 heavy {r0:.3f}")
axs[2].set_xticks(x); axs[2].set_xticklabels([n.replace("_desc", "") for n in names], fontsize=7, rotation=30); axs[2].set_ylabel("marginal 35->200 / average at 35 (medians)"); axs[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_deaccession.png"), dpi=120); plt.close()
R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("check_080", "survival_factor_quantiles", "TC_base_080", "drop_heavy_a0.5_mesh35", "ratio_rel_change_heavy_a0.5", "dscan", "hypotheses")}, ensure_ascii=False, indent=1, default=float))
print(json.dumps({k: v for k, v in R["deaccession"].items() if k != "cases"}, ensure_ascii=False, indent=1))
for tag in VK: print(tag, {p: {a: (round(CE[tag][p][a]["ratio_median"], 3)) for a in ("a0.0", "a0.5")} for p in ("total", "heavy")}, round(TC[tag]["heavy_a0.5_mesh35"], 4))
print("runtime", R["runtime_s"])
