"""研究079：寄贈した冊子が収蔵される確率 p_acc の数値（open_problems 203）。
計算のみ（資料は sources/ に保存済み、本スクリプトはネットに接続しない）。事前登録：PREREGISTRATION.md。
TC/V0 と費用あたりは生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。
078 の sim.py を「# ---------- 仮説 ----------」の前まで exec し、078 の CE を差0で再現してから p_acc だけを置き換える。新しい乱数は seed 20261079。"""
import json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, "..")
p78 = os.path.abspath(os.path.join(RS, "078-shipping-acceptance", "sim.py")); s = open(p78, encoding="utf8").read()
G = {"__file__": p78, "__name__": "sim078"}; exec(compile(s[: s.rindex("# ---------- 仮説 ----------")], p78, "exec"), G)
r78 = json.load(open(os.path.join(RS, "078-shipping-acceptance", "results.json"), encoding="utf8"))
R = {"study": "079-gift-acceptance-rate", "seed": 20261079, "note": "TC/V0 と費用あたりは生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。"}
# ---------- 検査：078 の CE を差0で再現 ----------
d = 0.0
for vn, V in r78["CE"].items():
    for part, P_ in V.items():
        for a, X in P_.items():
            for kk in ("avg35", "marg35_200"):
                A = np.asarray(G["CE"][vn][part][a][kk]); B = np.asarray(X[kk]); d = max(d, float(np.max(np.abs(A - B) / np.abs(B))))
R["check_078"] = {"CE_max_rel_diff": d, "reproduced": d < 1e-12}; assert R["check_078"]["reproduced"], R["check_078"]
Cost = G["Cost"]; get = G["get"]; qf = G["qf"]; DV = G["DV"]; M1 = G["M1"]; ALPHAS = G["ALPHAS"]; C_T = G["C_T"]
prn = G["prn"]; ship = G["ship"]; NC = prn.size; VERS78 = G["VERS"]; dom = G["R"]["postage_stats"]["domestic_usps_media"]
# ---------- 公開の受け入れ率（sources/acceptance.json） ----------
AC = json.load(open(os.path.join(HERE, "sources", "acceptance.json"), encoding="utf8"))
a_i = np.array([c["a"] for c in AC["cases"]]); LD = AC["legal_deposit"]
R["acceptance"] = {"n": int(a_i.size), "a_i": a_i.tolist(), "median": float(np.median(a_i)), "min": float(a_i.min()), "max": float(a_i.max()),
                   "pooled_counted": float(sum(c["added"] for c in AC["cases"] if c["added"]) / sum(c["received"] for c in AC["cases"] if c["added"]))}
R["legal_deposit"] = {"n": len(LD), "domestic_only": sum(x["domestic_only"] for x in LD), "share_domestic_only": sum(x["domestic_only"] for x in LD) / len(LD)}
# ---------- 新しい p_acc ----------
rg = np.random.default_rng(20261079); u = rg.uniform(size=NC)
main_ok = a_i.size >= 3
pacc = (a_i.min() + (a_i.max() - a_i.min()) * u) if main_ok else (0.5 + 0.4 * u)
base = prn + ship
VERS = {"main": base / pacc,
        "median_fixed_desc": base / np.median(a_i),
        "foreign_x0.5_desc": base / (0.5 * pacc),
        "old_078_main": VERS78["main"]}
R["main_uses_data"] = bool(main_ok)
R["per_P_quantiles"] = {k: qf(v) for k, v in VERS.items()}
R["per_P_median_ratio_main_vs_078"] = float(np.median(VERS["main"]) / np.median(VERS78["main"]))
a_, u_ = M1.split("/")
def ce(pp35, pp200, part, a):
    vt = get("main", M1)["TAB"][part]; vtc = get("main", M1)["TC"][part]
    t35 = DV[f"mesh35_a{a}"][a_][u_]["TC"][part]; t200 = DV[f"mesh200_a{a}"][a_][u_]["TC"][part]
    avg = (t35 - vtc) / vt * C_T / Cost(35, pp35); mg = (t200 - t35) / vt * C_T / (Cost(200, pp200) - Cost(35, pp35))
    return {"avg35": qf(avg), "marg35_200": qf(mg), "ratio_median": float(np.median(mg) / np.median(avg)), "share_marg_gt_avg": float(np.mean(mg > avg))}
CE = {vn: {part: {f"a{a}": ce(pp, pp, part, a) for a in ALPHAS} for part in ("total", "heavy")} for vn, pp in VERS.items()}
# 法定納本の経路（記述）：出版国の納本の n_ld 部は p_acc=1・国内料金、残りは主の p_acc。部数で平均した1部の費用。
# TC（置き方）は変えない。納本の部は1国に集まるので、置き方の価値は過大（上限の側）。
for n_ld in (2, 6):
    ld = prn + dom
    pp = lambda n: ((n - n_ld) * VERS["main"] + n_ld * ld) / n
    CE[f"legal_deposit_{n_ld}_desc"] = {part: {f"a{a}": ce(pp(35), pp(200), part, a) for a in ALPHAS} for part in ("total", "heavy")}
R["CE"] = CE
for part in ("total", "heavy"):
    for a in ("a0.0", "a0.5"):
        x = CE["old_078_main"][part][a]["ratio_median"]; y = r78["CE"]["main"][part][a]["ratio_median"]; assert abs(x - y) < 1e-12 * abs(y), (part, a)
R["check_old_078_main"] = "ratio_median equals 078 main (rel < 1e-12)"
# 分かれ目：主の式で p_acc を固定値にしたとき、重い崩壊 α=0.5 の限界/平均 = 1 となる p_acc
grid = np.linspace(0.05, 1.0, 96)
def rp(p, part="heavy", a=0.5): return ce(base / p, base / p, part, a)["ratio_median"]
scan = {"p": grid.tolist(), "heavy_a0.5": [rp(p) for p in grid], "heavy_a0.0": [rp(p, a=0.0) for p in grid], "total_a0.5": [rp(p, "total") for p in grid], "total_a0.0": [rp(p, "total", 0.0) for p in grid]}
def cross(v):
    v = np.asarray(v) - 1; i = np.where(np.diff(np.sign(v)) != 0)[0]
    if i.size == 0: return None
    j = i[0]; return float(grid[j] + (grid[j + 1] - grid[j]) * v[j] / (v[j] - v[j + 1]))
scan["p_cross_ratio1"] = {k: cross(v) for k, v in scan.items() if k != "p"}
R["pacc_scan"] = scan
R["share_main_samples_below_heavy_cross"] = float(np.mean(pacc < scan["p_cross_ratio1"]["heavy_a0.5"])) if scan["p_cross_ratio1"]["heavy_a0.5"] else None
# ---------- 仮説 ----------
H = {}
H["H91a"] = {"class": "確認的", "value": {"n": int(a_i.size), "median": float(np.median(a_i))}, "rule": "n>=3 and median a_i < 0.5", "supported": a_i.size >= 3 and float(np.median(a_i)) < 0.5}
v = R["legal_deposit"]["share_domestic_only"]; H["H91b"] = {"class": "確認的", "value": v, "rule": "share of legal-deposit regimes limited to domestic publications > 0.5", "supported": v > 0.5}
v = CE["main"]["heavy"]["a0.5"]["ratio_median"]; H["H91c"] = {"class": "予測", "value": v, "rule": "heavy, alpha=0.5: marginal/average (medians) > 1", "supported": v > 1}
v = CE["main"]["total"]["a0.5"]["ratio_median"]; H["H91d"] = {"class": "予測", "value": v, "rule": "total, alpha=0.5: marginal/average (medians) < 1", "supported": v < 1}
for x in H.values(): x["supported"] = bool(x["supported"])
R["hypotheses"] = H
# ---------- 図 ----------
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
axs[0].barh(range(a_i.size), a_i, color="C0"); axs[0].set_yticks(range(a_i.size)); axs[0].set_yticklabels([c["lib"][:40] for c in AC["cases"]], fontsize=7)
axs[0].axvspan(0.5, 0.9, color="r", alpha=.12, label="078 prior U(0.5,0.9)"); axs[0].axvline(np.median(a_i), color="k", ls=":", label="median")
xc = scan["p_cross_ratio1"]["heavy_a0.5"]
if xc: axs[0].axvline(xc, color="g", label=f"heavy a=0.5 cross {xc:.2f}")
axs[0].set_xlim(0, 1); axs[0].set_xlabel("share of received gifts added (a_i)"); axs[0].legend(fontsize=7)
for k in ("heavy_a0.5", "heavy_a0.0", "total_a0.5", "total_a0.0"): axs[1].plot(grid, scan[k], label=k)
axs[1].axhline(1, color="k", lw=.8, ls=":"); axs[1].axvspan(a_i.min(), a_i.max(), color="C0", alpha=.12, label="079 main p_acc range")
axs[1].set_yscale("log"); axs[1].set_xlabel("p_acc (fixed)"); axs[1].set_ylabel("marginal 35->200 / average at 35 (medians)"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_pacc.png"), dpi=120); plt.close()
R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("check_078", "acceptance", "legal_deposit", "main_uses_data", "per_P_quantiles", "per_P_median_ratio_main_vs_078", "share_main_samples_below_heavy_cross", "hypotheses")}, ensure_ascii=False, indent=1, default=float))
print("p_cross", scan["p_cross_ratio1"])
for vn in CE: print(vn, {p: {a: (round(CE[vn][p][a]["avg35"][1], 2), round(CE[vn][p][a]["marg35_200"][1], 2), round(CE[vn][p][a]["ratio_median"], 3), round(CE[vn][p][a]["share_marg_gt_avg"], 3)) for a in ("a0.0", "a0.5")} for p in ("total", "heavy")})
print("runtime", R["runtime_s"])
