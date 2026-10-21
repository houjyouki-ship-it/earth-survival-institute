"""研究077：1部ごとの費用 per_P の公開の見積もり（open_problems 198）。
計算のみ（価格の資料は sources/ に保存済みのものを使い、本スクリプトはネットに接続しない）。事前登録：PREREGISTRATION.md。
TC/V0 は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。
076 の sim.py を「# ---------- 順位 ----------」の前まで exec し、076 の CE を差0で再現してから per_P だけを置き換える。新しい乱数は seed 20261077。"""
import json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, "..")
p76 = os.path.abspath(os.path.join(RS, "076-copy-attraction-marginal", "sim.py")); s = open(p76, encoding="utf8").read()
G = {"__file__": p76, "__name__": "sim076"}; exec(compile(s[: s.index("# ---------- 順位 ----------")], p76, "exec"), G)
r76 = json.load(open(os.path.join(RS, "076-copy-attraction-marginal", "results.json"), encoding="utf8"))
R = {"study": "077-per-copy-cost", "seed": 20261077, "note": "TC/V0 は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。"}
# ---------- 検査：076 の CE と TC を差0で再現 ----------
d = 0.0
for part in ("total", "heavy"):
    for k, v in r76["CE"][part].items():
        for kk, vv in v.items():
            a = np.atleast_1d(G["CE"][part][k][kk]); b = np.atleast_1d(vv); d = max(d, float(np.max(np.abs(a - b) / np.maximum(1e-300, np.abs(b)))))
dtc = max(abs(G["R"]["heavy_TC"][k] - r76["heavy_TC"][k]) for k in r76["heavy_TC"])
R["check_076"] = {"CE_max_rel_diff": d, "heavyTC_max_abs_diff": dtc, "reproduced": d < 1e-12 and dtc < 1e-12}; assert R["check_076"]["reproduced"], R["check_076"]
find = G["find"]; per_P = np.asarray(G["per_P"]); f_pg = np.asarray(G["f_pg"]); prep_P = np.asarray(find("prep_P")); prep_pr = np.asarray(find("prep_pr"))
C_T = G["C_T"]; get = G["get"]; qf = G["qf"]; DV = G["DV"]; M1 = G["M1"]; ALPHAS = G["ALPHAS"]; NC = per_P.size
def Cost(n, pp): return prep_P + n * pp + 2 * (prep_pr + 100 * pp * f_pg + (n - 100) * pp * f_pg)
for n in (35, 100, 200):
    dd = float(np.max(np.abs(Cost(n, per_P) - G["Cost"](n)) / G["Cost"](n))); assert dd < 1e-12, (n, dd)
R["check_cost_formula"] = "Cost(n, per_P_061) == 076 Cost(n) (rel diff < 1e-12)"
# ---------- 公開の価格（sources/prices.json、頁の表示どおり） ----------
P = json.load(open(os.path.join(HERE, "sources", "prices.json"), encoding="utf8")); R["prices"] = P
book = [x["usd_100p"] for x in P["book"]]; acid = [x["usd_100p"] for x in P["book"] if x["acid_free_claim"]]; plain = [x["usd_100p"] for x in P["book"] if not x["acid_free_claim"]]
lo, hi = min(x["low"] for x in P["book"]), max(x["high"] for x in P["book"])
R["book_stats"] = {"n": len(book), "min": lo, "max": hi, "median": float(np.median(book)), "median_acid_free": float(np.median(acid)), "median_plain": float(np.median(plain)),
                   "ratio_acid_plain": float(np.median(acid) / np.median(plain)), "microfilm_reel_dup_usd": [x["usd"] for x in P["microfilm"]]}
rg = np.random.default_rng(20261077); u1 = rg.uniform(size=NC); u2 = rg.uniform(size=NC)
prn = np.exp(np.log(lo) + u1 * np.log(hi / lo)); ship = 25 + 25 * u2
VERS = {"main": prn + ship, "hardcover_x3_desc": 3 * prn + ship, "no_shipping_desc": prn, "shipping_x2_desc": prn + 2 * ship,
        "old_061": per_P, "old_x0.1_076": per_P * 0.1, "microfilm_reel_desc": np.exp(np.log(min(R["book_stats"]["microfilm_reel_dup_usd"])) + u1 * np.log(max(R["book_stats"]["microfilm_reel_dup_usd"]) / min(R["book_stats"]["microfilm_reel_dup_usd"]))) + ship}
R["per_P_quantiles"] = {k: qf(v) for k, v in VERS.items()}; R["per_P_median_ratio_new_old"] = float(np.median(VERS["main"]) / np.median(per_P))
a_, u_ = M1.split("/"); CE = {}
for vn, pp in VERS.items():
    CE[vn] = {}
    for part in ("total", "heavy"):
        vt = get("main", M1)["TAB"][part]; vtc = get("main", M1)["TC"][part]; CE[vn][part] = {}
        for a in ALPHAS:
            t35 = DV[f"mesh35_a{a}"][a_][u_]["TC"][part]; t200 = DV[f"mesh200_a{a}"][a_][u_]["TC"][part]
            avg = (t35 - vtc) / vt * C_T / Cost(35, pp); mg = (t200 - t35) / vt * C_T / (Cost(200, pp) - Cost(35, pp))
            CE[vn][part][f"a{a}"] = {"avg35": qf(avg), "marg35_200": qf(mg), "ratio_median": float(np.median(mg) / np.median(avg)), "share_marg_gt_avg": float(np.mean(mg > avg))}
R["CE"] = CE
cs = CE["old_x0.1_076"]["total"]["a0.5"]["avg35"][1]; R["check_x0.1_vs_076"] = [cs, r76["CE"]["total"]["a0.5_perPx0.1"]["avg35"][1]]
assert abs(cs - R["check_x0.1_vs_076"][1]) < 1e-9 * cs
# 費用の内訳（主、中央値、35部と200部）
R["cost_breakdown_median"] = {str(n): {"fixed_prep": float(np.median(prep_P + 2 * prep_pr)), "per_copy_part": float(np.median(n * VERS["main"] * (1 + 2 * f_pg))), "share_shipping_in_perP": float(np.median(ship / VERS["main"]))} for n in (35, 200)}
# ---------- 仮説 ----------
H = {}
v = R["book_stats"]["median"]; H["H89a"] = {"class": "確認的", "value": v, "rule": "median per-copy print cost (~100 pp) < $100", "supported": v < 100}
v = R["book_stats"]["ratio_acid_plain"]; H["H89b"] = {"class": "確認的", "value": v, "rule": "median acid-free-claim / median plain < 3", "supported": v < 3}
v = CE["main"]["total"]["a0.5"]["ratio_median"]; H["H89c"] = {"class": "予測", "value": v, "rule": "total, alpha=0.5: median marginal 35->200 / median avg35 < 1", "supported": v < 1}
v = CE["main"]["heavy"]["a0.5"]["ratio_median"]; H["H89d"] = {"class": "予測", "value": v, "rule": "heavy, alpha=0.5: median marginal 35->200 / median avg35 > 1", "supported": v > 1}
for x in H.values(): x["supported"] = bool(x["supported"])
R["hypotheses"] = H
# ---------- 図 ----------
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
names = list(VERS); axs[0].boxplot([VERS[k] for k in names], whis=(5, 95), showfliers=False); axs[0].set_xticks(range(1, len(names) + 1)); axs[0].set_xticklabels(names, rotation=30, fontsize=7)
axs[0].set_yscale("log"); axs[0].set_ylabel("per-copy cost per_P (USD, 5-95%)")
for x in P["book"]: axs[0].plot(0.7, x["usd_100p"], "k.", ms=4)
x = np.arange(len(names)); w = 0.2
for i, (part, a) in enumerate((("total", 0.0), ("total", 0.5), ("heavy", 0.0), ("heavy", 0.5))):
    axs[1].bar(x + (i - 1.5) * w, [CE[k][part][f"a{a}"]["ratio_median"] for k in names], w, label=f"{part}, alpha={a}")
axs[1].axhline(1, color="k", lw=.8, ls=":"); axs[1].set_xticks(x); axs[1].set_xticklabels(names, rotation=30, fontsize=7); axs[1].set_yscale("log")
axs[1].set_ylabel("marginal CE 35->200 / average CE at 35 (medians)"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_perP_ce.png"), dpi=120); plt.close()
R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("check_076", "book_stats", "per_P_quantiles", "per_P_median_ratio_new_old", "cost_breakdown_median", "hypotheses")}, ensure_ascii=False, indent=1, default=float))
for vn in VERS: print(vn, {p: {a: (round(CE[vn][p][a]["avg35"][1], 2), round(CE[vn][p][a]["marg35_200"][1], 2), round(CE[vn][p][a]["ratio_median"], 3)) for a in ("a0.0", "a0.5")} for p in ("total", "heavy")})
print("runtime", R["runtime_s"])
