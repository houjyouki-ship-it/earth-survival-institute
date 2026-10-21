"""研究080：事前の申し出つきの受け入れの割合 b_j（open_problems 205）。
計算のみ（資料は sources/ に保存済み、本スクリプトはネットに接続しない）。事前登録：PREREGISTRATION.md。
TC/V0 と費用あたりは生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。
079 の sim.py を「# ---------- 仮説 ----------」の前まで exec し（切り出しは rindex）、079 の CE を差0で再現してから p_acc の版を足す。新しい乱数は seed 20261080。"""
import json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, "..")
p79 = os.path.abspath(os.path.join(RS, "079-gift-acceptance-rate", "sim.py")); s = open(p79, encoding="utf8").read()
G = {"__file__": p79, "__name__": "sim079"}; exec(compile(s[: s.rindex("# ---------- 仮説 ----------")], p79, "exec"), G)
r79 = json.load(open(os.path.join(RS, "079-gift-acceptance-rate", "results.json"), encoding="utf8"))
R = {"study": "080-offer-acceptance-rate", "seed": 20261080, "note": "TC/V0 と費用あたりは生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。"}
# ---------- 検査：079 の CE を差0で再現 ----------
d = 0.0
for vn, V in r79["CE"].items():
    for part, P_ in V.items():
        for a, X in P_.items():
            for kk in ("avg35", "marg35_200"):
                A = np.asarray(G["CE"][vn][part][a][kk]); B = np.asarray(X[kk]); d = max(d, float(np.max(np.abs(A - B) / np.abs(B))))
R["check_079"] = {"CE_max_rel_diff": d, "reproduced": d < 1e-12}; assert R["check_079"]["reproduced"], R["check_079"]
ce = G["ce"]; base = G["base"]; pacc79 = G["pacc"]; qf = G["qf"]; ALPHAS = G["ALPHAS"]; NC = base.size
cross79 = r79["pacc_scan"]["p_cross_ratio1"]["heavy_a0.5"]
# ---------- 申し出の受け入れの割合（sources/offers.json） ----------
OF = json.load(open(os.path.join(HERE, "sources", "offers.json"), encoding="utf8"))
bq = np.array([c["accepted"] / c["offered"] for c in OF["qualifying_cases"]]) if OF["qualifying_cases"] else np.array([])
bb = np.array([c.get("b", (c["accepted"] / c["offered"]) if c.get("offered") else np.nan) for c in OF["broad_cases_desc"]])
R["offers"] = {"n_qualifying": int(bq.size), "b_qualifying": bq.tolist(),
               "n_broad_desc": int(bb.size), "b_broad_desc": bb.tolist(), "broad_median": float(np.median(bb)), "broad_min": float(bb.min()), "broad_max": float(bb.max()),
               "n_no_ratio": len(OF["no_ratio_found"])}
main_ok = bq.size >= 2
rg = np.random.default_rng(20261080); u = rg.uniform(size=NC); u2 = rg.uniform(size=NC)
p_main = (bq.min() + (bq.max() - bq.min()) * u) if main_ok else pacc79
p_broad = bb.min() + (bb.max() - bb.min()) * u2
VERS = {"main": base / p_main,
        "broad_offer_only_desc": base / p_broad,            # 申し出の段だけ（受け入れ後の選別なし）
        "broad_offer_x_receipt_desc": base / (p_broad * pacc79),  # 申し出の段 × 079 の受け取った後の比（両段がかかる下限の側）
        "broad_offer_foreign_x0.5_desc": base / (0.5 * p_broad)}
R["main_uses_new_data"] = bool(main_ok)
R["p_acc_quantiles"] = {"main": qf(p_main), "broad_offer_only_desc": qf(p_broad), "broad_offer_x_receipt_desc": qf(p_broad * pacc79), "broad_offer_foreign_x0.5_desc": qf(0.5 * p_broad)}
R["per_P_quantiles"] = {k: qf(v) for k, v in VERS.items()}
CE = {vn: {part: {f"a{a}": ce(pp, pp, part, a) for a in ALPHAS} for part in ("total", "heavy")} for vn, pp in VERS.items()}
R["CE"] = CE
if not main_ok:
    for part in ("total", "heavy"):
        for a in ("a0.0", "a0.5"):
            assert abs(CE["main"][part][a]["ratio_median"] - r79["CE"]["main"][part][a]["ratio_median"]) < 1e-12
    R["check_main_equals_079"] = True
R["share_below_heavy_cross"] = {k: float(np.mean(v < cross79)) for k, v in {"main": p_main, "broad_offer_only_desc": p_broad, "broad_offer_x_receipt_desc": p_broad * pacc79, "broad_offer_foreign_x0.5_desc": 0.5 * p_broad}.items()}
R["heavy_cross_p_from_079"] = cross79
# 申し出の段の割合がいくら以上なら、受け取った後の比 a（079 の中央値）を掛けても分かれ目を越えるか
R["b_needed_if_both_stages_with_a_median"] = float(cross79 / np.median(G["a_i"]))
# ---------- 仮説 ----------
H = {}
H["H92a"] = {"class": "確認的", "value": {"n": int(bq.size), "median": float(np.median(bq)) if bq.size else None}, "rule": "n>=2 and median b_j > 0.30",
             "supported": bool(bq.size >= 2 and np.median(bq) > 0.30)}
H["H92b"] = {"class": "確認的", "value": {"n": int(bq.size), "min": float(bq.min()) if bq.size else None}, "rule": "n>=2 and min b_j > 0.296",
             "supported": bool(bq.size >= 2 and bq.min() > 0.296)}
v = CE["main"]["heavy"]["a0.5"]["ratio_median"]; H["H92c"] = {"class": "予測", "value": v, "rule": "heavy, alpha=0.5: marginal/average (medians) > 1", "supported": bool(v > 1)}
v = CE["main"]["total"]["a0.5"]["ratio_median"]; H["H92d"] = {"class": "予測", "value": v, "rule": "total, alpha=0.5: marginal/average (medians) < 1", "supported": bool(v < 1)}
R["hypotheses"] = H
# ---------- 図 ----------
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
labs = [c["inst"][:38] for c in OF["broad_cases_desc"]] + [c["lib"][:38] for c in G["AC"]["cases"]]
vals = list(bb) + list(G["a_i"]); cols = ["C1"] * bb.size + ["C0"] * G["a_i"].size
axs[0].barh(range(len(vals)), vals, color=cols); axs[0].set_yticks(range(len(vals))); axs[0].set_yticklabels(labs, fontsize=7)
axs[0].axvline(cross79, color="g", label=f"heavy a=0.5 cross {cross79:.3f}")
axs[0].set_xlim(0, 1); axs[0].set_xlabel("orange: share of offers accepted (museums/archives, desc.)  blue: share of received gifts added (079)"); axs[0].legend(fontsize=7)
names = list(VERS); x = np.arange(len(names))
for j, (part, a) in enumerate([("heavy", "a0.5"), ("heavy", "a0.0"), ("total", "a0.5")]):
    axs[1].bar(x + (j - 1) * 0.27, [CE[n][part][a]["ratio_median"] for n in names], 0.27, label=f"{part} {a}")
axs[1].axhline(1, color="k", ls=":"); axs[1].set_yscale("log"); axs[1].set_xticks(x); axs[1].set_xticklabels([n.replace("_desc", "").replace("broad_", "") for n in names], fontsize=7, rotation=12)
axs[1].set_ylabel("marginal 35->200 / average at 35 (medians)"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_offer_pacc.png"), dpi=120); plt.close()
R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("check_079", "offers", "main_uses_new_data", "p_acc_quantiles", "per_P_quantiles", "share_below_heavy_cross", "b_needed_if_both_stages_with_a_median", "hypotheses")}, ensure_ascii=False, indent=1, default=float))
for vn in CE: print(vn, {p: {a: (round(CE[vn][p][a]["avg35"][1], 2), round(CE[vn][p][a]["marg35_200"][1], 2), round(CE[vn][p][a]["ratio_median"], 3), round(CE[vn][p][a]["share_marg_gt_avg"], 3)) for a in ("a0.0", "a0.5")} for p in ("total", "heavy")})
print("runtime", R["runtime_s"])
