"""研究078：送付・受け入れの1部ごとの費用と受け入れの可否（open_problems 200）。
計算のみ（料金・方針の資料は sources/ に保存済み、本スクリプトはネットに接続しない）。事前登録：PREREGISTRATION.md。
TC/V0 と費用あたりは生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。
077 の sim.py を「# ---------- 仮説 ----------」の前まで exec し、077 の CE を差0で再現してから送付分と受け入れの確率だけを置き換える。新しい乱数は seed 20261078。"""
import json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, "..")
p77 = os.path.abspath(os.path.join(RS, "077-per-copy-cost", "sim.py")); s = open(p77, encoding="utf8").read()
G = {"__file__": p77, "__name__": "sim077"}; exec(compile(s[: s.index("# ---------- 仮説 ----------")], p77, "exec"), G)
r77 = json.load(open(os.path.join(RS, "077-per-copy-cost", "results.json"), encoding="utf8"))
R = {"study": "078-shipping-acceptance", "seed": 20261078, "note": "TC/V0 と費用あたりは生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。"}
# ---------- 検査：077 の CE を差0で再現 ----------
d = 0.0
for vn, V in r77["CE"].items():
    for part, P_ in V.items():
        for a, X in P_.items():
            for kk in ("avg35", "marg35_200"):
                A = np.asarray(G["CE"][vn][part][a][kk]); B = np.asarray(X[kk]); d = max(d, float(np.max(np.abs(A - B) / np.abs(B))))
R["check_077"] = {"CE_max_rel_diff": d, "reproduced": d < 1e-12}; assert R["check_077"]["reproduced"], R["check_077"]
Cost = G["Cost"]; get = G["get"]; qf = G["qf"]; DV = G["DV"]; M1 = G["M1"]; ALPHAS = G["ALPHAS"]; C_T = G["C_T"]
prn = G["prn"]; NC = prn.size; VERS77 = G["VERS"]
# ---------- 公開の料金（sources/rates.json）と方針（sources/policies.json） ----------
RT = json.load(open(os.path.join(HERE, "sources", "rates.json"), encoding="utf8")); PO = json.load(open(os.path.join(HERE, "sources", "policies.json"), encoding="utf8"))
fx = RT["fx_per_eur"]; usd = lambda v, c: v / fx[c] * fx["USD"]
def table(key):
    out = []
    for x in RT[key]:
        vals = [usd(v, x["cur"]) for v in x["zones"].values()]; out.append({"op": x["op"], "class": x["class"], "usd": vals, "op_median": float(np.median(vals))})
    return out
CH = table("cheapest"); TR = table("tracked"); R["postage_cheapest_usd"] = CH; R["postage_tracked_usd"] = TR
ch_all = [v for x in CH for v in x["usd"]]; tr_all = [v for x in TR for v in x["usd"]]
R["postage_stats"] = {"cheapest_op_medians": [x["op_median"] for x in CH], "cheapest_median_of_op_medians": float(np.median([x["op_median"] for x in CH])),
                      "cheapest_min": min(ch_all), "cheapest_max": max(ch_all), "tracked_min": min(tr_all), "tracked_max": max(tr_all),
                      "tracked_median_of_op_medians": float(np.median([x["op_median"] for x in TR])), "domestic_usps_media": RT["domestic"][0]["value"]}
L = PO["libs"]; nL = len(L)
R["policy_stats"] = {"n": nL, "S": sum(x["S"] for x in L), "P": sum(x["P"] for x in L), "N": sum(x["N"] for x in L), "scope_national": sum(x["scope"] for x in L),
                     "S_or_P_or_N": sum(x["S"] or x["P"] or x["N"] for x in L), "share_S_or_P_or_N": sum(x["S"] or x["P"] or x["N"] for x in L) / nL,
                     "numeric_acceptance_rate_found": 0}
# ---------- 新しい per_P ----------
rg = np.random.default_rng(20261078); u1, u2, u3, u4 = (rg.uniform(size=NC) for _ in range(4))
lu = lambda lo, hi, u: np.exp(np.log(lo) + u * np.log(hi / lo))
ship = lu(min(ch_all), max(ch_all), u1); ship_tr = lu(min(tr_all), max(tr_all), u1)
pacc = 0.5 + 0.4 * u2; pacc_low = 0.2 + 0.3 * u2; pacc_post = 0.05 + 0.15 * u2
VERS = {"main": (prn + ship) / pacc,
        "pacc1_desc": prn + ship,
        "pacc_0.2-0.5_desc": (prn + ship) / pacc_low,
        "contact_+15_desc": (prn + ship + 15.0) / pacc,
        "domestic_desc": (prn + R["postage_stats"]["domestic_usps_media"]) / pacc,
        "tracked_desc": (prn + ship_tr) / pacc,
        "posthoc_pacc_0.05-0.2_desc": (prn + ship) / pacc_post,
        "old_077_main": VERS77["main"], "old_077_no_shipping": VERS77["no_shipping_desc"]}
R["per_P_quantiles"] = {k: qf(v) for k, v in VERS.items()}
R["per_P_median_ratio_main_vs_077"] = float(np.median(VERS["main"]) / np.median(VERS77["main"]))
R["share_print_in_main"] = float(np.median(prn / (prn + ship)))
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
for part in ("total", "heavy"):
    for a in ("a0.0", "a0.5"):
        x = CE["old_077_main"][part][a]["ratio_median"]; y = r77["CE"]["main"][part][a]["ratio_median"]; assert abs(x - y) < 1e-12 * abs(y), (part, a)
R["check_old_077_main"] = "ratio_median equals 077 main (rel < 1e-12)"
# 部数の判断の分かれ目：重い崩壊 α=0.5 で限界/平均 = 1 となる per_P の倍率（主の per_P を k 倍）
def ratio_k(k, part="heavy", a=0.5):
    pp = VERS["main"] * k; vt = get("main", M1)["TAB"][part]; vtc = get("main", M1)["TC"][part]
    t35 = DV[f"mesh35_a{a}"][a_][u_]["TC"][part]; t200 = DV[f"mesh200_a{a}"][a_][u_]["TC"][part]
    return float(np.median((t200 - t35) / vt * C_T / (Cost(200, pp) - Cost(35, pp))) / np.median((t35 - vtc) / vt * C_T / Cost(35, pp)))
ks = np.exp(np.linspace(np.log(0.1), np.log(10), 81)); rk = {"heavy_a0.5": [ratio_k(k) for k in ks], "heavy_a0.0": [ratio_k(k, a=0.0) for k in ks], "total_a0.5": [ratio_k(k, "total") for k in ks]}
def cross(v):
    v = np.asarray(v); i = np.where(np.diff(np.sign(v - 1)) != 0)[0]
    return None if i.size == 0 else float(np.exp(np.interp(0, [v[i[0]] - 1, v[i[0] + 1] - 1], [np.log(ks[i[0]]), np.log(ks[i[0] + 1])])))
R["perP_scale_scan"] = {"k": ks.tolist(), **rk, "k_cross_ratio1": {kk: cross(v) for kk, v in rk.items()}, "per_P_main_median": float(np.median(VERS["main"]))}
# ---------- 仮説 ----------
H = {}
v = R["postage_stats"]["cheapest_median_of_op_medians"]; H["H90a"] = {"class": "確認的", "value": v, "rule": "median (over operators) of cheapest ~300 g international postage in USD within [10, 30]", "supported": 10 <= v <= 30}
v = R["policy_stats"]["share_S_or_P_or_N"]; H["H90b"] = {"class": "確認的", "value": v, "rule": "share of policies with S or P or N > 0.5", "supported": v > 0.5}
v = CE["main"]["heavy"]["a0.5"]["ratio_median"]; H["H90c"] = {"class": "予測", "value": v, "rule": "heavy, alpha=0.5: marginal/average (medians) > 1", "supported": v > 1}
v = CE["main"]["total"]["a0.5"]["ratio_median"]; H["H90d"] = {"class": "予測", "value": v, "rule": "total, alpha=0.5: marginal/average (medians) < 1", "supported": v < 1}
for x in H.values(): x["supported"] = bool(x["supported"])
R["hypotheses"] = H
# ---------- 図 ----------
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
for i, x in enumerate(CH): axs[0].plot([i] * len(x["usd"]), x["usd"], "o", label=x["op"])
for i, x in enumerate(TR): axs[0].plot([i + 4.3] * len(x["usd"]), x["usd"], "x", label=x["op"] + " (tracked/parcel)")
axs[0].axhspan(25, 50, color="r", alpha=.12, label="061/077 U($25,$50)"); axs[0].set_yscale("log"); axs[0].set_ylabel("USD per ~300 g book, international"); axs[0].set_xticks([]); axs[0].legend(fontsize=6)
names = list(VERS); axs[1].boxplot([VERS[k] for k in names], whis=(5, 95), showfliers=False); axs[1].set_xticks(range(1, len(names) + 1)); axs[1].set_xticklabels(names, rotation=35, fontsize=6, ha="right")
axs[1].set_yscale("log"); axs[1].set_ylabel("per accepted copy per_P (USD, 5-95%)")
for kk, v in rk.items(): axs[2].plot(ks * np.median(VERS["main"]), v, label=kk)
axs[2].axhline(1, color="k", lw=.8, ls=":"); axs[2].axvline(np.median(VERS["main"]), color="g", lw=.8, label="main per_P median"); axs[2].set_xscale("log"); axs[2].set_yscale("log")
axs[2].set_xlabel("per_P median (USD, main scaled)"); axs[2].set_ylabel("marginal 35->200 / average at 35 (medians)"); axs[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_postage_acceptance.png"), dpi=120); plt.close()
R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("check_077", "postage_stats", "policy_stats", "per_P_quantiles", "per_P_median_ratio_main_vs_077", "share_print_in_main", "hypotheses")}, ensure_ascii=False, indent=1, default=float))
print("k_cross", R["perP_scale_scan"]["k_cross_ratio1"])
for vn in VERS: print(vn, {p: {a: (round(CE[vn][p][a]["avg35"][1], 2), round(CE[vn][p][a]["marg35_200"][1], 2), round(CE[vn][p][a]["ratio_median"], 3), round(CE[vn][p][a]["share_marg_gt_avg"], 3)) for a in ("a0.0", "a0.5")} for p in ("total", "heavy")})
print("runtime", R["runtime_s"])
