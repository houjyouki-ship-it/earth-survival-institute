"""研究083：確認・補充の仕組みが続く期間（open_problems 210）＋ 082 の再現。
計算のみ（資料は sources/ に保存済み、本スクリプトはネットに接続しない）。事前登録：PREREGISTRATION.md（2026-10-21T16:40Z）。
TC/V0 と費用あたりは生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。
082 の sim.py を「# ---------- 仮説 ----------」の前まで exec し（rindex）、082 の versions 全値を差0で再現してから、
Wikidata の学会・財団の寿命（Kaplan–Meier）から標本ごとの仕組みの寿命 L を引き、補充の比を置き直す。seed 20261083。"""
import json, os, time, csv
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, "..")
p82 = os.path.abspath(os.path.join(RS, "082-replenish-unique-weeding", "sim.py")); s = open(p82, encoding="utf8").read()
G = {"__file__": p82, "__name__": "sim082"}; exec(compile(s[: s.rindex("# ---------- 仮説 ----------")], p82, "exec"), G)
r82 = json.load(open(os.path.join(RS, "082-replenish-unique-weeding", "results.json"), encoding="utf8"))
print("082 loaded", round(time.time() - T0), "s", flush=True)
R = {"study": "083-mechanism-lifespan", "seed": 20261083, "note": "TC/V0 と費用あたりは生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。"}
# ---------- 検査：082 の versions を差0で再現 ----------
def walk(a, b, acc):
    if isinstance(b, dict):
        for k in b: walk(a[k], b[k], acc)
    elif isinstance(b, list):
        A = np.asarray(a, float); B = np.asarray(b, float); acc.append(float(np.max(np.abs(A - B) / np.maximum(np.abs(B), 1e-300))))
    else:
        acc.append(abs(float(a) - float(b)) / max(abs(float(b)), 1e-300))
acc = []; walk(json.loads(json.dumps(G["OUT"], default=float)), r82["versions"], acc)
R["check_082"] = {"versions_max_rel_diff": max(acc), "n_values": len(acc), "reproduced": max(acc) < 1e-12}; assert R["check_082"]["reproduced"], R["check_082"]
run_k, RT, M1, inc_ce, qf, Cost, pp_main = G["run_k"], G["RT"], G["M1"], G["inc_ce"], G["qf"], G["Cost"], G["pp_main"]
Delta, c_chk, bN, pN, d_main, to_NC, N = G["Delta"], G["c_chk"], G["bN"], G["pN"], G["d_main"], G["to_NC"], G["N"]
def presence2(d, k, p, L):
    """082 の presence を標本ごとの寿命 L（配列、∞可）に拡張。"""
    n = np.minimum(np.floor(Delta / k), np.floor(L / k))
    sv = np.exp(-d * k); pi = np.ones_like(Delta); cost = np.zeros_like(Delta)
    for j in range(1, int(n.max()) + 1):
        act = j <= n; miss = 1 - pi * sv
        cost = cost + act * (c_chk + miss * bN)
        pi = np.where(act, pi * sv + miss * p, pi)
    return pi * np.exp(-d * (Delta - n * k)), cost
for life in (30.0, 100.0):
    xa, ca = G["presence"](d_main, 10, pN, life, 0.0); xb, cb = presence2(d_main, 10, pN, np.full(N, life))
    assert np.array_equal(xa, xb) and np.array_equal(ca, cb)
R["check_presence2_equals_082_fixed_life"] = True
# ---------- 寿命の資料 ----------
def load(t):
    y0, y1 = [], []
    for x in csv.DictReader(open(os.path.join(HERE, "sources", f"raw_{t}.csv"), encoding="utf8")):
        try: a = int(x["y0"])
        except ValueError: continue
        if not 1800 <= a <= 1975: continue
        b = int(x["y1"]) if x["y1"] else None
        if b is not None and b < a: continue
        y0.append(a); y1.append(b)
    y0 = np.array(y0); dead = np.array([b is not None for b in y1]); T = np.array([(b if b is not None else 2026) - a for a, b in zip(y0, y1)], float)
    return T, dead
def km(T, dead):
    ts = np.unique(T[dead]); S = 1.0; out_t, out_S = [0.0], [1.0]
    for t in ts:
        at = np.sum(T >= t); dth = np.sum((T == t) & dead); S *= 1 - dth / at; out_t.append(t); out_S.append(S)
    return np.array(out_t), np.array(out_S)
def S_at(t_, S_, x): return S_[np.searchsorted(t_, x, side="right") - 1]
def inv_km(t_, S_, u):
    L = np.full(u.shape, np.inf); i = np.searchsorted(-S_, -u, side="left")  # 最初に S<=u となる位置
    ok = i < S_.size; L[ok] = t_[i[ok]]; return L
DATA = {}
for t in ("society", "foundation"):
    T, dead = load(t); t_, S_ = km(T, dead)
    DATA[t] = {"T": T, "dead": dead, "t": t_, "S": S_}
    R[f"lifespan_{t}"] = {"n": int(T.size), "n_dissolved": int(dead.sum()), "median_dissolved": float(np.median(T[dead])) if dead.any() else None,
                          "q25_q75_dissolved": np.percentile(T[dead], [25, 75]).tolist() if dead.any() else None,
                          "KM_S": {str(a): float(S_at(t_, S_, a)) for a in (10, 30, 50, 100, 150, 200)}, "KM_last_t": float(t_[-1]), "KM_last_S": float(S_[-1])}
rg = np.random.default_rng(20261083); u = rg.uniform(size=N)
Ts = DATA["society"]["T"][DATA["society"]["dead"]]
LIFE = {"soc_km": inv_km(DATA["society"]["t"], DATA["society"]["S"], u),
        "found_km_desc": inv_km(DATA["foundation"]["t"], DATA["foundation"]["S"], u),
        "soc_dissolved_only_desc": np.quantile(Ts, u)}
R["life_sample_share_inf"] = {k: float(np.mean(~np.isfinite(v))) for k, v in LIFE.items()}
R["life_sample_share_ge_Delta"] = {k: float(np.mean(v >= Delta)) for k, v in LIFE.items()}
VERS = {f"soc_km_k{k}": ("soc_km", k) for k in (5, 10, 25, 50)}
VERS["found_km_k10_desc"] = ("found_km_desc", 10); VERS["soc_dissolved_only_k10_desc"] = ("soc_dissolved_only_desc", 10)
FUT = {}
for tag, (ln, k) in VERS.items():
    x, c = presence2(d_main, k, pN, LIFE[ln]); FUT[tag] = {"x": x, "c": to_NC(c)}
    run_k(tag, x, 35, "0.5"); print("ver", tag, round(time.time() - T0), "s", flush=True)
C35, C200 = Cost(35, pp_main), Cost(200, pp_main)
OUT = {}
for tag in VERS:
    o = {}
    for part in ("heavy", "total"):
        C35R = C35 + 35 * FUT[tag]["c"]
        ceR = inc_ce("mesh35_a0.5_none", f"mesh35_a0.5_{tag}", C35, C35R, part)
        ce200 = inc_ce("mesh35_a0.5_none", "mesh200_a0.5_none", C35, C200, part)
        o[f"{part}_a0.5"] = {"TC_none35": RT["mesh35_a0.5_none"][M1][part], "TC_R35": RT[f"mesh35_a0.5_{tag}"][M1][part], "TC_none200": RT["mesh200_a0.5_none"][M1][part],
                             "ce_inc_R35": qf(ceR), "ce_inc_none200": qf(ce200), "ratio_median": float(np.median(ceR) / np.median(ce200)),
                             "share_R_gt_200": float(np.mean(ceR > ce200)), "extra_cost_R35_median_usd": float(np.median(35 * FUT[tag]["c"]))}
    o["presence_quantiles"] = qf(FUT[tag]["x"]); OUT[tag] = o
R["versions"] = OUT
R["ref_082"] = {t: r82["versions"][t]["heavy_a0.5"]["ratio_median"] for t in ("k10", "k10_life30_desc", "k10_life100_desc")}
# ---------- 仮説 ----------
H = {}; ls = R["lifespan_society"]
H["H95a"] = {"class": "確認的", "value": {"n_dissolved": ls["n_dissolved"], "median": ls["median_dissolved"]}, "rule": "n_dissolved>=30 and median<50", "supported": bool(ls["n_dissolved"] >= 30 and ls["median_dissolved"] < 50)}
H["H95b"] = {"class": "確認的", "value": {"n": ls["n"], "KM_S50": ls["KM_S"]["50"]}, "rule": "n>=30 and KM S(50)>=0.70", "supported": bool(ls["n"] >= 30 and ls["KM_S"]["50"] >= 0.70)}
H["H95c"] = {"class": "予測", "value": OUT["soc_km_k10"]["heavy_a0.5"]["ratio_median"], "rule": "ratio(soc_km,k10,heavy a0.5)>1", "supported": bool(OUT["soc_km_k10"]["heavy_a0.5"]["ratio_median"] > 1)}
H["H95d"] = {"class": "予測", "value": OUT["soc_dissolved_only_k10_desc"]["heavy_a0.5"]["ratio_median"], "rule": "ratio(dissolved-only,k10,heavy a0.5)<1", "supported": bool(OUT["soc_dissolved_only_k10_desc"]["heavy_a0.5"]["ratio_median"] < 1)}
R["hypotheses"] = H
# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for t, c in (("society", "C0"), ("foundation", "C1")):
    axs[0].step(DATA[t]["t"], DATA[t]["S"], where="post", color=c, label=f"{t} KM (Wikidata, founded 1800-1975, n={DATA[t]['T'].size})")
xs = np.sort(Ts); axs[0].plot(xs, 1 - np.arange(1, xs.size + 1) / xs.size, "C0:", label="society, dissolved-only")
axs[0].set_xlim(0, 230); axs[0].set_xlabel("years since founding"); axs[0].set_ylabel("share still existing"); axs[0].legend(fontsize=7)
nm = list(OUT) + ["082 k10 (forever)", "082 life30", "082 life100"]
vals = [OUT[t]["heavy_a0.5"]["ratio_median"] for t in OUT] + list(R["ref_082"].values())
axs[1].bar(np.arange(len(nm)), vals); axs[1].axhline(1, color="k", ls=":"); axs[1].set_xticks(np.arange(len(nm))); axs[1].set_xticklabels([n.replace("_desc", "") for n in nm], rotation=35, fontsize=7)
axs[1].set_ylabel("CE(35->35+replenish)/CE(35->200), heavy a0.5 (not a survival prob.)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_lifespan.png"), dpi=120); plt.close()
R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("check_082", "lifespan_society", "lifespan_foundation", "life_sample_share_inf", "life_sample_share_ge_Delta", "ref_082", "hypotheses")}, ensure_ascii=False, indent=1, default=float))
for tag, o in OUT.items():
    v = o["heavy_a0.5"]; print(tag, "TC none35 %.4f R35 %.4f none200 %.4f ratio %.2f share %.2f extraR %.0f | total ratio %.2f" % (v["TC_none35"], v["TC_R35"], v["TC_none200"], v["ratio_median"], v["share_R_gt_200"], v["extra_cost_R35_median_usd"], o["total_a0.5"]["ratio_median"]))
print("runtime", R["runtime_s"])
