"""研究028：重要 repo だけを小辞書で二重書きしたときの q・ΔV と費用（open_problems 80）
事前登録：PREREGISTRATION.md。入力：data/census.json（collect.py）、data/compress.json（compress.py）、
../025-q-xz-window/data/sample.json・data/compress.json、../024-content-value-reader-guide/sources/text_sizes.json。
seed 固定（024 の系列 20261024、025 の系列 20261026、本回 20261028）。numpy＋matplotlib のみ。"""
import json, time, sys, platform, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
KiB, MiB = 1024, 1024 ** 2
CEN = json.load(open(f"{HERE}/data/census.json"))
CMP = json.load(open(f"{HERE}/data/compress.json"))
S25 = json.load(open(f"{HERE}/../025-q-xz-window/data/sample.json"))
C25 = json.load(open(f"{HERE}/../025-q-xz-window/data/compress.json"))
REEL = 21e12 / 186
D_ARCH, S0_LC3 = 8 * MiB, 16 * KiB
S0_LC0 = S0_LC3 - 2 * (6144 - 768)          # 5,632 B（lzma-specification：1846 + 768·2^lc 個の16ビット確率）
CFG = {"c1_64KiB_lc3": (64 * KiB, S0_LC3), "c2_64KiB_lc0": (64 * KiB, S0_LC0), "c3_16KiB_lc0": (16 * KiB, S0_LC0),
       "c4_4KiB_lc0": (4 * KiB, S0_LC0), "u_uncompressed": (0, 1 * KiB)}
R = {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
     "inputs": {"reel_bytes": REEL, "S0_lc3": S0_LC3, "S0_lc0": S0_LC0, "configs": {k: list(v) for k, v in CFG.items()}}}

# ---- 圧縮計測 ----
cm = CMP["repos"]
tarA = np.array([r["tar_bytes_actual"] for r in cm], float)
xz = {c: np.array([r["xz_bytes"][c] for r in cm], float) for c in cm[0]["xz_bytes"]}
rho = {c: float(xz[c].sum() / tarA.sum()) for c in xz}            # バイト重みの圧縮率
rho["u_uncompressed"] = 1.0
ovh = {c: (xz[c] / xz["c0_8MiB_lc3"] - 1) for c in xz}
R["compress"] = {"n": len(cm), "n_featured": sum(r["group"] == "featured" for r in cm), "skipped_over_cap": CMP["skipped_over_cap"],
    "fetch_failed": CMP["fetch_failed"], "total_tar": float(tarA.sum()), "rho_bytes": rho,
    "overhead_total_vs_c0": {c: float(xz[c].sum() / xz["c0_8MiB_lc3"].sum() - 1) for c in xz},
    "overhead_median_vs_c0": {c: float(np.median(ovh[c])) for c in xz},
    "overhead_p10_p90_vs_c0": {c: [float(np.percentile(ovh[c], 10)), float(np.percentile(ovh[c], 90))] for c in xz},
    "lc0_vs_lc3_64KiB_median": float(np.median(xz["c2_64KiB_lc0"] / xz["c1_64KiB_lc3"] - 1)),
    "lc0_vs_lc3_64KiB_total": float(xz["c2_64KiB_lc0"].sum() / xz["c1_64KiB_lc3"].sum() - 1),
    "tree_est_over_actual_median": float(np.median([r["tar_bytes_tree_est"] / r["tar_bytes_actual"] for r in cm])),
    "by_group_overhead_median_c1": {g: float(np.median([r["xz_bytes"]["c1_64KiB_lc3"] / r["xz_bytes"]["c0_8MiB_lc3"] - 1 for r in cm if r["group"] == g])) for g in ("featured", "random20k")},
    "per_repo": [{"name": r["full_name"], "group": r["group"], "tar": r["tar_bytes_actual"],
                  **{c: r["xz_bytes"][c] / r["xz_bytes"]["c0_8MiB_lc3"] - 1 for c in xz}} for r in cm]}
meas = {r["full_name"]: r for r in cm}
# 025 の30件による感度用の圧縮率（lc0 は 028 の lc0/lc3 比を掛ける）
c25 = [r for r in C25["repos"] if r["status"] == "ok"]
t25 = sum(r["tar_bytes_actual"] for r in c25)
r25 = {"c0_8MiB_lc3": sum(r["xz_bytes_by_dict"][str(8 * MiB)] for r in c25) / t25,
       "c1_64KiB_lc3": sum(r["xz_bytes_by_dict"]["65536"] for r in c25) / t25,
       "c4_4KiB_lc3": sum(r["xz_bytes_by_dict"]["4096"] for r in c25) / t25}
lc_fac = rho["c2_64KiB_lc0"] / rho["c1_64KiB_lc3"]
rho25 = {"c1_64KiB_lc3": r25["c1_64KiB_lc3"], "c2_64KiB_lc0": r25["c1_64KiB_lc3"] * lc_fac,
         "c3_16KiB_lc0": (r25["c1_64KiB_lc3"] * r25["c4_4KiB_lc3"]) ** 0.5 * lc_fac, "c4_4KiB_lc0": r25["c4_4KiB_lc3"] * lc_fac,
         "u_uncompressed": 1.0}
R["compress"]["rho_from_025_sensitivity"] = rho25

# ---- 母集団（高層 ≥250 星）：≥20,000 星は全数、それ未満は 025 の標本の重み ----
cen = [r for r in CEN["repos"] if r.get("status") == "ok"]
R["census"] = {"search_total": CEN["search_total_count"], "listed": CEN["n_listed"], "ok": len(cen),
               "not_ok": sum(1 for r in CEN["repos"] if r.get("status") != "ok"),
               "truncated_after_walk": sum(1 for r in cen if r["truncated"]),
               "featured_found": sorted(r["full_name"] for r in cen if r["featured"])}
items = []
for r in cen:
    items.append({"name": r["full_name"], "src": "census", "w": 1.0, "stars": float(r["stars_now"]), "tar": float(r["tar_bytes"]),
                  "featured": r["featured"], "ge20k": r["stars_now"] >= 20000, "ge5k": r["stars_now"] >= 5000})
CHROM_TAR = float(os.environ.get("CHROM_TAR", 3e9))   # chromium/chromium：走査中断、TAR 不明【推測】
for r in CEN["repos"]:
    if r["status"] == "walk_aborted":
        items.append({"name": r["full_name"], "src": "census", "w": 1.0, "stars": float(r["stars_now"]), "tar": CHROM_TAR,
                      "featured": False, "ge20k": True, "ge5k": True})
R["census"]["excluded_not_ok_stars"] = float(sum(r["stars_now"] for r in CEN["repos"] if r["status"] not in ("ok", "walk_aborted")))
R["census"]["chromium_tar_assumed"] = CHROM_TAR
hi = [r for r in S25["repos"] if r.get("status") == "ok" and r["stratum"] == "high" and r["bin"] != "20000-None"]
bins = S25["bins"]; nb = {}
for r in hi: nb[r["bin"]] = nb.get(r["bin"], 0) + 1
for r in hi:
    items.append({"name": r["full_name"], "src": "sample025", "w": bins[r["bin"]]["total_count_now"] / nb[r["bin"]],
                  "stars": float(r["stars_now"]), "tar": float(r["tar_bytes"]), "featured": False, "ge20k": False,
                  "ge5k": r["bin"] == "5000-19999"})
w = np.array([i["w"] for i in items]); st = np.array([i["stars"] for i in items]); tar = np.array([i["tar"] for i in items])
sw = w * st; W = sw.sum()
src_c = np.array([i["src"] == "census" for i in items])
need0 = S0_LC3 + np.minimum(D_ARCH, tar)
def needc(c): d, s0 = CFG[c]; return s0 + (np.minimum(d, tar) if d else 0)
def cost_bytes(c, rhod=rho):
    """二重書き版の1 repo あたりバイト（計測済みは実測×木の推定比で補正せず実測を使う）"""
    out = tar * rhod[c]
    if rhod is rho:
        for k, it in enumerate(items):
            m = meas.get(it["name"])
            if m and it["src"] == "census":
                out[k] = m["tar_bytes_actual"] if c == "u_uncompressed" else m["xz_bytes"][c]
    return out
rank = np.argsort(-st[src_c]); cidx = np.where(src_c)[0][rank]
def topN(n): m = np.zeros(len(items), bool); m[cidx[:n]] = True; return m
SETS = {"featured15": np.array([i["featured"] for i in items]), "top100": topN(100), "top300": topN(300), "top1000": topN(1000),
        "all_ge20k": np.array([i["ge20k"] for i in items]), "all_ge5k": np.array([i["ge5k"] for i in items]),
        "all_high": np.ones(len(items), bool)}
R["star_share"] = {k: float(sw[m].sum() / W) for k, m in SETS.items()}
R["repo_count"] = {k: float(w[m].sum()) for k, m in SETS.items()}
R["tar_bytes_set"] = {k: float((w * tar)[m].sum()) for k, m in SETS.items()}
R["archive_total_high_tar_est"] = float((w * tar).sum())

# ---- 024 の V_of（024・025 と同じ式・同じ乱数系列） ----
rng2 = np.random.default_rng(20261024)
N = 200_000; H = 1000.0
def logu(lo, hi, n=N): return np.exp(rng2.uniform(np.log(lo), np.log(hi), n))
SIZES = json.load(open(f"{HERE}/../024-content-value-reader-guide/sources/text_sizes.json"))
G_LO = SIZES["wp_total"] / 1e6; G_HI = 2 * SIZES["rdh4"]["utf8"] / 1e6; P_H = 0.9305; D_HAND = 1.9
Z = rng2.standard_normal((3, N))
T1 = 20 * np.exp(0.7 * Z[0]); T2 = T1 + 30 * np.exp(0.7 * Z[1])
claim = logu(500, 2000); f_store = logu(0.3, 1.0)
life = claim * f_store / np.log(2) ** 0.5 * (-np.log(rng2.uniform(size=N))) ** 0.5
g_req = logu(G_LO, G_HI); P_max = rng2.uniform(0.3, 0.9, N); kappa = logu(0.5, 2.0)
t_build = 20 * np.exp(0.7 * rng2.standard_normal(N))
U_t = rng2.uniform(size=N); U_h = rng2.uniform(size=N); U_sep = rng2.uniform(size=N); p_sep = rng2.uniform(0.1, 0.5, N)
L_core = 50 * 11 / 12 * logu(1, 10); Vv = logu(550, 2400); t_tube = logu(0.5, 5); Uup = logu(3, 30)
N_min = Uup * (L_core + Vv * 3 * t_tube / 250) / (10.0 * logu(0.1, 0.4) * logu(0.005, 0.05))
pop_ok = N_min < logu(0.5e6, 5e6)
def V_of(m23, qv, g_hand, tablet=False):
    T3 = T2 + m23 * np.exp(0.8 * Z[2]); gen = (T3 < life) & (T3 <= H)
    cov = np.zeros(N); avail = np.zeros(N, bool)
    if g_hand > 0:
        ho = U_h < P_H; cov = np.where(ho, np.minimum(1.0, g_hand / g_req), cov); avail |= ho
    if tablet:
        to = U_sep > p_sep; cov = np.where(to, 1.0, cov); avail |= to
    Tt = T2 + t_build
    tgt = avail & pop_ok & (U_t < P_max * cov ** kappa) & (Tt < life) & (Tt <= H)
    return float(np.mean(np.maximum(qv * tgt, gen.astype(float))))
V0 = V_of(943, np.zeros(N), 0)
def dV(qv): return V_of(943, qv, 0, tablet=True) - V0

def q_from_need(e, Marr, mask_w=None):
    o = np.argsort(e); cw = np.cumsum(sw[o]) / W
    k = np.searchsorted(e[o], Marr, side="right"); return np.where(k > 0, cw[np.maximum(k - 1, 0)], 0.0)

# 検証：025 の B（標本のみの母集団・025 の M 系列）を再現
rng3 = np.random.default_rng(20261026)
_ = rng3.uniform(np.log(128), np.log(1152), N)
M25 = np.exp(rng3.uniform(np.log(4 * KiB), np.log(144 * KiB), N))
h25 = [r for r in S25["repos"] if r.get("status") == "ok" and r["stratum"] == "high"]
nb25 = {}
for r in h25: nb25[r["bin"]] = nb25.get(r["bin"], 0) + 1
sw25 = np.array([bins[r["bin"]]["total_count_now"] / nb25[r["bin"]] * r["stars_now"] for r in h25])
nd25 = S0_LC3 + np.minimum(D_ARCH, np.array([float(r["tar_bytes"]) for r in h25]))
o = np.argsort(nd25); cw = np.cumsum(sw25[o]) / sw25.sum(); k = np.searchsorted(nd25[o], M25, side="right")
q25 = np.where(k > 0, cw[np.maximum(k - 1, 0)], 0.0)
R["check_025_B"] = {"mean_q": float(q25.mean()), "dV_P2": dV(q25), "025_reported": 0.011543904358109425}

# 本回の読者（B：logU[4 KiB,144 KiB]）
rng4 = np.random.default_rng(20261028)
M = np.exp(rng4.uniform(np.log(4 * KiB), np.log(144 * KiB), N))
q_arch = q_from_need(need0, M)
R["as_archived"] = {"mean_q": float(q_arch.mean()), "dV_P2": dV(q_arch)}
R["dualwrite"] = {}
for c in CFG:
    nc = needc(c); cb = cost_bytes(c); cb25 = tar * rho25[c]
    R["dualwrite"][c] = {}
    for s, m in SETS.items():
        e = np.where(m, np.minimum(need0, nc), need0); qv = q_from_need(e, M)
        R["dualwrite"][c][s] = {"mean_q": float(qv.mean()), "dq": float(qv.mean() - q_arch.mean()), "dV_P2": dV(qv),
                                "cost_bytes": float((w * cb)[m].sum()), "cost_reels": float((w * cb)[m].sum() / REEL),
                                "cost_reels_rho025": float((w * cb25)[m].sum() / REEL)}
best_c = max(CFG, key=lambda c: R["dualwrite"][c]["all_high"]["dV_P2"])
R["best_config_all_high"] = best_c

# ---- 予算つき選択：星順 vs 欲張り（読者分布での期待増分/費用） ----
def budget_curve(c, budgets):
    nc = needc(c); cb = w * cost_bytes(c)
    Ms = np.sort(M)
    pM = lambda x: np.searchsorted(Ms, x, side="right") / N
    gain = sw * np.clip(pM(need0 - 1) - pM(nc - 1), 0, None) / W   # P(nc ≤ M < need0)
    orders = {"stars": np.lexsort((-tar, -st)), "greedy": np.argsort(-(gain / np.maximum(cb, 1)))}
    out = {}
    for lab, od in orders.items():
        cc = np.cumsum(cb[od]); gg = np.cumsum(gain[od]); res = []
        for B in budgets:
            k = np.searchsorted(cc, B * REEL)
            if k >= len(od): res.append(float(gg[-1])); continue
            prev_c = cc[k - 1] if k else 0.0; prev_g = gg[k - 1] if k else 0.0
            frac = (B * REEL - prev_c) / cb[od][k] if items[od[k]]["src"] == "sample025" else 0.0  # 標本の束は分割可、全数の repo は不可分
            res.append(float(prev_g + frac * gain[od][k]))
        out[lab] = res
    return out
BUD = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
R["budget"] = {"budgets_reels": BUD, **{c: budget_curve(c, BUD) for c in CFG}}

# ---- 仮説 ----
DW = R["dualwrite"]
R["hypotheses"] = {
 "H40a": {"value": R["star_share"]["featured15"], "rule": "<0.02", "supported": R["star_share"]["featured15"] < 0.02},
 "H40b": {"value": R["star_share"]["all_ge20k"], "rule": ">=0.20", "supported": R["star_share"]["all_ge20k"] >= 0.20},
 "H40c": {"value": DW["c2_64KiB_lc0"]["all_ge20k"]["cost_reels"], "rule": "<=1 reel", "supported": DW["c2_64KiB_lc0"]["all_ge20k"]["cost_reels"] <= 1},
 "H40d": {"value": R["compress"]["overhead_median_vs_c0"]["c1_64KiB_lc3"], "rule": ">=0.10", "supported": R["compress"]["overhead_median_vs_c0"]["c1_64KiB_lc3"] >= 0.10},
 "H40e": {"value": R["compress"]["lc0_vs_lc3_64KiB_median"], "rule": "<=0.05", "supported": R["compress"]["lc0_vs_lc3_64KiB_median"] <= 0.05},
 "H40f": {"value_all_high_best": DW[best_c]["all_high"]["dV_P2"], "value_ge20k_best": DW[best_c]["all_ge20k"]["dV_P2"],
          "rule": "all_high>=0.05 and ge20k>=0.03 (best config chosen on all_high)",
          "supported": DW[best_c]["all_high"]["dV_P2"] >= 0.05 and DW[best_c]["all_ge20k"]["dV_P2"] >= 0.03},
 "H40g": {"config_judged": "c2_64KiB_lc0 (H40c と同じ主設定。登録文で設定が未指定のため)",
          "greedy": R["budget"]["c2_64KiB_lc0"]["greedy"][BUD.index(0.1)], "stars": R["budget"]["c2_64KiB_lc0"]["stars"][BUD.index(0.1)],
          "rule": "greedy >= 2x stars at 0.1 reel",
          "supported": R["budget"]["c2_64KiB_lc0"]["greedy"][BUD.index(0.1)] >= 2 * R["budget"]["c2_64KiB_lc0"]["stars"][BUD.index(0.1)]}}

# ---- 図 ----
Mg = np.exp(np.linspace(np.log(1 * KiB), np.log(16 * MiB), 400))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].plot(Mg, q_from_need(need0, Mg), "k-", lw=2, label="as archived (xz 8 MiB)")
for s, ls in [("all_ge20k", "--"), ("all_high", "-")]:
    for c, col in [("c2_64KiB_lc0", "C0"), ("c3_16KiB_lc0", "C1"), ("c4_4KiB_lc0", "C2")]:
        e = np.where(SETS[s], np.minimum(need0, needc(c)), need0)
        ax[0].plot(Mg, q_from_need(e, Mg), color=col, ls=ls, label=f"{s} + {c}")
ax[0].axvspan(4 * KiB, 144 * KiB, color="gray", alpha=0.12, label="reader B: 4-144 KiB")
ax[0].set_xscale("log"); ax[0].set_ylim(0, 1); ax[0].set_xlabel("reader RAM (bytes)"); ax[0].set_ylabel("q (high tier, star weight)")
ax[0].legend(fontsize=7)
for c, col in [("c1_64KiB_lc3", "C3"), ("c2_64KiB_lc0", "C0"), ("c3_16KiB_lc0", "C1"), ("c4_4KiB_lc0", "C2"), ("u_uncompressed", "C4")]:
    xs = [DW[c][s]["cost_reels"] for s in SETS]; ys = [DW[c][s]["dV_P2"] for s in SETS]
    ax[1].plot(xs, ys, "o-", color=col, label=c)
for s in ("featured15", "all_ge20k", "all_high"):
    ax[1].annotate(s, (DW["c3_16KiB_lc0"][s]["cost_reels"], DW["c3_16KiB_lc0"][s]["dV_P2"]), fontsize=7)
ax[1].axhline(R["as_archived"]["dV_P2"], color="k", ls=":", label="as archived")
ax[1].set_xscale("log"); ax[1].set_xlabel("extra reels for dual-written copies (1 reel = 21 TB/186)"); ax[1].set_ylabel("dV (tablet guide), reader B")
ax[1].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{HERE}/fig_q_and_cost_dV.png", dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(7, 4.6))
for c, col in [("c2_64KiB_lc0", "C0"), ("c3_16KiB_lc0", "C1"), ("c4_4KiB_lc0", "C2")]:
    ax.plot(BUD, R["budget"][c]["greedy"], "-o", color=col, label=f"greedy {c}")
    ax.plot(BUD, R["budget"][c]["stars"], "--", color=col, label=f"by stars {c}")
ax.set_xscale("log"); ax.set_xlabel("budget (reels)"); ax.set_ylabel("expected q gain, reader B"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_budget_selection.png", dpi=120); plt.close()

R["runtime_seconds"] = time.time() - t0
OUTN = os.environ.get("OUT_JSON", "results.json")
json.dump(R, open(f"{HERE}/{OUTN}", "w"), ensure_ascii=False, indent=1)
pr = {k: R[k] for k in ("census", "star_share", "repo_count", "check_025_B", "as_archived", "best_config_all_high", "hypotheses")}
pr["compress"] = {k: v for k, v in R["compress"].items() if k != "per_repo"}
pr["dualwrite_summary"] = {c: {s: {k: round(v, 4) for k, v in d.items()} for s, d in DW[c].items()} for c in DW}
pr["budget"] = R["budget"]
print(json.dumps(pr, ensure_ascii=False, indent=1)); print("runtime", round(R["runtime_seconds"], 1))
