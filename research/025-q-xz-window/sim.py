"""研究025：狙い撃ち読取装置が届ける価値の割合 q(M) と、024 の ΔV の再評価（open_problems 79）
事前登録：PREREGISTRATION.md。入力は data/sample.json・data/compress.json（collect.py・compress.py の出力）と
../024-content-value-reader-guide/sources/text_sizes.json。seed 固定、numpy＋matplotlib のみ。"""
import json, time, sys, platform, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
SEED = 20261025
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f"{HERE}/data/sample.json"))
C = json.load(open(f"{HERE}/data/compress.json"))
KiB, MiB = 1024, 1024 ** 2
S0_MAIN, D_MAIN = 16 * KiB, 8 * MiB
ANCHORS = {"Manchester Baby 32x32bit": 128, "EDSAC 512x18bit": 1152, "Whirlwind 2048x16bit": 4096,
           "IBM 704 4096x36bit": 18432, "1 MiB": MiB, "xz -6 decoder 9 MiB": 9 * MiB}
R = {"seed": SEED, "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
     "inputs": {"S0_main": S0_MAIN, "D_main": D_MAIN, "anchors_bytes": ANCHORS}}

ok = [r for r in S["repos"] if r.get("status") == "ok"]
R["sample_counts"] = {st: {"attempted": sum(1 for r in S["repos"] if r["stratum"] == st),
                           "ok": sum(1 for r in ok if r["stratum"] == st),
                           "truncated": sum(1 for r in ok if r["stratum"] == st and r["truncated"])}
                      for st in ("high", "low0", "low1")}

def arr(rs, k): return np.array([float(r[k]) for r in rs])

# ---- 層ごとの重み ----
high = [r for r in ok if r["stratum"] == "high"]
bins = S["bins"]
nb = {b: sum(1 for r in high if r["bin"] == b) for b in bins}
w_cnt_h = np.array([bins[r["bin"]]["total_count_now"] / nb[r["bin"]] for r in high])
tar_h = arr(high, "tar_bytes"); stars_h = arr(high, "stars_now")
bin_h = np.array([r["bin"] for r in high])
low = {st: [r for r in ok if r["stratum"] == st] for st in ("low0", "low1")}
# 低層：日をランダムに選びその日の結果から1件 → 選ばれる確率 ∝ 1/その日の件数。重み＝その日の件数
w_low = {st: arr(rs, "day_total_now") for st, rs in low.items()}
tar_low = {st: arr(rs, "tar_bytes") for st, rs in low.items()}

def need(tar, D=D_MAIN, S0=S0_MAIN): return S0 + np.minimum(D, tar)
def qfrac(tar, w, M, D=D_MAIN, S0=S0_MAIN): return float(np.sum(w * (need(tar, D, S0) <= M)) / np.sum(w))

WEIGHTS = {"high_star": (tar_h, w_cnt_h * stars_h), "high_repo": (tar_h, w_cnt_h), "high_bytes": (tar_h, w_cnt_h * tar_h),
           "low0_repo": (tar_low["low0"], w_low["low0"]), "low0_bytes": (tar_low["low0"], w_low["low0"] * tar_low["low0"]),
           "low1_repo": (tar_low["low1"], w_low["low1"]), "low1_bytes": (tar_low["low1"], w_low["low1"] * tar_low["low1"])}
M_GRID = np.unique(np.concatenate([np.logspace(np.log10(128), np.log10(128 * MiB), 61), list(ANCHORS.values()),
                                   [64 * KiB, 144 * KiB, MiB + 16 * KiB]]))
rng = np.random.default_rng(SEED)
B = 2000
def boot_q(name, M, D=D_MAIN, S0=S0_MAIN):
    tar, w = WEIGHTS[name]
    n = len(tar); vals = []
    if name.startswith("high"):
        idx_b = [np.where(bin_h == b)[0] for b in bins]
        for _ in range(B):
            ii = np.concatenate([rng.choice(ix, len(ix)) for ix in idx_b if len(ix)])
            vals.append(qfrac(tar[ii], w[ii], M, D, S0))
    else:
        for _ in range(B):
            ii = rng.integers(0, n, n); vals.append(qfrac(tar[ii], w[ii], M, D, S0))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

R["q_curves_main"] = {k: {str(int(M)): qfrac(*WEIGHTS[k], M) for M in M_GRID} for k in WEIGHTS}
KEYM = {"Baby_128B": 128, "EDSAC_1152B": 1152, "Whirlwind_4KiB": 4096, "IBM704_18432B": 18432, "64KiB": 64 * KiB,
        "144KiB": 144 * KiB, "1MiB_plus_S0": MiB + 16 * KiB, "9MiB": 9 * MiB}
R["q_at_anchors"] = {k: {m: {"q": qfrac(*WEIGHTS[k], M), "ci95": boot_q(k, M)} for m, M in KEYM.items()} for k in WEIGHTS}
R["q_sensitivity"] = {}
for lab, D, S0 in [("D256KiB_xz-0", 256 * KiB, S0_MAIN), ("D64MiB_xz-9", 64 * MiB, S0_MAIN), ("S0_12KiB", D_MAIN, 12 * KiB)]:
    R["q_sensitivity"][lab] = {k: {m: qfrac(*WEIGHTS[k], M, D, S0) for m, M in KEYM.items()} for k in WEIGHTS}
R["tar_size_quantiles"] = {k: {f"p{p}": float(np.percentile(v, p)) for p in (10, 25, 50, 75, 90)}
                           for k, v in [("high", tar_h), ("low0", tar_low["low0"]), ("low1", tar_low["low1"])]}
R["high_by_bin_median_tar"] = {b: float(np.median(tar_h[bin_h == b])) if nb[b] else None for b in bins}
R["truncated_lower_bound_below_D"] = int(sum(1 for r in ok if r["truncated"] and r["tar_bytes"] < D_MAIN))

# ---- 圧縮の部分標本：辞書を小さくする費用 ----
cr = [r for r in C["repos"] if r["status"] == "ok"]
tot = {d: sum(r["xz_bytes_by_dict"][str(d)] for r in cr) for d in C["dicts"]}
R["compress"] = {"n": len(cr), "n_failed": sum(1 for r in C["repos"] if r["status"] != "ok"),
    "total_tar_bytes": sum(r["tar_bytes_actual"] for r in cr),
    "total_xz_bytes_by_dict": {str(d): v for d, v in tot.items()},
    "overhead_vs_8MiB": {str(d): tot[d] / tot[8 * MiB] - 1 for d in C["dicts"]},
    "ratio_tar_over_xz8MiB": sum(r["tar_bytes_actual"] for r in cr) / tot[8 * MiB],
    "per_repo_overhead_64KiB": {f"p{p}": float(np.percentile([r["xz_bytes_by_dict"]["65536"] / r["xz_bytes_by_dict"][str(8 * MiB)] - 1 for r in cr], p)) for p in (10, 50, 90)},
    "tree_est_vs_actual_tar_median_ratio": float(np.median([r["tar_bytes_tree_est"] / r["tar_bytes_actual"] for r in cr]))}

# ---- 024 のモデルを q(M) で回す（024 の sim.py と同じ乱数の順序・式） ----
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
rng3 = np.random.default_rng(SEED + 1)  # 本回追加の乱数は別系列（024 の系列を乱さない）
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
def dV(qv, m23=943):
    v0 = V_of(m23, qv, 0)
    return {"P0": v0, "dV_P1_f1": V_of(m23, qv, D_HAND) - v0, "dV_P2_tablet": V_of(m23, qv, 0, tablet=True) - v0}
R["check_024_q0.3"] = dV(0.3)
tar_s, w_s = WEIGHTS["high_star"]; need_s = need(tar_s)
def q_of_M(Marr, tar=tar_s, w=w_s, D=D_MAIN):
    nd = S0_MAIN + np.minimum(D, tar); o = np.argsort(nd); cw = np.cumsum(w[o]) / np.sum(w)
    k = np.searchsorted(nd[o], Marr, side="right"); return np.where(k > 0, cw[np.maximum(k - 1, 0)], 0.0)
SCEN = {"A_Baby_EDSAC_128B-1152B": (128, 1152), "B_Whirlwind_704ext_4KiB-144KiB": (4 * KiB, 144 * KiB),
        "C_144KiB-1MiB": (144 * KiB, MiB), "D_1MiB-9MiB": (MiB, 9 * MiB)}
R["dV_by_reader_ram"] = {}
for lab, (lo, hi) in SCEN.items():
    M = np.exp(rng3.uniform(np.log(lo), np.log(hi), N))
    qv = q_of_M(M); R["dV_by_reader_ram"][lab] = {"mean_q": float(qv.mean()), **dV(qv)}
    qv_r = q_of_M(M, D=64 * KiB)  # 運営者が辞書 64 KiB で再圧縮した場合
    R["dV_by_reader_ram"][lab]["recompressed_64KiB"] = {"mean_q": float(qv_r.mean()), **dV(qv_r)}
# 低層の repo 重み（一般 repo の代表）でも
tl, wl = WEIGHTS["low1_repo"]
for lab, (lo, hi) in SCEN.items():
    M = np.exp(rng3.uniform(np.log(lo), np.log(hi), N)); qv = q_of_M(M, tl, wl)
    R["dV_by_reader_ram"][lab]["low1_repo_weight"] = {"mean_q": float(qv.mean()), **dV(qv)}

QA = R["q_at_anchors"]
H37 = {
 "H37a": {"value": QA["high_star"]["64KiB"]["q"], "rule": "<0.10", "supported": QA["high_star"]["64KiB"]["q"] < 0.10},
 "H37b": {"value": QA["low1_repo"]["64KiB"]["q"], "value_low0": QA["low0_repo"]["64KiB"]["q"], "rule": ">=0.30 (low strata, repo weight)",
          "supported": min(QA["low1_repo"]["64KiB"]["q"], QA["low0_repo"]["64KiB"]["q"]) >= 0.30},
 "H37c": {"value": max(QA[k]["IBM704_18432B"]["q"] for k in WEIGHTS), "rule": "<0.05 all", "supported": max(QA[k]["IBM704_18432B"]["q"] for k in WEIGHTS) < 0.05},
 "H37d": {"value": R["compress"]["overhead_vs_8MiB"]["65536"], "rule": "<0.15", "supported": R["compress"]["overhead_vs_8MiB"]["65536"] < 0.15},
 "H37e": {"value": R["dV_by_reader_ram"]["B_Whirlwind_704ext_4KiB-144KiB"]["dV_P2_tablet"], "rule": "<0.03",
          "supported": R["dV_by_reader_ram"]["B_Whirlwind_704ext_4KiB-144KiB"]["dV_P2_tablet"] < 0.03},
 "H37f": {"value": QA["high_star"]["1MiB_plus_S0"]["q"], "rule": ">=0.30", "supported": QA["high_star"]["1MiB_plus_S0"]["q"] >= 0.30},
}
R["hypotheses"] = H37

# ---- 図 ----
fig, ax = plt.subplots(figsize=(9, 5))
sty = {"high_star": ("C0", "-"), "high_repo": ("C0", "--"), "high_bytes": ("C0", ":"), "low0_repo": ("C2", "--"),
       "low1_repo": ("C1", "--"), "low1_bytes": ("C1", ":"), "low0_bytes": ("C2", ":")}
for k, (c, ls) in sty.items():
    ax.plot(M_GRID, [R["q_curves_main"][k][str(int(M))] for M in M_GRID], color=c, ls=ls, label=k)
for n, v in ANCHORS.items():
    ax.axvline(v, color="gray", lw=0.6); ax.text(v, 1.01, n, rotation=60, fontsize=7, va="bottom")
ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_xlabel("reader RAM M (bytes)")
ax.set_ylabel("q(M): share decompressible (xz -6, need = 16 KiB + min(8 MiB, tar))"); ax.legend(fontsize=8, loc="center right")
plt.tight_layout(); plt.savefig(f"{HERE}/fig_q_vs_ram.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ds = C["dicts"]
ax[0].plot(ds, [R["compress"]["overhead_vs_8MiB"][str(d)] * 100 for d in ds], "o-")
ax[0].set_xscale("log"); ax[0].set_xlabel("LZMA2 dictionary (bytes)"); ax[0].set_ylabel("extra xz bytes vs 8 MiB dict (%)")
ax[0].set_title(f"subsample n={len(cr)} repos (sum of bytes)")
labs = list(SCEN); x = np.arange(len(labs))
ax[1].bar(x - 0.27, [R["dV_by_reader_ram"][l]["dV_P2_tablet"] for l in labs], 0.27, label="as archived (8 MiB), star weight")
ax[1].bar(x, [R["dV_by_reader_ram"][l]["recompressed_64KiB"]["dV_P2_tablet"] for l in labs], 0.27, label="if recompressed 64 KiB")
ax[1].bar(x + 0.27, [R["dV_by_reader_ram"][l]["low1_repo_weight"]["dV_P2_tablet"] for l in labs], 0.27, label="as archived, low-star repo weight")
ax[1].axhline(R["check_024_q0.3"]["dV_P2_tablet"], color="k", ls=":", label="024 (q=0.3 assumed)")
ax[1].set_xticks(x); ax[1].set_xticklabels([l.split("_", 1)[1] for l in labs], fontsize=8, rotation=10)
ax[1].set_ylabel("dV (tablet guide), m23=943"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_dict_cost_and_dV.png", dpi=120); plt.close()

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({"counts": R["sample_counts"], "q_at_anchors": {k: {m: round(v["q"], 3) for m, v in d.items()} for k, d in QA.items()},
                  "tar_q": R["tar_size_quantiles"], "compress": R["compress"], "check024": R["check_024_q0.3"],
                  "dV": R["dV_by_reader_ram"], "H37": H37}, ensure_ascii=False, indent=1))
print("runtime", round(R["runtime_seconds"], 1))
