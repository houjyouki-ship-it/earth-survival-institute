"""研究026：窓を遅い補助記憶に置く LZMA 復号の時間と q（open_problems 81）。
入力：data/counts.json（tools/measure.py）、../025-q-xz-window/data/sample.json、../024-content-value-reader-guide/sources/text_sizes.json。
seed=20261026。numpy＋matplotlib のみ。事前登録：PREREGISTRATION.md。"""
import json, time, sys, platform, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
t0 = time.time(); SEED = 20261026
HERE = os.path.dirname(os.path.abspath(__file__)); R25 = f"{HERE}/../025-q-xz-window"
KiB, MiB = 1024, 1024 ** 2; YEAR = 365.25 * 86400
CNT = json.load(open(f"{HERE}/data/counts.json")); reps = CNT["repos"]
S = json.load(open(f"{R25}/data/sample.json"))
R = {"seed": SEED, "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()}
assert all(r["roundtrip_ok"] for r in reps); R["n_sample"] = len(reps); R["roundtrip_all_ok"] = True
CS = [c["C"] for c in reps[0]["C"]]; KC = {c: i for i, c in enumerate(CS)}
# ---- 事実入力（sources/） ----
T_ADD, T_MUL = 1 / 42000, 1 / 5000            # IBM 709
AUX = {"drum": {"L": 2.5e-3, "rate": 50 * 5 / 4.8e-3},   # IBM 650
       "disk": {"L": 0.600, "rate": 8800.0},              # IBM 350
       "tape": {"start_stop": 0.010, "dens": 200.0, "rew": 500.0, "rate": 15000.0}}  # IBM 727
BLK = CNT["lru"]["block"]; S0 = 16 * KiB; LRU_RAM = CNT["lru"]["K"] * BLK
R["inputs"] = {"t_add": T_ADD, "t_mul": T_MUL, "aux": AUX, "block": BLK, "S0": S0, "lru_ram": LRU_RAM, "C_grid": CS}
MAIN = {"n_ops": 15.0, "n_byte": 8.0, "s": 1.0}

def cpu_sec(r, n_ops, n_byte, s):
    return (r["nbit"] * (T_MUL + n_ops * T_ADD) + r["ndirect"] * 6 * T_ADD + r["out"] * n_byte * T_ADD) / s
def aux_sec(r, C, kind):
    c = r["C"][KC[C]]
    if kind == "tape":
        a = AUX["tape"]; return c["far"] * a["start_stop"] + 2 * c["fardist"] / (a["dens"] * a["rew"]) + c["farbytes"] / a["rate"]
    a = AUX[kind]; return c["lru_miss"] * (a["L"] + BLK / a["rate"])

# ---- 部分標本の統計 ----
per = []
for r in reps:
    row = {"name": r["full_name"], "tar": r["tar_bytes"], "out": r["out"], "bits_per_byte": r["nbit"] / r["out"],
           "direct_per_byte": r["ndirect"] / r["out"], "ratio": r["lzma_bytes"] / r["tar_bytes"],
           "cpu_main_s": cpu_sec(r, **MAIN)}
    row["cpu_min_per_MB"] = row["cpu_main_s"] / 60 / (r["out"] / 1e6)
    for C in CS:
        c = r["C"][KC[C]]
        row[f"far_per_MB_{C}"] = c["far"] / (r["out"] / 1e6); row[f"blk_per_MB_{C}"] = c["lru_miss"] / (r["out"] / 1e6)
        for k in AUX: row[f"{k}_s_{C}"] = aux_sec(r, C, k)
    per.append(row)
R["per_repo"] = per
big = [p for p in per if p["tar"] > 64 * KiB]
R["n_big"] = len(big)
R["cpu_min_per_MB_quantiles"] = {q: float(np.quantile([p["cpu_min_per_MB"] for p in per], q)) for q in (0.1, 0.5, 0.9)}
R["bits_per_byte_quantiles"] = {q: float(np.quantile([p["bits_per_byte"] for p in per], q)) for q in (0.1, 0.5, 0.9)}
R["ratio_aux_cpu_C64K"] = {k: {"median": float(np.median([p[f"{k}_s_65536"] / p["cpu_main_s"] for p in big])),
                               "frac_le_0.25": float(np.mean([p[f"{k}_s_65536"] / p["cpu_main_s"] <= 0.25 for p in big])),
                               "q10": float(np.quantile([p[f"{k}_s_65536"] / p["cpu_main_s"] for p in big], 0.1)),
                               "q90": float(np.quantile([p[f"{k}_s_65536"] / p["cpu_main_s"] for p in big], 0.9))} for k in AUX}
# 距離分布（全標本合計、参照数）
H = np.sum([r["hist_log2_dist"] for r in reps], axis=0).astype(float)
R["pooled_ref_dist_ccdf"] = {f"2^{h}": float(H[h + 1:].sum() / H.sum()) for h in range(0, 24)}
R["pooled_far_frac_refs"] = {str(C): float(sum(r["C"][KC[C]]["far"] for r in reps) / sum(r["C"][0]["far"] for r in reps)) for C in CS}
R["pooled_far_bytes_frac_out"] = {str(C): float(sum(r["C"][KC[C]]["farbytes"] for r in reps) / sum(r["out"] for r in reps)) for C in CS}
R["pooled_blk_per_MB"] = {str(C): float(sum(r["C"][KC[C]]["lru_miss"] for r in reps) / sum(r["out"] for r in reps) * 1e6) for C in CS}

# ---- 母集団（025 の高層） ----
ok = [r for r in S["repos"] if r.get("status") == "ok"]; high = [r for r in ok if r["stratum"] == "high"]
bins = S["bins"]; nb = {b: sum(1 for r in high if r["bin"] == b) for b in bins}
w_cnt = np.array([bins[r["bin"]]["total_count_now"] / nb[r["bin"]] for r in high])
tar_h = np.array([float(r["tar_bytes"]) for r in high]); stars = np.array([float(r["stars_now"]) for r in high]); w_star = w_cnt * stars
EDGES = [0, 1e4, 1e5, 1e6, 4e6, 1.6e7, 4.1e7]
def cls(t): return min(int(np.searchsorted(EDGES, t, side="left")) - 1, len(EDGES) - 2)
samp_cls = np.array([cls(r["tar_bytes_tree_est"]) for r in reps]); pop_cls = np.array([cls(max(t, 1)) for t in tar_h])
def class_rates(n_ops, n_byte, s):
    """区分ごとのバイトあたり率（中央値）：cpu, 各 C・補助の aux"""
    out = {}
    for k in range(len(EDGES) - 1):
        ii = [i for i in range(len(reps)) if samp_cls[i] == k]
        cpu = np.median([cpu_sec(reps[i], n_ops, n_byte, s) / reps[i]["out"] for i in ii])
        aux = {(C, a): float(np.median([aux_sec(reps[i], C, a) / reps[i]["out"] for i in ii])) for C in CS for a in AUX}
        out[k] = (float(cpu), aux)
    return out
def repo_times(C, kind, rates):
    cpu = np.array([rates[k][0] for k in pop_cls]) * tar_h
    aux = np.array([rates[k][1][(C, kind)] for k in pop_cls]) * tar_h
    aux = np.where(tar_h <= C, 0.0, aux)   # 窓が高速 RAM に収まる
    return cpu + aux
def q_budget(T, Y):
    o = np.argsort(-(stars / np.maximum(T, 1e-9))); cw = np.cumsum((w_cnt * T)[o]); ws = w_star[o]
    budget = Y * YEAR; full = cw <= budget
    q = ws[full].sum()
    k = int(full.sum())
    if k < len(o):  # 境界の repo は割合で
        prev = cw[k - 1] if k > 0 else 0.0; q += ws[k] * max(0.0, (budget - prev) / (w_cnt[o][k] * T[o][k]))
    return float(q / w_star.sum())
rates_main = class_rates(**MAIN)
R["population"] = {"n_high_sample": len(high), "repos_represented": float(w_cnt.sum())}
YS = [0.1, 1, 3, 10, 30, 100]
R["q_main"] = {}
for kind in AUX:
    R["q_main"][kind] = {}
    for C in CS:
        T = repo_times(C, kind, rates_main)
        R["q_main"][kind][str(C)] = {"total_machine_years_all": float((w_cnt * T).sum() / YEAR), **{f"Y{y}": q_budget(T, y) for y in YS},
                                      "median_repo_days": float(np.median(T) / 86400)}
T_cpu_only = np.array([rates_main[k][0] for k in pop_cls]) * tar_h
R["q_cpu_only"] = {"total_machine_years_all": float((w_cnt * T_cpu_only).sum() / YEAR), **{f"Y{y}": q_budget(T_cpu_only, y) for y in YS}}
# 補助記憶の容量の壁（窓 min(8 MiB, TAR) を置ける容量 A）
CAP = {"IBM650_drum_4000w_35KB": 35000, "UNIVAC_drum_135KB": 135000, "IBM350_1unit_3.75MB": 3.75e6, "IBM350_2units": 7.5e6, "IBM350_3units": 11.25e6}
R["q_capacity"] = {k: float(w_star[np.minimum(8 * MiB, tar_h) <= A].sum() / w_star.sum()) for k, A in CAP.items()}
R["units_for_8MiB"] = {k: float(np.ceil(8 * MiB / A)) for k, A in CAP.items() if "units" not in k}
# CPU 等の不確実性（200 組）
rng = np.random.default_rng(SEED)
def logu(lo, hi, n=None): return np.exp(rng.uniform(np.log(lo), np.log(hi), n))
draws = []
for _ in range(200):
    th = {"n_ops": float(logu(8, 30)), "n_byte": float(logu(4, 15)), "s": float(logu(0.25, 1))}
    rt = class_rates(**th)
    draws.append({kind: q_budget(repo_times(65536, kind, rt), 10) for kind in AUX})
R["q_C64K_Y10_uncert"] = {kind: {q: float(np.quantile([d[kind] for d in draws], q)) for q in (0.05, 0.5, 0.95)} for kind in AUX}

# ---- 024 のモデル（025 の sim.py と同じ式・同じ乱数系列）で ΔV ----
rng2 = np.random.default_rng(20261024)
N = 200_000; Hh = 1000.0
def logu2(lo, hi, n=N): return np.exp(rng2.uniform(np.log(lo), np.log(hi), n))
SIZES = json.load(open(f"{HERE}/../024-content-value-reader-guide/sources/text_sizes.json"))
G_LO = SIZES["wp_total"] / 1e6; G_HI = 2 * SIZES["rdh4"]["utf8"] / 1e6; P_H = 0.9305; D_HAND = 1.9
Z = rng2.standard_normal((3, N))
T1 = 20 * np.exp(0.7 * Z[0]); T2 = T1 + 30 * np.exp(0.7 * Z[1])
claim = logu2(500, 2000); f_store = logu2(0.3, 1.0)
life = claim * f_store / np.log(2) ** 0.5 * (-np.log(rng2.uniform(size=N))) ** 0.5
g_req = logu2(G_LO, G_HI); P_max = rng2.uniform(0.3, 0.9, N); kappa = logu2(0.5, 2.0)
t_build = 20 * np.exp(0.7 * rng2.standard_normal(N))
U_t = rng2.uniform(size=N); U_h = rng2.uniform(size=N); U_sep = rng2.uniform(size=N); p_sep = rng2.uniform(0.1, 0.5, N)
L_core = 50 * 11 / 12 * logu2(1, 10); Vv = logu2(550, 2400); t_tube = logu2(0.5, 5); Uup = logu2(3, 30)
N_min = Uup * (L_core + Vv * 3 * t_tube / 250) / (10.0 * logu2(0.1, 0.4) * logu2(0.005, 0.05))
pop_ok = N_min < logu2(0.5e6, 5e6)
def V_of(m23, qv, g_hand, tablet=False):
    T3 = T2 + m23 * np.exp(0.8 * Z[2]); gen = (T3 < life) & (T3 <= Hh)
    cov = np.zeros(N); avail = np.zeros(N, bool)
    if g_hand > 0:
        ho = U_h < P_H; cov = np.where(ho, np.minimum(1.0, g_hand / g_req), cov); avail |= ho
    if tablet:
        to = U_sep > p_sep; cov = np.where(to, 1.0, cov); avail |= to
    Tt = T2 + t_build
    tgt = avail & pop_ok & (U_t < P_max * cov ** kappa) & (Tt < life) & (Tt <= Hh)
    return float(np.mean(np.maximum(qv * tgt, gen.astype(float))))
def dV(qv, m23=943):
    v0 = V_of(m23, qv, 0); return {"P0": v0, "dV_P1_f1": V_of(m23, qv, D_HAND) - v0, "dV_P2_tablet": V_of(m23, qv, 0, tablet=True) - v0}
R["check_024_q0.3"] = dV(0.3)
def q025(M):  # 025 の q(M)（窓を RAM に置く）
    nd = S0 + np.minimum(8 * MiB, tar_h); o = np.argsort(nd); cw = np.cumsum(w_star[o]) / w_star.sum()
    k = np.searchsorted(nd[o], M, side="right"); return np.where(k > 0, cw[np.maximum(k - 1, 0)], 0.0)
rng3 = np.random.default_rng(SEED + 1)
M = np.exp(rng3.uniform(np.log(4 * KiB), np.log(144 * KiB), N))
Cav = M - S0 - LRU_RAM
q_base = q025(M)
R["dV_reader_4K_144K"] = {"ram_only_025": {"mean_q": float(q_base.mean()), **dV(q_base)}}
for kind in AUX:
    qC = {C: R["q_main"][kind][str(C)]["Y10"] for C in CS}
    qv = np.zeros(N)
    for C in CS:  # 使える最大の格子 C（保守的）
        qv = np.where(Cav >= C, qC[C], qv)
    qv = np.where(Cav < 0, q_base, np.maximum(qv, q_base))
    R["dV_reader_4K_144K"][f"aux_{kind}_Y10"] = {"mean_q": float(qv.mean()), **dV(qv)}
    # 容量の制約つき（ドラムは UNIVAC 135 KB×n 台、ここでは 1 台と 8 台）
if True:
    for lab, A in [("drum_cap_135KB", 135000), ("drum_cap_8x135KB", 8 * 135000), ("disk_cap_3.75MB", 3.75e6)]:
        kind = "drum" if "drum" in lab else "disk"
        fits = np.minimum(8 * MiB, tar_h) <= A
        T = repo_times(65536, kind, rates_main); T = np.where(fits, T, np.inf)
        qY = q_budget(np.where(np.isfinite(T), T, 1e30), 10)
        qv = np.where(Cav >= 65536, qY, q_base); qv = np.maximum(qv, q_base)
        R["dV_reader_4K_144K"][f"{lab}_C64K_Y10"] = {"q_at_C64K": qY, "mean_q": float(qv.mean()), **dV(qv)}

D = R["dV_reader_4K_144K"]; QM = R["q_main"]; RA = R["ratio_aux_cpu_C64K"]
H38 = {
 "H38a": {"value": RA["drum"]["frac_le_0.25"], "rule": ">=0.80 of repos with aux/cpu<=0.25 (drum, C=64KiB)", "supported": RA["drum"]["frac_le_0.25"] >= 0.80},
 "H38b": {"value": RA["disk"]["median"], "rule": "median aux/cpu > 1 (disk)", "supported": RA["disk"]["median"] > 1},
 "H38c": {"value": R["cpu_min_per_MB_quantiles"][0.5], "rule": "5<=median min/MB<=100", "supported": 5 <= R["cpu_min_per_MB_quantiles"][0.5] <= 100},
 "H38d": {"value": QM["drum"]["65536"]["Y10"], "rule": ">=0.9", "supported": QM["drum"]["65536"]["Y10"] >= 0.9},
 "H38e": {"value": QM["disk"]["65536"]["Y10"], "rule": ">=0.5", "supported": QM["disk"]["65536"]["Y10"] >= 0.5},
 "H38f": {"value": QM["tape"]["65536"]["Y10"], "rule": "<0.3", "supported": QM["tape"]["65536"]["Y10"] < 0.3},
 "H38g": {"value": D["aux_drum_Y10"]["dV_P2_tablet"], "rule": ">=0.06", "supported": D["aux_drum_Y10"]["dV_P2_tablet"] >= 0.06},
}
R["H38"] = H38

# ---- 図 ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
xs = np.arange(0, 24); ax[0].semilogy(2.0 ** xs, [R["pooled_ref_dist_ccdf"][f"2^{h}"] for h in xs], "k-o", ms=3, label="pooled refs: P(dist > x)")
ax[0].set_xscale("log", base=2); ax[0].set_xlabel("match distance x (bytes)"); ax[0].set_ylabel("P(distance > x)")
ax2 = ax[0].twinx()
for p in per:
    if p["tar"] > 64 * KiB:
        ax2.plot([max(C, 256) for C in CS], [max(p[f"blk_per_MB_{C}"], 1) for C in CS], color="tab:blue", alpha=0.25, lw=0.8)
ax2.set_yscale("log"); ax2.set_ylabel("aux block fetches per MB (LRU, each repo >64KiB)", color="tab:blue")
ax[0].set_title("LZMA (8 MiB dict) reference distances, 30 real repos")
for kind, col in zip(AUX, ["tab:green", "tab:orange", "tab:red"]):
    ax[1].plot(YS, [QM[kind]["65536"][f"Y{y}"] for y in YS], "-o", color=col, label=f"window on {kind}, C=64KiB")
ax[1].plot(YS, [R["q_cpu_only"][f"Y{y}"] for y in YS], "k--", label="CPU only (no aux cost)")
ax[1].axhline(float(q025(np.array([84 * KiB]))[0]), color="gray", ls=":", label="025: window in RAM, M=84KiB")
ax[1].set_xscale("log"); ax[1].set_xlabel("decode budget Y (machine-years, IBM 709-class)"); ax[1].set_ylabel("q (star weight, >=250 stars)")
ax[1].set_ylim(0, 1.02); ax[1].legend(fontsize=8); ax[1].set_title("q vs time budget")
plt.tight_layout(); plt.savefig(f"{HERE}/fig_distance_and_q_time.png", dpi=120); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
labs = list(CAP.keys()); ax[0].bar(range(len(labs)), [R["q_capacity"][l] for l in labs], color="tab:purple")
ax[0].set_xticks(range(len(labs))); ax[0].set_xticklabels(labs, rotation=20, fontsize=7); ax[0].set_ylabel("q if aux capacity = A"); ax[0].set_title("capacity wall: window min(8MiB,TAR) must fit aux store")
kk = list(D.keys()); ax[1].bar(range(len(kk)), [D[k]["dV_P2_tablet"] for k in kk], color="tab:cyan")
ax[1].axhline(R["check_024_q0.3"]["dV_P2_tablet"], color="k", ls=":", label="024 (q=0.3)")
ax[1].set_xticks(range(len(kk))); ax[1].set_xticklabels(kk, rotation=25, fontsize=7); ax[1].set_ylabel("dV (tablet guide), reader RAM 4-144 KiB"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{HERE}/fig_capacity_and_dV.png", dpi=120); plt.close()
R["runtime_seconds"] = round(time.time() - t0, 2)
json.dump(R, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: R[k] for k in ["cpu_min_per_MB_quantiles", "bits_per_byte_quantiles", "ratio_aux_cpu_C64K", "pooled_far_frac_refs", "pooled_blk_per_MB",
                  "q_cpu_only", "q_capacity", "units_for_8MiB", "q_C64K_Y10_uncert", "check_024_q0.3", "dV_reader_4K_144K", "H38", "runtime_seconds"]}, ensure_ascii=False, indent=1))
print(json.dumps({k: {C: v for C, v in R["q_main"][k].items() if C in ("0", "4096", "65536", "1048576")} for k in R["q_main"]}, indent=1))
