"""研究027：テープに窓を置く LZMA 復号の手順と時間（open_problems 83）。
入力：data/traces.npz（tools/measure_trace.py：距離 >16 KiB の参照の (出力位置, 距離, 長さ)）、
../026-lzma-aux-window/data/counts.json、../025-q-xz-window/data/sample.json、../024-content-value-reader-guide/sources/text_sizes.json。
seed=20261027（026 と同じ乱数系列を ΔV に使う）。numpy＋matplotlib のみ。事前登録：PREREGISTRATION.md。"""
import json, time, sys, platform, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
t0 = time.time(); SEED = 20261027
HERE = os.path.dirname(os.path.abspath(__file__)); R25 = f"{HERE}/../025-q-xz-window"; R26 = f"{HERE}/../026-lzma-aux-window"
KiB, MiB = 1024, 1024 ** 2; YEAR = 365.25 * 86400
CNT = json.load(open(f"{R26}/data/counts.json")); reps = CNT["repos"]
TR = np.load(f"{HERE}/data/traces.npz")
S = json.load(open(f"{R25}/data/sample.json"))
R = {"seed": SEED, "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()}
CS26 = [c["C"] for c in reps[0]["C"]]; KC = {c: i for i, c in enumerate(CS26)}
# ---- 事実入力（sources/IBM_727.wiki、IBM 709 は 026） ----
DENS, V_RW, V_REW, T_SS, RATE = 200.0, 75.0, 500.0, 0.010, 15000.0
T_ADD, T_MUL = 1 / 42000, 1 / 5000
BLK, LRU_K, S0 = 512, 8, 16 * KiB; LRU_RAM = LRU_K * BLK; WCH = 32 * KiB
MAIN = {"n_ops": 15.0, "n_byte": 8.0, "s": 1.0}
R["inputs"] = {"dens_cpi": DENS, "v_rw_ips": V_RW, "v_rewind_ips": V_REW, "t_start_stop": T_SS, "rate_cps": RATE, "block": BLK, "lru_k": LRU_K, "write_chunk": WCH}
def cpu_sec(r, n_ops, n_byte, s):
    return (r["nbit"] * (T_MUL + n_ops * T_ADD) + r["ndirect"] * 6 * T_ADD + r["out"] * n_byte * T_ADD) / s

# ---- LRU（026 の C++ と同じ規則）を Python で再計算し、取得列を作る ----
def misses(idx, C):
    pos = TR[f"pos_{idx}"]; dist = TR[f"dist_{idx}"]; ln = TR[f"len_{idx}"]
    sel = dist > C; pos, dist, ln = pos[sel], dist[sel], ln[sel]
    lru = []; mp = []; mb = []
    for p, d, L in zip(pos.tolist(), dist.tolist(), ln.tolist()):
        src = p - d; L = min(L, d)
        for b in range(src // BLK, (src + L - 1) // BLK + 1):
            if b in lru:
                lru.remove(b); lru.insert(0, b)
            else:
                mp.append(p); mb.append(b); lru.insert(0, b)
                if len(lru) > LRU_K: lru.pop()
    return np.array(mp, dtype=np.int64), np.array(mb, dtype=np.int64)
CGRID = [16 * KiB, 64 * KiB, 256 * KiB, 1 * MiB]
MISS = {}; val = {}
for r in reps:
    for C in CGRID:
        MISS[(r["idx"], C)] = misses(r["idx"], C)
        val[(r["idx"], C)] = len(MISS[(r["idx"], C)][0]) == r["C"][KC[C]]["lru_miss"]
R["validation_lru_identical_026"] = bool(all(val.values())); R["validation_n_checks"] = len(val)
# N0 の再計算（026 の素朴モデル、計数対象の参照のみ）と 026 の式の値
def n0_sec(idx, C):
    dist = TR[f"dist_{idx}"]; ln = TR[f"len_{idx}"]; cnt = TR[f"cnt_{idx}"]
    s = (dist > C) & (cnt == 1)
    return float(s.sum() * T_SS + 2 * dist[s].sum() / (DENS * V_REW) + ln[s].sum() / RATE)
def n0_026(r, C):
    c = r["C"][KC[C]]; return c["far"] * T_SS + 2 * c["fardist"] / (DENS * V_REW) + c["farbytes"] / RATE

# ---- テープ手順のシミュレーション ----
def tape_sec(idx, C, out, K=1, L=None, v_sp=V_RW, gap=0.0):
    """K 台。区画 j（長さ L バイト）を装置 j mod K に追記。K=1 は S1（L は無関係）。
    戻り値：(合計秒, 内訳 dict)。位置はインチ。"""
    mp, mb = MISS[(idx, C)]
    if L is None or K == 1: L = 1 << 40
    bl_in = BLK / DENS + gap
    def loc(byte):  # 装置と装置内のバイト位置
        j = byte // L; return int(j % K), int((j // K) * L + byte % L)
    head = [0.0] * K; end = [0.0] * K
    t_move = t_read = t_write = 0.0; n_move = 0
    def move(d, x):
        nonlocal t_move, n_move
        h = head[d]
        if h == x: return
        c = T_SS + abs(x - h) / v_sp
        if x < h: c = min(c, T_SS + h / V_REW + T_SS + x / v_sp)
        t_move += c; n_move += 1; head[d] = x
    def write_chunk(w0, n):
        nonlocal t_write
        d, off = loc(w0); move(d, end[d]); nb = n / BLK
        t_write += T_SS + n / RATE + nb * gap / V_RW; end[d] += nb * bl_in; head[d] = end[d]
    w_done = 0; i = 0; nm = len(mp)
    while True:
        nxt = w_done + WCH
        # 位置 < nxt の取得を先に処理（その時点までに書かれた範囲だけを読む）
        while i < nm and mp[i] < nxt:
            d, off = loc(int(mb[i]) * BLK); x = (off // BLK) * bl_in
            move(d, x); t_read += T_SS + BLK / RATE; head[d] = x + bl_in; i += 1
        if nxt > out:
            if i >= nm: break
            nxt = out
        if nxt <= w_done: break
        write_chunk(w_done, min(WCH, out - w_done)); w_done = min(nxt, out)
        if w_done >= out and i >= nm: break
    return t_move + t_read + t_write, {"move": t_move, "read": t_read, "write": t_write, "n_move": n_move, "n_fetch": nm}

big = [r for r in reps if r["tar_bytes"] > 64 * KiB]
CPU = {r["idx"]: cpu_sec(r, **MAIN) for r in reps}
R["n_big"] = len(big)
R["validation_N0"] = {"sum_trace": sum(n0_sec(r["idx"], 65536) for r in reps), "sum_026": sum(n0_026(r, 65536) for r in reps)}
R["validation_N0"]["rel_diff"] = R["validation_N0"]["sum_trace"] / R["validation_N0"]["sum_026"] - 1

KS = [2, 3, 4, 8]; LS = [64 * KiB, 256 * KiB, 1 * MiB, 4 * MiB]
CONF = {"N0": None, "S1": dict(K=1)}
for K in KS:
    for L in LS: CONF[f"K{K}_L{L // KiB}K"] = dict(K=K, L=L)
def run(conf, C, v_sp=V_RW, gap=0.0):
    out = {}
    for r in reps:
        if conf is None: out[r["idx"]] = (n0_sec(r["idx"], C), {})
        else: out[r["idx"]] = tape_sec(r["idx"], C, r["out"], v_sp=v_sp, gap=gap, **conf)
    return out
def med_ratio(res): return float(np.median([res[r["idx"]][0] / CPU[r["idx"]] for r in big]))
RES = {}; R["ratio_aux_cpu_C64K_main"] = {}
for name, conf in CONF.items():
    RES[(name, 65536, "main")] = run(conf, 65536)
    rr = [RES[(name, 65536, "main")][r["idx"]][0] / CPU[r["idx"]] for r in big]
    R["ratio_aux_cpu_C64K_main"][name] = {"median": float(np.median(rr)), "q10": float(np.quantile(rr, 0.1)), "q90": float(np.quantile(rr, 0.9)),
                                          "sum_aux_h": sum(v[0] for v in RES[(name, 65536, "main")].values()) / 3600}
SK = [n for n in CONF if n.startswith("K")]
BEST = min(SK, key=lambda n: R["ratio_aux_cpu_C64K_main"][n]["median"]); R["best_SK"] = BEST
# 内訳（S1 と最良）
R["breakdown_C64K_main"] = {n: {k: float(sum(v[1].get(k, 0) for v in RES[(n, 65536, "main")].values())) for k in ("move", "read", "write", "n_move", "n_fetch")} for n in ("S1", BEST)}
# 感度：楽観的な位置合わせ速度、ブロック間の隙間
SENS = {}
for n in ("S1", BEST):
    for lab, kw in [("v500", dict(v_sp=V_REW)), ("gap0.75", dict(gap=0.75))]:
        RES[(n, 65536, lab)] = run(CONF[n], 65536, **kw); SENS[f"{n}_{lab}"] = med_ratio(RES[(n, 65536, lab)])
R["ratio_sensitivity_C64K"] = SENS
# 他の C（ΔV 用）
for n in ("N0", "S1", BEST):
    for C in CGRID:
        if (n, C, "main") not in RES: RES[(n, C, "main")] = run(CONF[n], C)

# ---- 母集団（025 の高層）と q（026 と同じ式） ----
ok = [r for r in S["repos"] if r.get("status") == "ok"]; high = [r for r in ok if r["stratum"] == "high"]
bins = S["bins"]; nb = {b: sum(1 for r in high if r["bin"] == b) for b in bins}
w_cnt = np.array([bins[r["bin"]]["total_count_now"] / nb[r["bin"]] for r in high])
tar_h = np.array([float(r["tar_bytes"]) for r in high]); stars = np.array([float(r["stars_now"]) for r in high]); w_star = w_cnt * stars
EDGES = [0, 1e4, 1e5, 1e6, 4e6, 1.6e7, 4.1e7]
def cls(t): return min(int(np.searchsorted(EDGES, t, side="left")) - 1, len(EDGES) - 2)
samp_cls = np.array([cls(r["tar_bytes_tree_est"]) for r in reps]); pop_cls = np.array([cls(max(t, 1)) for t in tar_h])
def class_rate(vals):  # vals[i] = 部分標本 i の秒数 → 区分ごとのバイトあたり中央値
    return {k: float(np.median([vals[i] / reps[i]["out"] for i in range(len(reps)) if samp_cls[i] == k])) for k in range(len(EDGES) - 1)}
cpu_rate = class_rate([CPU[r["idx"]] for r in reps])
def repo_times(res, C):
    ar = class_rate([res[r["idx"]][0] for r in reps])
    cpu = np.array([cpu_rate[k] for k in pop_cls]) * tar_h
    aux = np.where(tar_h <= C, 0.0, np.array([ar[k] for k in pop_cls]) * tar_h)
    return cpu + aux
def q_budget(T, Y):
    o = np.argsort(-(stars / np.maximum(T, 1e-9))); cw = np.cumsum((w_cnt * T)[o]); ws = w_star[o]
    budget = Y * YEAR; full = cw <= budget
    q = ws[full].sum(); k = int(full.sum())
    if k < len(o):
        prev = cw[k - 1] if k > 0 else 0.0; q += ws[k] * max(0.0, (budget - prev) / (w_cnt[o][k] * T[o][k]))
    return float(q / w_star.sum())
YS = [1, 3, 10, 30, 100]
R["q"] = {}
for (n, C, lab), res in RES.items():
    T = repo_times(res, C)
    R["q"][f"{n}|{C}|{lab}"] = {"total_machine_years_all": float((w_cnt * T).sum() / YEAR), **{f"Y{y}": q_budget(T, y) for y in YS}}
R["q_026_tape_C64K_Y10"] = json.load(open(f"{R26}/results.json"))["q_main"]["tape"]["65536"]["Y10"]
R["q_C64K_Y10_main"] = {n: R["q"][f"{n}|65536|main"]["Y10"] for n in CONF}

# ---- ΔV（026 と同じ 024 モデル・同じ乱数系列）。読者 RAM logU[4,144] KiB ----
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
def q025(M):
    nd = S0 + np.minimum(8 * MiB, tar_h); o = np.argsort(nd); cw = np.cumsum(w_star[o]) / w_star.sum()
    k = np.searchsorted(nd[o], M, side="right"); return np.where(k > 0, cw[np.maximum(k - 1, 0)], 0.0)
rng3 = np.random.default_rng(20261026 + 1)   # 026 と同じ読者 RAM の標本
M = np.exp(rng3.uniform(np.log(4 * KiB), np.log(144 * KiB), N)); Cav = M - S0 - LRU_RAM
q_base = q025(M)
# 逸脱（report に記載）：トレースは距離 >16 KiB のみなので、C<16 KiB の読者は 025 の RAM のみの q とする（026 は C=0〜4 KiB のテープ q を使った）
R["dV_reader_4K_144K"] = {"ram_only_025": {"mean_q": float(q_base.mean()), **dV(q_base)}}
for n in ("N0", "S1", BEST):
    qv = q_base.copy()
    for C in CGRID: qv = np.where(Cav >= C, np.maximum(q_base, R["q"][f"{n}|{C}|main"]["Y10"]), qv)
    R["dV_reader_4K_144K"][f"{n}_Y10"] = {"mean_q": float(qv.mean()), "frac_readers_Cav_ge_16K": float((Cav >= 16 * KiB).mean()), **dV(qv)}
# 参考：026 の格子のままの値（026 の C=0〜4 KiB を含む）
R["dV_026_tape_reported"] = json.load(open(f"{R26}/results.json"))["dV_reader_4K_144K"]["aux_tape_Y10"]

RA = R["ratio_aux_cpu_C64K_main"]; Q = R["q_C64K_Y10_main"]; DV = R["dV_reader_4K_144K"]
H39 = {
 "H39a": {"value": RA["S1"]["median"] / RA["N0"]["median"], "rule": "median(S1)/median(N0) >= 2", "supported": RA["S1"]["median"] / RA["N0"]["median"] >= 2},
 "H39b": {"value": RA[BEST]["median"] / RA["S1"]["median"], "rule": "best SK / S1 <= 1/3", "supported": RA[BEST]["median"] / RA["S1"]["median"] <= 1 / 3},
 "H39c": {"value": Q["S1"], "rule": "q(S1) < 0.45", "supported": Q["S1"] < 0.45},
 "H39d": {"value": Q[BEST], "rule": "q(best SK) >= 0.54", "supported": Q[BEST] >= 0.54},
 "H39e": {"value": R["validation_N0"]["rel_diff"], "rule": "|N0 trace / 026 - 1| <= 0.05", "supported": abs(R["validation_N0"]["rel_diff"]) <= 0.05},
 "H39f": {"value": DV[f"{BEST}_Y10"]["dV_P2_tablet"], "rule": "dV_P2(best SK) >= 0.06", "supported": DV[f"{BEST}_Y10"]["dV_P2_tablet"] >= 0.06},
}
R["H39"] = H39

# ---- 図 ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
mat = np.array([[RA[f"K{K}_L{L // KiB}K"]["median"] for L in LS] for K in KS])
im = ax[0].imshow(np.log10(mat), cmap="viridis", aspect="auto")
ax[0].set_xticks(range(len(LS))); ax[0].set_xticklabels([f"{L // KiB}K" for L in LS]); ax[0].set_yticks(range(len(KS))); ax[0].set_yticklabels([f"K={K}" for K in KS])
for a in range(len(KS)):
    for b in range(len(LS)): ax[0].text(b, a, f"{mat[a, b]:.0f}", ha="center", va="center", color="w", fontsize=9)
ax[0].set_xlabel("segment length L"); ax[0].set_title(f"median tape/CPU time, C=64KiB (S1={RA['S1']['median']:.0f}, N0={RA['N0']['median']:.0f})")
plt.colorbar(im, ax=ax[0], label="log10 ratio")
names = ["N0", "S1", BEST]; cols = ["gray", "tab:red", "tab:blue"]
for n, c in zip(names, cols):
    ax[1].plot(YS, [R["q"][f"{n}|65536|main"][f"Y{y}"] for y in YS], "-o", color=c, label=n)
for n, c in zip(["S1", BEST], ["tab:red", "tab:blue"]):
    ax[1].plot(YS, [R["q"][f"{n}|65536|v500"][f"Y{y}"] for y in YS], ":", color=c, label=f"{n}, spacing 500 ips")
ax[1].set_xscale("log"); ax[1].set_ylim(0, 1); ax[1].set_xlabel("decode budget Y (machine-years)"); ax[1].set_ylabel("q (star weight, >=250 stars)")
ax[1].legend(fontsize=8); ax[1].set_title("q with window on IBM 727-class tape, C=64KiB")
plt.tight_layout(); plt.savefig(f"{HERE}/fig_tape_procedures_q.png", dpi=120); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
bd = R["breakdown_C64K_main"]
for j, n in enumerate(("S1", BEST)):
    b = bd[n]; ax[0].bar([j - 0.2, j, j + 0.2], [b["move"] / 3600, b["read"] / 3600, b["write"] / 3600], width=0.2, color=["tab:orange", "tab:green", "tab:purple"])
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["S1", BEST]); ax[0].set_yscale("log"); ax[0].set_ylabel("hours, 30 repos (move / read / write)"); ax[0].set_title("where tape time goes")
kk = list(DV.keys()); ax[1].bar(range(len(kk)), [DV[k]["dV_P2_tablet"] for k in kk], color="tab:cyan")
ax[1].axhline(R["dV_026_tape_reported"]["dV_P2_tablet"], color="k", ls=":", label="026 tape (reported)")
ax[1].axhline(R["check_024_q0.3"]["dV_P2_tablet"], color="r", ls="--", label="024 (q=0.3)")
ax[1].set_xticks(range(len(kk))); ax[1].set_xticklabels(kk, rotation=20, fontsize=8); ax[1].set_ylabel("dV (tablet guide)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_breakdown_dV.png", dpi=120); plt.close()
R["runtime_seconds"] = round(time.time() - t0, 2)
json.dump(R, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: R[k] for k in ["validation_lru_identical_026", "validation_N0", "ratio_aux_cpu_C64K_main", "best_SK", "breakdown_C64K_main", "ratio_sensitivity_C64K",
                  "q_C64K_Y10_main", "q_026_tape_C64K_Y10", "dV_reader_4K_144K", "dV_026_tape_reported", "H39", "runtime_seconds"]}, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in R["q"].items() if "|65536|" in k}, indent=1))
