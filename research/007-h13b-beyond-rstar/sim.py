"""007: r>r*・恒久乗法での H13 再検定（H13b）＋人口回復の人口学的制約（H15）＋臨界致死割合（H16）

再現: python3 sim.py > stdout.txt  (seed=20261014) -> results.json, fig_*.png

背景: 研究006は r∈{10,20}≪r*∈(50,75) で非致死δが S(1e4) をむしろ上げたため H13 の前提が空だった。
また 006 は致死打撃後の人口回復時定数に 005 の「作物収量回復」r_yield(4–22年) を流用しており、
人口の再増殖速度（出生による）を制約していなかった。

事前登録仮説（本回の実行前に本ファイルへ記載。第三者のタイムスタンプはない）:
 H13b【独自仮説】: 非致死δのみ（φ=0）で S_joint(1e4) がショックなし基線より有意に下がる格子
      （drop_nl > 2·SE）では、Xia 型重症度・φ∈{0.1,0.3}・全人口回復モデルで、
      致死追加低下 addon = S_joint(φ=0) − S_joint(φ) ≤ drop_nl が成り立つ。予測: 全セルで成立。
 H15【独自仮説】: 人口回復をロジスティック成長 g=0.023/年（OWID: 世界人口増加率の歴史的ピーク2.3%/年, 1963）
      に置き換えると、λ0=1・φ=0.3 の P(E_collapse) は 006型（収量回復）の10倍以上になる。
      一方 P(E_extinct) は λ0≤3・φ≤0.3・Xia重症度の全条件で 1e-3 未満にとどまる。
 H16【独自仮説】: 代表セル（恒久乗法, λ0=1, δ=2, factor, φ=0.3, g=0.023）で、1回あたり致死割合 D を固定して
      走査したとき、絶滅経由の addon が drop_nl に達する臨界 D* は 0.999 を超える。

モデル: 研究006と同じ Ord 要因・事前登録 δ_k 倍率・型混合・減衰ショック λ(t)=λ0 e^{-t/τ}（τ=200）。
 ハザード回復: 加法 M_k=1+Σ δ m e^{-(t-s)/r}（r∈{20,50,75,100,200,∞}）、
               恒久乗法 M_k=Π(1+δ m)（mult_perm）。各セルで E[S(100)]=5/6 に再較正。
 人口: yield（006と同一: 打撃後 r_yield で P0へ指数回復）／logistic g∈{0.01,0.023,0.04}。
 終端: S_joint = E[exp(-Cum)·1{P≥5000}], S_strict = E[exp(-Cum)·1{P≥0.1P0 を一度も割らない}]。
 共通乱数: ショック時刻・型・重症度・致死一様乱数 U を λ0 ごとに1回引き、φ間で lethal={U<φ} を入れ子にする。
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20261014
OUT = Path(__file__).resolve().parent
TAU = 200.0
H_NAT = 1 / 87_000
ORD_TOTAL = 1 / 6
P0 = 6_700_000_000
N = 20_000
T = 10_000.0
EXT = 5000.0
COL = 0.1 * P0

ORD_KEYS = ["unaligned_AI", "engineered_pandemics", "unforeseen_anthropogenic",
            "other_anthropogenic", "nuclear_war", "climate_change", "other_environmental"]
ORD = [1/10, 1/30, 1/30, 1/50, 1/1000, 1/1000, 1/1000]
F = {k: i for i, k in enumerate(ORD_KEYS)}
NF = len(ORD_KEYS)
MULT = {  # 研究006の事前登録値をそのまま使用
    "war": {"nuclear_war": 1.0, "climate_change": 0.4, "other_environmental": 0.3,
            "unaligned_AI": 0.15, "engineered_pandemics": 0.15,
            "unforeseen_anthropogenic": 0.2, "other_anthropogenic": 0.2},
    "pandemic": {"engineered_pandemics": 1.0, "unaligned_AI": 0.25,
                 "unforeseen_anthropogenic": 0.3, "nuclear_war": 0.1,
                 "climate_change": 0.1, "other_environmental": 0.1, "other_anthropogenic": 0.2},
    "tech_cascade": {"unaligned_AI": 1.0, "unforeseen_anthropogenic": 0.5,
                     "other_anthropogenic": 0.4, "engineered_pandemics": 0.2,
                     "nuclear_war": 0.15, "climate_change": 0.1, "other_environmental": 0.1},
}
TYPE_NAMES = ["war", "pandemic", "tech_cascade"]
TYPE_PROBS = np.array([0.4, 0.3, 0.3])
M_MAT = np.zeros((3, NF))
for ti, tn in enumerate(TYPE_NAMES):
    for k, v in MULT[tn].items():
        M_MAT[ti, F[k]] = v

DEAD_FRAC = np.array([0.03805970149253731, 0.1382089552238806, 0.2128358208955224,
                      0.3105970149253731, 0.37492537313432833, 0.7971641791044776])
R_YIELD = np.array([4.35, 8.5, 13.8, 18.9, 21.95, 18.0])
SEV_W = np.array([0.05, 0.10, 0.25, 0.25, 0.20, 0.15])
DELTA_FRAG_150 = 3.9300956585724784

LAM0S = [0.3, 1.0, 3.0]
DELTAS = [1.0, 2.0, DELTA_FRAG_150]
RECOVS = ["add_r20", "add_r50", "add_r75", "add_r100", "add_r200", "add_rinf", "mult_perm"]
PHIS = [0.0, 0.1, 0.3]
POPS = ["yield", "logit_g0.01", "logit_g0.023", "logit_g0.04"]
D_SCAN = [0.7971641791044776, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]

t0 = time.time()


def draw(lam0, rng):
    lam = lam0 / 100
    Lam = lam * TAU * (-math.expm1(-T / TAU))
    k = rng.poisson(Lam, size=N)
    m = max(int(k.max()), 1)
    u = rng.uniform(size=(N, m))
    t = -TAU * np.log1p(-u * (-math.expm1(-T / TAU)))
    t = np.where(np.arange(m)[None, :] < k[:, None], t, np.inf)
    s = np.sort(t, axis=1)
    types = np.where(np.isfinite(s), rng.choice(3, size=s.shape, p=TYPE_PROBS), -1)
    sev = rng.choice(6, size=s.shape, p=SEV_W)
    U = rng.uniform(size=s.shape)
    return s, types, sev, U


def deltas(types, d):
    out = np.zeros(types.shape + (NF,))
    for ti in range(3):
        out[types == ti] = d * M_MAT[ti]
    return out


def I_factor(s, dk, recov, Th):
    """∫_0^Th e^{-t/τ} M_k(t) dt for one factor; dk:(n,m) per-shock δ·m."""
    if recov == "mult_perm":
        n, m = s.shape
        sc = np.minimum(s, Th)
        a = np.concatenate([np.zeros((n, 1)), sc], axis=1)
        b = np.concatenate([sc, np.full((n, 1), Th)], axis=1)
        f = np.where(np.isfinite(s) & (s < Th), 1.0 + dk, 1.0)
        C = np.concatenate([np.ones((n, 1)), np.cumprod(f, axis=1)], axis=1)
        seg = TAU * (np.exp(-a / TAU) - np.exp(-b / TAU))
        return (C * seg).sum(axis=1)
    r = float(recov.split("_r")[1]) if recov != "add_rinf" else math.inf
    base = TAU * (-math.expm1(-Th / TAU))
    sj = np.where(np.isfinite(s) & (s < Th), s, np.nan)
    if math.isinf(r):
        c = TAU * (np.exp(-sj / TAU) - math.exp(-Th / TAU))
    else:
        kk = 1 / TAU + 1 / r
        c = np.exp(sj / r) * (np.exp(-kk * sj) - math.exp(-kk * Th)) / kk
    return base + np.nansum(dk * c, axis=1)


def bisect(fun, target):
    lo, hi = 0.0, 1.0
    while fun(hi) < target and hi < 1e15:
        hi *= 2
    for _ in range(64):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fun(mid) < target else (lo, mid)
    return (lo + hi) / 2


def cum_hazard(s, dl, recov):
    """Calibrate B_k to Ord marginals and α to 1/6 at T=100; return Cum(1e4) and Cum(100)."""
    I100 = [I_factor(s, dl[:, :, k], recov, 100.0) for k in range(NF)]
    Bk = np.array([bisect(lambda b, I=I: 1 - np.mean(np.exp(-b * I)), ORD[k]) for k, I in enumerate(I100)])
    C100 = sum(Bk[k] * I100[k] for k in range(NF))
    alpha = bisect(lambda a: 1 - np.mean(np.exp(-a * C100 - 100 * H_NAT)), ORD_TOTAL)
    C1e4 = alpha * sum(Bk[k] * I_factor(s, dl[:, :, k], recov, T) for k in range(NF))
    return C1e4, alpha * C100


def pop_sim(s, lethal, sev, model, D_fixed=None):
    n, m = s.shape
    P = np.full(n, float(P0))
    ext = np.zeros(n, bool)
    col = np.zeros(n, bool)
    tp = np.zeros(n)
    rcur = np.full(n, 15.0)
    g = float(model.split("_g")[1]) if model.startswith("logit") else None
    for j in range(m):
        sj = s[:, j]
        act = np.isfinite(sj) & (sj < T) & ~ext
        if not act.any():
            continue
        dt = sj[act] - tp[act]
        if g is None:
            P[act] = P[act] + (P0 - P[act]) * (1 - np.exp(-dt / rcur[act]))
        else:
            P[act] = P0 / (1 + (P0 / P[act] - 1) * np.exp(-g * dt))
        hit = act & lethal[:, j]
        if hit.any():
            fi = sev[hit, j]
            dfrac = DEAD_FRAC[fi] if D_fixed is None else D_fixed
            P[hit] *= (1 - dfrac)
            rcur[hit] = R_YIELD[fi]
            ext[hit] |= P[hit] < EXT
            col[hit] |= P[hit] < COL
        tp[act] = sj[act]
    return ext, col


# analytic no-shock baseline
I0 = lambda Th: TAU * (-math.expm1(-Th / TAU))
B0 = (-math.log(5 / 6) - 100 * H_NAT) / I0(100)
S_NOSHOCK = math.exp(-B0 * I0(T) - H_NAT * T)

res = {"seed": SEED, "study": "007-h13b-beyond-rstar", "N": N, "tau": TAU, "T": T,
       "S_noshock_1e4": S_NOSHOCK,
       "preregistered": {
           "H13b": "drop_nl>2SE のセルで、Xia重症度・φ∈{0.1,0.3}・全人口モデルにつき addon ≤ drop_nl（予測: 全セル成立）",
           "H15": "logistic g=0.023 で λ0=1, φ=0.3 の P(E_collapse) が yield の≥10倍、かつ P(E_extinct)<1e-3（λ0≤3, φ≤0.3, Xia）",
           "H16": "恒久乗法 λ0=1 δ=2 φ=0.3 g=0.023 で addon=drop_nl となる臨界 D* > 0.999",
       },
       "pop_growth_source": {"url": "https://ourworldindata.org/population-growth",
                             "fact": "世界人口増加率は1963年に2.3%/年でピーク（OWID図注記）。g=0.01, 0.04 は感度用の【推測】"},
       }

pop_out, haz_out, h13b = {}, {}, []
cache = {}
for lam0 in LAM0S:
    rng = np.random.default_rng([SEED, int(lam0 * 1000), 7])
    s, types, sev, U = draw(lam0, rng)
    valid = np.isfinite(s)
    cache[lam0] = (s, types, sev, U)
    # population outcomes (independent of hazard model)
    pops = {}
    for phi in PHIS:
        lethal = (U < phi) & valid
        for pm in POPS:
            ext, col = pop_sim(s, lethal, sev, pm)
            pops[(phi, pm)] = (ext, col)
            pop_out[f"lam{lam0}_phi{phi}_{pm}"] = {"P_extinct": float(ext.mean()),
                                                   "P_collapse": float(col.mean())}
    for d in DELTAS:
        dl = deltas(types, d)
        for rc in RECOVS:
            C, C100 = cum_hazard(s, dl, rc)
            w = np.exp(-C - H_NAT * T)
            S100 = float(np.mean(np.exp(-C100 - 100 * H_NAT)))
            S0 = float(w.mean())
            se0 = float(w.std(ddof=1) / math.sqrt(N))
            drop_nl = S_NOSHOCK - S0
            key = f"lam{lam0}_d{d:.3g}_{rc}"
            cell = {"lam0": lam0, "delta": d, "recov": rc, "S100_check": S100,
                    "S_ord_1e4": S0, "SE": se0, "drop_nl": drop_nl,
                    "nl_drops_significantly": bool(drop_nl > 2 * se0), "addon": {}}
            for phi in PHIS[1:]:
                for pm in POPS:
                    ext, col = pops[(phi, pm)]
                    addon = float((w * ext).mean())        # S0 - S_joint(φ)
                    addon_strict = float((w * col).mean())  # S0 - S_strict(φ)
                    cell["addon"][f"phi{phi}_{pm}"] = {"addon_extinct": addon,
                                                       "addon_collapse": addon_strict}
                    if cell["nl_drops_significantly"]:
                        h13b.append({"key": key, "phi": phi, "pop": pm, "drop_nl": drop_nl,
                                     "addon": addon, "ok": bool(addon <= drop_nl),
                                     "addon_collapse": addon_strict,
                                     "ok_if_collapse_counts": bool(addon_strict <= drop_nl)})
            haz_out[key] = cell
    print(f"lam0={lam0} done t={time.time()-t0:.1f}s", flush=True)

res["population"] = pop_out
res["hazard_cells"] = haz_out

# r* per (λ0, δ): first additive r where S_ord(1e4) < S_noshock
rstar = {}
for lam0 in LAM0S:
    for d in DELTAS:
        seq = [(rc, haz_out[f"lam{lam0}_d{d:.3g}_{rc}"]["drop_nl"]) for rc in RECOVS]
        add = [(rc, x) for rc, x in seq if rc.startswith("add")]
        bracket = None
        for (r1, x1), (r2, x2) in zip(add, add[1:]):
            if x1 <= 0 < x2:
                bracket = [r1, r2]
                break
        rstar[f"lam{lam0}_d{d:.3g}"] = {"bracket": bracket,
                                        "drop_nl_by_recov": {rc: x for rc, x in seq}}
res["r_star"] = rstar

n_sig = len(h13b)
n_ok = sum(h["ok"] for h in h13b)
n_ok_col = sum(h["ok_if_collapse_counts"] for h in h13b)
res["H13b"] = {"n_tests": n_sig, "n_ok": n_ok, "n_ok_if_collapse_counts": n_ok_col,
               "n_cells_nl_drops": len({h['key'] for h in h13b}),
               "n_cells_total": len(haz_out),
               "max_ratio_addon_over_drop": max((h["addon"] / h["drop_nl"] for h in h13b), default=None),
               "max_ratio_collapse_addon_over_drop": max((h["addon_collapse"] / h["drop_nl"] for h in h13b), default=None),
               "verdict": ("支持" if n_sig > 0 and n_ok == n_sig else ("検定不能" if n_sig == 0 else "棄却")),
               "violations": [h for h in h13b if not h["ok"]][:20],
               "collapse_violations_sample": [h for h in h13b if not h["ok_if_collapse_counts"]][:10]}

# H15
pc_y = pop_out["lam1.0_phi0.3_yield"]["P_collapse"]
pc_l = pop_out["lam1.0_phi0.3_logit_g0.023"]["P_collapse"]
max_ext = max(v["P_extinct"] for v in pop_out.values())
ratio = (pc_l / pc_y) if pc_y > 0 else (math.inf if pc_l > 0 else None)
res["H15"] = {"P_collapse_yield": pc_y, "P_collapse_logit023": pc_l, "ratio": ratio,
              "max_P_extinct_all": max_ext,
              "part1_collapse_x10": bool(ratio is not None and ratio >= 10),
              "part2_extinct_lt_1e-3": bool(max_ext < 1e-3)}
res["H15"]["verdict"] = ("支持" if res["H15"]["part1_collapse_x10"] and res["H15"]["part2_extinct_lt_1e-3"]
                         else "部分支持" if (res["H15"]["part1_collapse_x10"] or res["H15"]["part2_extinct_lt_1e-3"])
                         else "棄却")

# H16: fixed per-shock death fraction scan
h16 = {}
for lam0 in [1.0, 3.0]:
    s, types, sev, U = cache[lam0]
    lethal = (U < 0.3) & np.isfinite(s)
    for rc in ["mult_perm", "add_r200"]:
        dl = deltas(types, 2.0)
        C, _ = cum_hazard(s, dl, rc)
        w = np.exp(-C - H_NAT * T)
        drop_nl = S_NOSHOCK - float(w.mean())
        rows = []
        for pm in ["logit_g0.023", "logit_g0.01"]:
            for D in D_SCAN:
                ext, col = pop_sim(s, lethal, sev, pm, D_fixed=D)
                rows.append({"pop": pm, "D": D, "P_extinct": float(ext.mean()),
                             "addon": float((w * ext).mean())})
        dstar = {}
        for pm in ["logit_g0.023", "logit_g0.01"]:
            rr = [x for x in rows if x["pop"] == pm]
            hit = [x["D"] for x in rr if x["addon"] >= drop_nl]
            dstar[pm] = min(hit) if hit else None
        h16[f"lam{lam0}_{rc}_d2"] = {"drop_nl": drop_nl, "scan": rows, "D_star": dstar}
res["H16"] = {"cells": h16}
ds = h16["lam1.0_mult_perm_d2"]["D_star"]["logit_g0.023"]
res["H16"]["D_star_primary"] = ds
res["H16"]["verdict"] = ("支持" if (ds is None or ds > 0.999) else "棄却")
res["H16"]["note"] = "D_star=None は走査範囲（≤0.999999）で addon が drop_nl に届かないことを意味する"
res["runtime_seconds"] = time.time() - t0

with open(OUT / "results.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False, default=float)

# ---------------- figures ----------------
xs = range(len(RECOVS))
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, lam0 in zip(axes, LAM0S):
    for d, mk in zip(DELTAS, "osd"):
        ys = [haz_out[f"lam{lam0}_d{d:.3g}_{rc}"]["S_ord_1e4"] for rc in RECOVS]
        ax.plot(xs, ys, "-" + mk, label=f"δ={d:.3g} φ=0")
        ys3 = [haz_out[f"lam{lam0}_d{d:.3g}_{rc}"]["S_ord_1e4"]
               - haz_out[f"lam{lam0}_d{d:.3g}_{rc}"]["addon"]["phi0.3_logit_g0.023"]["addon_collapse"]
               for rc in RECOVS]
        ax.plot(xs, ys3, ":" + mk, alpha=0.6, label=f"δ={d:.3g} φ=0.3 S_strict(g=.023)")
    ax.axhline(S_NOSHOCK, color="k", ls="--", lw=1, label="no shock")
    ax.set_xticks(list(xs)); ax.set_xticklabels([r.replace("add_", "") for r in RECOVS], rotation=40, fontsize=8)
    ax.set_title(f"λ0={lam0}/century"); ax.set_xlabel("hazard recovery")
axes[0].set_ylabel("S(10,000 y)"); axes[0].legend(fontsize=6)
plt.suptitle("007: non-lethal S_ord(1e4) vs recovery (r* = crossing of dashed line); dotted = collapse counted")
plt.tight_layout(); plt.savefig(OUT / "fig_S_vs_recovery.png", dpi=120); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for key, mk in zip(h16, "osd^"):
    for pm, ls in [("logit_g0.023", "-"), ("logit_g0.01", "--")]:
        rr = [x for x in h16[key]["scan"] if x["pop"] == pm]
        xx = [1 - x["D"] for x in rr]
        axes[0].loglog(xx, [max(x["P_extinct"], 1e-6) for x in rr], ls + mk, label=f"{key} {pm}", ms=4)
        axes[1].loglog(xx, [max(x["addon"], 1e-7) for x in rr], ls + mk, ms=4)
    axes[1].axhline(h16[key]["drop_nl"], color="gray", lw=0.6)
for ax in axes:
    ax.invert_xaxis(); ax.set_xlabel("survivor fraction per lethal shock (1−D)")
axes[0].set_ylabel("P(E_extinct)"); axes[1].set_ylabel("lethal add-on to S_joint(1e4)")
axes[1].set_title("gray = non-lethal drop (H16 threshold)")
axes[0].legend(fontsize=6)
plt.tight_layout(); plt.savefig(OUT / "fig_Dstar_scan.png", dpi=120); plt.close()

print("=== 007 ===")
print(f"seed={SEED} N={N} runtime={res['runtime_seconds']:.1f}s S_noshock={S_NOSHOCK:.4f}")
for k, v in rstar.items():
    print("r*", k, v["bracket"], {rc: round(x, 4) for rc, x in v["drop_nl_by_recov"].items()})
print("H13b", {k: v for k, v in res["H13b"].items() if k not in ("violations", "collapse_violations_sample")})
print("H15", res["H15"])
for k, v in h16.items():
    print("H16", k, "drop_nl=%.4f" % v["drop_nl"], v["D_star"],
          [(x["pop"][-5:], x["D"], x["P_extinct"], round(x["addon"], 5)) for x in v["scan"]])
print("H16 verdict", res["H16"]["verdict"])
for k, v in pop_out.items():
    if v["P_collapse"] > 0 or v["P_extinct"] > 0:
        print("pop", k, v)
