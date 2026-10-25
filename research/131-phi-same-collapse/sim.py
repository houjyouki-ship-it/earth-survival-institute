"""研究131：phi（段を免れた標本の停滞の倍率）の格子で F_heavy を出す。130 の explore_kdelay.py と同じ仕組み。
注：F_heavy・V は生存確率や復興確率ではない。再現：phi=0・0.5・1 で 130 の explore_kdelay.json と差 0（assert）。"""
import json, time, os
import numpy as np
t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
P127 = os.path.join(RES, "127-stag-corr-horizon", "sim.py")
src = open(P127, encoding="utf8").read(); body = src[:src.index("\nSL = \"slow_Tasmania\"")]
G = {"__file__": P127, "__name__": "sim127"}
exec(compile(body, P127, "exec"), G)
E130 = json.load(open(os.path.join(RES, "130-kept-stagnation", "explore_kdelay.json"), encoding="utf8"))["OUT"]
CJ = json.load(open(os.path.join(HERE, "cases.json"), encoding="utf8"))
stag_of, lost, KAPS, LK, r_pop = (G[k] for k in ("stag_of", "lost", "KAPS", "LK", "r_pop"))
T2o = G["T2_stag"]; FIXED = G["FIXED"]
PHI = [0.0]
def T2x(k, st, stag, r):
    t = T2o(k, st, stag, r)
    if k in FIXED: return t
    return t + np.where((st == 2) | (st == 3), PHI[0] * stag, 0.0)
G["T2_stag"] = T2x
SL = "slow_Tasmania"; MAIN3 = ("D2rz|none", "D1|main", "D4|main"); PK = 1/6
named = {"common_start(France)": CJ["phi_common_start"], "new_median_peak": CJ["phi_new_median_peak"],
         "all_median_peak": CJ["phi_all_median_peak"], "Italy_peak": CJ["pairs_peak"]["Italy"]}
grid = sorted(set([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5] + list(named.values())))
OUT = {}
S = stag_of(lambda k: np.where(lost[k], 0.9, PK)); vb = G["base_val_k"](S, r_pop)
for phi in grid:
    PHI[0] = phi
    for vn in MAIN3: OUT[f"{phi:.6f}|{vn}"] = G["run_k"](KAPS[SL], LK[vn], S, r_pop, False, vb)
    print(round(phi, 4), {vn[:4]: round(OUT[f'{phi:.6f}|{vn}']['F_heavy'], 4) for vn in MAIN3}, flush=True)
chk = {}
for phi, key130 in ((0.0, "0.0|0.0000"), (0.5, "0.5|0.1667"), (1.0, "1.0|0.1667")):
    for vn in MAIN3:
        d = abs(OUT[f"{phi:.6f}|{vn}"]["F_heavy"] - E130[f"{key130}|{vn}"]["F_heavy"]); chk[f"{phi}|{vn}"] = d; assert d < 1e-12, (phi, vn, d)
NAMED = {n: {vn: OUT[f"{p:.6f}|{vn}"]["F_heavy"] for vn in MAIN3} | {"phi": p} for n, p in named.items()}
# phi で F_heavy が 1（写しの利得ゼロ）を割る点（線形補間）
cross = {}
for vn in MAIN3:
    xs = grid; ys = [OUT[f"{p:.6f}|{vn}"]["F_heavy"] for p in xs]
    cross[vn] = next((xs[i] + (1 - ys[i]) * (xs[i+1] - xs[i]) / (ys[i+1] - ys[i]) for i in range(len(xs) - 1) if ys[i] >= 1 > ys[i+1]), None)
json.dump({"note": "F_heavy・V は生存確率や復興確率ではない", "grid": grid, "OUT": OUT, "named": NAMED, "cross_F1": cross,
           "check_vs_130": chk, "runtime_s": time.time() - t0}, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4.2))
for vn, c in zip(MAIN3, ("C0", "C1", "C2")):
    ax.plot(grid, [OUT[f"{p:.6f}|{vn}"]["F_heavy"] for p in grid], "o-", color=c, label=vn.split("|")[0])
for n, p in named.items(): ax.axvline(p, color="gray", ls=":", lw=0.8); ax.text(p, ax.get_ylim()[1] if False else 2.25, n, rotation=90, fontsize=7, va="top")
ax.axhline(1, color="k", lw=0.6); ax.set_xlabel("phi (stagnation of copy-protected samples / lost)"); ax.set_ylabel("F_heavy (slow kappa, p|lost 0.9)")
ax.set_title("131: F_heavy vs phi (not a survival probability)"); ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_phi.png"), dpi=120)
print("cross_F1", cross, "runtime", round(time.time() - t0))
