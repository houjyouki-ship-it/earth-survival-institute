"""研究132：stag と phi を「谷からの遅れ」で揃えた F_heavy（open_problems 307）。
注：F_heavy・V は生存確率や復興確率ではない。事前登録 PREREGISTRATION.md（05:48 JST）。
127 の sim.py の前半を exec（126 の assert 通過）、VA0 を 131 の results.json（phi=0）と差 0 で再現（assert）。"""
import json, time, os, math
import numpy as np
t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
P127 = os.path.join(RES, "127-stag-corr-horizon", "sim.py")
src = open(P127, encoding="utf8").read(); body = src[:src.index("\nSL = \"slow_Tasmania\"")]
G = {"__file__": P127, "__name__": "sim127"}
exec(compile(body, P127, "exec"), G)
print("prefix", round(time.time() - t0), "s", flush=True)
R131 = json.load(open(os.path.join(RES, "131-phi-same-collapse", "results.json"), encoding="utf8"))
CJ = json.load(open(os.path.join(RES, "131-phi-same-collapse", "cases.json"), encoding="utf8"))
EX = json.load(open(os.path.join(RES, "131-phi-same-collapse", "explore_posttrough.json"), encoding="utf8"))
r = CJ["R_MED"]
stag_of, lost, KAPS, LK, r_pop, COMPS, N = (G[k] for k in ("stag_of", "lost", "KAPS", "LK", "r_pop", "COMPS", "N"))
T2o = G["T2_stag"]; FIXED = G["FIXED"]
A_vals = np.asarray(G["V2"]["stag_years_no_immig"], float)
lag_G = 300 - math.log(6e5 / 3e5) / r; lag_B = EX["lag_years"]["Britain"]
B_vals = np.array([lag_G, lag_B, 650.0])
assert abs(A_vals[0] - lag_G - 200) < 1e-6 and abs(A_vals[1] - lag_B - 200) < 1e-6
idx = np.minimum((G["U_pick"] * 3).astype(int), 2)
assert np.array_equal(G["pick"], A_vals[idx])
PHI = [0.0]; XS = [None]
def T2x(k, st, stag, r_):
    t = T2o(k, st, stag, r_)
    if k in FIXED: return t
    return t + np.where((st == 2) | (st == 3), PHI[0] * XS[0][k], 0.0)
G["T2_stag"] = T2x
SL = "slow_Tasmania"; MAIN3 = ("D2rz|none", "D1|main", "D4|main"); PK = 1/6
def S_with(vals):
    G["pick"] = vals[idx]; S = stag_of(lambda k: np.where(lost[k], 0.9, PK)); G["pick"] = A_vals[idx]; return S
SA, SB = S_with(A_vals), S_with(B_vals)
ph_mean = float(np.mean(list(EX["phi"].values()))); ph_it = EX["phi"]["Italy"]
VERS = {"VA0": (SA, SA, 0.0), "VB0": (SB, SB, 0.0), "VB_mean": (SB, SB, ph_mean), "VB_it": (SB, SB, ph_it),
        "VAB_mean": (SA, SB, ph_mean), "VAB_it": (SA, SB, ph_it)}
OUT = {}
for v, (S, X, ph) in VERS.items():
    PHI[0] = 0.0; XS[0] = X; vb = G["base_val_k"](S, r_pop)
    PHI[0] = ph
    OUT[v] = {"phi": ph, **{vn: G["run_k"](KAPS[SL], LK[vn], S, r_pop, False, vb) for vn in MAIN3}}
    print(v, round(ph, 3), {vn[:4]: round(OUT[v][vn]["F_heavy"], 4) for vn in MAIN3}, flush=True)
ref = R131["OUT"][f"{0.0:.6f}|D2rz|none"]["F_heavy"]
d = abs(OUT["VA0"]["D2rz|none"]["F_heavy"] - ref); assert d < 1e-12, d
for vn in MAIN3:
    assert abs(OUT["VA0"][vn]["F_heavy"] - R131["OUT"][f"{0.0:.6f}|{vn}"]["F_heavy"]) < 1e-12
FH = {v: {vn: OUT[v][vn]["F_heavy"] for vn in MAIN3} for v in OUT}
Fa = FH["VA0"]["D2rz|none"]
H = {"H144a": FH["VB0"]["D2rz|none"] < 0.95 * Fa,
     "H144b": all(FH[v]["D2rz|none"] >= 1.6 for v in ("VB0", "VB_mean", "VB_it")),
     "H144c": all(FH[v]["D2rz|none"] >= FH[v]["D1|main"] >= FH[v]["D4|main"] for v in FH),
     "H144d": FH["VB_it"]["D2rz|none"] <= 0.9 * FH["VB0"]["D2rz|none"]}
json.dump({"note": "F_heavy・V は生存確率や復興確率ではない", "B_vals": B_vals.tolist(), "A_vals": A_vals.tolist(),
           "F_heavy": FH, "rel_to_VA0_D2rz": {v: FH[v]["D2rz|none"] / Fa - 1 for v in FH}, "hyp": H,
           "check_VA0_vs_131": d, "OUT": OUT, "runtime_s": time.time() - t0}, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.5, 4)); vv = list(FH); x = np.arange(len(vv))
for i, (vn, c) in enumerate(zip(MAIN3, ("C0", "C1", "C2"))):
    ax.bar(x + (i - 1) * 0.27, [FH[v][vn] for v in vv], 0.27, color=c, label=vn.split("|")[0])
ax.axhline(1, color="k", lw=0.6); ax.axhspan(1.27, 1.43, color="gray", alpha=0.2, label="131 peak-rule range")
ax.set_xticks(x); ax.set_xticklabels(vv); ax.set_ylabel("F_heavy (slow kappa, p|lost 0.9)")
ax.set_title("132: stag & phi on the trough-lag definition (not a survival probability)", fontsize=9); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_trough_lag.png"), dpi=120)
print(json.dumps(H), "runtime", round(time.time() - t0))
